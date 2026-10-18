use std::collections::HashMap;

use clap::Parser;
use ndarray::ArrayD;
use tch::{nn, Device, Kind, Tensor};

use app::models::universeg::UniverSeg;
use app::models::Segmenter;
use app::pipelines::base::{
    finalize, run_crf, run_mc_dropout, run_noisy, run_normal, run_tta, save_array,
    save_sample_outputs, SampleData, UncertaintyOutput,
};
use app::tensor_util::{
    load_g_channel_tensor, load_mask_array, load_mask_tensor, resize_nearest, tensor_to_array,
};
use tfm_core::config::{FusionConfig, PipelineConfig};
use tfm_core::dataset::{load_test_indices, recover_image_mask_pairs};
use tfm_core::fusion::weighted_average_with_uncertainty;
use tfm_core::metrics::{certainty_score, compute_dice, compute_iou, compute_metrics};

#[derive(Parser)]
struct Args {
    #[clap(long, default_value = "./configs/foundation_universeg.yaml")]
    config: String,
    #[clap(long, default_value = "64")]
    context_size: usize,
    #[clap(long, default_value = "test_indices.json")]
    test_indices: String,
    #[clap(long)]
    skip_support: bool,
}

const INPUT_SIZE: (i64, i64) = (128, 128);

fn metric_map(iou: f64, dice: f64, cert: f64, m: &tfm_core::metrics::Metrics) -> HashMap<String, f64> {
    let mut map = HashMap::new();
    map.insert("iou".to_string(), iou);
    map.insert("dice".to_string(), dice);
    map.insert("certainty".to_string(), cert);
    map.insert("nll".to_string(), m.nll);
    map.insert("nll_fg".to_string(), m.nll_fg);
    map.insert("ece".to_string(), m.ece);
    map.insert("brier".to_string(), m.brier);
    map.insert("accuracy".to_string(), m.accuracy);
    map.insert("precision".to_string(), m.precision);
    map.insert("recall".to_string(), m.recall);
    map
}

struct FusionOutput {
    prob: ArrayD<f64>,
    uncert: ArrayD<f64>,
    mask: ArrayD<f64>,
    metrics: HashMap<String, f64>,
}

fn run_fusion_foundation(
    mc: &UncertaintyOutput,
    tta: &UncertaintyOutput,
    noisy: &UncertaintyOutput,
    gt_mask: &ArrayD<f64>,
    cfg_fusion: &FusionConfig,
) -> FusionOutput {
    let mut valid_means: Vec<ArrayD<f64>> = Vec::new();
    let mut valid_uncerts: Vec<ArrayD<f64>> = Vec::new();
    for res in [mc, tta, noisy] {
        let dice = res.metrics.get("dice").copied().unwrap_or(0.0);
        if res.mean_prediction.ndim() > 0 && dice > 0.0 {
            valid_means.push(res.mean_prediction.clone());
            valid_uncerts.push(res.entropy.clone());
        }
    }

    let (prob, uncert) = if valid_means.is_empty() {
        let zeros = ArrayD::<f64>::zeros(gt_mask.shape().to_vec());
        (zeros.clone(), ArrayD::<f64>::ones(gt_mask.shape().to_vec()))
    } else {
        while valid_means.len() < 3 {
            valid_means.push(valid_means[valid_means.len() - 1].clone());
            valid_uncerts.push(valid_uncerts[valid_uncerts.len() - 1].clone());
        }
        let mcm = valid_means[0].clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let mcu = valid_uncerts[0].clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let ttm = valid_means[1].clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let ttu = valid_uncerts[1].clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let nm = valid_means[2].clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let nu = valid_uncerts[2].clone().into_dimensionality::<ndarray::Ix2>().unwrap();
        let (p, u, _m) = weighted_average_with_uncertainty(
            &mcm, &mcu, &ttm, &ttu, &nm, &nu,
            &cfg_fusion.weighting_method,
            cfg_fusion.beta,
            cfg_fusion.alpha,
            &cfg_fusion.threshold_method,
            cfg_fusion.percentile,
            cfg_fusion.k,
            cfg_fusion.epsilon,
        );
        (p.into_dyn(), u.into_dyn())
    };

    let mask = prob.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
    let iou = compute_iou(&mask.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let dice = compute_dice(&mask.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let m = compute_metrics(&prob.view().into_dyn(), &gt_mask.view().into_dyn(), 1e-8);
    let cert = certainty_score(&uncert.view().into_dyn(), &gt_mask.view().into_dyn(), 2);

    FusionOutput {
        prob,
        uncert,
        mask,
        metrics: metric_map(iou, dice, cert, &m),
    }
}

fn eval_pairs(
    seg: &Segmenter,
    config: &PipelineConfig,
    device: Device,
    pairs: &[(String, String)],
    label: &str,
    save_path: &str,
) -> HashMap<String, HashMap<String, HashMap<String, f64>>> {
    let mut overall = HashMap::new();
    for (idx, (img_path, mask_path)) in pairs.iter().enumerate() {
        let image = load_g_channel_tensor(img_path, 3, device).unsqueeze(0);
        let mut gt_mask = load_mask_array(mask_path);
        if gt_mask.shape()[..] != [128, 128] {
            gt_mask = resize_nearest(&gt_mask, 128, 128);
        }

        let sample_dir = format!("{}/{}_{}", save_path, label, idx);
        let _ = std::fs::create_dir_all(&sample_dir);
        save_array(&tensor_to_array(&image), &format!("{}/original_image.png", sample_dir));
        save_array(&gt_mask, &format!("{}/ground_truth.png", sample_dir));

        let mut sample_metrics = HashMap::new();

        let normal = run_normal(seg, &image, &gt_mask, &config.inference.activation);
        sample_metrics.insert("normal".to_string(), normal.metrics.clone());
        save_sample_outputs(&sample_dir, "original", &SampleData::from_run(&normal));

        let mc = run_mc_dropout(seg, &image, &gt_mask, &config.inference, config.mc_dropout.p);
        sample_metrics.insert("mc_dropout".to_string(), mc.metrics.clone());
        save_sample_outputs(&sample_dir, "mc_dropout", &SampleData::from_uncertainty(&mc));

        let tta = run_tta(seg, &image, &gt_mask, &config.inference, true);
        sample_metrics.insert("tta".to_string(), tta.metrics.clone());
        save_sample_outputs(&sample_dir, "tta", &SampleData::from_uncertainty(&tta));

        let noisy = run_noisy(seg, &image, &gt_mask, &config.inference, config.noisy.noise_std);
        sample_metrics.insert("noisy".to_string(), noisy.metrics.clone());
        save_sample_outputs(&sample_dir, "noisy", &SampleData::from_uncertainty(&noisy));

        let fusion = run_fusion_foundation(&mc, &tta, &noisy, &gt_mask, &config.fusion);
        sample_metrics.insert("fusion".to_string(), fusion.metrics.clone());
        save_sample_outputs(
            &sample_dir,
            "fusion",
            &SampleData {
                prob: Some(fusion.prob.clone()),
                mask: Some(fusion.mask.clone()),
                uncertainty: Some(fusion.uncert.clone()),
                ..Default::default()
            },
        );

        let crf = run_crf(&image, &fusion.prob, &fusion.uncert, &gt_mask, &config.inference, &config.crf);
        sample_metrics.insert("crf".to_string(), crf.metrics.clone());
        save_sample_outputs(&sample_dir, "refined", &SampleData::from_run(&crf));

        println!("  {} {} — Dice: norm={:.3}", label, idx, normal.metrics["dice"]);
        overall.insert(format!("{}_{}", label, idx), sample_metrics);
    }
    overall
}

fn main() {
    let args = Args::parse();
    let config = PipelineConfig::from_yaml(&args.config).expect("load config");
    let device = Device::cuda_if_available();
    let save_path = config.paths.save_path.clone();
    let root_path = config.paths.root_path.clone();

    let (pairs, _) = recover_image_mask_pairs(&root_path);
    let test_idx = load_test_indices(&args.test_indices).expect("load test indices");
    let test_set: std::collections::HashSet<usize> = test_idx.iter().copied().collect();
    let train_pairs: Vec<(String, String)> = (0..pairs.len())
        .filter(|&i| !test_set.contains(&i))
        .map(|i| pairs[i].clone())
        .collect();
    let test_pairs: Vec<(String, String)> = test_idx.iter().map(|&i| pairs[i].clone()).collect();
    println!("Train pairs: {}, Test pairs: {}", train_pairs.len(), test_pairs.len());

    let context_pairs: Vec<(String, String)> = train_pairs.iter().take(args.context_size).cloned().collect();

    let mut ctx_images = Vec::new();
    let mut ctx_masks = Vec::new();
    for (ip, mp) in &context_pairs {
        ctx_images.push(load_g_channel_tensor(ip, 3, device));
        ctx_masks.push(load_mask_tensor(mp, device));
    }
    let images = Tensor::stack(&ctx_images, 0);
    let gray = images.mean_dim(&[1i64][..], true, Kind::Float);
    let gray = gray.upsample_bilinear2d(&[INPUT_SIZE.0, INPUT_SIZE.1], false, None, None);
    let support_images = gray.unsqueeze(0);

    let masks = Tensor::stack(&ctx_masks, 0);
    let masks = masks.upsample_bilinear2d(&[INPUT_SIZE.0, INPUT_SIZE.1], false, None, None);
    let support_labels = masks.unsqueeze(0);

    let mut vs = nn::VarStore::new(device);
    let model = UniverSeg::new(&vs.root());
    vs.load("universeg.safetensors").expect("load universeg weights");
    let seg = Segmenter::UniverSeg {
        model,
        support_images,
        support_labels,
        input_size: INPUT_SIZE,
    };

    let support_pairs: Vec<(String, String)> = context_pairs.clone();
    let mut overall = HashMap::new();
    if !args.skip_support {
        println!("\n--- SUPPORT SET ({} images) ---", support_pairs.len());
        overall = eval_pairs(&seg, &config, device, &support_pairs, "support", &save_path);
    }

    let n_samples = config
        .subset
        .subset_size
        .unwrap_or(test_pairs.len() as i64)
        .min(test_pairs.len() as i64) as usize;
    let test_subset: Vec<(String, String)> = test_pairs.iter().take(n_samples).cloned().collect();
    println!("\n--- TEST SET ({} images) ---", test_subset.len());
    let test_overall = eval_pairs(&seg, &config, device, &test_subset, "test", &save_path);

    overall.extend(test_overall);
    finalize(&overall, &save_path);
}

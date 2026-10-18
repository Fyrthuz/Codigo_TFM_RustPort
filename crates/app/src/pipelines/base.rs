use std::collections::HashMap;
use std::path::Path;

use ndarray::ArrayD;
use tch::Tensor;

use tfm_core::config::PipelineConfig;
use tfm_core::crf::refine_with_crf_uncertainty;
use tfm_core::fusion::weighted_average_with_uncertainty;
use tfm_core::io::{save_detailed_metrics_csv, save_image, save_metrics_csv};
use tfm_core::metrics::{certainty_score, compute_dice, compute_iou, compute_metrics, Metrics};

use crate::models::Segmenter;
use crate::tensor_util::{squeeze, tensor_to_array};
use crate::uncertainty::mc_dropout::mc_dropout_inference;
use crate::uncertainty::noise::noisy_inference;
use crate::uncertainty::tta::tta_inference;

pub const METHODS: [&str; 6] = ["normal", "mc_dropout", "tta", "noisy", "fusion", "crf"];
pub const METRICS: [&str; 10] = [
    "iou", "dice", "nll", "nll_fg", "ece", "brier", "accuracy", "precision", "recall", "certainty",
];

fn save(arr: &ArrayD<f64>, path: &str) {
    if let Some(parent) = Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    save_image(&arr.view(), path);
}

pub fn save_array(arr: &ArrayD<f64>, path: &str) {
    save(arr, path);
}

fn metric_map(iou: f64, dice: f64, cert: f64, m: &Metrics) -> HashMap<String, f64> {
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

pub struct RunResult {
    pub prob: ArrayD<f64>,
    pub mask: ArrayD<f64>,
    pub uncertainty: ArrayD<f64>,
    pub metrics: HashMap<String, f64>,
}

pub fn run_normal(
    seg: &Segmenter,
    image: &Tensor,
    gt_mask: &ArrayD<f64>,
    activation: &str,
) -> RunResult {
    let output = tch::no_grad(|| seg.forward(image, 0.0));
    let output = output.sigmoid();
    let prob = squeeze(&tensor_to_array(&output));
    let mask = prob.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
    let iou = compute_iou(&mask.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let dice = compute_dice(&mask.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let m = compute_metrics(&prob.view().into_dyn(), &gt_mask.view().into_dyn(), 1e-8);
    let uncertainty = prob.mapv(|p| 1.0 - p);
    let cert = certainty_score(&uncertainty.view().into_dyn(), &gt_mask.view().into_dyn(), 2);
    let _ = activation;
    RunResult {
        prob,
        mask,
        uncertainty,
        metrics: metric_map(iou, dice, cert, &m),
    }
}

pub struct UncertaintyOutput {
    pub images: Vec<ArrayD<f64>>,
    pub masks: Vec<ArrayD<f64>>,
    pub mean_prediction: ArrayD<f64>,
    pub entropy: ArrayD<f64>,
    pub mask_pred: ArrayD<f64>,
    pub metrics: HashMap<String, f64>,
}

fn uncertainty_metrics(mean: &ArrayD<f64>, entropy: &ArrayD<f64>, gt_mask: &ArrayD<f64>) -> HashMap<String, f64> {
    let mask_pred = mean.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
    let iou = compute_iou(&mask_pred.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let dice = compute_dice(&mask_pred.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let m = compute_metrics(&mean.view().into_dyn(), &gt_mask.view().into_dyn(), 1e-8);
    let cert = certainty_score(&entropy.view().into_dyn(), &gt_mask.view().into_dyn(), 2);
    metric_map(iou, dice, cert, &m)
}

pub fn run_mc_dropout(
    seg: &Segmenter,
    image: &Tensor,
    gt_mask: &ArrayD<f64>,
    cfg_inf: &tfm_core::config::InferenceConfig,
    p: f64,
) -> UncertaintyOutput {
    let res = mc_dropout_inference(seg, image, cfg_inf.num_samples, &cfg_inf.activation, p);
    let mask_pred = res.mean_probs.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
    let metrics = uncertainty_metrics(&res.mean_probs, &res.entropy, gt_mask);
    UncertaintyOutput {
        images: res.images,
        masks: res.masks,
        mean_prediction: res.mean_probs,
        entropy: res.entropy,
        mask_pred,
        metrics,
    }
}

pub fn run_tta(
    seg: &Segmenter,
    image: &Tensor,
    gt_mask: &ArrayD<f64>,
    cfg_inf: &tfm_core::config::InferenceConfig,
    resize_safe: bool,
) -> UncertaintyOutput {
    let res = tta_inference(seg, image, &cfg_inf.activation, resize_safe);
    let mask_pred = res.mean_probs.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
    let metrics = uncertainty_metrics(&res.mean_probs, &res.entropy, gt_mask);
    UncertaintyOutput {
        images: res.images,
        masks: res.masks,
        mean_prediction: res.mean_probs,
        entropy: res.entropy,
        mask_pred,
        metrics,
    }
}

pub fn run_noisy(
    seg: &Segmenter,
    image: &Tensor,
    gt_mask: &ArrayD<f64>,
    cfg_inf: &tfm_core::config::InferenceConfig,
    noise_std: f64,
) -> UncertaintyOutput {
    let res = noisy_inference(seg, image, cfg_inf.num_samples, noise_std, &cfg_inf.activation);
    let mask_pred = res.mean_probs.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
    let metrics = uncertainty_metrics(&res.mean_probs, &res.entropy, gt_mask);
    UncertaintyOutput {
        images: res.images,
        masks: res.masks,
        mean_prediction: res.mean_probs,
        entropy: res.entropy,
        mask_pred,
        metrics,
    }
}

pub fn run_fusion(
    mc: &UncertaintyOutput,
    tta: &UncertaintyOutput,
    noisy: &UncertaintyOutput,
    gt_mask: &ArrayD<f64>,
    cfg_fusion: &tfm_core::config::FusionConfig,
) -> RunResult {
    let mcm = mc.mean_prediction.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let mcu = mc.entropy.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let ttm = tta.mean_prediction.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let ttu = tta.entropy.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let nm = noisy.mean_prediction.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let nu = noisy.entropy.clone().into_dimensionality::<ndarray::Ix2>().unwrap();

    let (prob, uncert, _mask) = weighted_average_with_uncertainty(
        &mcm, &mcu, &ttm, &ttu, &nm, &nu,
        &cfg_fusion.weighting_method,
        cfg_fusion.beta,
        cfg_fusion.alpha,
        &cfg_fusion.threshold_method,
        cfg_fusion.percentile,
        cfg_fusion.k,
        cfg_fusion.epsilon,
    );

    let fusion_mask = prob.mapv(|p| if p > 0.5 { 1.0 } else { 0.0 });
    let iou = compute_iou(&fusion_mask.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let dice = compute_dice(&fusion_mask.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let m = compute_metrics(&prob.view().into_dyn(), &gt_mask.view().into_dyn(), 1e-8);
    let cert = certainty_score(&uncert.view().into_dyn(), &gt_mask.view().into_dyn(), 2);

    RunResult {
        prob: prob.into_dyn(),
        mask: fusion_mask.into_dyn(),
        uncertainty: uncert.into_dyn(),
        metrics: metric_map(iou, dice, cert, &m),
    }
}

pub fn run_crf(
    image: &Tensor,
    consensus_prob: &ArrayD<f64>,
    consensus_uncertainty: &ArrayD<f64>,
    gt_mask: &ArrayD<f64>,
    cfg_inf: &tfm_core::config::InferenceConfig,
    cfg_crf: &tfm_core::config::CRFConfig,
) -> RunResult {
    let image_arr = tensor_to_array(image);
    let sdims = (cfg_crf.sdims[0], cfg_crf.sdims[1]);
    let schan = cfg_crf.schan[0];
    let (probs, seg, uncert) = refine_with_crf_uncertainty(
        &image_arr.view(),
        &consensus_prob.view(),
        &consensus_uncertainty.view(),
        sdims,
        schan,
        cfg_crf.n_iters as usize,
        cfg_crf.epsilon,
        cfg_crf.w_g,
        cfg_crf.w_b,
    );

    let n_classes = probs.shape()[0];
    let crf_mean_prob = if cfg_inf.activation == "sigmoid" || n_classes == 2 {
        probs.index_axis(ndarray::Axis(0), 1).to_owned().into_dyn()
    } else {
        probs.map_axis(ndarray::Axis(0), |ax| ax.fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
            .into_dyn()
    };

    let seg_float = seg.mapv(|v| v as f64);
    let iou = compute_iou(&seg_float.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let dice = compute_dice(&seg_float.view().into_dyn(), &gt_mask.view().into_dyn(), 0.5);
    let m = compute_metrics(&crf_mean_prob.view().into_dyn(), &gt_mask.view().into_dyn(), 1e-8);
    let cert = certainty_score(&uncert.view().into_dyn(), &gt_mask.view().into_dyn(), 2);

    RunResult {
        prob: crf_mean_prob,
        mask: seg_float.into_dyn(),
        uncertainty: uncert.into_dyn(),
        metrics: metric_map(iou, dice, cert, &m),
    }
}

pub fn save_sample_outputs(sample_dir: &str, name: &str, data: &SampleData) {
    let method_dir = Path::new(sample_dir).join(name);
    let _ = std::fs::create_dir_all(&method_dir);
    let md = method_dir.to_str().unwrap();

    if let Some(p) = &data.prob {
        save(p, &format!("{}/probability.png", md));
    }
    if let Some(m) = &data.mask {
        save(m, &format!("{}/mask.png", md));
    }
    if let Some(u) = &data.uncertainty {
        save(u, &format!("{}/uncertainty.png", md));
    }

    if let (Some(images), Some(masks)) = (&data.images, &data.masks) {
        let images_dir = format!("{}/images", md);
        let predictions_dir = format!("{}/predictions", md);
        for (i, img) in images.iter().enumerate() {
            save(img, &format!("{}/image_{}.png", images_dir, i));
        }
        for (i, m) in masks.iter().enumerate() {
            save(m, &format!("{}/prediction_{}.png", predictions_dir, i));
        }
    }

    if let Some(mp) = &data.mean_prediction {
        save(mp, &format!("{}/mean_prediction.png", md));
    }
    if let Some(e) = &data.entropy {
        save(e, &format!("{}/uncertainty.png", md));
    }
    if let Some(mp) = &data.mask_pred {
        save(mp, &format!("{}/mean_mask_prediction.png", md));
    }
}

#[derive(Default)]
pub struct SampleData {
    pub prob: Option<ArrayD<f64>>,
    pub mask: Option<ArrayD<f64>>,
    pub uncertainty: Option<ArrayD<f64>>,
    pub images: Option<Vec<ArrayD<f64>>>,
    pub masks: Option<Vec<ArrayD<f64>>>,
    pub mean_prediction: Option<ArrayD<f64>>,
    pub entropy: Option<ArrayD<f64>>,
    pub mask_pred: Option<ArrayD<f64>>,
}

impl SampleData {
    pub fn from_run(r: &RunResult) -> Self {
        SampleData {
            prob: Some(r.prob.clone()),
            mask: Some(r.mask.clone()),
            uncertainty: Some(r.uncertainty.clone()),
            ..Default::default()
        }
    }
    pub fn from_uncertainty(u: &UncertaintyOutput) -> Self {
        SampleData {
            images: Some(u.images.clone()),
            masks: Some(u.masks.clone()),
            mean_prediction: Some(u.mean_prediction.clone()),
            entropy: Some(u.entropy.clone()),
            mask_pred: Some(u.mask_pred.clone()),
            ..Default::default()
        }
    }
}

pub fn finalize(
    overall_metrics: &HashMap<String, HashMap<String, HashMap<String, f64>>>,
    save_path: &str,
) {
    let viz_dir = Path::new(save_path).join("visualizations");
    let _ = std::fs::create_dir_all(&viz_dir);
    let vd = viz_dir.to_str().unwrap();

    let active_methods: Vec<String> = METHODS.iter().map(|s| s.to_string()).collect();
    let active_metrics: Vec<String> = METRICS.iter().map(|s| s.to_string()).collect();

    let mut mean_results: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for method in &active_methods {
        let mut mmap = HashMap::new();
        for metric in &active_metrics {
            let mut vals = Vec::new();
            for sample_data in overall_metrics.values() {
                if let Some(m) = sample_data.get(method) {
                    if let Some(&v) = m.get(metric) {
                        if v.is_finite() {
                            vals.push(v);
                        }
                    }
                }
            }
            let mean = if vals.is_empty() {
                0.0
            } else {
                vals.iter().sum::<f64>() / vals.len() as f64
            };
            mmap.insert(metric.clone(), mean);
        }
        mean_results.insert(method.clone(), mmap);
    }

    save_metrics_csv(&mean_results, &active_methods, &active_metrics, vd, "metrics_summary.csv");
    save_detailed_metrics_csv(overall_metrics, &active_methods, &active_metrics, vd, "detailed_metrics.csv");

    println!("\nOverall Metrics:");
    for method in &active_methods {
        println!("Method: {}", method);
        for metric in &active_metrics {
            let val = mean_results[method][metric];
            println!("  {}: {:.4}", metric, val);
        }
    }
}

pub fn process_sample(
    seg: &Segmenter,
    config: &PipelineConfig,
    image: &Tensor,
    gt_mask: &ArrayD<f64>,
    sample_dir: &str,
    tta_resize_safe: bool,
) -> HashMap<String, HashMap<String, f64>> {
    let _ = std::fs::create_dir_all(sample_dir);
    let mut sample_metrics = HashMap::new();

    save(&tensor_to_array(image), &format!("{}/original_image.png", sample_dir));
    save(gt_mask, &format!("{}/ground_truth.png", sample_dir));

    let cfg_inf = &config.inference;

    let normal = run_normal(seg, image, gt_mask, &cfg_inf.activation);
    sample_metrics.insert("normal".to_string(), normal.metrics.clone());
    save_sample_outputs(sample_dir, "original", &SampleData::from_run(&normal));

    let mc = run_mc_dropout(seg, image, gt_mask, cfg_inf, config.mc_dropout.p);
    sample_metrics.insert("mc_dropout".to_string(), mc.metrics.clone());
    save_sample_outputs(sample_dir, "mc_dropout", &SampleData::from_uncertainty(&mc));

    let tta = run_tta(seg, image, gt_mask, cfg_inf, tta_resize_safe);
    sample_metrics.insert("tta".to_string(), tta.metrics.clone());
    save_sample_outputs(sample_dir, "tta", &SampleData::from_uncertainty(&tta));

    let noisy = run_noisy(seg, image, gt_mask, cfg_inf, config.noisy.noise_std);
    sample_metrics.insert("noisy".to_string(), noisy.metrics.clone());
    save_sample_outputs(sample_dir, "noisy", &SampleData::from_uncertainty(&noisy));

    let fusion = run_fusion(&mc, &tta, &noisy, gt_mask, &config.fusion);
    sample_metrics.insert("fusion".to_string(), fusion.metrics.clone());
    save_sample_outputs(sample_dir, "fusion", &SampleData::from_run(&fusion));

    let crf = run_crf(image, &fusion.prob, &fusion.uncertainty, gt_mask, cfg_inf, &config.crf);
    sample_metrics.insert("crf".to_string(), crf.metrics.clone());
    save_sample_outputs(sample_dir, "refined", &SampleData::from_run(&crf));

    sample_metrics
}

use std::collections::HashMap;

use clap::Parser;
use ndarray::ArrayD;
use tch::{nn, Device};

use app::models::unet::Unet;
use app::models::Segmenter;
use app::pipelines::base::{finalize, process_sample};
use app::tensor_util::{load_image_rgb_tensor, load_mask_array};
use tfm_core::config::PipelineConfig;
use tfm_core::dataset::{load_test_indices, recover_image_mask_pairs};

#[derive(Parser)]
struct Args {
    #[clap(long, default_value = "./configs/pipeline_2d.yaml")]
    config: String,
    #[clap(long, default_value = "unet_model.safetensors")]
    checkpoint: String,
    #[clap(long, default_value = "test_indices.json")]
    test_indices: String,
    /// BatchNorm en modo entrenamiento (reproduce el `run_unet.py` original, que
    /// omite `model.eval()`). Por defecto usa modo evaluación.
    #[clap(long)]
    train_mode: bool,
}

fn main() {
    let args = Args::parse();
    let config = PipelineConfig::from_yaml(&args.config).expect("load config");
    let device = Device::cuda_if_available();

    let root_path = config.paths.root_path.clone();
    let save_path = config.paths.save_path.clone();

    let (all_pairs, _) = recover_image_mask_pairs(&root_path);
    let test_idx = load_test_indices(&args.test_indices).expect("load test indices");
    let pairs: Vec<(String, String)> = test_idx.iter().map(|&i| all_pairs[i].clone()).collect();
    println!("Using test set: {} samples (from {})", pairs.len(), args.test_indices);

    let mut vs = nn::VarStore::new(device);
    let model = Unet::new(
        &vs.root(),
        config.model.in_channels,
        config.model.out_channels,
        config.model.init_features,
    );
    if std::path::Path::new(&args.checkpoint).exists() {
        vs.load(&args.checkpoint).expect("load checkpoint");
        println!("Model loaded from {}", args.checkpoint);
    } else {
        println!("Warning: checkpoint {} not found, using untrained model", args.checkpoint);
    }
    let seg = Segmenter::Unet { model, train_mode: args.train_mode };

    let mut n_samples = pairs.len();
    if let Some(ss) = config.subset.subset_size {
        n_samples = n_samples.min(ss as usize);
    }

    let mut overall_metrics: HashMap<String, HashMap<String, HashMap<String, f64>>> = HashMap::new();
    for idx in 0..n_samples {
        let (img_path, mask_path) = &pairs[idx];
        let image = load_image_rgb_tensor(img_path, device).unsqueeze(0);
        let gt_mask: ArrayD<f64> = load_mask_array(mask_path);
        let sample_dir = format!("{}/sample_{}", save_path, idx);
        let metrics = process_sample(&seg, &config, &image, &gt_mask, &sample_dir, false);
        overall_metrics.insert(format!("sample_{}", idx), metrics);
    }

    finalize(&overall_metrics, &save_path);
}

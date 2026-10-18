use clap::Parser;
use tch::{nn, nn::OptimizerConfig, Device, Kind, Tensor};

use app::models::unet::Unet;
use app::tensor_util::array_to_tensor;
use tfm_core::dataset::{save_test_indices, split_by_patient, LGGSegmentationDataset};

#[derive(Parser)]
struct Args {
    #[clap(long, default_value = "./MRI/filtered_data")]
    data_root: String,
    #[clap(long, default_value = "60")]
    epochs: i64,
    #[clap(long, default_value = "16")]
    batch_size: i64,
    #[clap(long, default_value = "1e-4")]
    lr: f64,
    #[clap(long, default_value = "unet_model.safetensors")]
    save_path: String,
    #[clap(long, default_value = "test_indices.json")]
    test_indices: String,
}

fn dice_loss(preds: &Tensor, targets: &Tensor, smooth: f64) -> Tensor {
    let preds = preds.sigmoid();
    let intersection = (&preds * targets).sum(Kind::Float);
    let denom = preds.sum(Kind::Float) + targets.sum(Kind::Float) + smooth;
    1.0 - (2.0 * intersection + smooth) / denom
}

fn combined_loss(preds: &Tensor, targets: &Tensor, bce_weight: f64) -> Tensor {
    let bce = preds.binary_cross_entropy_with_logits::<&Tensor>(targets, None, None, tch::Reduction::Mean);
    let dice = dice_loss(preds, targets, 1.0);
    bce_weight * bce + (1.0 - bce_weight) * dice
}

fn compute_iou(preds: &Tensor, targets: &Tensor, threshold: f64) -> f64 {
    let p = preds.sigmoid().gt(threshold).to_kind(Kind::Float);
    let intersection = (&p * targets).sum(Kind::Float);
    let union = p.sum(Kind::Float) + targets.sum(Kind::Float) - &intersection;
    let u = f64::try_from(&union).unwrap();
    if u > 0.0 {
        f64::try_from(intersection / union).unwrap()
    } else {
        1.0
    }
}

fn augment(img: &Tensor, mask: &Tensor) -> (Tensor, Tensor) {
    let mut img = img.shallow_clone();
    let mut mask = mask.shallow_clone();
    if rand::random::<f64>() < 0.5 {
        img = img.flip(&[3]);
        mask = mask.flip(&[3]);
    }
    if rand::random::<f64>() < 0.5 {
        img = img.flip(&[2]);
        mask = mask.flip(&[2]);
    }
    (img, mask)
}

fn main() {
    let args = Args::parse();
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let dataset = LGGSegmentationDataset::new(&args.data_root);
    let (train_idx, val_idx, test_idx) = split_by_patient(&dataset.patient_ids, 0.70, 0.15, 42);
    println!(
        "Patients: {} total | Train: {} samples, Val: {} samples, Test: {} samples",
        dataset.patient_ids.iter().collect::<std::collections::HashSet<_>>().len(),
        train_idx.len(),
        val_idx.len(),
        test_idx.len()
    );
    save_test_indices(&test_idx, &args.test_indices).expect("save test indices");

    let vs = nn::VarStore::new(device);
    let model = Unet::new(&vs.root(), 3, 1, 32);
    let mut opt = nn::Adam::default().build(&vs, args.lr).expect("adam");

    let mut rng = rand::thread_rng();
    let mut best_val_iou = 0.0;
    let mut wait = 0;

    for epoch in 1..=args.epochs {
        let mut running_loss = 0.0;
        let mut running_iou = 0.0;
        let mut n_batches = 0i64;

        let mut order = train_idx.clone();
        for i in 0..order.len() {
            let j = rand::Rng::gen_range(&mut rng, i..order.len());
            order.swap(i, j);
        }

        let mut batch_images: Vec<Tensor> = Vec::new();
        let mut batch_masks: Vec<Tensor> = Vec::new();
        for &idx in &order {
            let (img_arr, mask_arr) = dataset.load(idx).expect("load sample");
            let img = array_to_tensor(&img_arr.into_dyn(), device);
            let mask = array_to_tensor(&mask_arr.into_dyn(), device);
            let (img, mask) = augment(&img, &mask);
            batch_images.push(img);
            batch_masks.push(mask);

            if batch_images.len() as i64 >= args.batch_size {
                let inputs = Tensor::stack(&batch_images, 0);
                let targets = Tensor::stack(&batch_masks, 0);
                let outputs = model.forward(&inputs, true, 0.0);
                let loss = combined_loss(&outputs, &targets, 0.5);
                opt.backward_step(&loss);
                running_loss += f64::try_from(loss).unwrap();
                running_iou += compute_iou(&outputs, &targets, 0.5);
                n_batches += 1;
                batch_images.clear();
                batch_masks.clear();
            }
        }

        let val_loss = validate(&model, &dataset, &val_idx, device);
        let val_iou = validate_iou(&model, &dataset, &val_idx, device);
        println!(
            "Epoch {}/{} - Train Loss: {:.4}, Train IoU: {:.4}, Val Loss: {:.4}, Val IoU: {:.4}",
            epoch,
            args.epochs,
            running_loss / n_batches.max(1) as f64,
            running_iou / n_batches.max(1) as f64,
            val_loss,
            val_iou
        );

        if val_iou > best_val_iou {
            best_val_iou = val_iou;
            vs.save(&args.save_path).expect("save model");
            wait = 0;
        } else {
            wait += 1;
            if wait >= 15 {
                println!("Early stopping at epoch {}", epoch);
                break;
            }
        }
    }

    println!("Best Val IoU: {:.4}", best_val_iou);
    println!("Model saved to {}", args.save_path);
}

fn validate(model: &Unet, dataset: &LGGSegmentationDataset, indices: &[usize], device: Device) -> f64 {
    let mut loss_sum = 0.0;
    let mut n = 0i64;
    for &idx in indices {
        let (img_arr, mask_arr) = dataset.load(idx).expect("load");
        let img = array_to_tensor(&img_arr.into_dyn(), device).unsqueeze(0);
        let mask = array_to_tensor(&mask_arr.into_dyn(), device).unsqueeze(0);
        let outputs = tch::no_grad(|| model.forward(&img, false, 0.0));
        loss_sum += f64::try_from(combined_loss(&outputs, &mask, 0.5)).unwrap();
        n += 1;
    }
    loss_sum / n.max(1) as f64
}

fn validate_iou(model: &Unet, dataset: &LGGSegmentationDataset, indices: &[usize], device: Device) -> f64 {
    let mut iou_sum = 0.0;
    let mut n = 0i64;
    for &idx in indices {
        let (img_arr, mask_arr) = dataset.load(idx).expect("load");
        let img = array_to_tensor(&img_arr.into_dyn(), device).unsqueeze(0);
        let mask = array_to_tensor(&mask_arr.into_dyn(), device).unsqueeze(0);
        let outputs = tch::no_grad(|| model.forward(&img, false, 0.0));
        iou_sum += compute_iou(&outputs, &mask, 0.5);
        n += 1;
    }
    iou_sum / n.max(1) as f64
}

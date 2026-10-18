use tch::Tensor;

use crate::models::Segmenter;
use crate::tensor_util::tensor_to_array;

use super::{aggregate_sigmoid, UncertaintyResult};

pub fn tta_inference(
    seg: &Segmenter,
    image: &Tensor,
    _activation: &str,
    resize_safe: bool,
) -> UncertaintyResult {
    let mut tta_predictions = Vec::new();
    let mut augmented_images = Vec::new();

    let _guard = tch::no_grad_guard();

    if resize_safe {
        for transform in resize_safe_combos() {
            let augmented = transform(image).clamp(0.0, 1.0);
            augmented_images.push(tensor_to_array(&augmented));
            let output = seg.forward(&augmented, 0.0);
            tta_predictions.push(output);
        }
    } else {
        let flips = [false, true];
        let scales = [0.5f64, 1.0, 2.0];
        let factors = [0.8f64, 0.9, 1.0, 1.1, 1.2];
        let (h, w) = (image.size()[2], image.size()[3]);
        for &flip in flips.iter() {
            for &scale in scales.iter() {
                for &factor in factors.iter() {
                    let mut aug = image.shallow_clone();
                    if flip {
                        aug = aug.flip(&[3]);
                    }
                    if scale != 1.0 {
                        let nh = (h as f64 * scale) as i64;
                        let nw = (w as f64 * scale) as i64;
                        aug = aug.upsample_nearest2d(&[nh, nw], None, None);
                    }
                    if factor != 1.0 {
                        aug = aug * factor;
                    }
                    augmented_images.push(tensor_to_array(&aug));

                    let output = seg.forward(&aug, 0.0);

                    let mut deaug = output;
                    if scale != 1.0 {
                        let dh = (deaug.size()[2] as f64 / scale) as i64;
                        let dw = (deaug.size()[3] as f64 / scale) as i64;
                        deaug = deaug.upsample_nearest2d(&[dh, dw], None, None);
                    }
                    if flip {
                        deaug = deaug.flip(&[3]);
                    }
                    tta_predictions.push(deaug);
                }
            }
        }
    }

    drop(_guard);
    aggregate_sigmoid(&tta_predictions, augmented_images)
}

fn resize_safe_combos() -> Vec<Box<dyn Fn(&Tensor) -> Tensor>> {
    vec![
        Box::new(|x: &Tensor| x.shallow_clone()),
        Box::new(|x: &Tensor| x * 0.9),
        Box::new(|x: &Tensor| x * 1.1),
        Box::new(|x: &Tensor| x.clamp(1e-6, 1.0).pow_tensor_scalar(0.85)),
        Box::new(|x: &Tensor| x.clamp(1e-6, 1.0).pow_tensor_scalar(1.15)),
        Box::new(|x: &Tensor| x - 0.03),
        Box::new(|x: &Tensor| x + 0.03),
        Box::new(|x: &Tensor| (x - 0.5) * 0.9 + 0.5),
        Box::new(|x: &Tensor| (x - 0.5) * 1.1 + 0.5),
    ]
}

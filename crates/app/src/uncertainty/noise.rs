use tch::Tensor;

use crate::models::Segmenter;
use crate::tensor_util::tensor_to_array;

use super::{aggregate_sigmoid, UncertaintyResult};

pub fn noisy_inference(
    seg: &Segmenter,
    image: &Tensor,
    num_samples: i64,
    noise_std: f64,
    _activation: &str,
) -> UncertaintyResult {
    let mut noisy_images = Vec::new();
    let mut outputs = Vec::new();

    let _guard = tch::no_grad_guard();
    for _ in 0..num_samples {
        let noise = Tensor::randn_like(image) * noise_std;
        let noisy = (image + noise).clamp(0.0, 1.0);
        noisy_images.push(tensor_to_array(&noisy));
        outputs.push(seg.forward(&noisy, 0.0));
    }
    drop(_guard);

    aggregate_sigmoid(&outputs, noisy_images)
}

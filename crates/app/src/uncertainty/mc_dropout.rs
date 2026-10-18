use tch::Tensor;

use crate::models::Segmenter;
use crate::tensor_util::tensor_to_array;

use super::{aggregate_sigmoid, UncertaintyResult};

pub fn mc_dropout_inference(
    seg: &Segmenter,
    image: &Tensor,
    num_samples: i64,
    _activation: &str,
    p: f64,
) -> UncertaintyResult {
    let mut outputs = Vec::new();
    let _guard = tch::no_grad_guard();
    for _ in 0..num_samples {
        outputs.push(seg.forward(image, p));
    }
    drop(_guard);

    let images = (0..num_samples).map(|_| tensor_to_array(image)).collect();
    aggregate_sigmoid(&outputs, images)
}

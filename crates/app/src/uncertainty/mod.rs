pub mod mc_dropout;
pub mod noise;
pub mod tta;

use ndarray::ArrayD;
use tch::{Kind, Tensor};

use crate::tensor_util::{binary_entropy, squeeze, tensor_to_array};

#[derive(Debug)]
pub struct UncertaintyResult {
    pub images: Vec<ArrayD<f64>>,
    pub masks: Vec<ArrayD<f64>>,
    pub mean_probs: ArrayD<f64>,
    pub entropy: ArrayD<f64>,
}

pub fn aggregate_sigmoid(outputs: &[Tensor], images: Vec<ArrayD<f64>>) -> UncertaintyResult {
    let stacked = Tensor::stack(outputs, 0);
    let sigmoid = stacked.sigmoid();
    let mean = sigmoid.mean_dim(&[0i64][..], false, Kind::Float);
    let mp = squeeze(&tensor_to_array(&mean));
    let entropy = binary_entropy(&mp);
    let masks = (0..outputs.len())
        .map(|i| {
            let m = sigmoid.select(0, i as i64).gt(0.5).to_kind(Kind::Float);
            tensor_to_array(&m)
        })
        .collect();
    UncertaintyResult {
        images,
        masks,
        mean_probs: mp,
        entropy,
    }
}

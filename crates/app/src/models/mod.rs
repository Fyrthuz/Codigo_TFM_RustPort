pub mod unet;
pub mod universeg;

use tch::{Kind, Tensor};

use self::unet::Unet;
use self::universeg::UniverSeg;

pub enum Segmenter {
    Unet { model: Unet, train_mode: bool },
    UniverSeg {
        model: UniverSeg,
        support_images: Tensor,
        support_labels: Tensor,
        input_size: (i64, i64),
    },
}

impl Segmenter {
    pub fn forward(&self, image: &Tensor, mc_p: f64) -> Tensor {
        match self {
            Segmenter::Unet { model, train_mode } => model.forward(image, *train_mode, mc_p),
            Segmenter::UniverSeg {
                model,
                support_images,
                support_labels,
                input_size,
            } => {
                let x = image.to_kind(Kind::Float);
                let gray = x.mean_dim(&[1i64][..], true, Kind::Float);
                let (h, w) = (gray.size()[2], gray.size()[3]);
                let resized = if (h, w) != *input_size {
                    gray.upsample_bilinear2d(&[input_size.0, input_size.1], false, None, None)
                } else {
                    gray
                };
                model.forward(&resized, support_images, support_labels, mc_p)
            }
        }
    }

    pub fn input_channels(&self) -> i64 {
        match self {
            Segmenter::Unet { .. } => 3,
            Segmenter::UniverSeg { .. } => 3,
        }
    }
}

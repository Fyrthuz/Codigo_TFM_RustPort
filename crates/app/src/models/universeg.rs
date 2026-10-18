use tch::{nn, nn::Module, Kind, Tensor};

#[derive(Debug)]
pub struct CrossConv2d {
    pub conv: nn::Conv2D,
}

impl CrossConv2d {
    fn new(p: &nn::Path, in_channels: i64, out_channels: i64, ksize: i64) -> Self {
        let conv = nn::conv2d(
            p / "cross_conv",
            in_channels,
            out_channels,
            ksize,
            nn::ConvConfig {
                padding: ksize / 2,
                ..Default::default()
            },
        );
        CrossConv2d { conv }
    }

    fn forward(&self, x: &Tensor, y: &Tensor) -> Tensor {
        let b = x.size()[0];
        let sx = x.size()[1];
        let sy = y.size()[1];
        let c2 = x.size()[2] + y.size()[2];
        let h = x.size()[3];
        let w = x.size()[4];

        let xs = x.unsqueeze(2).repeat(&[1, 1, sy, 1, 1, 1]);
        let ys = y.unsqueeze(1).repeat(&[1, sx, 1, 1, 1, 1]);
        let xy = Tensor::cat(&[xs, ys], 3);
        let batched = xy.reshape(&[b * sx * sy, c2, h, w]);
        let out = self.conv.forward(&batched);
        let co = out.size()[1];
        out.reshape(&[b, sx, sy, co, h, w])
    }
}

#[derive(Debug)]
pub struct CrossBlock {
    cross: CrossConv2d,
    target_conv: nn::Conv2D,
    support_conv: nn::Conv2D,
}

fn vmap_conv_leaky(conv: &nn::Conv2D, x: &Tensor, mc_p: f64) -> Tensor {
    let b = x.size()[0];
    let s = x.size()[1];
    let c = x.size()[2];
    let h = x.size()[3];
    let w = x.size()[4];
    let flat = x.reshape(&[b * s, c, h, w]);
    let out = conv.forward(&flat).leaky_relu();
    let out = if mc_p > 0.0 {
        let mask = out.rand_like().gt(mc_p).to_kind(Kind::Float);
        out * mask
    } else {
        out
    };
    let co = out.size()[1];
    out.reshape(&[b, s, co, h, w])
}

impl CrossBlock {
    fn new(p: &nn::Path, cross_in: i64, cross_features: i64, conv_features: i64) -> Self {
        let cross = CrossConv2d::new(&(p / "cross"), cross_in, cross_features, 3);
        let target_conv = nn::conv2d(
            p / "target" / "vmapped" / "conv",
            cross_features,
            conv_features,
            3,
            nn::ConvConfig { padding: 1, ..Default::default() },
        );
        let support_conv = nn::conv2d(
            p / "support" / "vmapped" / "conv",
            cross_features,
            conv_features,
            3,
            nn::ConvConfig { padding: 1, ..Default::default() },
        );
        CrossBlock { cross, target_conv, support_conv }
    }

    fn forward(&self, target: &Tensor, support: &Tensor, mc_p: f64) -> (Tensor, Tensor) {
        let interaction = self.cross.forward(target, support).squeeze_dim(1);
        let interaction = if mc_p > 0.0 {
            let mask = interaction.rand_like().gt(mc_p).to_kind(Kind::Float);
            interaction * mask
        } else {
            interaction
        };
        let new_target = interaction.mean_dim(&[1i64][..], true, Kind::Float);
        let target_out = vmap_conv_leaky(&self.target_conv, &new_target, mc_p);
        let support_out = vmap_conv_leaky(&self.support_conv, &interaction, mc_p);
        (target_out, support_out)
    }
}

#[derive(Debug)]
pub struct UniverSeg {
    enc_blocks: Vec<CrossBlock>,
    dec_blocks: Vec<CrossBlock>,
    out_conv: nn::Conv2D,
}

impl UniverSeg {
    pub fn new(p: &nn::Path) -> Self {
        let mut enc_blocks = Vec::new();
        let mut cross_in = 3i64;
        for i in 0..4 {
            let bp = p / "enc_blocks" / i.to_string();
            enc_blocks.push(CrossBlock::new(&bp, cross_in, 64, 64));
            cross_in = 128;
        }

        let mut dec_blocks = Vec::new();
        for i in 0..3 {
            let bp = p / "dec_blocks" / i.to_string();
            dec_blocks.push(CrossBlock::new(&bp, 256, 64, 64));
        }

        let out_conv = nn::conv2d(p / "out_conv" / "conv", 64, 1, 1, Default::default());

        UniverSeg {
            enc_blocks,
            dec_blocks,
            out_conv,
        }
    }

    fn vmap_maxpool(&self, x: &Tensor) -> Tensor {
        let b = x.size()[0];
        let s = x.size()[1];
        let c = x.size()[2];
        let h = x.size()[3];
        let w = x.size()[4];
        let flat = x.reshape(&[b * s, c, h, w]);
        let out = flat.max_pool2d_default(2);
        let oh = out.size()[2];
        let ow = out.size()[3];
        out.reshape(&[b, s, c, oh, ow])
    }

    fn vmap_upsample(&self, x: &Tensor) -> Tensor {
        let b = x.size()[0];
        let s = x.size()[1];
        let c = x.size()[2];
        let h = x.size()[3];
        let w = x.size()[4];
        let flat = x.reshape(&[b * s, c, h, w]);
        let out = flat.upsample_bilinear2d(&[h * 2, w * 2], true, None, None);
        out.reshape(&[b, s, c, h * 2, w * 2])
    }

    pub fn forward(
        &self,
        target_image: &Tensor,
        support_images: &Tensor,
        support_labels: &Tensor,
        mc_p: f64,
    ) -> Tensor {
        let target = target_image.unsqueeze(2);
        let support = Tensor::cat(&[support_images, support_labels], 2);

        let mut pass_through: Vec<(Tensor, Tensor)> = Vec::new();

        let (mut target, mut support) = (target, support);
        for (i, block) in self.enc_blocks.iter().enumerate() {
            let (t, s) = block.forward(&target, &support, mc_p);
            target = t;
            support = s;
            if i == self.enc_blocks.len() - 1 {
                break;
            }
            pass_through.push((target.copy(), support.copy()));
            target = self.vmap_maxpool(&target);
            support = self.vmap_maxpool(&support);
        }

        for block in self.dec_blocks.iter() {
            let (target_skip, support_skip) = pass_through.pop().unwrap();
            let up_target = self.vmap_upsample(&target);
            let up_support = self.vmap_upsample(&support);
            target = Tensor::cat(&[up_target, target_skip], 2);
            support = Tensor::cat(&[up_support, support_skip], 2);
            let (t, s) = block.forward(&target, &support, mc_p);
            target = t;
            support = s;
        }

        let b = target.size()[0];
        let c = target.size()[2];
        let h = target.size()[3];
        let w = target.size()[4];
        let target = target.reshape(&[b, c, h, w]);
        let out = self.out_conv.forward(&target);
        if mc_p > 0.0 {
            let mask = out.rand_like().gt(mc_p).to_kind(Kind::Float);
            out * mask
        } else {
            out
        }
    }
}

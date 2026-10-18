use tch::{nn, nn::Module, nn::ModuleT, Kind, Tensor};

#[derive(Debug)]
pub struct ConvBNReLU {
    pub conv: nn::Conv2D,
    pub bn: nn::BatchNorm,
}

impl ConvBNReLU {
    fn new(p: &nn::Path, in_c: i64, out_c: i64, ksize: i64) -> Self {
        let conv = nn::conv2d(
            p / "conv",
            in_c,
            out_c,
            ksize,
            nn::ConvConfig {
                padding: ksize / 2,
                bias: false,
                ..Default::default()
            },
        );
        let bn = nn::batch_norm2d(p / "norm", out_c, Default::default());
        ConvBNReLU { conv, bn }
    }

    pub fn forward(&self, x: &Tensor, train: bool, mc_p: f64) -> Tensor {
        let y = self.bn.forward_t(&self.conv.forward(x), train).relu();
        if mc_p > 0.0 {
            let mask = y.rand_like().gt(mc_p).to_kind(Kind::Float);
            y * mask
        } else {
            y
        }
    }
}

fn double_conv(p: &nn::Path, in_c: i64, out_c: i64, mid_c: i64) -> (ConvBNReLU, ConvBNReLU) {
    let dc = p / "double_conv";
    let c0 = ConvBNReLU::new(&(&dc / "0"), in_c, mid_c, 3);
    let c1 = ConvBNReLU::new(&(&dc / "1"), mid_c, out_c, 3);
    (c0, c1)
}

#[derive(Debug)]
pub struct Unet {
    inc: (ConvBNReLU, ConvBNReLU),
    down1: (ConvBNReLU, ConvBNReLU),
    down2: (ConvBNReLU, ConvBNReLU),
    down3: (ConvBNReLU, ConvBNReLU),
    down4: (ConvBNReLU, ConvBNReLU),
    up1: (ConvBNReLU, ConvBNReLU),
    up2: (ConvBNReLU, ConvBNReLU),
    up3: (ConvBNReLU, ConvBNReLU),
    up4: (ConvBNReLU, ConvBNReLU),
    outc: nn::Conv2D,
}

impl Unet {
    pub fn new(p: &nn::Path, in_channels: i64, out_channels: i64, init_features: i64) -> Self {
        let f = init_features;
        let inc = double_conv(&(p / "inc"), in_channels, f, f);

        let d1 = &(p / "down1");
        let down1 = double_conv(&(d1 / "maxpool_conv" / "1"), f, f * 2, f * 2);
        let d2 = &(p / "down2");
        let down2 = double_conv(&(d2 / "maxpool_conv" / "1"), f * 2, f * 4, f * 4);
        let d3 = &(p / "down3");
        let down3 = double_conv(&(d3 / "maxpool_conv" / "1"), f * 4, f * 8, f * 8);
        let d4 = &(p / "down4");
        let down4 = double_conv(&(d4 / "maxpool_conv" / "1"), f * 8, f * 8, f * 8);

        let up1 = double_conv(&(p / "up1" / "conv"), f * 16, f * 4, f * 8);
        let up2 = double_conv(&(p / "up2" / "conv"), f * 8, f * 2, f * 4);
        let up3 = double_conv(&(p / "up3" / "conv"), f * 4, f, f * 2);
        let up4 = double_conv(&(p / "up4" / "conv"), f * 2, f, f);

        let outc = nn::conv2d(p / "outc" / "conv", f, out_channels, 1, Default::default());

        Unet {
            inc,
            down1,
            down2,
            down3,
            down4,
            up1,
            up2,
            up3,
            up4,
            outc,
        }
    }

    fn up_step(&self, x1: &Tensor, x2: &Tensor, conv: &(ConvBNReLU, ConvBNReLU), train: bool, mc_p: f64) -> Tensor {
        let (_, _, h, w) = x1.size4().unwrap();
        let up = x1.upsample_bilinear2d(&[h * 2, w * 2], true, None, None);
        let (_, _, h2, w2) = x2.size4().unwrap();
        let dy = h2 - h * 2;
        let dx = w2 - w * 2;
        let up = up.pad(&[dx / 2, dx - dx / 2, dy / 2, dy - dy / 2], "constant", 0.0);
        let x = Tensor::cat(&[x2, &up], 1);
        conv.1.forward(&conv.0.forward(&x, train, mc_p), train, mc_p)
    }

    pub fn forward(&self, x: &Tensor, train: bool, mc_p: f64) -> Tensor {
        let x1 = self.inc.1.forward(&self.inc.0.forward(x, train, mc_p), train, mc_p);

        let d = &self.down1;
        let x2 = d.1.forward(&d.0.forward(&x1.max_pool2d_default(2), train, mc_p), train, mc_p);
        let d = &self.down2;
        let x3 = d.1.forward(&d.0.forward(&x2.max_pool2d_default(2), train, mc_p), train, mc_p);
        let d = &self.down3;
        let x4 = d.1.forward(&d.0.forward(&x3.max_pool2d_default(2), train, mc_p), train, mc_p);
        let d = &self.down4;
        let x5 = d.1.forward(&d.0.forward(&x4.max_pool2d_default(2), train, mc_p), train, mc_p);

        let x = self.up_step(&x5, &x4, &self.up1, train, mc_p);
        let x = self.up_step(&x, &x3, &self.up2, train, mc_p);
        let x = self.up_step(&x, &x2, &self.up3, train, mc_p);
        let x = self.up_step(&x, &x1, &self.up4, train, mc_p);

        let y = self.outc.forward(&x);
        if mc_p > 0.0 {
            let mask = y.rand_like().gt(mc_p).to_kind(Kind::Float);
            y * mask
        } else {
            y
        }
    }
}

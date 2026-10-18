use ndarray::{ArrayD, IxDyn};
use tch::{Device, Kind, Tensor};

pub fn tensor_to_array(t: &Tensor) -> ArrayD<f64> {
    let t = t.to_kind(Kind::Float).to_device(Device::Cpu);
    let shape: Vec<usize> = t.size().iter().map(|&s| s as usize).collect();
    let numel: usize = shape.iter().product();
    let mut data = vec![0f32; numel.max(1)];
    t.copy_data(&mut data, numel);
    let data: Vec<f64> = data.into_iter().map(|v| v as f64).collect();
    ArrayD::from_shape_vec(IxDyn(&shape), data).unwrap()
}

pub fn array_to_tensor(a: &ArrayD<f64>, device: Device) -> Tensor {
    let shape: Vec<i64> = a.shape().iter().map(|&s| s as i64).collect();
    let data: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    Tensor::from_slice(&data).reshape(&shape).to_device(device)
}

pub fn load_image_rgb_tensor(path: &str, device: Device) -> Tensor {
    let img = image::open(path).expect("open image").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; 3 * h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            data[0 * h * w + y * w + x] = p[0] as f32 / 255.0;
            data[1 * h * w + y * w + x] = p[1] as f32 / 255.0;
            data[2 * h * w + y * w + x] = p[2] as f32 / 255.0;
        }
    }
    Tensor::from_slice(&data)
        .reshape(&[3, h as i64, w as i64])
        .to_device(device)
}

pub fn load_g_channel_tensor(path: &str, repeat: i64, device: Device) -> Tensor {
    let img = image::open(path).expect("open image").to_rgb8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            data[y * w + x] = p[1] as f32 / 255.0;
        }
    }
    let g = Tensor::from_slice(&data).reshape(&[1, h as i64, w as i64]);
    g.repeat(&[repeat, 1, 1]).to_device(device)
}

pub fn load_mask_array(path: &str) -> ArrayD<f64> {
    let img = image::open(path).expect("open mask").to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f64; h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            data[y * w + x] = p[0] as f64 / 255.0;
        }
    }
    ArrayD::from_shape_vec(IxDyn(&[h, w]), data).unwrap()
}

pub fn load_mask_tensor(path: &str, device: Device) -> Tensor {
    let img = image::open(path).expect("open mask").to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut data = vec![0f32; h * w];
    for y in 0..h {
        for x in 0..w {
            let p = img.get_pixel(x as u32, y as u32);
            data[y * w + x] = p[0] as f32 / 255.0;
        }
    }
    Tensor::from_slice(&data)
        .reshape(&[1, h as i64, w as i64])
        .to_device(device)
}

pub fn binary_entropy(probs: &ArrayD<f64>) -> ArrayD<f64> {
    probs.mapv(|p| -(p * (p + 1e-8).ln() + (1.0 - p) * (1.0 - p + 1e-8).ln()))
}

pub fn squeeze(a: &ArrayD<f64>) -> ArrayD<f64> {
    let shape: Vec<usize> = a.shape().iter().cloned().filter(|&d| d != 1).collect();
    a.clone().into_shape_with_order(IxDyn(&shape)).unwrap()
}

pub fn resize_nearest(a: &ArrayD<f64>, out_h: usize, out_w: usize) -> ArrayD<f64> {
    let a = a.clone().into_dimensionality::<ndarray::Ix2>().unwrap();
    let (in_h, in_w) = a.dim();
    let mut out = ndarray::Array2::<f64>::zeros((out_h, out_w));
    for i in 0..out_h {
        let si = (i * in_h / out_h).min(in_h - 1);
        for j in 0..out_w {
            let sj = (j * in_w / out_w).min(in_w - 1);
            out[[i, j]] = a[[si, sj]];
        }
    }
    out.into_dyn()
}

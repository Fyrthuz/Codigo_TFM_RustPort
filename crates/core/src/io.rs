use std::collections::HashMap;
use std::path::Path;

use ndarray::{ArrayD, ArrayViewD, Axis, IxDyn};

pub fn save_image(array: &ArrayViewD<f64>, path: &str) {
    let mut a = array.to_owned();

    if a.ndim() == 5 {
        a = a.index_axis_move(Axis(0), 0);
    }
    if a.ndim() == 4 {
        let mid = a.shape()[1] / 2;
        let view = a.view();
        a = view.index_axis(Axis(1), mid).to_owned();
    }
    if a.ndim() == 3 {
        let c = a.shape()[0];
        if c == 1 || c == 3 || c == 4 {
            a = permute_hwc(&a);
        } else {
            let mid = a.shape()[0] / 2;
            let view = a.view();
            a = view.index_axis(Axis(0), mid).to_owned();
        }
    }
    if a.ndim() == 3 && a.shape()[2] == 1 {
        let view = a.view();
        a = view.index_axis(Axis(2), 0).to_owned();
    }

    // squeeze trailing/leading singleton dims
    let shape: Vec<usize> = a.shape().iter().cloned().filter(|&d| d != 1).collect();
    if let Ok(reshaped) = a.clone().into_shape_with_order(IxDyn(&shape)) {
        a = reshaped;
    }

    if a.ndim() == 0 {
        let v = a.iter().next().copied().unwrap_or(0.0);
        a = ArrayD::from_shape_vec(IxDyn(&[1, 1]), vec![v]).unwrap();
    }
    if a.ndim() == 1 {
        let v: Vec<f64> = a.iter().copied().collect();
        let w = v.len();
        a = ArrayD::from_shape_vec(IxDyn(&[1, w]), v).unwrap();
    }

    let h = a.shape()[0];
    let w = a.shape()[1];
    let is_rgb = a.ndim() == 3 && a.shape()[2] == 3;

    if is_rgb {
        let mut buf = Vec::with_capacity(h * w * 3);
        for i in 0..h {
            for j in 0..w {
                for c in 0..3 {
                    let v = a[[i, j, c]];
                    buf.push(to_u8(v));
                }
            }
        }
        if let Some(img) = image::RgbImage::from_raw(w as u32, h as u32, buf) {
            let _ = img.save(path);
        }
    } else {
        let mut buf = Vec::with_capacity(h * w);
        for i in 0..h {
            for j in 0..w {
                buf.push(to_u8(a[[i, j]]));
            }
        }
        if let Some(img) = image::GrayImage::from_raw(w as u32, h as u32, buf) {
            let _ = img.save(path);
        }
    }
}

fn to_u8(v: f64) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0).round() as u8
}

fn permute_hwc(a: &ArrayD<f64>) -> ArrayD<f64> {
    let (c, h, w) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    let mut out = ArrayD::zeros(IxDyn(&[h, w, c]));
    for i in 0..h {
        for j in 0..w {
            for k in 0..c {
                out[[i, j, k]] = a[[k, i, j]];
            }
        }
    }
    out
}

pub fn save_metrics_csv(
    mean_results: &HashMap<String, HashMap<String, f64>>,
    methods: &[String],
    metrics: &[String],
    dir: &str,
    filename: &str,
) {
    let path = Path::new(dir).join(filename);
    let mut w = csv::Writer::from_path(&path).expect("open csv");
    let mut header: Vec<String> = vec![String::new()];
    header.extend(metrics.iter().cloned());
    w.write_record(&header).expect("write header");
    for method in methods {
        let mut row: Vec<String> = vec![method.clone()];
        for metric in metrics {
            let v = mean_results
                .get(method)
                .and_then(|m| m.get(metric))
                .copied()
                .unwrap_or(f64::NAN);
            row.push(format_float(v));
        }
        w.write_record(&row).expect("write row");
    }
    let _ = w.flush();
}

pub fn save_detailed_metrics_csv(
    overall_metrics: &HashMap<String, HashMap<String, HashMap<String, f64>>>,
    methods: &[String],
    metrics: &[String],
    dir: &str,
    filename: &str,
) {
    let path = Path::new(dir).join(filename);
    let mut w = csv::Writer::from_path(&path).expect("open csv");
    w.write_record(&["sample", "method", "metric", "value"]).expect("header");
    let mut samples: Vec<&String> = overall_metrics.keys().collect();
    samples.sort();
    for sample in samples {
        let sample_data = &overall_metrics[sample];
        for method in methods {
            if !sample_data.contains_key(method) {
                continue;
            }
            for metric in metrics {
                let v = sample_data[method].get(metric).copied().unwrap_or(f64::NAN);
                let value = if v.is_nan() {
                    String::new()
                } else {
                    format_float(v)
                };
                w.write_record(&[sample.as_str(), method.as_str(), metric.as_str(), value.as_str()])
                    .expect("write row");
            }
        }
    }
    let _ = w.flush();
}

pub fn format_float(v: f64) -> String {
    if v.is_nan() {
        String::new()
    } else if v == v.trunc() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

pub fn ensure_dir(path: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(path)
}

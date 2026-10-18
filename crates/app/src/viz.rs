use std::collections::HashMap;

use plotters::prelude::*;
use rand::SeedableRng;

const COLORS: [RGBColor; 6] = [
    RGBColor(0x1f, 0x77, 0xb4),
    RGBColor(0xff, 0x7f, 0x0e),
    RGBColor(0x2c, 0xa0, 0x2c),
    RGBColor(0xd6, 0x27, 0x28),
    RGBColor(0x94, 0x67, 0xbd),
    RGBColor(0x8c, 0x56, 0x4b),
];

fn color_for(method: &str) -> RGBColor {
    const ORDER: [&str; 6] = ["normal", "mc_dropout", "tta", "noisy", "fusion", "crf"];
    ORDER.iter().position(|m| *m == method).map(|i| COLORS[i]).unwrap_or(RGBColor(0, 0, 0))
}

pub fn plot_metrics_comparison(
    mean_results: &HashMap<String, HashMap<String, f64>>,
    methods: &[String],
    metrics: &[String],
    dir: &str,
) {
    let path = format!("{}/metrics_summary.png", dir);
    let n = metrics.len();
    let cols: usize = 3;
    let rows: usize = (n as usize + cols - 1) / cols;
    let root = BitMapBackend::new(&path, (500 * cols as u32, 350 * rows as u32)).into_drawing_area();
    let _ = root.fill(&WHITE);
    let areas = root.split_evenly((rows, cols));
    for (i, metric) in metrics.iter().enumerate() {
        let max_v = mean_results
            .values()
            .filter_map(|m| m.get(metric))
            .cloned()
            .fold(0.0f64, f64::max);
        let mut chart = ChartBuilder::on(&areas[i])
            .caption(metric.as_str(), ("sans-serif", 18))
            .margin(10)
            .build_cartesian_2d(0f64..methods.len() as f64, 0.0..(max_v * 1.15).max(1e-6))
            .unwrap();
        let _ = chart.configure_mesh().disable_mesh().draw();
        for (j, method) in methods.iter().enumerate() {
            let v = mean_results.get(method).and_then(|m| m.get(metric)).copied().unwrap_or(0.0);
            let c = color_for(method);
            let _ = chart.draw_series(std::iter::once(Rectangle::new(
                [(j as f64, 0.0), (j as f64 + 1.0, v)],
                c.filled(),
            )));
        }
    }
    let _ = root.present();
    println!("Saved {}", path);
}

pub fn plot_box_comparison(
    overall_metrics: &HashMap<String, HashMap<String, HashMap<String, f64>>>,
    methods: &[String],
    metrics: &[String],
    dir: &str,
) {
    let path = format!("{}/box_plot_comparison.png", dir);
    let n = metrics.len();
    let cols: usize = 3;
    let rows: usize = (n as usize + cols - 1) / cols;
    let root = BitMapBackend::new(&path, (500 * cols as u32, 350 * rows as u32)).into_drawing_area();
    let _ = root.fill(&WHITE);
    let areas = root.split_evenly((rows, cols));
    for (i, metric) in metrics.iter().enumerate() {
        let mut series: Vec<Vec<f64>> = Vec::new();
        let mut max_v = 0.0f64;
        for method in methods.iter() {
            let mut vals: Vec<f64> = overall_metrics
                .values()
                .filter_map(|sd| sd.get(method).and_then(|m| m.get(metric)).copied())
                .filter(|v| v.is_finite())
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if let Some(&mx) = vals.last() {
                max_v = max_v.max(mx);
            }
            series.push(vals);
        }
        let mut chart = ChartBuilder::on(&areas[i])
            .caption(metric.as_str(), ("sans-serif", 18))
            .margin(10)
            .build_cartesian_2d(0f64..methods.len() as f64, 0.0..(max_v * 1.15).max(1e-6))
            .unwrap();
        let _ = chart.configure_mesh().disable_mesh().draw();
        for (j, vals) in series.iter().enumerate() {
            if vals.is_empty() {
                continue;
            }
            let (min, q1, med, q3, max) = quartiles(vals);
            let c = color_for(&methods[j]);
            let x0 = j as f64;
            let x1 = j as f64 + 1.0;
            let cx = (x0 + x1) / 2.0;
            let _ = chart.draw_series(std::iter::once(Rectangle::new(
                [(x0 + 0.2, q1), (x1 - 0.2, q3)],
                c.filled(),
            )));
            let _ = chart.draw_series(std::iter::once(PathElement::new(
                vec![(x0 + 0.2, med), (x1 - 0.2, med)],
                BLACK,
            )));
            let _ = chart.draw_series(std::iter::once(PathElement::new(
                vec![(cx, min), (cx, q1)],
                BLACK,
            )));
            let _ = chart.draw_series(std::iter::once(PathElement::new(
                vec![(cx, q3), (cx, max)],
                BLACK,
            )));
        }
    }
    let _ = root.present();
    println!("Saved {}", path);
}

fn quartiles(vals: &[f64]) -> (f64, f64, f64, f64, f64) {
    let n = vals.len();
    let min = vals[0];
    let max = vals[n - 1];
    let med = percentile(vals, 50.0);
    let q1 = percentile(vals, 25.0);
    let q3 = percentile(vals, 75.0);
    (min, q1, med, q3, max)
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    let idx = (q / 100.0) * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (sorted[hi] - sorted[lo]) * (idx - lo as f64)
    }
}

pub struct ForestRow {
    pub comparison: String,
    pub delta: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub p_wilcoxon: f64,
}

pub fn plot_forest(rows: &[ForestRow], path: &str, title: &str) {
    let n = rows.len();
    let lo = rows.iter().map(|r| r.ci_low).fold(f64::INFINITY, f64::min).min(0.0);
    let hi = rows.iter().map(|r| r.ci_high).fold(f64::NEG_INFINITY, f64::max).max(0.0);
    let span = (hi - lo).max(1e-6);
    let x_lo = lo - 0.15 * span;
    let x_hi = hi + 0.15 * span;
    let root = BitMapBackend::new(path, (700, 60 + 40 * n as u32)).into_drawing_area();
    let _ = root.fill(&WHITE);
    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 16))
        .margin(10)
        .build_cartesian_2d(x_lo..x_hi, 0f64..n as f64)
        .unwrap();
    let _ = chart.configure_mesh().disable_mesh().draw();
    for (i, r) in rows.iter().enumerate() {
        let y = n as f64 - i as f64 - 0.5;
        let sig = r.p_wilcoxon < 0.05;
        let c = if sig {
            if r.delta > 0.0 {
                RGBColor(0x1a, 0x7a, 0x1a)
            } else {
                RGBColor(0xb3, 0x00, 0x00)
            }
        } else {
            RGBColor(0x8a, 0x8a, 0x8a)
        };
        let _ = chart.draw_series(std::iter::once(PathElement::new(
            vec![(r.ci_low, y), (r.ci_high, y)],
            c,
        )));
        let _ = chart.draw_series(std::iter::once(Circle::new((r.delta, y), 4, c.filled())));
    }
    let _ = root.present();
    println!("Saved {}", path);
}

// ---------------------------------------------------------------------------
// Qualitative panels and per-method distributions
// ---------------------------------------------------------------------------

const METHOD_COLUMNS: [(&str, &str, &str); 6] = [
    ("normal", "Normal", "original/mask.png"),
    ("mc_dropout", "MC Dropout", "mc_dropout/mean_mask_prediction.png"),
    ("tta", "TTA", "tta/mean_mask_prediction.png"),
    ("noisy", "Noisy", "noisy/mean_mask_prediction.png"),
    ("fusion", "Fusion", "fusion/mask.png"),
    ("crf", "CRF", "refined/probability.png"),
];

fn load_gray(path: &str) -> (Vec<u8>, usize, usize) {
    let img = image::open(path).unwrap_or_else(|_| panic!("open {}", path)).to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let data = img.into_raw();
    (data, w, h)
}

fn load_mask_bin(path: &str, tw: usize, th: usize) -> Vec<bool> {
    let img = image::open(path).unwrap_or_else(|_| panic!("open {}", path)).to_luma8();
    let (w, h) = (img.width() as usize, img.height() as usize);
    let raw = img.into_raw();
    let mut out = vec![false; tw * th];
    for y in 0..th {
        let sy = (y * h / th).min(h - 1);
        for x in 0..tw {
            let sx = (x * w / tw).min(w - 1);
            out[y * tw + x] = raw[sy * w + sx] > 127;
        }
    }
    out
}

fn is_edge(mask: &[bool], i: usize, w: usize, h: usize) -> bool {
    if !mask[i] {
        return false;
    }
    let (x, y) = (i % w, i / w);
    if x > 0 && !mask[i - 1] {
        return true;
    }
    if x + 1 < w && !mask[i + 1] {
        return true;
    }
    if y > 0 && !mask[i - w] {
        return true;
    }
    if y + 1 < h && !mask[i + w] {
        return true;
    }
    false
}

fn composite_rgb(gray: &[u8], mask: &[bool], gt: &[bool], w: usize, h: usize) -> Vec<u8> {
    let mut buf = vec![0u8; w * h * 3];
    for i in 0..(w * h) {
        let g = gray[i] as f64;
        let mut r = g;
        let mut gg = g;
        let mut b = g;
        if mask[i] {
            r = r * 0.58 + 242.25 * 0.42;
            gg = gg * 0.58 + 38.25 * 0.42;
            b = b * 0.58 + 38.25 * 0.42;
        }
        if is_edge(gt, i, w, h) {
            r = 255.0;
            gg = 255.0;
            b = 0.0;
        }
        buf[i * 3] = r.clamp(0.0, 255.0) as u8;
        buf[i * 3 + 1] = gg.clamp(0.0, 255.0) as u8;
        buf[i * 3 + 2] = b.clamp(0.0, 255.0) as u8;
    }
    buf
}

fn draw_cell(root: &DrawingArea<BitMapBackend, plotters::coord::Shift>, x: i32, y: i32, cell: usize, buf: &[u8], src_w: usize, src_h: usize) {
    let rgb = resize_rgb(buf, src_w, src_h, cell, cell);
    let el = BitMapElement::with_ref((x, y), (cell as u32, cell as u32), &rgb);
    if let Some(el) = el {
        let _ = root.draw(&el);
    }
}

fn resize_rgb(buf: &[u8], sw: usize, sh: usize, tw: usize, th: usize) -> Vec<u8> {
    let mut out = vec![0u8; tw * th * 3];
    for y in 0..th {
        let sy = (y * sh / th).min(sh - 1);
        for x in 0..tw {
            let sx = (x * sw / tw).min(sw - 1);
            let si = (sy * sw + sx) * 3;
            let di = (y * tw + x) * 3;
            out[di] = buf[si];
            out[di + 1] = buf[si + 1];
            out[di + 2] = buf[si + 2];
        }
    }
    out
}

pub fn plot_qualitative_panel(results_dir: &str, samples: &[(String, String)], out_path: &str) {
    let n_rows = samples.len();
    let n_cols = 2 + METHOD_COLUMNS.len() + 1;
    let cell = 180usize;
    let label_w = 150i32;
    let title_h = 30i32;
    let width = label_w + n_cols as i32 * cell as i32;
    let height = title_h + n_rows as i32 * cell as i32;

    let root = BitMapBackend::new(out_path, (width as u32, height as u32)).into_drawing_area();
    let _ = root.fill(&WHITE);

    let titles = ["Imagen", "Ground truth", "Normal", "MC Dropout", "TTA", "Noisy", "Fusion", "CRF", "Incertidumbre"];
    for (c, t) in titles.iter().enumerate() {
        let x = label_w + c as i32 * cell as i32 + cell as i32 / 2;
        let _ = root.draw(&Text::new(*t, (x, title_h / 2 - 8), ("sans-serif", 14).into_font().color(&BLACK)));
    }

    for (r, (kind, sample)) in samples.iter().enumerate() {
        let dir = format!("{}/{}", results_dir, sample);
        let (gray, gw, gh) = load_gray(&format!("{}/original_image.png", dir));
        let gt = load_mask_bin(&format!("{}/ground_truth.png", dir), gw, gh);
        let empty = vec![false; gw * gh];

        let ry = title_h + r as i32 * cell as i32;
        let _ = root.draw(&Text::new(kind.as_str(), (4, ry + cell as i32 / 2), ("sans-serif", 13).into_font().color(&BLACK)));

        let img_buf = composite_rgb(&gray, &empty, &empty, gw, gh);
        draw_cell(&root, label_w, ry, cell, &img_buf, gw, gh);

        let gt_buf = composite_rgb(&gray, &empty, &gt, gw, gh);
        draw_cell(&root, label_w + cell as i32, ry, cell, &gt_buf, gw, gh);

        for (m, (_, _, rel)) in METHOD_COLUMNS.iter().enumerate() {
            let mask = load_mask_bin(&format!("{}/{}", dir, rel), gw, gh);
            let buf = composite_rgb(&gray, &mask, &gt, gw, gh);
            let x = label_w + (m as i32 + 2) * cell as i32;
            draw_cell(&root, x, ry, cell, &buf, gw, gh);
        }

        let (unc, uw, uh) = load_gray(&format!("{}/fusion/uncertainty.png", dir));
        let unc_rgb = {
            let mut buf = vec![0u8; uw * uh * 3];
            for i in 0..(uw * uh) {
                let v = unc[i];
                buf[i * 3] = v;
                buf[i * 3 + 1] = v;
                buf[i * 3 + 2] = v;
            }
            buf
        };
        let x = label_w + (n_cols as i32 - 1) * cell as i32;
        draw_cell(&root, x, ry, cell, &unc_rgb, uw, uh);
    }
    let _ = root.present();
    println!("Saved {}", out_path);
}

pub fn plot_metric_distributions(
    overall: &HashMap<String, HashMap<String, HashMap<String, f64>>>,
    methods: &[String],
    out_path: &str,
) {
    let dist_metrics = [("dice", "Dice"), ("iou", "IoU"), ("nll_fg", "NLL_fg"), ("ece", "ECE")];
    let root = BitMapBackend::new(out_path, (4 * 340, 340)).into_drawing_area();
    let _ = root.fill(&WHITE);
    let areas = root.split_evenly((1, 4));

    for (i, (metric, label)) in dist_metrics.iter().enumerate() {
        let mut series: Vec<Vec<f64>> = Vec::new();
        let mut max_v = 0.0f64;
        for method in methods.iter() {
            let mut vals: Vec<f64> = overall
                .values()
                .filter_map(|sd| sd.get(method).and_then(|m| m.get(*metric)).copied())
                .filter(|v| v.is_finite())
                .collect();
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if let Some(&mx) = vals.last() {
                max_v = max_v.max(mx);
            }
            series.push(vals);
        }
        let y_cap = if *metric == "nll_fg" {
            let p99 = tfm_core::fusion::percentile(&series.concat(), 99.0);
            (p99 * 1.05).max(1e-6)
        } else if *metric == "dice" || *metric == "iou" {
            1.02
        } else {
            max_v * 1.10
        };

        let mut chart = ChartBuilder::on(&areas[i])
            .caption(*label, ("sans-serif", 18))
            .margin(10)
            .build_cartesian_2d(0f64..methods.len() as f64, 0.0..y_cap)
            .unwrap();
        let _ = chart.configure_mesh().disable_mesh().draw();

        for (j, vals) in series.iter().enumerate() {
            if vals.is_empty() {
                continue;
            }
            let (min, q1, med, q3, max) = quartiles(vals);
            let c = color_for(&methods[j]);
            let x0 = j as f64;
            let x1 = j as f64 + 1.0;
            let cx = (x0 + x1) / 2.0;
            let _ = chart.draw_series(std::iter::once(Rectangle::new(
                [(x0 + 0.2, q1), (x1 - 0.2, q3)],
                c.filled(),
            )));
            let _ = chart.draw_series(std::iter::once(PathElement::new(vec![(x0 + 0.2, med), (x1 - 0.2, med)], BLACK)));
            let _ = chart.draw_series(std::iter::once(PathElement::new(vec![(cx, min), (cx, q1)], BLACK)));
            let _ = chart.draw_series(std::iter::once(PathElement::new(vec![(cx, q3), (cx, max)], BLACK)));
            let mut rng = rand::rngs::StdRng::seed_from_u64(0);
            for v in vals {
                let jx = cx + rand::Rng::gen_range(&mut rng, -0.14..0.14);
                let _ = chart.draw_series(std::iter::once(Circle::new((jx, *v), 2, BLACK.mix(0.4).filled())));
            }
        }
    }
    let _ = root.present();
    println!("Saved {}", out_path);
}

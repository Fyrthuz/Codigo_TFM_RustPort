use ndarray::{Array2, Array3, ArrayViewD};

pub fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    let idx = (q / 100.0) * (n as f64 - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        v[lo]
    } else {
        v[lo] + (v[hi] - v[lo]) * (idx - lo as f64)
    }
}

fn otsu_threshold(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n_bins = 256usize;
    let mut hist = vec![0u64; n_bins];
    for &v in values {
        let mut b = (v * (n_bins as f64 - 1.0)).floor() as usize;
        if b >= n_bins {
            b = n_bins - 1;
        }
        hist[b] += 1;
    }
    let total = values.len() as f64;
    let mut sum = 0.0f64;
    for (i, &c) in hist.iter().enumerate() {
        sum += (i as f64 + 0.5) / n_bins as f64 * c as f64;
    }

    let mut sum_b = 0.0f64;
    let mut w_b = 0.0f64;
    let mut max_var = -1.0f64;
    let mut threshold = 0.0f64;
    for i in 0..n_bins {
        w_b += hist[i] as f64;
        if w_b == 0.0 {
            continue;
        }
        let w_f = total - w_b;
        if w_f == 0.0 {
            break;
        }
        sum_b += (i as f64 + 0.5) / n_bins as f64 * hist[i] as f64;
        let m_b = sum_b / w_b;
        let m_f = (sum - sum_b) / w_f;
        let var_between = w_b * w_f * (m_b - m_f) * (m_b - m_f);
        if var_between > max_var {
            max_var = var_between;
            threshold = (i as f64 + 0.5) / n_bins as f64;
        }
    }
    threshold
}

pub fn dynamic_threshold_multiclass(
    prob_maps: &ArrayViewD<f64>,
    method: &str,
    percentile_q: f64,
    k: f64,
) -> Vec<f64> {
    let m = method.to_lowercase();
    if prob_maps.ndim() == 2 {
        let vals: Vec<f64> = prob_maps.iter().copied().collect();
        let t = match m.as_str() {
            "otsu" => otsu_threshold(&vals),
            "percentile" => percentile(&vals, percentile_q),
            "mean_std" => {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                mean + k * var.sqrt()
            }
            _ => panic!("Unknown method. Choose from 'otsu', 'percentile', or 'mean_std'."),
        };
        vec![t]
    } else if prob_maps.ndim() == 3 {
        let n = prob_maps.shape()[0];
        let mut thresholds = Vec::with_capacity(n);
        for c in 0..n {
            let vals: Vec<f64> = prob_maps
                .index_axis(ndarray::Axis(0), c)
                .iter()
                .copied()
                .collect();
            let t = match m.as_str() {
                "otsu" => otsu_threshold(&vals),
                "percentile" => percentile(&vals, percentile_q),
                "mean_std" => {
                    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                    let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
                        / vals.len() as f64;
                    mean + k * var.sqrt()
                }
                _ => panic!("Unknown method. Choose from 'otsu', 'percentile', or 'mean_std'."),
            };
            thresholds.push(t);
        }
        thresholds
    } else {
        panic!("prob_maps must be either a 2D (binary) or 3D (multiclass) array.");
    }
}

#[allow(clippy::too_many_arguments)]
pub fn weighted_average_with_uncertainty(
    mc_mean: &Array2<f64>,
    mc_uncert: &Array2<f64>,
    tta_mean: &Array2<f64>,
    tta_uncert: &Array2<f64>,
    noise_mean: &Array2<f64>,
    noise_uncert: &Array2<f64>,
    weighting_method: &str,
    beta: f64,
    alpha: f64,
    threshold_method: &str,
    percentile_q: f64,
    k: f64,
    epsilon: f64,
) -> (Array2<f64>, Array2<f64>, Array2<u8>) {
    let shape = mc_mean.shape();
    let mut prob_maps = Array3::<f64>::zeros((3, shape[0], shape[1]));
    let mut uncertainty_maps = Array3::<f64>::zeros((3, shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            prob_maps[[0, i, j]] = mc_mean[[i, j]];
            prob_maps[[1, i, j]] = tta_mean[[i, j]];
            prob_maps[[2, i, j]] = noise_mean[[i, j]];
            uncertainty_maps[[0, i, j]] = mc_uncert[[i, j]];
            uncertainty_maps[[1, i, j]] = tta_uncert[[i, j]];
            uncertainty_maps[[2, i, j]] = noise_uncert[[i, j]];
        }
    }

    let weights = match weighting_method.to_lowercase().as_str() {
        "inverse" => uncertainty_maps.mapv(|u| 1.0 / (u + epsilon)),
        "exponential" => uncertainty_maps.mapv(|u| (-beta * u).exp()),
        "powerlaw" => uncertainty_maps.mapv(|u| (1.0 - u).powf(alpha)),
        _ => panic!("Unsupported weighting method."),
    };

    let mut consensus_prob = Array2::<f64>::zeros((shape[0], shape[1]));
    let mut consensus_uncertainty = Array2::<f64>::zeros((shape[0], shape[1]));
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            let w_sum = weights[[0, i, j]] + weights[[1, i, j]] + weights[[2, i, j]] + epsilon;
            let w0 = weights[[0, i, j]] / w_sum;
            let w1 = weights[[1, i, j]] / w_sum;
            let w2 = weights[[2, i, j]] / w_sum;
            consensus_prob[[i, j]] = prob_maps[[0, i, j]] * w0
                + prob_maps[[1, i, j]] * w1
                + prob_maps[[2, i, j]] * w2;
            consensus_uncertainty[[i, j]] = uncertainty_maps[[0, i, j]] * w0
                + uncertainty_maps[[1, i, j]] * w1
                + uncertainty_maps[[2, i, j]] * w2;
        }
    }

    let consensus_mask = if threshold_method == "naive" {
        consensus_prob.mapv(|p| if p > 0.5 { 1u8 } else { 0u8 })
    } else {
        let thresholds = dynamic_threshold_multiclass(
            &consensus_prob.view().into_dyn(),
            threshold_method,
            percentile_q,
            k,
        );
        consensus_prob.mapv(|p| if p > thresholds[0] { 1u8 } else { 0u8 })
    };

    (consensus_prob, consensus_uncertainty, consensus_mask)
}

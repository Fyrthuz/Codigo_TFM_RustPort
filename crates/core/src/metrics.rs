use ndarray::{ArrayD, ArrayViewD, IxDyn};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics {
    pub nll: f64,
    pub nll_fg: f64,
    pub ece: f64,
    pub brier: f64,
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
}

fn squeeze(a: &ArrayViewD<f64>) -> ArrayD<f64> {
    let shape: Vec<usize> = a
        .shape()
        .iter()
        .cloned()
        .filter(|&d| d != 1)
        .collect();
    a.to_owned().into_shape_with_order(IxDyn(&shape)).unwrap()
}

pub fn compute_iou(pred: &ArrayViewD<f64>, target: &ArrayViewD<f64>, threshold: f64) -> f64 {
    let pred = squeeze(pred).mapv(|v| v > threshold);
    let target = squeeze(target).mapv(|v| v > 0.5);
    let mut intersection: u64 = 0;
    let mut union: u64 = 0;
    for (p, t) in pred.iter().zip(target.iter()) {
        if *p && *t {
            intersection += 1;
        }
        if *p || *t {
            union += 1;
        }
    }
    if union == 0 {
        return if intersection == 0 { 1.0 } else { 0.0 };
    }
    intersection as f64 / union as f64
}

pub fn compute_dice(pred: &ArrayViewD<f64>, target: &ArrayViewD<f64>, threshold: f64) -> f64 {
    let pred = squeeze(pred).mapv(|v| v > threshold);
    let target = squeeze(target).mapv(|v| v > 0.5);
    let mut intersection: u64 = 0;
    let mut pred_sum: u64 = 0;
    let mut target_sum: u64 = 0;
    for (p, t) in pred.iter().zip(target.iter()) {
        if *p {
            pred_sum += 1;
        }
        if *t {
            target_sum += 1;
        }
        if *p && *t {
            intersection += 1;
        }
    }
    if pred_sum + target_sum == 0 {
        return 1.0;
    }
    (2 * intersection) as f64 / (pred_sum + target_sum) as f64
}

fn digitize_right(x: f64, bins: &[f64]) -> i64 {
    let n = bins.len();
    for i in 1..n {
        if bins[i - 1] < x && x <= bins[i] {
            return i as i64;
        }
    }
    if x <= bins[0] {
        0
    } else {
        n as i64
    }
}

pub fn classwise_ece(prob_flat: &[f64], gt_flat: &[i64], n_bins: usize) -> f64 {
    let n = prob_flat.len().max(1);
    let mut bin_edges: Vec<f64> = Vec::with_capacity(n_bins + 1);
    for i in 0..=n_bins {
        bin_edges.push(i as f64 / n_bins as f64);
    }
    let mut ece_total = 0.0;
    for cls in 0..=1i64 {
        let scores: Vec<f64> = prob_flat
            .iter()
            .map(|&p| if cls == 1 { p } else { 1.0 - p })
            .collect();
        let correct: Vec<f64> = gt_flat
            .iter()
            .map(|&g| if g == cls { 1.0 } else { 0.0 })
            .collect();
        let bin_indices: Vec<i64> = scores.iter().map(|&s| digitize_right(s, &bin_edges)).collect();
        let mut ece_cls = 0.0;
        for b in 1..=n_bins as i64 {
            let mut bin_size = 0usize;
            let mut conf_sum = 0.0;
            let mut freq_sum = 0.0;
            for i in 0..bin_indices.len() {
                if bin_indices[i] == b {
                    bin_size += 1;
                    conf_sum += scores[i];
                    freq_sum += correct[i];
                }
            }
            if bin_size == 0 {
                continue;
            }
            let conf = conf_sum / bin_size as f64;
            let freq = freq_sum / bin_size as f64;
            ece_cls += (freq - conf).abs() * bin_size as f64;
        }
        ece_total += ece_cls / n as f64;
    }
    ece_total / 2.0
}

pub fn compute_metrics(prob: &ArrayViewD<f64>, gt_mask: &ArrayViewD<f64>, epsilon: f64) -> Metrics {
    let mut prob = prob.to_owned();
    let gt = gt_mask.to_owned();

    while prob.ndim() > gt.ndim() && prob.shape()[0] == 1 {
        prob = prob.index_axis_move(ndarray::Axis(0), 0);
    }
    if prob.ndim() > gt.ndim() && prob.shape()[0] > 1 {
        prob = prob
            .view()
            .map_axis(ndarray::Axis(0), |axis| {
                axis.fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            })
            .to_owned();
    }

    let mut prob_flat: Vec<f64> = prob.iter().copied().collect();
    let gt_flat: Vec<i64> = gt.iter().map(|&v| if v > 0.0 { 1 } else { 0 }).collect();

    for p in prob_flat.iter_mut() {
        let clipped = p.clamp(epsilon, 1.0 - epsilon);
        *p = if clipped.is_nan() { 0.5 } else { clipped };
    }

    let nll = -prob_flat
        .iter()
        .zip(gt_flat.iter())
        .map(|(&p, &g)| g as f64 * p.ln() + (1.0 - g as f64) * (1.0 - p).ln())
        .sum::<f64>()
        / prob_flat.len() as f64;

    let fg: Vec<f64> = prob_flat
        .iter()
        .zip(gt_flat.iter())
        .filter(|(_, &g)| g == 1)
        .map(|(&p, _)| p)
        .collect();
    let nll_fg = if fg.is_empty() {
        f64::NAN
    } else {
        -fg.iter().map(|p| p.ln()).sum::<f64>() / fg.len() as f64
    };

    let brier = prob_flat
        .iter()
        .zip(gt_flat.iter())
        .map(|(&p, &g)| (p - g as f64).powi(2))
        .sum::<f64>()
        / prob_flat.len() as f64;

    let mut tp = 0.0f64;
    let mut fp = 0.0f64;
    let mut tn = 0.0f64;
    let mut fn_ = 0.0f64;
    for (&p, &g) in prob_flat.iter().zip(gt_flat.iter()) {
        let pred = if p > 0.5 { 1 } else { 0 };
        match (pred, g) {
            (1, 1) => tp += 1.0,
            (1, 0) => fp += 1.0,
            (0, 0) => tn += 1.0,
            (0, 1) => fn_ += 1.0,
            _ => {}
        }
    }

    let accuracy = (tp + tn) / (tp + tn + fp + fn_ + epsilon);
    let precision = tp / (tp + fp + epsilon);
    let recall = tp / (tp + fn_ + epsilon);

    let ece = classwise_ece(&prob_flat, &gt_flat, 10);

    Metrics {
        nll,
        nll_fg,
        ece,
        brier,
        accuracy,
        precision,
        recall,
    }
}

pub fn certainty_score(
    uncertainty_map: &ArrayViewD<f64>,
    ground_truth: &ArrayViewD<f64>,
    num_classes: usize,
) -> f64 {
    let mut uncertainty = uncertainty_map.to_owned();
    let gt = squeeze(ground_truth);
    let gt = gt.mapv(|v| v > 0.5);

    if uncertainty.ndim() > gt.ndim() {
        uncertainty = uncertainty.mean_axis(ndarray::Axis(0)).unwrap();
    }
    let mut uncertainty = squeeze(&uncertainty.view());

    if gt.iter().filter(|&&v| v).count() == 0 {
        return f64::NAN;
    }

    for v in uncertainty.iter_mut() {
        if v.is_nan() {
            *v = 0.0;
        } else if *v == f64::INFINITY {
            *v = 1.0;
        } else if *v == f64::NEG_INFINITY {
            *v = 0.0;
        }
    }

    let max_entropy = (num_classes as f64).ln();
    if max_entropy == 0.0 {
        return 1.0;
    }

    let mut count = 0usize;
    let mut sum = 0.0;
    for (u, g) in uncertainty.iter().zip(gt.iter()) {
        if *g {
            let norm = 1.0 - (*u / max_entropy);
            let norm = norm.clamp(0.0, 1.0);
            sum += norm;
            count += 1;
        }
    }
    sum / count as f64
}

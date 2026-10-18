use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::Deserialize;

use crate::dataset::{load_test_indices, recover_image_mask_pairs};
use crate::fusion::percentile as np_percentile;

#[derive(Debug, Clone, Deserialize)]
pub struct MetricRow {
    pub sample: String,
    pub method: String,
    pub metric: String,
    pub value: Option<f64>,
}

pub fn read_detailed_metrics(path: &str) -> anyhow::Result<Vec<MetricRow>> {
    let mut reader = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for rec in reader.records() {
        let rec = rec?;
        if rec.len() < 4 {
            continue;
        }
        let value = if rec[3].trim().is_empty() {
            None
        } else {
            rec[3].trim().parse::<f64>().ok()
        };
        rows.push(MetricRow {
            sample: rec[0].to_string(),
            method: rec[1].to_string(),
            metric: rec[2].to_string(),
            value,
        });
    }
    Ok(rows)
}

fn rankdata_average(x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && x[order[j + 1]] == x[order[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            ranks[k] = avg;
        }
        i = j + 1;
    }
    ranks
}

fn wilcoxon_distr(n: usize) -> Vec<f64> {
    let mut c = vec![1.0];
    for k in 1..=n {
        let m = c.len();
        let new_len = k * (k + 1) / 2 + 1;
        let mut nc = vec![0.0; new_len];
        for i in 0..m {
            nc[i] = c[i] * 0.5;
        }
        for i in 0..m {
            nc[new_len - m + i] += c[i] * 0.5;
        }
        c = nc;
    }
    c
}

fn normal_sf(x: f64) -> f64 {
    0.5 * erfc(x / std::f64::consts::SQRT_2)
}

fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn erfc(x: f64) -> f64 {
    1.0 - erf(x)
}

pub fn wilcoxon_pvalue(diffs: &[f64]) -> f64 {
    let nz: Vec<f64> = diffs.iter().copied().filter(|&d| d != 0.0).collect();
    let n = nz.len();
    if n == 0 {
        return f64::NAN;
    }
    let abs_d: Vec<f64> = nz.iter().map(|d| d.abs()).collect();
    let ranks = rankdata_average(&abs_d);
    let mut r_plus = 0.0;
    for (i, &d) in nz.iter().enumerate() {
        if d > 0.0 {
            r_plus += ranks[i];
        }
    }
    let has_ties = {
        let mut s = abs_d.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s.windows(2).any(|w| w[0] == w[1])
    };

    if !has_ties {
        let pmf = wilcoxon_distr(n);
        let mn = (n * (n + 1)) as f64 / 4.0;
        let r_floor = r_plus.floor() as usize;
        let r_ceil = r_plus.ceil() as usize;
        let sf = wilcoxon_sf(&pmf, mn, r_floor);
        let cdf = wilcoxon_cdf(&pmf, mn, r_ceil);
        (2.0 * sf.min(cdf)).clamp(0.0, 1.0)
    } else {
        let count = n as f64;
        let mn = count * (count + 1.0) * 0.25;
        let mut se_raw = count * (count + 1.0) * (2.0 * count + 1.0);
        let mut tie_correct = 0.0;
        let mut s = abs_d.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mut i = 0;
        while i < s.len() {
            let mut j = i;
            while j + 1 < s.len() && s[j + 1] == s[i] {
                j += 1;
            }
            let t = (j - i + 1) as f64;
            tie_correct += t * t * t - t;
            i = j + 1;
        }
        se_raw -= tie_correct / 2.0;
        let se = (se_raw / 24.0).sqrt();
        let z = (r_plus - mn) / se;
        let z_adj = z - z.signum() * 0.5 / se;
        2.0 * normal_sf(z_adj.abs())
    }
}

fn wilcoxon_sf(pmf: &[f64], mn: f64, k: usize) -> f64 {
    if (k as f64) <= mn {
        pmf[k..].iter().sum()
    } else {
        1.0 - pmf[..k].iter().sum::<f64>()
    }
}

fn wilcoxon_cdf(pmf: &[f64], mn: f64, k: usize) -> f64 {
    if (k as f64) <= mn {
        pmf[..=k].iter().sum()
    } else {
        1.0 - pmf[k + 1..].iter().sum::<f64>()
    }
}

#[derive(Debug, Clone)]
pub struct ComparisonResult {
    pub comparison: String,
    pub metric: String,
    pub delta: f64,
    pub ci_low: f64,
    pub ci_high: f64,
    pub p_wilcoxon: f64,
    pub n_patients: usize,
    pub n_slices: usize,
    pub pct_patients_improved: f64,
    pub pct_slices_improved: f64,
    pub cohen_dz: f64,
}

pub fn compare_methods(
    rows: &[MetricRow],
    patient_map: &HashMap<String, String>,
    method_a: &str,
    method_b: &str,
    metric: &str,
    n_boot: usize,
    seed: u64,
    alpha: f64,
) -> ComparisonResult {
    let mut a_map: HashMap<&str, f64> = HashMap::new();
    let mut b_map: HashMap<&str, f64> = HashMap::new();
    for r in rows {
        if r.metric != metric {
            continue;
        }
        if let Some(v) = r.value {
            if r.method == method_a {
                a_map.insert(&r.sample, v);
            } else if r.method == method_b {
                b_map.insert(&r.sample, v);
            }
        }
    }

    let mut samples: Vec<&str> = a_map.keys().copied().collect();
    samples.sort();

    let mut diff_values: Vec<f64> = Vec::new();
    let mut patient_labels: Vec<String> = Vec::new();
    for s in samples {
        if let (Some(&a), Some(&b)) = (a_map.get(s), b_map.get(s)) {
            if let Some(pid) = patient_map.get(s) {
                diff_values.push(a - b);
                patient_labels.push(pid.clone());
            }
        }
    }

    let mut diffs_by_patient: HashMap<&str, Vec<f64>> = HashMap::new();
    let mut key_order: Vec<&str> = Vec::new();
    for (d, p) in diff_values.iter().zip(patient_labels.iter()) {
        if !diffs_by_patient.contains_key(p.as_str()) {
            key_order.push(p.as_str());
        }
        diffs_by_patient.entry(p.as_str()).or_default().push(*d);
    }
    let n_patients = key_order.len();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut boot = vec![0.0; n_boot];
    for b in boot.iter_mut() {
        let mut total = 0.0;
        let mut count = 0usize;
        for _ in 0..n_patients {
            let j = rng.gen_range(0..n_patients);
            let vals = &diffs_by_patient[key_order[j]];
            total += vals.iter().sum::<f64>();
            count += vals.len();
        }
        *b = total / count as f64;
    }
    let ci_low = np_percentile(&boot, 100.0 * alpha / 2.0);
    let ci_high = np_percentile(&boot, 100.0 * (1.0 - alpha / 2.0));

    let patient_means: Vec<f64> = key_order
        .iter()
        .map(|k| {
            let v = &diffs_by_patient[k];
            v.iter().sum::<f64>() / v.len() as f64
        })
        .collect();

    let all_zero = patient_means
        .iter()
        .all(|&v| (v - 0.0).abs() <= 1e-8);
    let p_wilcoxon = if all_zero { 1.0 } else { wilcoxon_pvalue(&patient_means) };

    let n = patient_means.len();
    let mean = patient_means.iter().sum::<f64>() / n as f64;
    let sd = if n > 1 {
        let var = patient_means.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };
    let cohen_dz = if sd > 0.0 { mean / sd } else { f64::NAN };

    let n_patients_improved = patient_means.iter().filter(|&&v| v > 0.0).count();
    let n_slices_improved = diff_values.iter().filter(|&&v| v > 0.0).count();

    ComparisonResult {
        comparison: format!("{} vs {}", method_a, method_b),
        metric: metric.to_string(),
        delta: diff_values.iter().sum::<f64>() / diff_values.len() as f64,
        ci_low,
        ci_high,
        p_wilcoxon,
        n_patients,
        n_slices: diff_values.len(),
        pct_patients_improved: 100.0 * n_patients_improved as f64 / n_patients as f64,
        pct_slices_improved: 100.0 * n_slices_improved as f64 / diff_values.len() as f64,
        cohen_dz,
    }
}

pub const DEFAULT_COMPARISONS: [(&str, &str); 4] = [
    ("fusion", "normal"),
    ("crf", "fusion"),
    ("tta", "normal"),
    ("noisy", "normal"),
];
pub const DEFAULT_METRICS: [&str; 4] = ["dice", "iou", "ece", "nll_fg"];

pub fn build_patient_map(
    root_dir: &str,
    test_indices_path: &str,
    kind: &str,
) -> anyhow::Result<(HashMap<String, String>, Vec<(String, String)>, Vec<usize>)> {
    let (pairs, patient_ids) = recover_image_mask_pairs(root_dir);
    let test_idx = load_test_indices(test_indices_path)?;
    let test_set: std::collections::HashSet<usize> = test_idx.iter().copied().collect();

    let mut mapping = HashMap::new();
    if kind == "unet" {
        for (k, &i) in test_idx.iter().enumerate() {
            mapping.insert(format!("sample_{}", k), patient_ids[i].clone());
        }
    } else if kind == "foundation" {
        for (k, &i) in test_idx.iter().enumerate() {
            mapping.insert(format!("test_{}", k), patient_ids[i].clone());
        }
        let mut train_idx: Vec<usize> = (0..pairs.len()).filter(|i| !test_set.contains(i)).collect();
        train_idx.truncate(64);
        for (k, &i) in train_idx.iter().enumerate() {
            mapping.insert(format!("support_{}", k), patient_ids[i].clone());
        }
    } else {
        anyhow::bail!("kind must be 'unet' or 'foundation'");
    }
    Ok((mapping, pairs, test_idx))
}

pub fn verify_sample_mapping(
    pairs: &[(String, String)],
    test_idx: &[usize],
    results_dir: &str,
    kind: &str,
) -> anyhow::Result<()> {
    let n = test_idx.len();
    let mut checks: Vec<usize> = vec![0, n / 2, n - 1];
    checks.sort_unstable();
    checks.dedup();
    for &k in checks.iter().take(3) {
        let source_path = &pairs[test_idx[k]].0;
        let source = image::open(source_path)?.to_rgb8();
        let name = if kind == "unet" {
            format!("sample_{}", k)
        } else {
            format!("test_{}", k)
        };
        let saved_path = format!("{}/{}/original_image.png", results_dir, name);
        let saved = image::open(&saved_path)?.to_luma8();
        let mut mae = 0.0f64;
        let mut count = 0usize;
        for (s_px, v_px) in source.pixels().zip(saved.pixels()) {
            let g = s_px[1] as f64;
            let s = v_px[0] as f64;
            mae += (s - g).abs();
            count += 1;
        }
        mae /= count as f64 * 255.0;
        if mae > 3.0 / 255.0 {
            anyhow::bail!(
                "Sample mapping mismatch for {}: MAE={:.4} > {:.4}. The dataset iteration order changed.",
                name,
                mae,
                3.0 / 255.0
            );
        }
    }
    Ok(())
}

pub fn run_for_pipeline(
    kind: &str,
    results_dir: &str,
    root_dir: &str,
    test_indices_path: &str,
    n_boot: usize,
    seed: u64,
) -> anyhow::Result<Vec<ComparisonResult>> {
    let (patient_map, pairs, test_idx) = build_patient_map(root_dir, test_indices_path, kind)?;
    verify_sample_mapping(&pairs, &test_idx, results_dir, kind)?;

    let csv_path = format!("{}/visualizations/detailed_metrics.csv", results_dir);
    let mut rows = read_detailed_metrics(&csv_path)?;
    if kind == "foundation" {
        rows.retain(|r| r.sample.starts_with("test_"));
    }

    let mut results = Vec::new();
    for (a, b) in DEFAULT_COMPARISONS {
        for metric in DEFAULT_METRICS {
            results.push(compare_methods(
                &rows, &patient_map, a, b, metric, n_boot, seed, 0.05,
            ));
        }
    }
    Ok(results)
}

pub fn format_table(results: &[ComparisonResult]) -> String {
    let mut lines = vec![
        "| Comparación | Métrica | Δ medio | IC 95% (bootstrap paciente) | p (Wilcoxon) | n pac. | Mejora pac. | Mejora slices | d_z |".to_string(),
        "|---|---|---:|---|---:|---:|---:|---:|---:|".to_string(),
    ];
    for r in results {
        lines.push(format!(
            "| {} | {} | {:+.4} | [{:+.4}, {:+.4}] | {:.3} | {} | {:.0}% | {:.0}% | {:+.2} |",
            r.comparison,
            r.metric,
            r.delta,
            r.ci_low,
            r.ci_high,
            r.p_wilcoxon,
            r.n_patients,
            r.pct_patients_improved,
            r.pct_slices_improved,
            r.cohen_dz
        ));
    }
    lines.join("\n")
}

pub fn write_statistical_tests_csv(results: &[ComparisonResult], dir: &str) -> anyhow::Result<()> {
    let path = std::path::Path::new(dir).join("statistical_tests.csv");
    let mut w = csv::Writer::from_path(&path)?;
    w.write_record(&[
        "comparison",
        "metric",
        "delta",
        "ci_low",
        "ci_high",
        "p_wilcoxon",
        "n_patients",
        "n_slices",
        "pct_patients_improved",
        "pct_slices_improved",
        "cohen_dz",
    ])?;
    for r in results {
        w.write_record(&[
            r.comparison.clone(),
            r.metric.clone(),
            crate::io::format_float(r.delta),
            crate::io::format_float(r.ci_low),
            crate::io::format_float(r.ci_high),
            crate::io::format_float(r.p_wilcoxon),
            r.n_patients.to_string(),
            r.n_slices.to_string(),
            crate::io::format_float(r.pct_patients_improved),
            crate::io::format_float(r.pct_slices_improved),
            crate::io::format_float(r.cohen_dz),
        ])?;
    }
    let _ = w.flush();
    Ok(())
}

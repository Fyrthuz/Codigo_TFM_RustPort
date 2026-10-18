use std::collections::HashMap;

use tfm_core::statistics::{compare_methods, MetricRow};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn synthetic(
    n_patients: usize,
    slices: usize,
    delta: f64,
    noise: f64,
    seed: u64,
) -> (Vec<MetricRow>, HashMap<String, String>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut rows = Vec::new();
    let mut patient_map = HashMap::new();
    for p in 0..n_patients {
        let base = rng.gen_range(0.7..0.95);
        for s in 0..slices {
            let name = format!("sample_{}", p * slices + s);
            patient_map.insert(name.clone(), format!("P{:02}", p));
            let n1: f64 = rng.gen_range(-noise..noise);
            let n2: f64 = rng.gen_range(-noise..noise);
            rows.push(MetricRow {
                sample: name.clone(),
                method: "fusion".to_string(),
                metric: "dice".to_string(),
                value: Some(base + delta + n1),
            });
            rows.push(MetricRow {
                sample: name,
                method: "normal".to_string(),
                metric: "dice".to_string(),
                value: Some(base + n2),
            });
        }
    }
    (rows, patient_map)
}

fn approx(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "{} != {} (tol {})", a, b, tol);
}

#[test]
fn test_detects_positive_effect() {
    let (rows, pm) = synthetic(12, 5, 0.05, 0.005, 0);
    let res = compare_methods(&rows, &pm, "fusion", "normal", "dice", 500, 42, 0.05);
    approx(res.delta, 0.05, 0.01);
    assert!(res.ci_low > 0.0, "ci_low {}", res.ci_low);
    assert!(res.p_wilcoxon < 0.01, "p {}", res.p_wilcoxon);
    assert_eq!(res.n_patients, 12);
    assert_eq!(res.n_slices, 60);
    assert_eq!(res.pct_patients_improved, 100.0);
}

#[test]
fn test_null_effect_not_significant() {
    let (rows, pm) = synthetic(12, 5, 0.0, 0.01, 0);
    let res = compare_methods(&rows, &pm, "fusion", "normal", "dice", 500, 42, 0.05);
    assert!(res.delta.abs() < 0.01, "delta {}", res.delta);
    assert!(res.p_wilcoxon > 0.05, "p {}", res.p_wilcoxon);
}

#[test]
fn test_ignores_samples_without_patient() {
    let (rows, pm) = synthetic(12, 5, 0.0, 0.01, 1);
    let rows: Vec<MetricRow> = rows
        .into_iter()
        .filter(|r| r.sample != "sample_0")
        .collect();
    let res = compare_methods(&rows, &pm, "fusion", "normal", "dice", 200, 1, 0.05);
    assert_eq!(res.n_slices, 59);
    assert_eq!(res.n_patients, 12);
}

#[test]
fn test_patient_clustering_reflected_in_n() {
    let (rows, pm) = synthetic(6, 8, 0.0, 0.01, 2);
    let res = compare_methods(&rows, &pm, "fusion", "normal", "dice", 200, 2, 0.05);
    assert_eq!(res.n_patients, 6);
    assert_eq!(res.n_slices, 48);
}

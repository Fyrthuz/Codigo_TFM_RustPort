use std::collections::HashMap;

use clap::Parser;
use serde::Deserialize;
use tfm_core::statistics::{read_detailed_metrics, MetricRow};

use app::viz::{
    plot_box_comparison, plot_forest, plot_metric_distributions, plot_metrics_comparison,
    plot_qualitative_panel, ForestRow,
};

#[derive(Parser)]
struct Args {
    #[clap(long, default_value = "./results")]
    results_dir: String,
    #[clap(long, default_value = "./docs")]
    out_dir: String,
    #[clap(long, default_value = "")]
    sample_prefix: String,
    #[clap(long, default_value = "unet")]
    tag: String,
}

#[derive(Deserialize)]
struct StatsRow {
    comparison: String,
    metric: String,
    delta: f64,
    ci_low: f64,
    ci_high: f64,
    p_wilcoxon: f64,
}

fn rebuild(
    rows: &[MetricRow],
    methods: &[String],
    metrics: &[String],
) -> (HashMap<String, HashMap<String, HashMap<String, f64>>>, HashMap<String, HashMap<String, f64>>) {
    let mut overall: HashMap<String, HashMap<String, HashMap<String, f64>>> = HashMap::new();
    for r in rows {
        if let Some(v) = r.value {
            overall
                .entry(r.sample.clone())
                .or_default()
                .entry(r.method.clone())
                .or_default()
                .insert(r.metric.clone(), v);
        }
    }

    let mut mean_results: HashMap<String, HashMap<String, f64>> = HashMap::new();
    for method in methods {
        let mut mm = HashMap::new();
        for metric in metrics {
            let mut vals = Vec::new();
            for sd in overall.values() {
                if let Some(m) = sd.get(method) {
                    if let Some(&v) = m.get(metric) {
                        if v.is_finite() {
                            vals.push(v);
                        }
                    }
                }
            }
            mm.insert(metric.clone(), if vals.is_empty() { 0.0 } else { vals.iter().sum::<f64>() / vals.len() as f64 });
        }
        mean_results.insert(method.clone(), mm);
    }
    (overall, mean_results)
}

fn pick_samples(rows: &[MetricRow], prefix: &str) -> Vec<(String, String)> {
    let mut sub: Vec<(String, f64)> = rows
        .iter()
        .filter(|r| r.method == "fusion" && r.metric == "dice")
        .filter(|r| prefix.is_empty() || r.sample.starts_with(prefix))
        .filter_map(|r| r.value.map(|v| (r.sample.clone(), v)))
        .collect();
    sub.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    if sub.is_empty() {
        return Vec::new();
    }
    let values: Vec<f64> = sub.iter().map(|(_, v)| *v).collect();
    let picks = [("facil", 85.0), ("tipico", 50.0), ("dificil", 12.0)];
    let mut used = std::collections::HashSet::new();
    let mut out = Vec::new();
    for (kind, pct) in picks {
        let target = tfm_core::fusion::percentile(&values, pct);
        let mut best: Option<(String, f64)> = None;
        let mut best_d = f64::INFINITY;
        for (s, v) in &sub {
            if used.contains(s) {
                continue;
            }
            if kind == "dificil" && *v < 0.4 {
                continue;
            }
            let d = (v - target).abs();
            if d < best_d {
                best_d = d;
                best = Some((s.clone(), *v));
            }
        }
        if let Some((s, _)) = best {
            used.insert(s.clone());
            out.push((kind.to_string(), s));
        }
    }
    out
}

fn main() {
    let args = Args::parse();
    std::fs::create_dir_all(&args.out_dir).unwrap();

    let methods: Vec<String> = ["normal", "mc_dropout", "tta", "noisy", "fusion", "crf"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    let metrics: Vec<String> = ["iou", "dice", "nll", "nll_fg", "ece", "brier", "accuracy", "precision", "recall", "certainty"]
        .iter()
        .map(|s| s.to_string())
        .collect();

    let csv_path = format!("{}/visualizations/detailed_metrics.csv", args.results_dir);
    let stats_path = format!("{}/visualizations/statistical_tests.csv", args.results_dir);

    if let Ok(rows) = read_detailed_metrics(&csv_path) {
        let (overall, mean_results) = rebuild(&rows, &methods, &metrics);
        plot_metrics_comparison(&mean_results, &methods, &metrics, &args.out_dir);
        plot_box_comparison(&overall, &methods, &metrics, &args.out_dir);

        let samples = pick_samples(&rows, &args.sample_prefix);
        if !samples.is_empty() {
            plot_qualitative_panel(
                &args.results_dir,
                &samples,
                &format!("{}/qualitative_{}.png", args.out_dir, args.tag),
            );
        }
        plot_metric_distributions(&overall, &methods, &format!("{}/distribution_{}.png", args.out_dir, args.tag));
    } else {
        println!("No detailed_metrics.csv found at {}", csv_path);
    }

    if let Ok(mut reader) = csv::Reader::from_path(&stats_path) {
        let mut forest: Vec<ForestRow> = Vec::new();
        for rec in reader.deserialize::<StatsRow>().flatten() {
            if rec.metric == "dice" {
                forest.push(ForestRow {
                    comparison: rec.comparison,
                    delta: rec.delta,
                    ci_low: rec.ci_low,
                    ci_high: rec.ci_high,
                    p_wilcoxon: rec.p_wilcoxon,
                });
            }
        }
        if !forest.is_empty() {
            plot_forest(&forest, &format!("{}/statistical_summary_{}.png", args.out_dir, args.tag), "Paired tests (dice)");
        }
    }
}

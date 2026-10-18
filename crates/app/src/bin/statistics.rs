use clap::Parser;

use tfm_core::statistics::{format_table, run_for_pipeline, write_statistical_tests_csv};

#[derive(Parser)]
struct Args {
    #[clap(long, default_value = "all")]
    pipeline: String,
    #[clap(long, default_value = "./results")]
    results_dir: String,
    #[clap(long, default_value = "./results_foundation_universeg")]
    foundation_results_dir: String,
    #[clap(long, default_value = "./MRI/filtered_data")]
    data_root: String,
    #[clap(long, default_value = "test_indices.json")]
    test_indices: String,
    #[clap(long, default_value = "10000")]
    n_boot: usize,
    #[clap(long, default_value = "42")]
    seed: u64,
}

fn main() {
    let args = Args::parse();

    let mut pipelines: Vec<(&str, String)> = Vec::new();
    if args.pipeline == "unet" || args.pipeline == "all" {
        pipelines.push(("unet", args.results_dir.clone()));
    }
    if args.pipeline == "foundation" || args.pipeline == "all" {
        pipelines.push(("foundation", args.foundation_results_dir.clone()));
    }

    for (kind, results_dir) in pipelines {
        println!(
            "\n{}\n{} — paired tests at patient level ({} bootstrap samples)\n{}",
            "=".repeat(80),
            kind.to_uppercase(),
            args.n_boot,
            "=".repeat(80)
        );
        match run_for_pipeline(kind, &results_dir, &args.data_root, &args.test_indices, args.n_boot, args.seed) {
            Ok(result) => {
                println!("{}", format_table(&result));
                if let Err(e) = write_statistical_tests_csv(&result, &format!("{}/visualizations", results_dir)) {
                    eprintln!("error writing csv: {}", e);
                }
                println!("\nSaved: {}/visualizations/statistical_tests.csv", results_dir);
            }
            Err(e) => {
                eprintln!("error: {}", e);
            }
        }
    }
}

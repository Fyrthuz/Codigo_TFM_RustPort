use tfm_core::config::PipelineConfig;

#[test]
fn test_default_values() {
    let c = PipelineConfig::default();
    assert_eq!(c.inference.num_samples, 30);
    assert_eq!(c.inference.activation, "sigmoid");
    assert_eq!(c.model.in_channels, 3);
    assert_eq!(c.mc_dropout.p, 0.01);
    assert_eq!(c.noisy.noise_std, 0.01);
    assert_eq!(c.crf.n_iters, 3);
    assert_eq!(c.crf.w_g, 0.5);
    assert_eq!(c.crf.w_b, 1.0);
}

#[test]
fn test_from_yaml() {
    let dir = std::env::temp_dir();
    let fname = dir.join(format!("cfg_{}.yaml", std::process::id()));
    let content = "inference:\n  num_samples: 20\n  activation: softmax\nmodel:\n  in_channels: 1\n  out_channels: 3\nnoisy:\n  noise_std: 0.05\n";
    std::fs::write(&fname, content).unwrap();
    let c = PipelineConfig::from_yaml(fname.to_str().unwrap()).unwrap();
    assert_eq!(c.inference.num_samples, 20);
    assert_eq!(c.inference.activation, "softmax");
    assert_eq!(c.model.in_channels, 1);
    assert_eq!(c.model.out_channels, 3);
    assert_eq!(c.noisy.noise_std, 0.05);
    let _ = std::fs::remove_file(&fname);
}

#[test]
fn test_crf_tuple_defaults() {
    let c = PipelineConfig::default();
    assert_eq!(c.crf.sdims, vec![2.0, 3.0]);
    assert_eq!(c.crf.schan, vec![0.15]);
}

#[test]
fn test_from_yaml_crf_lists() {
    let dir = std::env::temp_dir();
    let fname = dir.join(format!("cfg_crf_{}.yaml", std::process::id()));
    let content = "crf:\n  sdims: [2, 3]\n  schan: [0.15]\n  n_iters: 3\n";
    std::fs::write(&fname, content).unwrap();
    let c = PipelineConfig::from_yaml(fname.to_str().unwrap()).unwrap();
    assert_eq!(c.crf.sdims, vec![2.0, 3.0]);
    assert_eq!(c.crf.schan, vec![0.15]);
    let _ = std::fs::remove_file(&fname);
}

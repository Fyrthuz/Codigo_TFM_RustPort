use serde::Deserialize;

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct InferenceConfig {
    pub num_samples: i64,
    pub num_classes: i64,
    pub activation: String,
    pub batch_size: i64,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        InferenceConfig {
            num_samples: 30,
            num_classes: 2,
            activation: "sigmoid".to_string(),
            batch_size: 1,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub name: String,
    pub in_channels: i64,
    pub out_channels: i64,
    pub init_features: i64,
    pub pretrained: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            name: "unet".to_string(),
            in_channels: 3,
            out_channels: 1,
            init_features: 32,
            pretrained: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct FusionConfig {
    pub weighting_method: String,
    pub beta: f64,
    pub alpha: f64,
    pub threshold_method: String,
    pub percentile: f64,
    pub k: f64,
    pub epsilon: f64,
}

impl Default for FusionConfig {
    fn default() -> Self {
        FusionConfig {
            weighting_method: "inverse".to_string(),
            beta: 1.0,
            alpha: 1.0,
            threshold_method: "naive".to_string(),
            percentile: 50.0,
            k: 0.5,
            epsilon: 1e-6,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CRFConfig {
    pub sdims: Vec<f64>,
    pub schan: Vec<f64>,
    pub n_iters: i64,
    pub epsilon: f64,
    pub w_g: f64,
    pub w_b: f64,
}

impl Default for CRFConfig {
    fn default() -> Self {
        CRFConfig {
            sdims: vec![2.0, 3.0],
            schan: vec![0.15],
            n_iters: 3,
            epsilon: 1e-8,
            w_g: 0.5,
            w_b: 1.0,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct NoisyConfig {
    pub noise_std: f64,
}

impl Default for NoisyConfig {
    fn default() -> Self {
        NoisyConfig { noise_std: 0.01 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MCDropoutConfig {
    pub p: f64,
}

impl Default for MCDropoutConfig {
    fn default() -> Self {
        MCDropoutConfig { p: 0.01 }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct SubsetConfig {
    pub subset_size: Option<i64>,
    pub size_context_set: Option<i64>,
}

impl Default for SubsetConfig {
    fn default() -> Self {
        SubsetConfig {
            subset_size: None,
            size_context_set: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CertaintyRefinementConfig {
    pub enabled: bool,
    pub alpha: f64,
    pub threshold: f64,
}

impl Default for CertaintyRefinementConfig {
    fn default() -> Self {
        CertaintyRefinementConfig {
            enabled: true,
            alpha: 0.6,
            threshold: 0.5,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PathsConfig {
    pub save_path: String,
    pub root_path: String,
}

impl Default for PathsConfig {
    fn default() -> Self {
        PathsConfig {
            save_path: "./results".to_string(),
            root_path: "./MRI/filtered_data".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct PipelineConfig {
    pub paths: PathsConfig,
    pub directories: Vec<String>,
    pub inference: InferenceConfig,
    pub model: ModelConfig,
    pub fusion: FusionConfig,
    pub crf: CRFConfig,
    pub noisy: NoisyConfig,
    pub mc_dropout: MCDropoutConfig,
    pub subset: SubsetConfig,
    pub certainty_refinement: CertaintyRefinementConfig,
    pub uncertainty_metric: serde_yaml::Value,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        PipelineConfig {
            paths: PathsConfig::default(),
            directories: vec![
                "mc_dropout".to_string(),
                "tta".to_string(),
                "noisy".to_string(),
                "base".to_string(),
                "refined".to_string(),
                "original".to_string(),
            ],
            inference: InferenceConfig::default(),
            model: ModelConfig::default(),
            fusion: FusionConfig::default(),
            crf: CRFConfig::default(),
            noisy: NoisyConfig::default(),
            mc_dropout: MCDropoutConfig::default(),
            subset: SubsetConfig::default(),
            certainty_refinement: CertaintyRefinementConfig::default(),
            uncertainty_metric: serde_yaml::Value::Null,
        }
    }
}

impl PipelineConfig {
    pub fn from_yaml(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let cfg: PipelineConfig = serde_yaml::from_str(&content)?;
        Ok(cfg)
    }
}

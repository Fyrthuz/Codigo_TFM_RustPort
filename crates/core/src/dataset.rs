use std::collections::HashMap;
use std::path::Path;

use ndarray::Array3;

use crate::pyrandom::Mt19937;

pub fn extract_patient_id(case_name: &str) -> String {
    case_name.split('_').take(3).collect::<Vec<_>>().join("_")
}

pub fn recover_image_mask_pairs(root_dir: &str) -> (Vec<(String, String)>, Vec<String>) {
    let mut pairs: Vec<(String, String)> = Vec::new();
    let mut patient_ids: Vec<String> = Vec::new();

    let root = Path::new(root_dir);
    if let Ok(entries) = std::fs::read_dir(root) {
        for case_entry in entries.flatten() {
            let case_path = case_entry.path();
            if !case_path.is_dir() {
                continue;
            }
            let case_name = case_path
                .file_name()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();
            let patient_id = extract_patient_id(&case_name);
            if let Ok(files) = std::fs::read_dir(&case_path) {
                for file_entry in files.flatten() {
                    let file_name = file_entry.file_name().to_string_lossy().to_string();
                    if file_name.to_lowercase().ends_with(".tif") && !file_name.contains("_mask") {
                        let image_path = file_entry.path();
                        let base = file_name
                            .rsplit_once('.')
                            .map(|(b, _)| b.to_string())
                            .unwrap_or_default();
                        let ext = Path::new(&file_name)
                            .extension()
                            .map(|e| e.to_string_lossy().to_string())
                            .unwrap_or_default();
                        let mask_file = format!("{}_mask.{}", base, ext);
                        let mask_path = case_path.join(&mask_file);
                        if mask_path.exists() {
                            pairs.push((
                                image_path.to_string_lossy().to_string(),
                                mask_path.to_string_lossy().to_string(),
                            ));
                            patient_ids.push(patient_id.clone());
                        }
                    }
                }
            }
        }
    }

    (pairs, patient_ids)
}

pub fn split_by_patient(
    patient_ids: &[String],
    train_ratio: f64,
    val_ratio: f64,
    seed: u32,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut patient_to_indices: HashMap<&str, Vec<usize>> = HashMap::new();
    let mut order: Vec<&str> = Vec::new();
    for (idx, pid) in patient_ids.iter().enumerate() {
        if !patient_to_indices.contains_key(pid.as_str()) {
            order.push(pid.as_str());
        }
        patient_to_indices.entry(pid.as_str()).or_default().push(idx);
    }

    let mut patients = order;
    let mut rng = Mt19937::py_seed(seed);
    rng.shuffle(&mut patients);

    let n = patients.len();
    let n_train = (n as f64 * train_ratio) as usize;
    let n_val = (n as f64 * val_ratio) as usize;

    let mut train_indices = Vec::new();
    let mut val_indices = Vec::new();
    let mut test_indices = Vec::new();
    for (i, p) in patients.iter().enumerate() {
        let idxs = &patient_to_indices[*p];
        if i < n_train {
            train_indices.extend_from_slice(idxs);
        } else if i < n_train + n_val {
            val_indices.extend_from_slice(idxs);
        } else {
            test_indices.extend_from_slice(idxs);
        }
    }
    train_indices.sort_unstable();
    val_indices.sort_unstable();
    test_indices.sort_unstable();

    (train_indices, val_indices, test_indices)
}

pub fn save_test_indices(test_indices: &[usize], path: &str) -> anyhow::Result<()> {
    let json = serde_json::to_string(test_indices)?;
    std::fs::write(path, json)?;
    Ok(())
}

pub fn load_test_indices(path: &str) -> anyhow::Result<Vec<usize>> {
    let content = std::fs::read_to_string(path)?;
    let v: Vec<usize> = serde_json::from_str(&content)?;
    Ok(v)
}

pub struct LGGSegmentationDataset {
    pub image_mask_pairs: Vec<(String, String)>,
    pub patient_ids: Vec<String>,
}

impl LGGSegmentationDataset {
    pub fn new(root_dir: &str) -> Self {
        let (mut pairs, patient_ids) = recover_image_mask_pairs(root_dir);
        pairs.sort_by(|a, b| a.0.cmp(&b.0));
        LGGSegmentationDataset {
            image_mask_pairs: pairs,
            patient_ids,
        }
    }

    pub fn len(&self) -> usize {
        self.image_mask_pairs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.image_mask_pairs.is_empty()
    }

    pub fn load(&self, idx: usize) -> anyhow::Result<(Array3<f64>, Array3<f64>)> {
        let (image_path, mask_path) = &self.image_mask_pairs[idx];
        let image = image::open(image_path)?.to_rgb8();
        let mask = image::open(mask_path)?.to_luma8();
        let (h, w) = (image.height() as usize, image.width() as usize);
        let mut img_arr = Array3::<f64>::zeros((3, h, w));
        for i in 0..h {
            for j in 0..w {
                let px = image.get_pixel(j as u32, i as u32);
                img_arr[[0, i, j]] = px[0] as f64 / 255.0;
                img_arr[[1, i, j]] = px[1] as f64 / 255.0;
                img_arr[[2, i, j]] = px[2] as f64 / 255.0;
            }
        }
        let mut mask_arr = Array3::<f64>::zeros((1, h, w));
        for i in 0..h {
            for j in 0..w {
                let px = mask.get_pixel(j as u32, i as u32);
                mask_arr[[0, i, j]] = px[0] as f64 / 255.0;
            }
        }
        Ok((img_arr, mask_arr))
    }
}

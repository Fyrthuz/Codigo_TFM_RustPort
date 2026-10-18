use tfm_core::dataset::{extract_patient_id, recover_image_mask_pairs, split_by_patient, LGGSegmentationDataset};
use ndarray::IxDyn;

fn make_temp_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("ds_{}_{}", std::process::id(), rand::random::<u32>()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn write_tif_case(root: &std::path::Path, case: &str, img_name: &str, with_mask: bool) {
    let case_dir = root.join(case);
    std::fs::create_dir_all(&case_dir).unwrap();
    let img_path = case_dir.join(img_name);
    image::RgbImage::from_pixel(256, 256, image::Rgb([128, 128, 128]))
        .save(&img_path)
        .unwrap();
    if with_mask {
        let base = img_name.rsplit_once('.').unwrap().0;
        let mask_path = case_dir.join(format!("{}_mask.tif", base));
        image::GrayImage::from_pixel(256, 256, image::Luma([0])).save(&mask_path).unwrap();
    }
}

#[test]
fn test_extract_patient_id() {
    assert_eq!(extract_patient_id("TCGA_CS_4941_19960909"), "TCGA_CS_4941");
}

#[test]
fn test_len() {
    let dir = make_temp_dir();
    write_tif_case(&dir, "TCGA_001", "image.tif", true);
    let ds = LGGSegmentationDataset::new(dir.to_str().unwrap());
    assert_eq!(ds.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_load_shape() {
    let dir = make_temp_dir();
    write_tif_case(&dir, "TCGA_001", "image.tif", true);
    let ds = LGGSegmentationDataset::new(dir.to_str().unwrap());
    let (img, mask) = ds.load(0).unwrap();
    assert_eq!(img.shape(), &[3, 256, 256]);
    assert_eq!(mask.shape(), &[1, 256, 256]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_empty_directory() {
    let dir = make_temp_dir();
    let ds = LGGSegmentationDataset::new(dir.to_str().unwrap());
    assert_eq!(ds.len(), 0);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_skip_missing_mask() {
    let dir = make_temp_dir();
    write_tif_case(&dir, "TCGA_001", "image.tif", true);
    write_tif_case(&dir, "TCGA_002", "image2.tif", false);
    let ds = LGGSegmentationDataset::new(dir.to_str().unwrap());
    assert_eq!(ds.len(), 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_recover_and_sort_consistency() {
    let dir = make_temp_dir();
    write_tif_case(&dir, "TCGA_BBB", "image.tif", true);
    write_tif_case(&dir, "TCGA_AAA", "image.tif", true);
    let (pairs, pids) = recover_image_mask_pairs(dir.to_str().unwrap());
    assert_eq!(pairs.len(), 2);
    assert_eq!(pids.len(), 2);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_split_by_patient_groups() {
    let pids: Vec<String> = (0..20).map(|i| format!("P{}", i)).collect();
    let (train, val, test) = split_by_patient(&pids, 0.70, 0.15, 42);
    let mut all = train.clone();
    all.extend(&val);
    all.extend(&test);
    all.sort_unstable();
    assert_eq!(all, (0..20).collect::<Vec<usize>>());
    assert_eq!(train.len(), 14);
    assert_eq!(val.len(), 3);
    assert_eq!(test.len(), 3);
}

#[test]
fn test_split_no_leakage() {
    let pids: Vec<String> = (0..100).map(|i| format!("P{}", i / 5)).collect();
    let (train, val, test) = split_by_patient(&pids, 0.70, 0.15, 42);
    let train_p: std::collections::HashSet<usize> =
        train.iter().map(|&i| i / 5).collect();
    let test_p: std::collections::HashSet<usize> =
        test.iter().map(|&i| i / 5).collect();
    assert!(train_p.is_disjoint(&test_p));
}

#[test]
fn test_roundtrip_indices() {
    let v = vec![1usize, 2, 3, 100];
    let dir = std::env::temp_dir();
    let f = dir.join(format!("idx_{}.json", std::process::id()));
    tfm_core::dataset::save_test_indices(&v, f.to_str().unwrap()).unwrap();
    let loaded = tfm_core::dataset::load_test_indices(f.to_str().unwrap()).unwrap();
    assert_eq!(v, loaded);
    let _ = std::fs::remove_file(&f);
}

#[test]
fn test_array_shape_build() {
    let _ = ndarray::ArrayD::<f64>::zeros(IxDyn(&[3, 256, 256]));
}

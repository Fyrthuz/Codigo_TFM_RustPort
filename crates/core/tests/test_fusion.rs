use tfm_core::fusion::{dynamic_threshold_multiclass, weighted_average_with_uncertainty};
use ndarray::{Array2, ArrayD, IxDyn};

fn a2d(v: Vec<f64>, h: usize, w: usize) -> Array2<f64> {
    Array2::from_shape_vec((h, w), v).unwrap()
}

#[test]
fn test_identical_inputs() {
    let prob = a2d(vec![0.8; 32 * 32], 32, 32);
    let uncert = a2d(vec![0.2; 32 * 32], 32, 32);
    let (cp, cu, cm) = weighted_average_with_uncertainty(
        &prob, &uncert, &prob, &uncert, &prob, &uncert,
        "inverse", 1.0, 1.0, "naive", 50.0, 0.5, 1e-6,
    );
    for (a, b) in cp.iter().zip(prob.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    for (a, b) in cu.iter().zip(uncert.iter()) {
        assert!((a - b).abs() < 1e-6);
    }
    assert_eq!(cm.dim(), (32, 32));
    for &v in cm.iter() {
        assert!(v == 0 || v == 1);
    }
}

#[test]
fn test_one_certain_source() {
    let prob_certain = a2d(vec![1.0; 16 * 16], 16, 16);
    let uncert_certain = a2d(vec![0.0; 16 * 16], 16, 16);
    let prob_uncertain = a2d(vec![0.0; 16 * 16], 16, 16);
    let uncert_uncertain = a2d(vec![1.0; 16 * 16], 16, 16);
    let (cp, _, _) = weighted_average_with_uncertainty(
        &prob_certain, &uncert_certain, &prob_uncertain, &uncert_uncertain,
        &prob_uncertain, &uncert_uncertain,
        "inverse", 1.0, 1.0, "naive", 50.0, 0.5, 1e-6,
    );
    for &v in cp.iter() {
        assert!((v - 1.0).abs() < 1e-4);
    }
}

#[test]
fn test_methods() {
    let prob = a2d(vec![0.7; 16 * 16], 16, 16);
    let uncert = a2d(vec![0.3; 16 * 16], 16, 16);
    for method in ["inverse", "exponential", "powerlaw"] {
        let (cp, cu, _) = weighted_average_with_uncertainty(
            &prob, &uncert, &prob, &uncert, &prob, &uncert,
            method, 1.0, 1.0, "naive", 50.0, 0.5, 1e-6,
        );
        assert_eq!(cp.dim(), (16, 16));
        assert_eq!(cu.dim(), (16, 16));
    }
}

#[test]
fn test_threshold_methods() {
    let prob = a2d(vec![0.7; 16 * 16], 16, 16);
    let uncert = a2d(vec![0.3; 16 * 16], 16, 16);
    for thresh in ["naive", "otsu", "percentile"] {
        let (_, _, cm) = weighted_average_with_uncertainty(
            &prob, &uncert, &prob, &uncert, &prob, &uncert,
            "inverse", 1.0, 1.0, thresh, 50.0, 0.5, 1e-6,
        );
        for &v in cm.iter() {
            assert!(v == 0 || v == 1);
        }
    }
}

#[test]
fn test_otsu_binary() {
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let v: Vec<f64> = (0..64 * 64).map(|_| rng.gen::<f64>()).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[64, 64]), v).unwrap();
    let t = dynamic_threshold_multiclass(&arr.view(), "otsu", 50.0, 0.5);
    assert_eq!(t.len(), 1);
    assert!(t[0] >= 0.0 && t[0] <= 1.0);
}

#[test]
fn test_percentile() {
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let v: Vec<f64> = (0..64 * 64).map(|_| rng.gen::<f64>()).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[64, 64]), v).unwrap();
    let t = dynamic_threshold_multiclass(&arr.view(), "percentile", 75.0, 0.5);
    assert_eq!(t.len(), 1);
    let t50 = dynamic_threshold_multiclass(&arr.view(), "percentile", 50.0, 0.5);
    assert!(t[0] >= t50[0]);
}

#[test]
fn test_multiclass() {
    let mut rng = rand::thread_rng();
    use rand::Rng;
    let v: Vec<f64> = (0..3 * 64 * 64).map(|_| rng.gen::<f64>()).collect();
    let arr = ArrayD::from_shape_vec(IxDyn(&[3, 64, 64]), v).unwrap();
    let t = dynamic_threshold_multiclass(&arr.view(), "otsu", 50.0, 0.5);
    assert_eq!(t.len(), 3);
}

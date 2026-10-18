use tfm_core::metrics::{certainty_score, classwise_ece, compute_dice, compute_iou, compute_metrics};
use ndarray::{Array2, ArrayD, IxDyn};

fn a2d(v: Vec<f64>, h: usize, w: usize) -> Array2<f64> {
    Array2::from_shape_vec((h, w), v).unwrap()
}

fn dyn2d(v: Vec<f64>, h: usize, w: usize) -> ArrayD<f64> {
    a2d(v, h, w).into_dyn()
}

fn approx(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "{} != {} (tol {})", a, b, tol);
}

#[test]
fn test_iou_perfect_overlap() {
    let pred = dyn2d(vec![1.0; 100], 10, 10);
    let target = dyn2d(vec![1.0; 100], 10, 10);
    assert_eq!(compute_iou(&pred.view(), &target.view(), 0.5), 1.0);
}

#[test]
fn test_iou_no_overlap() {
    let pred = dyn2d(vec![1.0; 100], 10, 10);
    let target = dyn2d(vec![0.0; 100], 10, 10);
    assert_eq!(compute_iou(&pred.view(), &target.view(), 0.5), 0.0);
}

#[test]
fn test_iou_half_overlap() {
    let mut pred = vec![0.0; 100];
    for i in 0..5 {
        for j in 0..10 {
            pred[i * 10 + j] = 1.0;
        }
    }
    let mut target = vec![0.0; 100];
    for i in 0..10 {
        for j in 0..5 {
            target[i * 10 + j] = 1.0;
        }
    }
    let iou = compute_iou(&dyn2d(pred, 10, 10).view(), &dyn2d(target, 10, 10).view(), 0.5);
    assert!(iou > 0.0 && iou < 1.0);
}

#[test]
fn test_iou_both_empty() {
    let pred = dyn2d(vec![0.0; 100], 10, 10);
    let target = dyn2d(vec![0.0; 100], 10, 10);
    assert_eq!(compute_iou(&pred.view(), &target.view(), 0.5), 1.0);
}

#[test]
fn test_iou_threshold() {
    let pred = dyn2d(vec![0.3; 100], 10, 10);
    let target = dyn2d(vec![1.0; 100], 10, 10);
    assert_eq!(compute_iou(&pred.view(), &target.view(), 0.5), 0.0);
}

#[test]
fn test_dice_perfect_overlap() {
    let pred = dyn2d(vec![1.0; 100], 10, 10);
    let target = dyn2d(vec![1.0; 100], 10, 10);
    assert_eq!(compute_dice(&pred.view(), &target.view(), 0.5), 1.0);
}

#[test]
fn test_dice_no_overlap() {
    let pred = dyn2d(vec![1.0; 100], 10, 10);
    let target = dyn2d(vec![0.0; 100], 10, 10);
    assert_eq!(compute_dice(&pred.view(), &target.view(), 0.5), 0.0);
}

#[test]
fn test_dice_both_empty() {
    let pred = dyn2d(vec![0.0; 100], 10, 10);
    let target = dyn2d(vec![0.0; 100], 10, 10);
    assert_eq!(compute_dice(&pred.view(), &target.view(), 0.5), 1.0);
}

#[test]
fn test_metrics_perfect_prediction() {
    let prob = dyn2d(vec![1.0; 100], 10, 10);
    let gt = dyn2d(vec![1.0; 100], 10, 10);
    let m = compute_metrics(&prob.view(), &gt.view(), 1e-8);
    approx(m.nll, 0.0, 1e-4);
    approx(m.nll_fg, 0.0, 1e-4);
    approx(m.brier, 0.0, 1e-4);
    approx(m.accuracy, 1.0, 1e-4);
    approx(m.precision, 1.0, 1e-4);
    approx(m.recall, 1.0, 1e-4);
    approx(m.ece, 0.0, 1e-2);
}

#[test]
fn test_metrics_worst_prediction() {
    let prob = dyn2d(vec![0.0; 100], 10, 10);
    let gt = dyn2d(vec![1.0; 100], 10, 10);
    let m = compute_metrics(&prob.view(), &gt.view(), 1e-8);
    assert_eq!(m.accuracy, 0.0);
    assert_eq!(m.recall, 0.0);
}

#[test]
fn test_metrics_half_correct() {
    let mut prob = vec![0.0; 100];
    for i in 0..5 {
        for j in 0..10 {
            prob[i * 10 + j] = 0.9;
        }
    }
    let mut gt = vec![0.0; 100];
    for i in 0..5 {
        for j in 0..10 {
            gt[i * 10 + j] = 1.0;
        }
    }
    let m = compute_metrics(&dyn2d(prob, 10, 10).view(), &dyn2d(gt, 10, 10).view(), 1e-8);
    assert!(m.accuracy >= 0.9 && m.accuracy <= 1.0);
}

#[test]
fn test_metrics_probability_clipping() {
    let mut prob = vec![1e-10; 100];
    prob[0] = 0.0;
    prob[1] = 1.0;
    let gt = dyn2d(vec![1.0; 100], 10, 10);
    let m = compute_metrics(&dyn2d(prob, 10, 10).view(), &gt.view(), 1e-8);
    assert!(m.nll.is_finite());
}

#[test]
fn test_ece_background_heavy_calibrated() {
    let mut prob = vec![0.02; 1000];
    let mut gt = vec![0.0; 1000];
    for i in 980..1000 {
        prob[i] = 0.9;
        gt[i] = 1.0;
    }
    let m = compute_metrics(&dyn2d(prob, 100, 10).view(), &dyn2d(gt, 100, 10).view(), 1e-8);
    assert!(m.ece < 0.05);
}

#[test]
fn test_ece_overconfident_wrong() {
    let prob = dyn2d(vec![0.99; 100], 10, 10);
    let gt = dyn2d(vec![0.0; 100], 10, 10);
    let m = compute_metrics(&prob.view(), &gt.view(), 1e-8);
    assert!(m.ece > 0.5);
}

#[test]
fn test_nll_fg_foreground_only() {
    let mut prob = vec![0.0; 100];
    let mut gt = vec![0.0; 100];
    for i in 0..20 {
        prob[i] = 0.5;
        gt[i] = 1.0;
    }
    let m = compute_metrics(&dyn2d(prob, 10, 10).view(), &dyn2d(gt, 10, 10).view(), 1e-8);
    approx(m.nll_fg, -(0.5f64).ln(), 1e-3);
}

#[test]
fn test_nll_fg_nan_without_foreground() {
    let prob = dyn2d(vec![0.0; 100], 10, 10);
    let gt = dyn2d(vec![0.0; 100], 10, 10);
    let m = compute_metrics(&prob.view(), &gt.view(), 1e-8);
    assert!(m.nll_fg.is_nan());
}

#[test]
fn test_certainty_max() {
    let unc = dyn2d(vec![0.0; 100], 10, 10);
    let gt = dyn2d(vec![1.0; 100], 10, 10);
    approx(certainty_score(&unc.view(), &gt.view(), 2), 1.0, 1e-2);
}

#[test]
fn test_certainty_max_uncertainty() {
    let unc = dyn2d(vec![2.0f64.ln(); 100], 10, 10);
    let gt = dyn2d(vec![1.0; 100], 10, 10);
    approx(certainty_score(&unc.view(), &gt.view(), 2), 0.0, 1e-2);
}

#[test]
fn test_certainty_empty_gt_nan() {
    let unc = dyn2d(vec![0.0; 100], 10, 10);
    let gt = dyn2d(vec![0.0; 100], 10, 10);
    assert!(certainty_score(&unc.view(), &gt.view(), 2).is_nan());
}

#[test]
fn test_classwise_ece_direct() {
    let prob = [0.5f64; 10];
    let gt = [1i64; 10];
    let ece = classwise_ece(&prob, &gt, 10);
    assert!(ece.is_finite());
}

#[test]
fn test_metrics_multiclass_max_reduce() {
    // prob (2,H,W) with a leading class dim; gt (H,W)
    let mut v = vec![0.0; 200];
    for i in 0..100 {
        v[i] = 0.1; // class 0
        v[100 + i] = 0.9; // class 1
    }
    let prob = ArrayD::from_shape_vec(IxDyn(&[2, 10, 10]), v).unwrap();
    let gt = dyn2d(vec![1.0; 100], 10, 10);
    let m = compute_metrics(&prob.view(), &gt.view(), 1e-8);
    assert!(m.recall > 0.9);
}

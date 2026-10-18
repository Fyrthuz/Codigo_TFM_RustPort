use tfm_core::crf::refine_with_crf_uncertainty;
use ndarray::{ArrayD, IxDyn};
use rand::Rng;

fn dice(a: &[u8], b: &[u8]) -> f64 {
    let mut inter = 0.0;
    let mut asum = 0.0;
    let mut bsum = 0.0;
    for (&x, &y) in a.iter().zip(b.iter()) {
        let xb = x > 0;
        let yb = y > 0;
        if xb && yb {
            inter += 1.0;
        }
        if xb {
            asum += 1.0;
        }
        if yb {
            bsum += 1.0;
        }
    }
    2.0 * inter / (asum + bsum + 1e-8)
}

#[test]
fn test_binary_input_shapes() {
    let mut rng = rand::thread_rng();
    let v: Vec<f64> = (0..3 * 32 * 32).map(|_| rng.gen_range(0.0..255.0)).collect();
    let image = ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), v).unwrap();
    let pv: Vec<f64> = (0..32 * 32).map(|_| rng.gen::<f64>()).collect();
    let prob = ArrayD::from_shape_vec(IxDyn(&[32, 32]), pv).unwrap();
    let uv: Vec<f64> = (0..32 * 32).map(|_| rng.gen::<f64>()).collect();
    let unc = ArrayD::from_shape_vec(IxDyn(&[32, 32]), uv).unwrap();

    let (probs, seg, uncert) = refine_with_crf_uncertainty(
        &image.view(), &prob.view(), &unc.view(), (2.0, 3.0), 0.15, 2, 1e-8, 0.5, 1.0,
    );
    assert_eq!(probs.dim(), (2, 32, 32));
    assert_eq!(seg.dim(), (32, 32));
    assert_eq!(uncert.dim(), (32, 32));
    for i in 0..32 {
        for j in 0..32 {
            let s = probs[[0, i, j]] + probs[[1, i, j]];
            assert!((s - 1.0).abs() < 1e-6);
        }
    }
}

#[test]
fn test_multiclass_input_shapes() {
    let mut rng = rand::thread_rng();
    let v: Vec<f64> = (0..3 * 32 * 32).map(|_| rng.gen_range(0.0..255.0)).collect();
    let image = ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), v).unwrap();
    let pv: Vec<f64> = (0..3 * 32 * 32).map(|_| rng.gen::<f64>()).collect();
    let mut prob = ArrayD::from_shape_vec(IxDyn(&[3, 32, 32]), pv).unwrap();
    for i in 0..32 {
        for j in 0..32 {
            let s = prob[[0, i, j]] + prob[[1, i, j]] + prob[[2, i, j]];
            prob[[0, i, j]] /= s;
            prob[[1, i, j]] /= s;
            prob[[2, i, j]] /= s;
        }
    }
    let uv: Vec<f64> = (0..32 * 32).map(|_| rng.gen::<f64>()).collect();
    let unc = ArrayD::from_shape_vec(IxDyn(&[32, 32]), uv).unwrap();

    let (probs, seg, uncert) = refine_with_crf_uncertainty(
        &image.view(), &prob.view(), &unc.view(), (2.0, 3.0), 0.15, 2, 1e-8, 0.5, 1.0,
    );
    assert_eq!(probs.dim(), (3, 32, 32));
    assert_eq!(seg.dim(), (32, 32));
    assert_eq!(uncert.dim(), (32, 32));
}

#[test]
fn test_prob_map_on_different_grid() {
    let mut rng = rand::thread_rng();
    let v: Vec<f64> = (0..3 * 128 * 128).map(|_| rng.gen::<f64>()).collect();
    let image = ArrayD::from_shape_vec(IxDyn(&[3, 128, 128]), v).unwrap();
    let pv: Vec<f64> = (0..64 * 64).map(|_| rng.gen::<f64>()).collect();
    let prob = ArrayD::from_shape_vec(IxDyn(&[64, 64]), pv).unwrap();
    let uv: Vec<f64> = (0..64 * 64).map(|_| rng.gen::<f64>()).collect();
    let unc = ArrayD::from_shape_vec(IxDyn(&[64, 64]), uv).unwrap();

    let (probs, seg, _) = refine_with_crf_uncertainty(
        &image.view(), &prob.view(), &unc.view(), (2.0, 3.0), 0.15, 1, 1e-8, 0.5, 1.0,
    );
    assert_eq!(probs.dim(), (2, 64, 64));
    assert_eq!(seg.dim(), (64, 64));
}

#[test]
fn test_clean_disk_is_preserved() {
    let h = 256usize;
    let w = 256usize;
    let mut gt = vec![0.0f64; h * w];
    let mut prob = vec![0.0f64; h * w];
    let mut unc = vec![0.0f64; h * w];
    for i in 0..h {
        for j in 0..w {
            let r = ((i as f64 - 128.0).powi(2) + (j as f64 - 128.0).powi(2)).sqrt();
            gt[i * w + j] = if r < 40.0 { 1.0 } else { 0.0 };
            let p = 1.0 / (1.0 + ((r - 40.0) / 2.0).exp());
            prob[i * w + j] = p;
            unc[i * w + j] = -(p * (p + 1e-8).ln() + (1.0 - p) * (1.0 - p + 1e-8).ln());
        }
    }
    let image_vals = vec![100.0 / 255.0; 3 * h * w];
    let image = ArrayD::from_shape_vec(IxDyn(&[3, h, w]), image_vals).unwrap();
    let prob_arr = ArrayD::from_shape_vec(IxDyn(&[h, w]), prob).unwrap();
    let unc_arr = ArrayD::from_shape_vec(IxDyn(&[h, w]), unc).unwrap();

    let (_, seg, _) = refine_with_crf_uncertainty(
        &image.view(), &prob_arr.view(), &unc_arr.view(), (2.0, 3.0), 0.15, 3, 1e-8, 0.5, 1.0,
    );

    let seg_flat: Vec<u8> = seg.iter().copied().collect();
    let gt_bin: Vec<u8> = gt.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();
    let prob_bin: Vec<u8> = prob_arr.iter().map(|&v| if v > 0.5 { 1 } else { 0 }).collect();

    let seg_area = seg_flat.iter().filter(|&&v| v > 0).count();
    let prob_area = prob_bin.iter().filter(|&&v| v > 0).count().max(1);
    let ratio = seg_area as f64 / prob_area as f64;

    assert!(dice(&seg_flat, &gt_bin) > 0.93, "dice too low");
    assert!(ratio >= 0.85 && ratio <= 1.10, "area ratio {}", ratio);
}

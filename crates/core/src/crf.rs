use ndarray::{Array2, Array3, ArrayViewD, Axis};

fn rgb_to_gray(rgb: &Array3<f64>) -> Array2<f64> {
    let (h, w, _) = rgb.dim();
    let mut gray = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            gray[[i, j]] = 0.299 * rgb[[i, j, 0]] + 0.587 * rgb[[i, j, 1]] + 0.114 * rgb[[i, j, 2]];
        }
    }
    gray
}

fn reflect101_pad(src: &Array2<f64>, r: usize) -> Array2<f64> {
    let (h, w) = src.dim();
    let (oh, ow) = (h + 2 * r, w + 2 * r);
    let mut out = Array2::<f64>::zeros((oh, ow));
    let idx = |x: isize, n: isize| -> usize {
        if x < 0 {
            (-x) as usize
        } else if x >= n {
            (2 * (n - 1) - x) as usize
        } else {
            x as usize
        }
    };
    for i in 0..oh {
        for j in 0..ow {
            let si = idx(i as isize - r as isize, h as isize);
            let sj = idx(j as isize - r as isize, w as isize);
            out[[i, j]] = src[[si, sj]];
        }
    }
    out
}

fn separable_blur(src: &Array2<f64>, kernel: &[f64]) -> Array2<f64> {
    let (h, w) = src.dim();
    let r = kernel.len() / 2;
    let padded = reflect101_pad(src, r);
    let mut tmp = Array2::<f64>::zeros((h + 2 * r, w));
    for i in 0..(h + 2 * r) {
        for j in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                acc += padded[[i, j + k]] * kv;
            }
            tmp[[i, j]] = acc;
        }
    }
    let mut out = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0;
            for (k, &kv) in kernel.iter().enumerate() {
                acc += tmp[[i + k, j]] * kv;
            }
            out[[i, j]] = acc;
        }
    }
    out
}

fn gaussian_kernel(sigma: f64) -> Vec<f64> {
    let r = (sigma * 3.0).round() as usize;
    let r = r.max(1);
    let mut k = Vec::with_capacity(2 * r + 1);
    let mut sum = 0.0;
    for i in 0..(2 * r + 1) {
        let x = i as f64 - r as f64;
        let v = (-(x * x) / (2.0 * sigma * sigma)).exp();
        k.push(v);
        sum += v;
    }
    for v in k.iter_mut() {
        *v /= sum;
    }
    k
}

fn sobel_x(src: &Array2<f64>) -> Array2<f64> {
    let (h, w) = src.dim();
    let padded = reflect101_pad(src, 1);
    let kx = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let mut out = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0;
            for di in 0..3 {
                for dj in 0..3 {
                    acc += padded[[i + di, j + dj]] * kx[di * 3 + dj];
                }
            }
            out[[i, j]] = acc;
        }
    }
    out
}

fn sobel_y(src: &Array2<f64>) -> Array2<f64> {
    let (h, w) = src.dim();
    let padded = reflect101_pad(src, 1);
    let ky = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let mut out = Array2::<f64>::zeros((h, w));
    for i in 0..h {
        for j in 0..w {
            let mut acc = 0.0;
            for di in 0..3 {
                for dj in 0..3 {
                    acc += padded[[i + di, j + dj]] * ky[di * 3 + dj];
                }
            }
            out[[i, j]] = acc;
        }
    }
    out
}

fn bilateral_filter(src: &Array2<f64>, sigma_color: f64, sigma_space: f64) -> Array2<f64> {
    let (h, w) = src.dim();
    let radius = ((sigma_space * 1.5).round() as usize).max(1);
    let color_scale = -0.5 / (sigma_color * sigma_color);
    let space_scale = -0.5 / (sigma_space * sigma_space);
    let mut out = Array2::<f64>::zeros((h, w));
    let padded = reflect101_pad(src, radius);
    for i in 0..h {
        for j in 0..w {
            let center = src[[i, j]];
            let mut acc = 0.0;
            let mut norm = 0.0;
            for di in 0..(2 * radius + 1) {
                for dj in 0..(2 * radius + 1) {
                    let si = di as isize - radius as isize;
                    let sj = dj as isize - radius as isize;
                    let val = padded[[i + di, j + dj]];
                    let diff = val - center;
                    let space = (si * si + sj * sj) as f64;
                    let wgt = (space_scale * space).exp() * (color_scale * diff * diff).exp();
                    acc += val * wgt;
                    norm += wgt;
                }
            }
            out[[i, j]] = acc / norm;
        }
    }
    out
}

fn resize_bilinear(src: &Array2<f64>, new_h: usize, new_w: usize) -> Array2<f64> {
    let (h, w) = src.dim();
    let mut out = Array2::<f64>::zeros((new_h, new_w));
    let sx = w as f64 / new_w as f64;
    let sy = h as f64 / new_h as f64;
    for i in 0..new_h {
        for j in 0..new_w {
            let fy = (i as f64 + 0.5) * sy - 0.5;
            let fx = (j as f64 + 0.5) * sx - 0.5;
            let y0 = fy.floor().max(0.0) as usize;
            let x0 = fx.floor().max(0.0) as usize;
            let y1 = (y0 + 1).min(h - 1);
            let x1 = (x0 + 1).min(w - 1);
            let dy = fy - y0 as f64;
            let dx = fx - x0 as f64;
            let v00 = src[[y0, x0]];
            let v01 = src[[y0, x1]];
            let v10 = src[[y1, x0]];
            let v11 = src[[y1, x1]];
            let top = v00 + (v01 - v00) * dx;
            let bot = v10 + (v11 - v10) * dx;
            out[[i, j]] = top + (bot - top) * dy;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn refine_with_crf_uncertainty(
    image: &ArrayViewD<f64>,
    prob_map: &ArrayViewD<f64>,
    _uncertainty_map: &ArrayViewD<f64>,
    sdims: (f64, f64),
    schan: f64,
    n_iters: usize,
    epsilon: f64,
    w_g: f64,
    w_b: f64,
) -> (Array3<f64>, Array2<u8>, Array2<f64>) {
    let mut img = image.to_owned();
    while img.ndim() > 3 && img.shape()[0] == 1 {
        img = img.index_axis_move(Axis(0), 0);
    }
    if img.ndim() == 4 {
        img = img.index_axis_move(Axis(0), 0);
    }
    if img.ndim() == 3 && (img.shape()[0] == 1 || img.shape()[0] == 3 || img.shape()[0] == 4) {
        let (c, h, w) = (img.shape()[0], img.shape()[1], img.shape()[2]);
        let mut t = Array3::<f64>::zeros((h, w, c));
        for i in 0..h {
            for j in 0..w {
                for k in 0..c {
                    t[[i, j, k]] = img[[k, i, j]];
                }
            }
        }
        img = t.into_dyn();
    }
    if img.ndim() == 3 && img.shape()[2] > 3 {
        let view = img.view();
        let (h, w, _) = (img.shape()[0], img.shape()[1], 3);
        let mut t = Array3::<f64>::zeros((h, w, 3));
        for i in 0..h {
            for j in 0..w {
                for k in 0..3 {
                    t[[i, j, k]] = view[[i, j, k]];
                }
            }
        }
        img = t.into_dyn();
    }

    let gray: Array2<f64> = if img.ndim() == 3 && img.shape()[2] == 3 {
        let view = img.view();
        let rgb = view
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap()
            .to_owned();
        rgb_to_gray(&rgb)
    } else if img.ndim() == 3 {
        let view = img.view();
        let (h, w, _) = (img.shape()[0], img.shape()[1], img.shape()[2]);
        let mut g = Array2::<f64>::zeros((h, w));
        for i in 0..h {
            for j in 0..w {
                g[[i, j]] = view[[i, j, 0]];
            }
        }
        g
    } else {
        img.view().into_dimensionality::<ndarray::Ix2>().unwrap().to_owned()
    };

    let is_uint8_scale = gray.iter().any(|&v| v > 1.0);
    let gray = if is_uint8_scale {
        gray.mapv(|v| v / 255.0)
    } else {
        gray.mapv(|v| v.clamp(0.0, 1.0))
    };

    let gx = sobel_x(&gray);
    let gy = sobel_y(&gray);
    let mut grad = Array2::<f64>::zeros(gx.dim());
    for i in 0..gx.dim().0 {
        for j in 0..gx.dim().1 {
            grad[[i, j]] = (gx[[i, j]].powi(2) + gy[[i, j]].powi(2)).sqrt();
        }
    }
    let grad_max = grad.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let mut edge_gate = grad.mapv(|g| (-((g / (grad_max + 1e-8)) / 0.15).powi(2)).exp());

    let prob_stack: Array3<f64> = if prob_map.ndim() == 2 {
        let p = prob_map
            .view()
            .into_dimensionality::<ndarray::Ix2>()
            .unwrap()
            .to_owned();
        let (h, w) = p.dim();
        let mut stack = Array3::<f64>::zeros((2, h, w));
        for i in 0..h {
            for j in 0..w {
                stack[[0, i, j]] = 1.0 - p[[i, j]];
                stack[[1, i, j]] = p[[i, j]];
            }
        }
        stack
    } else {
        prob_map
            .view()
            .into_dimensionality::<ndarray::Ix3>()
            .unwrap()
            .to_owned()
    };

    let n_classes = prob_stack.shape()[0];
    let ph = prob_stack.shape()[1];
    let pw = prob_stack.shape()[2];

    if edge_gate.dim() != (ph, pw) {
        edge_gate = resize_bilinear(&edge_gate, ph, pw);
    }

    let mut unary = prob_stack.mapv(|v| v.clamp(epsilon, 1.0 - epsilon));
    for i in 0..ph {
        for j in 0..pw {
            let s = (0..n_classes).map(|c| unary[[c, i, j]]).sum::<f64>() + epsilon;
            for c in 0..n_classes {
                unary[[c, i, j]] /= s;
            }
        }
    }

    let mut q = unary.clone();
    for _ in 0..n_iters {
        let mut message = Array3::<f64>::zeros((n_classes, ph, pw));
        for c in 0..n_classes {
            let qc = q.index_axis(Axis(0), c).to_owned();
            let mut kernel = Array2::<f64>::zeros((ph, pw));
            if w_g > 0.0 {
                let gk = gaussian_kernel(sdims.0);
                let blurred = separable_blur(&qc, &gk);
                for i in 0..ph {
                    for j in 0..pw {
                        kernel[[i, j]] += w_g * blurred[[i, j]];
                    }
                }
            }
            if w_b > 0.0 {
                let bf = bilateral_filter(&qc, schan, sdims.1);
                for i in 0..ph {
                    for j in 0..pw {
                        kernel[[i, j]] += w_b * bf[[i, j]];
                    }
                }
            }
            for i in 0..ph {
                for j in 0..pw {
                    message[[c, i, j]] = kernel[[i, j]] * edge_gate[[i, j]];
                }
            }
        }

        for c in 0..n_classes {
            for i in 0..ph {
                for j in 0..pw {
                    q[[c, i, j]] = (unary[[c, i, j]] * message[[c, i, j]].exp()).clamp(epsilon, 1.0 - epsilon);
                }
            }
        }
        for i in 0..ph {
            for j in 0..pw {
                let s = (0..n_classes).map(|c| q[[c, i, j]]).sum::<f64>() + epsilon;
                for c in 0..n_classes {
                    q[[c, i, j]] /= s;
                }
            }
        }
    }

    let mut refined_segmentation = Array2::<u8>::zeros((ph, pw));
    for i in 0..ph {
        for j in 0..pw {
            let mut best = 0usize;
            let mut bestv = f64::NEG_INFINITY;
            for c in 0..n_classes {
                if q[[c, i, j]] > bestv {
                    bestv = q[[c, i, j]];
                    best = c;
                }
            }
            refined_segmentation[[i, j]] = best as u8;
        }
    }

    let mut refined_uncertainty = Array2::<f64>::zeros((ph, pw));
    for i in 0..ph {
        for j in 0..pw {
            let mut s = 0.0;
            for c in 0..n_classes {
                s -= q[[c, i, j]] * (q[[c, i, j]] + epsilon).ln();
            }
            refined_uncertainty[[i, j]] = s;
        }
    }

    (q, refined_segmentation, refined_uncertainty)
}

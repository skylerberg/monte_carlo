//! Symmetric eigendecomposition, which CMA-ES needs and nothing else here does.
//!
//! Cyclic Jacobi rather than anything cleverer: the covariance matrix is as
//! wide as the parameter vector — a couple of dozen at most — and it is
//! decomposed once every several generations, each of which costs hundreds of
//! games. At that size Jacobi is microseconds, needs no pivoting or balancing,
//! and is short enough to read, which is worth more here than an asymptotic
//! advantage that never arrives.

/// Eigenvalues and eigenvectors of the symmetric `n`×`n` matrix in `matrix`,
/// row-major.
///
/// Returns `(values, vectors)` with `vectors` row-major and each eigenvector
/// stored as a *column*: `vectors[row * n + k]` is component `row` of
/// eigenvector `k`, matching `values[k]`. That is the layout CMA-ES wants for
/// `B`, where `C = B diag(values) Bᵀ`.
pub fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    debug_assert_eq!(matrix.len(), n * n);

    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }

    // Jacobi converges quadratically once the off-diagonal is small; fifty
    // sweeps is far past what any covariance matrix of this size needs, and is
    // here only so a matrix full of NaN cannot spin forever.
    const MAX_SWEEPS: usize = 50;
    for _ in 0..MAX_SWEEPS {
        let mut off = 0.0;
        for p in 0..n {
            for q in (p + 1)..n {
                off += a[p * n + q] * a[p * n + q];
            }
        }
        if off <= 1e-30 {
            break;
        }

        for p in 0..(n.saturating_sub(1)) {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-300 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];

                // The rotation that zeroes (p, q), taking the smaller root so
                // the transformation stays well conditioned.
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (theta * theta + 1.0).sqrt())
                } else {
                    -1.0 / (-theta + (theta * theta + 1.0).sqrt())
                };
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;

                for k in 0..n {
                    let akp = a[k * n + p];
                    let akq = a[k * n + q];
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[p * n + k];
                    let aqk = a[q * n + k];
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[k * n + p];
                    let vkq = v[k * n + q];
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }

    let values = (0..n).map(|i| a[i * n + i]).collect();
    (values, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `B diag(values) Bᵀ` has to reproduce the matrix that went in, which is
    /// the property CMA-ES relies on and the one a sign or transpose slip in
    /// the rotation would break.
    fn assert_reconstructs(matrix: &[f64], n: usize) {
        let (values, vectors) = symmetric_eigen(matrix, n);
        for row in 0..n {
            for col in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += vectors[row * n + k] * values[k] * vectors[col * n + k];
                }
                assert!(
                    (sum - matrix[row * n + col]).abs() < 1e-9,
                    "reconstruction differs at ({row}, {col}): {sum} vs {}",
                    matrix[row * n + col]
                );
            }
        }
    }

    #[test]
    fn diagonal_matrix_is_its_own_decomposition() {
        let m = vec![4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 1.0];
        let (values, _) = symmetric_eigen(&m, 3);
        let mut sorted = values.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 1.0).abs() < 1e-12);
        assert!((sorted[1] - 4.0).abs() < 1e-12);
        assert!((sorted[2] - 9.0).abs() < 1e-12);
        assert_reconstructs(&m, 3);
    }

    #[test]
    fn eigenvectors_are_orthonormal() {
        let m = vec![2.0, 1.0, 0.5, 1.0, 3.0, -0.4, 0.5, -0.4, 1.5];
        let (_, vectors) = symmetric_eigen(&m, 3);
        for i in 0..3 {
            for j in 0..3 {
                let mut dot = 0.0;
                for k in 0..3 {
                    dot += vectors[k * 3 + i] * vectors[k * 3 + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((dot - expected).abs() < 1e-10, "({i},{j}) dot = {dot}");
            }
        }
    }

    #[test]
    fn reconstructs_a_dense_correlated_matrix() {
        let m = vec![
            1.0, 0.8, 0.3, 0.1, //
            0.8, 2.0, 0.5, 0.2, //
            0.3, 0.5, 1.5, 0.9, //
            0.1, 0.2, 0.9, 3.0,
        ];
        assert_reconstructs(&m, 4);
    }

    #[test]
    fn handles_one_by_one() {
        let (values, vectors) = symmetric_eigen(&[7.0], 1);
        assert!((values[0] - 7.0).abs() < 1e-12);
        assert!((vectors[0] - 1.0).abs() < 1e-12);
    }
}

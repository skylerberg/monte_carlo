use mcts::rand_core::SeedableRng;
use wyrand::WyRand;

use crate::eigen::symmetric_eigen;
use crate::optimizer::Optimizer;
use crate::resume::{exact, ResumeError, Snapshot};
use crate::sampling::standard_normal;
use crate::tunable::Tunable;

/// Knobs for [`CmaEs`]. The defaults are the standard ones and are rarely worth
/// touching — the strategy derives almost everything from the dimension.
#[derive(Debug, Clone)]
pub struct CmaParams {
    /// Candidates per generation. Zero picks the standard `4 + floor(3 ln n)`.
    ///
    /// Raising it buys robustness against noise at a directly proportional cost
    /// in games. When fitness is as noisy as a few hundred games make it, the
    /// cheaper fix is usually more games per candidate rather than more
    /// candidates: it attacks the noise itself rather than averaging over it.
    pub population: usize,
    /// Initial step size, as a multiple of [`Tunable::gene_scales`]. 0.3 means
    /// a generation's first candidates sit about 30% of a gene's own magnitude
    /// away from the seed.
    pub initial_sigma: f64,
    pub seed: u64,
}

impl Default for CmaParams {
    fn default() -> Self {
        Self {
            population: 0,
            initial_sigma: 0.3,
            seed: 0x5EED,
        }
    }
}

/// Covariance Matrix Adaptation Evolution Strategy, `(mu/mu_w, lambda)` with
/// rank-one and rank-mu updates and cumulative step-size adaptation.
///
/// The reason to prefer it over [`crate::Ga`] on a vector of continuous weights
/// is that it learns the *shape* of the region worth sampling. Evaluation
/// weights are rarely independent — raise what a castle is worth and what
/// threatening one is worth should move with it — and a genetic algorithm's
/// uniform crossover has no way to express that, so it proposes combinations
/// that break the relationship and spends games discovering they are bad. CMA-ES
/// accumulates a covariance matrix that encodes the correlation, and samples
/// along it.
///
/// Per-parameter scale is carried in the initial covariance, `diag(scales²)`,
/// which is what lets a weight near 12 and a probability near 0.05 be tuned in
/// the same run without the small one being swamped.
pub struct CmaEs {
    n: usize,
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    chi_n: f64,

    mean: Vec<f64>,
    sigma: f64,
    /// Covariance, row-major and symmetric.
    cov: Vec<f64>,
    /// Eigenvectors of `cov` as columns.
    basis: Vec<f64>,
    /// Square roots of the eigenvalues, matching `basis`.
    spread: Vec<f64>,
    p_sigma: Vec<f64>,
    p_c: Vec<f64>,

    generation: u32,
    rng: WyRand,
    best_genes: Vec<f64>,
    best_fitness: f64,
}

/// What a checkpoint carries: the state the search has *evolved*, and nothing
/// else.
///
/// Everything derived — the population size, the recombination weights, the
/// learning rates — is a pure function of the dimension and [`CmaParams`], so
/// the constructor recomputes it and a checkpoint never carries it. That is not
/// only tidier. A derived constant that travels through a file can come back
/// disagreeing with the code that recomputes it, and a recombination weight one
/// bit out is enough to move every candidate after the resume.
#[derive(serde::Serialize, serde::Deserialize)]
struct CmaState {
    /// Checked rather than restored: resuming with a different population would
    /// change the strategy's whole character while looking like a continuation.
    lambda: usize,
    #[serde(with = "exact::vector")]
    mean: Vec<f64>,
    #[serde(with = "exact::scalar")]
    sigma: f64,
    #[serde(with = "exact::vector")]
    cov: Vec<f64>,
    #[serde(with = "exact::vector")]
    basis: Vec<f64>,
    #[serde(with = "exact::vector")]
    spread: Vec<f64>,
    #[serde(with = "exact::vector")]
    p_sigma: Vec<f64>,
    #[serde(with = "exact::vector")]
    p_c: Vec<f64>,
    generation: u32,
    rng: WyRand,
    #[serde(with = "exact::vector")]
    best_genes: Vec<f64>,
    #[serde(with = "exact::scalar")]
    best_fitness: f64,
}

impl CmaEs {
    /// Start a search centred on `base`.
    pub fn new<T: Tunable>(base: &T, params: CmaParams) -> Self {
        let mean = base.to_genes();
        let scales = base.gene_scales();
        let n = mean.len();
        assert!(
            n > 0,
            "mcts-tune: nothing to tune — `to_genes` returned no genes"
        );
        assert_eq!(
            scales.len(),
            n,
            "mcts-tune: `gene_scales` and `to_genes` disagree on the dimension"
        );
        assert!(
            params.initial_sigma > 0.0 && params.initial_sigma.is_finite(),
            "mcts-tune: initial sigma must be finite and positive"
        );

        let lambda = if params.population == 0 {
            4 + (3.0 * (n as f64).ln()).floor() as usize
        } else {
            params.population
        }
        .max(4);
        let mu = (lambda / 2).max(1);

        // Log-decreasing recombination weights: the best candidate counts for
        // more than the mu-th, rather than the selected half counting equally.
        let raw: Vec<f64> = (0..mu)
            .map(|i| (mu as f64 + 0.5).ln() - ((i + 1) as f64).ln())
            .collect();
        let total: f64 = raw.iter().sum();
        let weights: Vec<f64> = raw.iter().map(|w| w / total).collect();
        let mu_eff = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        let nf = n as f64;
        let c_sigma = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
        let d_sigma =
            1.0 + 2.0 * (0.0f64).max(((mu_eff - 1.0) / (nf + 1.0)).sqrt() - 1.0) + c_sigma;
        let c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
        let c_1 = 2.0 / ((nf + 1.3) * (nf + 1.3) + mu_eff);
        let c_mu = (1.0 - c_1)
            .min(2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0) * (nf + 2.0) + mu_eff));
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        // C = diag(scales²) makes the initial sampling ellipsoid match each
        // parameter's own units, so `initial_sigma` is one relative number
        // rather than one absolute number that can only suit one parameter.
        let mut cov = vec![0.0; n * n];
        let mut basis = vec![0.0; n * n];
        for i in 0..n {
            let scale = if scales[i].is_finite() && scales[i] > 0.0 {
                scales[i]
            } else {
                1e-3
            };
            cov[i * n + i] = scale * scale;
            basis[i * n + i] = 1.0;
        }
        let spread: Vec<f64> = (0..n).map(|i| cov[i * n + i].sqrt()).collect();

        Self {
            n,
            lambda,
            mu,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            best_genes: mean.clone(),
            mean,
            sigma: params.initial_sigma,
            cov,
            basis,
            spread,
            p_sigma: vec![0.0; n],
            p_c: vec![0.0; n],
            generation: 0,
            rng: WyRand::seed_from_u64(params.seed),
            best_fitness: f64::NEG_INFINITY,
        }
    }

    /// `basis * vector`, the map out of the sampling ellipsoid's own axes.
    fn rotate_out(&self, vector: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|row| {
                (0..self.n)
                    .map(|k| self.basis[row * self.n + k] * vector[k])
                    .sum()
            })
            .collect()
    }

    /// `basisᵀ * vector`, the map into them.
    fn rotate_in(&self, vector: &[f64]) -> Vec<f64> {
        (0..self.n)
            .map(|k| {
                (0..self.n)
                    .map(|row| self.basis[row * self.n + k] * vector[row])
                    .sum()
            })
            .collect()
    }

    /// Refresh the decomposition the sampler and the step-size path both read.
    ///
    /// Done every generation rather than on the usual lazy schedule: the
    /// standard condition amounts to "every generation" at the dimensions a
    /// hand-written evaluation has, and a Jacobi sweep over a matrix this size
    /// is nothing beside the hundreds of games each generation costs.
    fn decompose(&mut self) {
        let n = self.n;
        // Numerical drift makes the two triangles disagree in the last bits;
        // Jacobi assumes exact symmetry.
        for i in 0..n {
            for j in (i + 1)..n {
                let mean = 0.5 * (self.cov[i * n + j] + self.cov[j * n + i]);
                self.cov[i * n + j] = mean;
                self.cov[j * n + i] = mean;
            }
        }

        let (values, vectors) = symmetric_eigen(&self.cov, n);
        let largest = values.iter().cloned().fold(f64::MIN, f64::max);

        // A covariance that has lost positive-definiteness to rounding would
        // otherwise produce a NaN step and silently end the run. Flooring the
        // eigenvalues rather than restarting keeps the shape that has been
        // learned so far.
        let floor = if largest.is_finite() && largest > 0.0 {
            largest * 1e-14
        } else {
            1e-20
        };
        self.spread = values
            .iter()
            .map(|value| {
                let clamped = if value.is_finite() {
                    value.max(floor)
                } else {
                    floor
                };
                clamped.sqrt()
            })
            .collect();
        self.basis = vectors;
    }
}

impl Optimizer for CmaEs {
    fn ask(&mut self) -> Vec<Vec<f64>> {
        (0..self.lambda)
            .map(|_| {
                let scaled: Vec<f64> = (0..self.n)
                    .map(|k| self.spread[k] * standard_normal(&mut self.rng))
                    .collect();
                let step = self.rotate_out(&scaled);
                (0..self.n)
                    .map(|i| self.mean[i] + self.sigma * step[i])
                    .collect()
            })
            .collect()
    }

    fn tell(&mut self, candidates: &[Vec<f64>], fitness: &[f64]) {
        assert_eq!(
            candidates.len(),
            fitness.len(),
            "mcts-tune: told {} candidates and {} fitnesses",
            candidates.len(),
            fitness.len()
        );
        if candidates.is_empty() {
            return;
        }
        let n = self.n;

        // Steps are recovered from the genes actually played, not from what
        // `ask` proposed, so that a repair which moved a candidate is reflected
        // in the distribution rather than attributed to a candidate that was
        // never tried.
        let steps: Vec<Vec<f64>> = candidates
            .iter()
            .map(|genes| {
                (0..n)
                    .map(|i| (genes[i] - self.mean[i]) / self.sigma)
                    .collect()
            })
            .collect();

        let mut ranked: Vec<(usize, f64)> = fitness.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("fitness is never NaN"));

        if ranked[0].1 > self.best_fitness {
            self.best_fitness = ranked[0].1;
            self.best_genes = candidates[ranked[0].0].clone();
        }

        let selected = self.mu.min(ranked.len());
        let mut recombined = vec![0.0; n];
        for (rank, &(candidate, _)) in ranked.iter().take(selected).enumerate() {
            for (slot, step) in recombined.iter_mut().zip(&steps[candidate]) {
                *slot += self.weights[rank] * step;
            }
        }

        // The step-size path is measured in the sphere C^(-1/2) maps the search
        // into, so that its length reports how far the mean is travelling
        // independently of the shape the covariance has taken.
        let rotated = self.rotate_in(&recombined);
        let whitened: Vec<f64> = (0..n).map(|k| rotated[k] / self.spread[k]).collect();
        let unrotated = self.rotate_out(&whitened);

        self.generation += 1;
        let sigma_gain = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for (path, step) in self.p_sigma.iter_mut().zip(&unrotated) {
            *path = (1.0 - self.c_sigma) * *path + sigma_gain * step;
        }
        let path_norm = self.p_sigma.iter().map(|v| v * v).sum::<f64>().sqrt();

        // Stalling the rank-one update while the path is long stops the
        // covariance inflating during the early generations, when consecutive
        // steps point the same way simply because the mean is far from anywhere
        // good.
        let decay = 1.0 - (1.0 - self.c_sigma).powi(2 * self.generation as i32);
        let expanding =
            path_norm / decay.max(1e-300).sqrt() < (1.4 + 2.0 / (n as f64 + 1.0)) * self.chi_n;
        let path_gain = if expanding {
            (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt()
        } else {
            0.0
        };
        for (path, step) in self.p_c.iter_mut().zip(&recombined) {
            *path = (1.0 - self.c_c) * *path + path_gain * step;
        }

        let correction = if expanding {
            0.0
        } else {
            self.c_c * (2.0 - self.c_c)
        };
        for i in 0..n {
            for j in 0..n {
                let rank_one = self.p_c[i] * self.p_c[j] + correction * self.cov[i * n + j];
                let mut rank_mu = 0.0;
                for (rank, &(candidate, _)) in ranked.iter().take(selected).enumerate() {
                    rank_mu += self.weights[rank] * steps[candidate][i] * steps[candidate][j];
                }
                self.cov[i * n + j] = (1.0 - self.c_1 - self.c_mu) * self.cov[i * n + j]
                    + self.c_1 * rank_one
                    + self.c_mu * rank_mu;
            }
        }

        for (centre, step) in self.mean.iter_mut().zip(&recombined) {
            *centre += self.sigma * step;
        }

        self.sigma *= ((self.c_sigma / self.d_sigma) * (path_norm / self.chi_n - 1.0)).exp();
        if !self.sigma.is_finite() || self.sigma <= 0.0 {
            // Only reachable if the game handed back a fitness the strategy
            // could not have produced a finite step from. Recovering beats
            // spending the rest of the run sampling NaN.
            self.sigma = 1e-8;
        }

        self.decompose();
    }

    fn best(&self) -> (&[f64], f64) {
        (&self.best_genes, self.best_fitness)
    }

    fn population(&self) -> usize {
        self.lambda
    }

    fn name(&self) -> &'static str {
        "cma-es"
    }

    fn snapshot(&self) -> serde_json::Value {
        let state = CmaState {
            lambda: self.lambda,
            mean: self.mean.clone(),
            sigma: self.sigma,
            cov: self.cov.clone(),
            basis: self.basis.clone(),
            spread: self.spread.clone(),
            p_sigma: self.p_sigma.clone(),
            p_c: self.p_c.clone(),
            generation: self.generation,
            rng: self.rng.clone(),
            best_genes: self.best_genes.clone(),
            best_fitness: self.best_fitness,
        };
        serde_json::to_value(Snapshot::new(self.name(), self.n, &state))
            .expect("a snapshot serializes")
    }

    fn restore(&mut self, snapshot: &serde_json::Value) -> Result<(), ResumeError> {
        let state: CmaState = Snapshot::open(snapshot, self.name(), self.n)?;
        if state.lambda != self.lambda {
            return Err(ResumeError::Malformed(format!(
                "the checkpoint ran {} candidates a generation and this run is set to {}",
                state.lambda, self.lambda
            )));
        }
        let n = self.n;
        for (field, found, expected) in [
            ("mean", state.mean.len(), n),
            ("sigma path", state.p_sigma.len(), n),
            ("covariance path", state.p_c.len(), n),
            ("best genes", state.best_genes.len(), n),
            ("eigenvalues", state.spread.len(), n),
            ("covariance", state.cov.len(), n * n),
            ("basis", state.basis.len(), n * n),
        ] {
            if found != expected {
                return Err(ResumeError::Malformed(format!(
                    "the checkpoint's {field} holds {found} entries where {expected} are expected"
                )));
            }
        }

        self.mean = state.mean;
        self.sigma = state.sigma;
        self.cov = state.cov;
        self.basis = state.basis;
        self.spread = state.spread;
        self.p_sigma = state.p_sigma;
        self.p_c = state.p_c;
        self.generation = state.generation;
        self.rng = state.rng;
        self.best_genes = state.best_genes;
        self.best_fitness = state.best_fitness;
        Ok(())
    }
}

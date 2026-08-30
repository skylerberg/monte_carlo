use mcts::rand_core::SeedableRng;
use wyrand::WyRand;

use crate::optimizer::Optimizer;
use crate::sampling::{index, standard_normal, uniform};
use crate::tunable::Tunable;

/// Knobs for [`Ga`].
#[derive(Debug, Clone)]
pub struct GaParams {
    pub population: usize,
    /// Candidates carried into the next generation untouched. Without at least
    /// one, a generation that measures its best candidate unluckily can lose it
    /// outright.
    pub elites: usize,
    /// Candidates drawn per tournament. Larger is greedier.
    pub tournament: usize,
    /// Probability that any one gene is perturbed when a child is made.
    pub mutation_rate: f64,
    /// Perturbation size, as a multiple of [`Tunable::gene_scales`].
    pub mutation_scale: f64,
    /// Spread of the initial population around the seed, same units.
    pub init_scale: f64,
    pub seed: u64,
}

impl Default for GaParams {
    fn default() -> Self {
        Self {
            population: 20,
            elites: 2,
            tournament: 3,
            mutation_rate: 0.15,
            mutation_scale: 0.25,
            // Deliberately far tighter than "scale every gene by a uniform draw
            // from [0.5, 2)", which is what makes an initial population where
            // every member has had all of its genes moved at once and is
            // therefore much worse than the seed. Generations then get spent
            // climbing back to the starting point.
            init_scale: 0.15,
            seed: 0x5EED,
        }
    }
}

/// A generational genetic algorithm: elitism, tournament selection, uniform
/// crossover, Gaussian mutation.
///
/// Mutation is **additive**, scaled by [`Tunable::gene_scales`], rather than
/// multiplicative. Multiplicative mutation cannot change a gene's sign and
/// cannot move a gene that has reached zero, so any parameter clamped to zero
/// by a repair is frozen there for the rest of the run — silently, and looking
/// exactly like a parameter the search decided to leave alone.
pub struct Ga {
    params: GaParams,
    scales: Vec<f64>,
    population: Vec<Vec<f64>>,
    rng: WyRand,
    best_genes: Vec<f64>,
    best_fitness: f64,
}

impl Ga {
    /// Seed a population around `base`.
    pub fn new<T: Tunable>(base: &T, params: GaParams) -> Self {
        assert!(
            params.population >= 2,
            "mcts-tune: a GA needs at least two candidates to breed from"
        );
        assert!(
            params.elites < params.population,
            "mcts-tune: {} elites in a population of {} leaves nothing to breed",
            params.elites,
            params.population
        );
        assert!(
            params.tournament >= 1,
            "mcts-tune: a tournament needs at least one entrant"
        );

        let seed_genes = base.to_genes();
        let scales = base.gene_scales();
        assert_eq!(
            scales.len(),
            seed_genes.len(),
            "mcts-tune: `gene_scales` and `to_genes` disagree on the dimension"
        );

        let mut rng = WyRand::seed_from_u64(params.seed);
        let mut population = Vec::with_capacity(params.population);

        // The seed itself is a member, so a run can never end up worse than
        // where it started for want of ever having measured the starting point.
        population.push(seed_genes.clone());
        for _ in 1..params.population {
            let mut genes = seed_genes.clone();
            for (gene, scale) in genes.iter_mut().zip(&scales) {
                *gene += standard_normal(&mut rng) * params.init_scale * scale;
            }
            T::repair(&mut genes);
            population.push(genes);
        }

        Self {
            params,
            scales,
            population,
            rng,
            best_genes: seed_genes,
            best_fitness: f64::NEG_INFINITY,
        }
    }

    fn tournament(&mut self, ranked: &[(usize, f64)]) -> usize {
        let mut best = index(&mut self.rng, ranked.len());
        for _ in 1..self.params.tournament {
            let challenger = index(&mut self.rng, ranked.len());
            if ranked[challenger].1 > ranked[best].1 {
                best = challenger;
            }
        }
        ranked[best].0
    }
}

impl Optimizer for Ga {
    fn ask(&mut self) -> Vec<Vec<f64>> {
        self.population.clone()
    }

    fn tell(&mut self, candidates: &[Vec<f64>], fitness: &[f64]) {
        assert_eq!(
            candidates.len(),
            fitness.len(),
            "mcts-tune: told {} candidates and {} fitnesses",
            candidates.len(),
            fitness.len()
        );

        // Adopt the repaired genes: those are what was actually measured, and
        // breeding from the unrepaired proposals would carry values back in
        // that the game cannot represent.
        self.population = candidates.to_vec();

        let mut ranked: Vec<(usize, f64)> = fitness.iter().copied().enumerate().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).expect("fitness is never NaN"));

        if ranked[0].1 > self.best_fitness {
            self.best_fitness = ranked[0].1;
            self.best_genes = self.population[ranked[0].0].clone();
        }

        let mut next = Vec::with_capacity(self.params.population);
        for &(candidate, _) in ranked.iter().take(self.params.elites) {
            next.push(self.population[candidate].clone());
        }

        while next.len() < self.params.population {
            let mother = self.tournament(&ranked);
            let father = self.tournament(&ranked);

            let mut child = Vec::with_capacity(self.scales.len());
            for gene in 0..self.scales.len() {
                let parent = if uniform(&mut self.rng) < 0.5 {
                    mother
                } else {
                    father
                };
                child.push(self.population[parent][gene]);
            }

            for (gene, slot) in child.iter_mut().enumerate() {
                if uniform(&mut self.rng) < self.params.mutation_rate {
                    *slot += standard_normal(&mut self.rng)
                        * self.params.mutation_scale
                        * self.scales[gene];
                }
            }

            next.push(child);
        }

        self.population = next;
    }

    fn best(&self) -> (&[f64], f64) {
        (&self.best_genes, self.best_fitness)
    }

    fn population(&self) -> usize {
        self.params.population
    }

    fn name(&self) -> &'static str {
        "ga"
    }
}

//! Model configuration (`AddiVortesConfig`): mandatory-seed constructor,
//! consuming `with_*` setters (never panic, never clamp), data-free
//! `validate()`, and the `fit()` convenience loop.

use crate::engine::data::{Data, Metric};
use crate::engine::error::{AddiVortesError, Result};
use crate::engine::model::FittedAddiVortes;
use crate::engine::model::ResponseFamily;

/// Configuration for an AddiVortes fit: plain data only. Mandatory seed;
/// hyperparameters at the paper's defaults (λ_c excepted, see the field
/// note); per-column metrics and the response family as enums. Component
/// wiring is crate-internal (the model files own it).
///
/// No `Default` impl on purpose: the seed is mandatory (reproducibility
/// contract).
/// Setters are consuming and never panic or clamp; all checking happens in
/// [`validate`](AddiVortesConfig::validate) (called by `fit`).
#[derive(Debug, Clone, PartialEq)]
pub struct AddiVortesConfig {
    /// Chain seed (expanded to the ChaCha8 key via splitmix64).
    pub(crate) seed: u64,
    /// Ensemble size m (paper default 200).
    pub(crate) m: usize,
    /// σ² prior degrees of freedom ν (paper default 6).
    pub(crate) nu: f64,
    /// σ² prior calibration quantile q (paper default 0.85).
    pub(crate) q: f64,
    /// μ prior spread parameter k (paper default 3): σ_μ = 0.5/(k√m).
    pub(crate) k: f64,
    /// Centre-coordinate prior/proposal spread σ_c (paper default 0.8).
    pub(crate) sigma_c: f64,
    /// Dimension-count prior parameter ω (paper default 3).
    pub(crate) omega: f64,
    /// Centre-count prior parameter λ_c (default 5, chosen by benchmark
    /// calibration of the exact shifted-Poisson prior; the paper reports 25).
    pub(crate) lambda_c: f64,
    /// Burn-in sweeps discarded by `fit` (default 200).
    pub(crate) burn_in: usize,
    /// Posterior draws kept by `fit` (default 1000).
    pub(crate) n_draws: usize,
    /// Thinning interval for `fit` (default 1 = keep every sweep).
    pub(crate) thinning: usize,
    /// Per-raw-column metrics; `None` = all Euclidean.
    pub(crate) metrics: Option<Vec<Metric>>,
    /// Response family (the predict-side half):
    /// selects the link and the per-variant predictive distribution.
    pub(crate) family: ResponseFamily,
    /// Cell-value prior SD σ_μ, set directly (scaled space); `None` = the
    /// k-rule σ_μ = 0.5/(k√m), or the family's own rule (BinaryProbit widens
    /// to 3/(k√m)). Crate-internal: the dial a model file turns when its
    /// derivation prescribes a σ_μ the k-rule cannot express.
    pub(crate) cell_prior_sd: Option<f64>,
}

impl AddiVortesConfig {
    /// A configuration with the shipped defaults (the paper's, except
    /// λ_c = 5) and the mandatory chain seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            m: 200,
            nu: 6.0,
            q: 0.85,
            k: 3.0,
            sigma_c: 0.8,
            omega: 3.0,
            lambda_c: 5.0,
            burn_in: 200,
            n_draws: 1000,
            thinning: 1,
            metrics: None,
            family: ResponseFamily::Gaussian,
            cell_prior_sd: None,
        }
    }

    /// Replace the chain seed (last wins).
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Ensemble size m.
    #[must_use]
    pub fn with_m(mut self, m: usize) -> Self {
        self.m = m;
        self
    }

    /// σ² prior degrees of freedom ν.
    #[must_use]
    pub fn with_nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    /// σ² prior calibration quantile q (Pr(σ < σ̂) = q).
    #[must_use]
    pub fn with_q(mut self, q: f64) -> Self {
        self.q = q;
        self
    }

    /// μ prior spread parameter k (σ_μ = 0.5/(k√m)).
    #[must_use]
    pub fn with_k(mut self, k: f64) -> Self {
        self.k = k;
        self
    }

    /// Centre-coordinate prior/proposal spread σ_c.
    #[must_use]
    pub fn with_sigma_c(mut self, sigma_c: f64) -> Self {
        self.sigma_c = sigma_c;
        self
    }

    /// Dimension-count prior parameter ω (must satisfy ω < p at fit).
    #[must_use]
    pub fn with_omega(mut self, omega: f64) -> Self {
        self.omega = omega;
        self
    }

    /// Centre-count prior parameter λ_c (default 5; the paper reports 25,
    /// reproduced with `with_lambda_c(25.0)`).
    #[must_use]
    pub fn with_lambda_c(mut self, lambda_c: f64) -> Self {
        self.lambda_c = lambda_c;
        self
    }

    /// Burn-in sweeps discarded by `fit`.
    #[must_use]
    pub fn with_burn_in(mut self, burn_in: usize) -> Self {
        self.burn_in = burn_in;
        self
    }

    /// Posterior draws kept by `fit`.
    #[must_use]
    pub fn with_draws(mut self, n_draws: usize) -> Self {
        self.n_draws = n_draws;
        self
    }

    /// Thinning interval for `fit` (keep every `thinning`-th sweep).
    #[must_use]
    pub fn with_thinning(mut self, thinning: usize) -> Self {
        self.thinning = thinning;
        self
    }

    /// Per-column metrics for the raw (pre-encoding) columns; the default
    /// is all-Euclidean. Last wins.
    #[must_use]
    pub fn with_metrics(mut self, metrics: Vec<Metric>) -> Self {
        self.metrics = Some(metrics);
        self
    }

    /// The response family (predict side; last wins; the default
    /// is [`ResponseFamily::Gaussian`], the paper's model). Selecting
    /// [`ResponseFamily::BinaryProbit`] turns the fit into Binary-AddiVortes:
    /// the response must be {0, 1} labels, the Albert–Chib augmentation and
    /// the pinned unit scale attach automatically, the cell-value prior widens to
    /// the ±3 latent range (σ_μ = 3/(k√m)), and every prediction entry
    /// speaks the probability scale through the probit link. Selecting
    /// [`ResponseFamily::RobustT`] keeps the identity link and the response
    /// scale but swaps the error law for Student-t of the given `df`: the
    /// scale-mixture augmentation, the weight-aware mean family and the
    /// precision-weighted σ² draw attach automatically, and prediction
    /// intervals come from the t-mixture predictive.
    #[must_use]
    pub fn with_response_family(mut self, family: ResponseFamily) -> Self {
        self.family = family;
        self
    }

    /// The configured response family (what
    /// [`with_response_family`](AddiVortesConfig::with_response_family) set,
    /// or the Gaussian default). The read-side bindings and loaders need to
    /// know which scale a fitted model's predictions speak.
    pub fn response_family(&self) -> ResponseFamily {
        self.family
    }

    /// The cell-value prior SD σ_μ, directly (last wins; scaled space; the
    /// default is the k-rule σ_μ = 0.5/(k√m), and BinaryProbit's family
    /// wiring widens to 3/(k√m)). The crate-internal width dial for model
    /// files whose derivation prescribes σ_μ itself; when set it wins over
    /// both the k-rule and the family rule, and `k` no longer reaches σ_μ.
    #[must_use]
    pub(crate) fn with_cell_prior_sd(mut self, sigma_mu: f64) -> Self {
        self.cell_prior_sd = Some(sigma_mu);
        self
    }

    /// Fit `n_chains` independent chains of the same model (the
    /// multi-chain entry the convergence diagnostics consume:
    /// [`diagnostics::r_hat`](crate::diagnostics::r_hat) and friends take
    /// per-chain draw vectors). Chain 0 runs this configuration's own seed,
    /// so its output is bit-identical to a single [`fit`]: the
    /// derivation never perturbs the single-chain contract; chains 1.. use
    /// seeds derived from it by successive splitmix64 outputs (pinned, so
    /// "seed S, chain k" names exactly one chain).
    ///
    /// [`fit`]: AddiVortesConfig::fit
    pub fn fit_chains(
        &self,
        x: &Data,
        y: &[f64],
        n_chains: usize,
    ) -> Result<Vec<FittedAddiVortes>> {
        if n_chains == 0 {
            return Err(AddiVortesError::InvalidHyperparameter {
                name: "n_chains".into(),
                reason: "must be at least 1".into(),
            });
        }
        let mut state = self.seed;
        (0..n_chains)
            .map(|chain| {
                let mut config = self.clone();
                if chain > 0 {
                    config.seed = crate::engine::sampler::splitmix64(&mut state);
                }
                config.fit(x, y)
            })
            .collect()
    }

    /// Data-free validation of every hyperparameter (fit calls this first).
    /// Never clamps; every failure is `InvalidHyperparameter` with the exact
    /// field name. ω is required positive here; the ω < p check needs data
    /// and happens at the fit boundary.
    pub fn validate(&self) -> Result<()> {
        let bad = |name: &str, reason: String| {
            Err(AddiVortesError::InvalidHyperparameter {
                name: name.into(),
                reason,
            })
        };
        if self.m < 1 {
            return bad("m", "must be at least 1".into());
        }
        if !(self.nu.is_finite() && self.nu > 0.0) {
            return bad(
                "nu",
                format!("must be finite and positive, got {}", self.nu),
            );
        }
        if !(self.q.is_finite() && self.q > 0.0 && self.q < 1.0) {
            return bad(
                "q",
                format!("must be in the open interval (0, 1), got {}", self.q),
            );
        }
        if !(self.k.is_finite() && self.k > 0.0) {
            return bad("k", format!("must be finite and positive, got {}", self.k));
        }
        if !(self.sigma_c.is_finite() && self.sigma_c > 0.0) {
            return bad(
                "sigma_c",
                format!("must be finite and positive, got {}", self.sigma_c),
            );
        }
        if !(self.omega.is_finite() && self.omega > 0.0) {
            return bad(
                "omega",
                format!("must be finite and positive, got {}", self.omega),
            );
        }
        if !(self.lambda_c.is_finite() && self.lambda_c > 0.0) {
            return bad(
                "lambda_c",
                format!("must be finite and positive, got {}", self.lambda_c),
            );
        }
        if let Some(sd) = self.cell_prior_sd {
            if !(sd.is_finite() && sd > 0.0) {
                return bad(
                    "cell_prior_sd",
                    format!("must be finite and positive, got {sd}"),
                );
            }
        }
        if self.n_draws < 1 {
            return bad("draws", "must be at least 1".into());
        }
        if self.thinning < 1 {
            return bad("thinning", "must be at least 1".into());
        }
        Ok(())
    }

    /// Fit the model: validate the configuration, run the boundary checks, and
    /// drive the [`Sampler`](crate::Sampler) for `burn_in + draws × thinning`
    /// sweeps, keeping every `thinning`-th sweep after burn-in.
    ///
    /// Consumes the configuration and stores it in the fitted model
    /// (self-contained).
    pub fn fit(self, x: &Data, y: &[f64]) -> Result<FittedAddiVortes> {
        crate::engine::model::fit(self, x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The dial validates like every other hyperparameter: never clamped,
    /// rejected with the exact field name.
    #[test]
    fn cell_prior_sd_validates_like_any_hyperparameter() {
        assert!(
            AddiVortesConfig::new(1)
                .with_cell_prior_sd(0.25)
                .validate()
                .is_ok()
        );
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let err = AddiVortesConfig::new(1)
                .with_cell_prior_sd(bad)
                .validate()
                .unwrap_err();
            assert!(matches!(
                err,
                AddiVortesError::InvalidHyperparameter { ref name, .. } if name == "cell_prior_sd"
            ));
        }
    }
}

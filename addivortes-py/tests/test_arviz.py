"""The ArviZ adapter: InferenceData shape and multi-chain diagnostics."""

import numpy as np
import pytest

az = pytest.importorskip("arviz")

from addivortes import AddiVortes
from addivortes.arviz import to_inference_data

SMALL = dict(m=8, burn_in=15, draws=25, omega=1.5)


def data(n=40, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(size=(n, 3))
    return x, 4.0 * x[:, 0] - 2.0 * x[:, 1] + rng.normal(scale=0.2, size=n)


def test_single_chain_inference_data():
    x, y = data()
    model = AddiVortes(seed=1, **SMALL).fit(x, y)
    idata = to_inference_data(model, y=y)
    assert idata.posterior.sigma.shape == (1, SMALL["draws"])
    assert idata.posterior.total_cells.shape == (1, SMALL["draws"])
    assert np.array_equal(idata.observed_data.y.values, y)


def test_multi_chain_inference_data_feeds_arviz_diagnostics():
    x, y = data(seed=2)
    chains = AddiVortes(seed=2, **SMALL).fit_chains(x, y, 3)
    idata = to_inference_data(chains)
    assert idata.posterior.sigma.shape == (3, SMALL["draws"])
    summary = az.summary(idata, var_names=["sigma"])
    # arviz 1.x formats summary cells as strings; float() covers both.
    assert np.isfinite(float(summary["r_hat"].values[0]))


def test_mismatched_chains_are_rejected():
    x, y = data(seed=3)
    a = AddiVortes(seed=3, **SMALL).fit(x, y)
    b = AddiVortes(seed=3, m=8, burn_in=15, draws=10, omega=1.5).fit(x, y)
    with pytest.raises(ValueError, match="draw counts"):
        to_inference_data([a, b])
    with pytest.raises(ValueError, match="at least one"):
        to_inference_data([])


def test_full_groups_with_x_and_y():
    x, y = data()
    chains = AddiVortes(seed=4, **SMALL).fit_chains(x, y, 2)
    idata = to_inference_data(chains, X=x, y=y, feature_names=["a", "b", "c"])
    n, d = len(y), SMALL["draws"]
    assert idata.posterior.mu.shape == (2, d, n)
    assert idata.posterior_predictive.y.shape == (2, d, n)
    assert idata.log_likelihood.y.shape == (2, d, n)
    assert idata.constant_data.X.shape == (n, 3)
    assert list(idata.constant_data.feature.values) == ["a", "b", "c"]
    assert np.array_equal(idata.observed_data.y.values, y)
    # mu's per-draw mean is the model's predict.
    assert np.allclose(
        idata.posterior.mu.values.mean(axis=(0, 1)),
        np.mean([c.predict(x) for c in chains], axis=0),
    )


def _elpd(result):
    """The elpd point estimate across ArviZ generations: 0.x ELPDData
    carries `elpd_loo` / `elpd_waic`; the 1.x (arviz-stats) ELPDData
    carries `elpd`."""
    for attr in ("elpd_loo", "elpd_waic", "elpd"):
        value = getattr(result, attr, None)
        if value is not None:
            return float(value)
    raise AssertionError(f"no elpd estimate found on {type(result)!r}")


def test_loo_and_waic_run_on_the_export():
    x, y = data(seed=4)
    chains = AddiVortes(seed=5, **SMALL).fit_chains(x, y, 2)
    idata = to_inference_data(chains, X=x, y=y)
    assert np.isfinite(_elpd(az.loo(idata)))
    # waic exists on 0.x; the 1.x line may drop it. The loo check above
    # is the primary assertion either way.
    if hasattr(az, "waic"):
        assert np.isfinite(_elpd(az.waic(idata)))


def test_plot_ppc_consumes_the_export():
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not hasattr(az, "plot_ppc"):
        pytest.skip(f"arviz {az.__version__} has no plot_ppc (moved to arviz-plots)")
    x, y = data(seed=5)
    model = AddiVortes(seed=6, **SMALL).fit(x, y)
    idata = to_inference_data(model, X=x, y=y)
    ax = az.plot_ppc(idata, num_pp_samples=10)
    assert ax is not None
    plt.close("all")


def test_probit_predictive_replicates_are_labels():
    x, y = data(seed=6)
    labels = (y > np.median(y)).astype(np.float64)
    model = AddiVortes(seed=7, **SMALL, response_family="binary_probit").fit(
        x, labels
    )
    idata = to_inference_data(model, X=x, y=labels)
    reps = idata.posterior_predictive.y.values
    assert set(np.unique(reps)) <= {0.0, 1.0}
    assert np.isfinite(_elpd(az.loo(idata)))


def test_posterior_predictive_sampling_is_seeded():
    x, y = data(seed=7)
    model = AddiVortes(seed=8, **SMALL).fit(x, y)
    a = to_inference_data(model, X=x, y=y, random_seed=3)
    b = to_inference_data(model, X=x, y=y, random_seed=3)
    c = to_inference_data(model, X=x, y=y, random_seed=4)
    assert np.array_equal(
        a.posterior_predictive.y.values, b.posterior_predictive.y.values
    )
    assert not np.array_equal(
        a.posterior_predictive.y.values, c.posterior_predictive.y.values
    )


def test_export_validation_errors():
    x, y = data(seed=8)
    model = AddiVortes(seed=9, **SMALL).fit(x, y)
    with pytest.raises(ValueError, match="rows"):
        to_inference_data(model, X=x, y=y[:-1])
    with pytest.raises(ValueError, match="feature_names"):
        to_inference_data(model, X=x, y=y, feature_names=["only_one"])
    robust = AddiVortes(seed=9, **SMALL, response_family="robust_t", t_df=4.0).fit(
        x, y
    )
    with pytest.raises(ValueError, match="response family"):
        to_inference_data([model, robust])

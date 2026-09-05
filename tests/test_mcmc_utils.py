import numpy as np

from quant_trafego.mcmc import _moment_match_beta


def test_moment_match_beta_preserves_mean():
    rng = np.random.default_rng(10)
    samples = rng.beta(30, 70, size=5000)
    post = _moment_match_beta(samples)
    assert abs(post.mean - samples.mean()) < 0.01
    assert post.alpha > 0
    assert post.beta > 0

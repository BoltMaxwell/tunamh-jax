import jax
import jax.numpy as jnp
import numpy as np

from tunamh_jax import logistic


def test_log_likelihood_matches_the_naive_formula():
    """In the range where the reference's sigmoid-then-log is accurate, agree with it."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 5))
    theta = rng.normal(size=5)
    y = rng.integers(0, 2, size=64)

    z = x @ theta
    predict = 1.0 / (1.0 + np.exp(-z))
    expected = y * np.log(predict) + (1 - y) * np.log1p(-predict)

    got = logistic.log_likelihood_terms(jnp.asarray(x), jnp.asarray(2 * y - 1), jnp.asarray(theta), 1.0)
    np.testing.assert_allclose(np.asarray(got), expected, atol=1e-12)


def test_log_likelihood_survives_where_the_naive_formula_underflows():
    """The reference returns -inf past |z| ~ 37; log_sigmoid stays finite."""
    x = jnp.asarray([[200.0], [-200.0]])
    y_sign = jnp.asarray([-1.0, 1.0])
    got = logistic.log_likelihood_terms(x, y_sign, jnp.asarray([1.0]), 1.0)
    np.testing.assert_allclose(np.asarray(got), [-200.0, -200.0], rtol=1e-12)


def test_temperature_scales_the_log_likelihood():
    rng = np.random.default_rng(2)
    x, theta = rng.normal(size=(8, 3)), rng.normal(size=3)
    signs = jnp.asarray(rng.choice([-1.0, 1.0], size=8))
    hot = logistic.log_likelihood(jnp.asarray(x), signs, jnp.asarray(theta), 2.5)
    cold = logistic.log_likelihood(jnp.asarray(x), signs, jnp.asarray(theta), 1.0)
    np.testing.assert_allclose(float(hot) * 2.5, float(cold), rtol=1e-12)


def test_lipschitz_constants_bound_the_energy_difference():
    """Assumption 1: |U_i(θ) - U_i(θ')| <= c_i ||θ - θ'||. Everything relies on this."""
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.normal(size=(200, 6)))
    signs = jnp.asarray(rng.choice([-1.0, 1.0], size=200))
    temperature = 0.7
    c = logistic.lipschitz_constants(x, temperature)

    for trial in range(20):
        sub = np.random.default_rng(trial)
        theta = jnp.asarray(sub.normal(size=6))
        theta_prime = jnp.asarray(theta + sub.normal(size=6) * 0.5)
        delta = logistic.log_likelihood_terms(
            x, signs, theta, temperature
        ) - logistic.log_likelihood_terms(x, signs, theta_prime, temperature)
        bound = c * jnp.linalg.norm(theta - theta_prime)
        assert bool(jnp.all(jnp.abs(delta) <= bound + 1e-9))


def test_log1mexp():
    """Checked against a 60-digit Decimal reference, because the two obvious
    float formulas -- ``log1p(-exp(-x))`` and ``log(-expm1(-x))`` -- are each
    inaccurate on one side of ``log 2``, which is why log1mexp is piecewise."""
    from decimal import Decimal, getcontext

    getcontext().prec = 60
    x = np.array([1e-12, 1e-6, 0.1, 0.6931471805599453, 1.0, 5.0, 40.0, 800.0])
    reference = np.array([float((1 - (-Decimal(float(v))).exp()).ln()) for v in x])
    got = np.asarray(logistic.log1mexp(jnp.asarray(x)))
    np.testing.assert_allclose(got, reference, rtol=1e-14)
    assert float(logistic.log1mexp(jnp.asarray(0.0))) == -np.inf


def test_kaiming_init_respects_its_bound():
    theta = logistic.kaiming_uniform_init(jax.random.key(0), 50)
    bound = np.sqrt(3.0) * np.sqrt(2.0 / 6.0) / np.sqrt(50)
    assert theta.shape == (50,)
    assert float(jnp.max(jnp.abs(theta))) <= bound


def test_accuracy_uses_the_reference_tie_break():
    """predict > 0 votes class 1; predict == 0 votes class 0."""
    test_x = jnp.asarray([[1.0], [-1.0], [0.0]])
    test_y = jnp.asarray([1, 0, 0])
    assert float(logistic.accuracy(jnp.asarray([1.0]), test_x, test_y)) == 1.0
    assert float(logistic.accuracy(jnp.asarray([0.0]), test_x, test_y)) == 2 / 3

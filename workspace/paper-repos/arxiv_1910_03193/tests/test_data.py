import numpy as np

from deeponet_repro.data import grf_samples, make_dataset, solve


def test_grf_shape_and_correlation():
    x, U = grf_samples(200, 0.2, 101, np.random.default_rng(0))
    assert U.shape == (200, 101)
    # neighbouring points are highly correlated, distant points much less
    c_near = np.corrcoef(U[:, 50], U[:, 51])[0, 1]
    c_far = np.corrcoef(U[:, 0], U[:, 100])[0, 1]
    assert c_near > 0.95 and abs(c_far) < 0.5


def test_antiderivative_of_linear_function():
    x = np.linspace(0, 1, 501)
    s = solve("antiderivative", x, 2 * x)
    assert np.allclose(s, x**2, atol=1e-5)


def test_nonlinear_ode_matches_tanh_solution():
    # ds/dx = 1 - s^2, s(0)=0  ->  s = tanh(x)
    x = np.linspace(0, 1, 201)
    s = solve("nonlinear_ode", x, np.ones_like(x))
    assert np.allclose(s, np.tanh(x), atol=1e-6)


def test_pendulum_zero_forcing_stays_at_rest():
    x = np.linspace(0, 1, 101)
    assert np.allclose(solve("pendulum", x, np.zeros_like(x)), 0)


def test_dataset_shapes():
    d = make_dataset("antiderivative", 5, 3, m=10, n_grid=51)
    assert d.u_sensors.shape == (15, 10) and d.y.shape == (15, 1) and d.s.shape == (15, 1)

# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["test_trivial_solver", "test_basic_solver", "test_hodlr_solver"]

import numpy as np

import george
from george.utils import nd_sort_samples
from george import kernels, modeling
from george import TrivialSolver, BasicSolver, HODLRSolver


def test_trivial_solver(N=300, seed=1234):
    # Sample some data.
    np.random.seed(seed)
    x = np.random.randn(N, 3)
    yerr = 1e-3 * np.ones(N)
    y = np.sin(np.sum(x, axis=1))

    solver = TrivialSolver()
    solver.compute(x, yerr)

    assert np.allclose(solver.log_determinant, 2*np.sum(np.log(yerr)))
    assert np.allclose(solver.apply_inverse(y), y / yerr**2)


def _test_solver(Solver, N=300, seed=1234, **kwargs):
    # Set up the solver.
    kernel = 1.0 * kernels.ExpSquaredKernel(1.0)
    solver = Solver(kernel, **kwargs)

    # Sample some data.
    np.random.seed(seed)
    x = np.atleast_2d(np.sort(10*np.random.randn(N))).T
    yerr = 0.1 * np.ones(N)
    solver.compute(x, yerr)

    # Build the matrix.
    K = kernel.get_value(x)
    K[np.diag_indices_from(K)] += yerr ** 2

    # Check the determinant.
    sgn, lndet = np.linalg.slogdet(K)
    assert sgn == 1.0, "Invalid determinant"
    assert np.allclose(solver.log_determinant, lndet), "Incorrect determinant"

    y = np.sin(x[:, 0])
    b0 = np.linalg.solve(K, y)
    b = solver.apply_inverse(y).flatten()
    assert np.allclose(b, b0)

    # Check the inverse.
    assert np.allclose(solver.apply_inverse(K), np.eye(N)), "Incorrect inverse"


def test_basic_solver(**kwargs):
    _test_solver(BasicSolver, **kwargs)


def test_hodlr_solver(**kwargs):
    _test_solver(HODLRSolver, tol=1e-10, **kwargs)

def test_strange_hodlr_bug():
    np.random.seed(1234)
    x = np.sort(np.random.uniform(0, 10, 50000))
    yerr = 0.1 * np.ones_like(x)
    y = np.sin(x)

    kernel = np.var(y) * kernels.ExpSquaredKernel(1.0)

    gp_hodlr = george.GP(kernel, solver=HODLRSolver, seed=42)
    n = 200
    gp_hodlr.compute(x[:n], yerr[:n])
    gp_hodlr.log_likelihood(y[:n])

def test_model_tutorial():

    class Model(modeling.Model):
        parameter_names = ("amp", "location", "log_sigma2")

        def get_value(self, t):
            return self.amp * np.exp(-0.5*(t.flatten()-self.location)**2 *
                                     np.exp(-self.log_sigma2))

    def generate_data(params, N, rng=(-5, 5)):
        gp = george.GP(0.1 * kernels.ExpSquaredKernel(3.3))
        t = rng[0] + np.diff(rng) * np.sort(np.random.rand(N))
        y = gp.sample(t)
        y += Model(**params).get_value(t)
        yerr = 0.05 + 0.05 * np.random.rand(N)
        y += yerr * np.random.randn(N)
        return t, y, yerr

    np.random.seed(1234)
    truth = dict(amp=-1.0, location=0.1, log_sigma2=np.log(0.4))
    t, y, yerr = generate_data(truth, 50)

    params = dict([('amp', -0.9084520230247749),
                   ('location', 0.014424970078361431),
                   ('log_sigma2', -0.91748937072607739)])
    kernel = kernels.ConstantKernel(log_constant=-2.826564897326464)
    kernel *= kernels.Matern32Kernel(7.369719781508781)
    gp = george.GP(kernel, mean=Model(**params), solver=george.HODLRSolver)
    gp0 = george.GP(kernel, mean=Model(**params))

    p = np.array([-0.89532215,  0.04577885, -1.32793447, -1.32874009,  3.39837512])
    gp.set_parameter_vector(p)
    gp0.set_parameter_vector(p)

    gp.compute(t, yerr)
    gp0.compute(t, yerr)
    x = np.linspace(-5, 5, 500)

    for _ in range(10):
        alpha0 = gp0.apply_inverse(y)
        alpha = gp.apply_inverse(y)
        assert np.allclose(alpha, alpha0)

    for _ in range(10):
        mu0, cov0 = gp0.predict(y, x)
        mu, cov = gp.predict(y, x)
        assert np.allclose(mu, mu0)
    print(gp0._alpha)
    assert 0

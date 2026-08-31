import numpy as np
import pytest

from surfStats.scaleMap.spec_utils import az_lambda, gen_cov, hanning2


def test_hanning2_shape_and_range():
    n = 8
    w = hanning2(n)
    assert w.shape == (n, n)
    assert w.min() == 0.0
    assert w[0, 0] == 0.0
    assert w[n // 2, n // 2] == w.max()


def test_hanning2_is_symmetric():
    w = hanning2(8)
    assert np.allclose(w, w[::-1, :])
    assert np.allclose(w, w[:, ::-1])
    assert np.allclose(w, w.T)


def test_az_lambda_folded_shapes_and_dc_wavelength():
    nx, ny, dx = 8, 8, 1.0
    az, L, kx, ky = az_lambda(nx, ny, dx, fold=True)
    # folded (rfft-style) grid keeps only nx//2 + 1 columns
    assert az.shape == (ny, nx // 2 + 1)
    assert L.shape == az.shape
    assert kx.shape == az.shape
    assert ky.shape == az.shape
    # the DC bin's wavelength is set explicitly, rather than from 2*pi/k
    assert L[0, 0] == pytest.approx(2 * nx * dx)


def test_az_lambda_unfolded_shape():
    nx, ny, dx = 8, 8, 1.0
    az, L, kx, ky = az_lambda(nx, ny, dx, fold=False)
    assert az.shape == (ny, nx)
    assert L.shape == (ny, nx)


def test_gen_cov_pure_x_variance():
    # points spread only along x (y all zero, uniform weight): all the
    # variance should land in covxx, with no x/y coupling.
    x = np.array([1.0, -1.0, 2.0, -2.0])
    y = np.zeros_like(x)
    W = np.ones_like(x)
    C = gen_cov(W, x, y)
    assert C.shape == (2, 2)
    assert C[0, 1] == pytest.approx(0.0)
    assert C[1, 0] == pytest.approx(0.0)
    assert C[1, 1] == pytest.approx(0.0)
    assert C[0, 0] == pytest.approx(2.5)


def test_gen_cov_explicit_sums_match_defaults():
    x = np.array([1.0, -1.0, 2.0, -2.0])
    y = np.array([0.5, -0.5, 1.0, -1.0])
    W = np.array([1.0, 2.0, 1.0, 2.0])
    C_default = gen_cov(W, x, y)
    C_explicit = gen_cov(
        W, x, y,
        xbar=np.sum(W * x) / np.sum(W),
        ybar=np.sum(W * y) / np.sum(W),
        sumW=np.sum(W),
        sumW2=np.sum(W * W),
    )
    assert np.allclose(C_default, C_explicit)

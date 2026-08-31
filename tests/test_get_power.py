import numpy as np
import pytest

from surfStats.scaleMap.get_power import get_power
from surfStats.scaleMap.spec_utils import az_lambda, hanning2
from surfStats.scale_map_fft2 import get_P_wrapper

try:
    import pyfftw  # noqa: F401
    HAVE_PYFFTW = True
except ImportError:
    HAVE_PYFFTW = False

N = 8


def _make_bins():
    az, L, kx, ky = az_lambda(N, N, 1, fold=True)
    scales = [2.0, 4.0]
    L_bins = [
        np.ravel_multi_index(np.nonzero((L >= s) & (L < 2 * s)), L.shape)
        for s in scales
    ]
    W = hanning2(N)
    Wsum = np.sum(W.ravel())
    Wsum2 = np.sum(W.ravel() ** 2)
    return W, L_bins, kx.ravel(), ky.ravel(), Wsum, Wsum2


def test_get_power_constant_image_gives_nan_power():
    W, L_bins, kx, ky, Wsum, Wsum2 = _make_bins()
    img = np.full((N, N), 3.0)
    P, az, R, bar, fft_time = get_power(img, W, L_bins, kx, ky, Wsum=Wsum, Wsum2=Wsum2)
    # a flat image carries no spectral power in any scale bin
    assert np.all(np.isnan(P))
    assert np.all(np.isnan(R))
    # bar (the weighted mean) recovers the constant value
    assert bar == pytest.approx(3.0)
    assert fft_time >= 0


def test_get_power_mean_matches_sum_divided_by_bin_size():
    W, L_bins, kx, ky, Wsum, Wsum2 = _make_bins()
    rng = np.random.default_rng(0)
    img = rng.normal(size=(N, N))
    P_sum, *_ = get_power(img, W, L_bins, kx, ky, Wsum=Wsum, Wsum2=Wsum2, use_mean=False)
    P_mean, *_ = get_power(img, W, L_bins, kx, ky, Wsum=Wsum, Wsum2=Wsum2, use_mean=True)
    bin_sizes = np.array([len(idx) for idx in L_bins]).reshape(-1, 1)
    assert np.allclose(P_sum / bin_sizes, P_mean)


@pytest.mark.parametrize(
    "use_fftw",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skipif(not HAVE_PYFFTW, reason="pyfftw not installed"),
        ),
    ],
)
def test_get_power_backends_agree(use_fftw):
    # the non-fftw path used to call pyfftw.rfftn, which does not exist
    W, L_bins, kx, ky, Wsum, Wsum2 = _make_bins()
    rng = np.random.default_rng(4)
    img = rng.normal(size=(N, N))
    P, az, R, bar, _ = get_power(
        img, W, L_bins, kx, ky, use_fftw=use_fftw, Wsum=Wsum, Wsum2=Wsum2
    )
    P_ref, _, _, bar_ref, _ = get_power(
        img, W, L_bins, kx, ky, use_fftw=False, Wsum=Wsum, Wsum2=Wsum2
    )
    assert np.allclose(P, P_ref)
    assert bar == pytest.approx(bar_ref)


def test_get_power_single_band_stack_matches_plain_2d_image():
    # a (1, N, N) stack must be treated as one image, not FFT'd over 3 axes
    W, L_bins, kx, ky, Wsum, Wsum2 = _make_bins()
    rng = np.random.default_rng(5)
    img = rng.normal(size=(N, N))
    P_2d, az_2d, R_2d, bar_2d, _ = get_power(img, W, L_bins, kx, ky, Wsum=Wsum, Wsum2=Wsum2)
    P_3d, az_3d, R_3d, bar_3d, _ = get_power(
        img[np.newaxis, :, :], W, L_bins, kx, ky, Wsum=Wsum, Wsum2=Wsum2
    )
    assert np.allclose(P_2d, P_3d)
    assert np.allclose(az_2d, az_3d)
    assert np.allclose(R_2d, R_3d)
    assert bar_2d == pytest.approx(bar_3d)


def test_get_power_output_shapes():
    W, L_bins, kx, ky, Wsum, Wsum2 = _make_bins()
    rng = np.random.default_rng(1)
    img = rng.normal(size=(N, N))
    P, az, R, bar, fft_time = get_power(img, W, L_bins, kx, ky, Wsum=Wsum, Wsum2=Wsum2)
    n_scales = len(L_bins)
    assert P.shape == (n_scales, 1)
    assert az.shape == (n_scales, 1)
    assert R.shape == (n_scales, 1)


def test_get_power_isotropic_case_accepts_xy_gradient_stack():
    W, L_bins, kx, ky, Wsum, Wsum2 = _make_bins()
    rng = np.random.default_rng(2)
    gx = rng.normal(size=(N, N))
    gy = 0.5 * gx
    img = np.stack([gx, gy])
    P, az, R, bar, fft_time = get_power(img, W, L_bins, kx, ky, Wsum=Wsum, Wsum2=Wsum2)
    assert P.shape == (len(L_bins), 1)
    assert np.all(np.isfinite(P))


def test_get_P_wrapper_matches_get_power_and_passes_through_indices():
    W, L_bins, kx, ky, Wsum, Wsum2 = _make_bins()
    rng = np.random.default_rng(3)
    img = rng.normal(size=(N, N))
    direct_P, direct_az, direct_R, direct_bar, _ = get_power(
        img, W, L_bins, kx, ky, use_fftw=False, Wsum=Wsum, Wsum2=Wsum2, use_mean=False
    )

    arg_list = [img, W, L_bins, kx, ky, False, Wsum, Wsum2, False, 3, 7]
    r_out, c_out, P, az, R, bar, fft_time = get_P_wrapper(arg_list)

    assert (r_out, c_out) == (3, 7)
    assert np.allclose(P, direct_P)
    assert np.allclose(az, direct_az)
    assert np.allclose(R, direct_R)
    assert bar == pytest.approx(direct_bar)

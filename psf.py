#!/usr/bin/env python
# coding: utf-8
"""
The EnMAP point-spread function (PSF), and Bayesian deconvolution of the
radiance field in the Fourier domain.

WHY. Every per-pixel retrieval (this work's ProfileML, Dozier 1981,
Dennison et al. 2006, Amici et al. 2022) models the spectrum recorded at
pixel p as the radiance of ground pixel p alone. The instrument records
the scene convolved with its PSF: with EnMAP's MTF requirement a fire
leaves on average 39% of its radiance in its own pixel and 10-13% in each
edge neighbour (VERIFICATION.md Section 25.1). On synthetic truth this
biases the per-pixel retrieval by +42 K, its intervals cover 27%, and it
reports fire in most fire-free neighbours of a fire (Section 25.3).

PSF MODEL. A Gaussian optics PSF (sigma from the MTF requirement at
Nyquist, mtf_to_sigma) integrated over the pixel aperture. A point fire at
sub-pixel offset u has the kernel psf_point_weights(u); a fire at an
unknown, uniformly distributed position has the position-averaged kernel
psf_kernel() - the operator used here - and the position variance
psf_kernel_variance() quantifies the error of using the average.

INVERSION. For each band, on a window of pixels around a fire,

    y = M (H x + n),   n ~ N(0, R),   x ~ N(m, s^2 I)                     (1)

H: convolution with the position-averaged kernel on a grid padded by an
unrecorded margin (whose radiance is estimated too, so radiance blurred in
from outside the window is accounted for); M: selects the recorded, usable
samples (saturated and flagged samples are left out); R: diagonal noise
variance; m: the window median (the background level - a fire window's mean
is pulled up by the fire). The posterior mean is

    x_hat = m + s^2 H^T M^T v,   (s^2 M H H^T M^T + R) v = y - M H m       (2)

(the data-space form of the Gaussian posterior). H and H H^T are
diagonalised by the discrete Fourier transform, so (2) is solved by
conjugate gradients in which each iteration is one FFT pair, with a Jacobi
preconditioner for the strongly varying noise (fire samples carry ~30x the
noise of background ones); for complete data and constant noise (2) is the
Wiener filter. All bands of a window are solved as one batch. (The
equivalent parameter-space normal equations, H^T W H + I/s^2, converge
hundreds of times more slowly here.)

PRIOR (empirical Bayes). White: the pixels are independent a priori with
variance s^2 about m - the natural prior for sparse, point-like fires (a
spatially correlated prior estimated from the window's own periodogram was
tried and rejected: in a small window dominated by one fire it couples the
fire to the unrecorded margin and loses ~6% of the fire's radiance to edge
artefacts; an uninformative mean level was rejected for the same reason).
s^2 per band is the type-II maximum-likelihood (marginal-likelihood)
estimate under the complete-data model, Y(k) ~ N(0, |K(k)|^2 s^2 + v N).
The blur only moves radiance between pixels, so for a fire inside the
window the deconvolved total equals the recorded total (test_psf.py).

UNCERTAINTY. Exact posterior draws by Matheron's rule (Hoffman & Ribak
1991; the "perturb-and-MAP" of Papandreou & Yuille 2010): with x0 a prior
draw and e a noise draw, x0 + s^2 H^T M^T S^-1 (y + e - M H x0) is
distributed as the posterior; the per-sample standard deviation is the
spread of such draws. Where a recorded sample is missing (a saturated core
band) the posterior falls back toward the prior and the standard deviation
says so - such samples are reported as not data-determined (`determined`).

KERNEL ERROR. The real fire sits at one position, not the average: the
recorded field differs from H x by sum_q dk_q(p - q) x_q, with variance
sum_q v(p - q) x_q^2 (v = position variance of the kernel). It is added to
the noise variance from a first pass (`kernel_variance=`, two passes).

LIMITS. The prior is Gaussian and stationary: no positivity, mild ringing
around the brightest sources, and a saturated core's missing bands are
estimated only as well as its neighbours' spill determines them. Results
on synthetic truth: VERIFICATION.md Section 28.

All FFTs use scipy.fft over every CPU core (workers=-1) unless told
otherwise; deconvolve_cube() processes fire windows in parallel.
"""

import numpy as np
from scipy import fft as _fft
from scipy.special import ndtr

__all__ = ['ENMAP_MTF_NYQUIST', 'mtf_to_sigma', 'default_sigmas', 'aperture_weights',
           'psf_point_weights', 'psf_kernel', 'psf_kernel_variance', 'kernel_otf',
           'estimate_prior_variance', 'bayes_deconvolve', 'deconvolve_cube']

ENMAP_MTF_NYQUIST = {'across': 0.25, 'along': 0.16}     # mission requirement (eoPortal; > these values)


# --------------------------------------------------------------------------
# PSF model
# --------------------------------------------------------------------------

def mtf_to_sigma(mtf_nyquist):
    """Optics Gaussian sigma (pixels) such that Gaussian x pixel-aperture MTF
    at Nyquist (0.5 cycles/pixel) equals `mtf_nyquist`:
    MTF = sinc(0.5) exp(-pi^2 sigma^2 / 2), sinc(0.5) = 2/pi."""
    r = float(mtf_nyquist) / (2.0 / np.pi)
    if not 0 < r < 1:
        raise ValueError(f'MTF {mtf_nyquist} not reachable by a Gaussian + aperture model')
    return float(np.sqrt(-2.0 * np.log(r)) / np.pi)


def default_sigmas(sigma_along=None, sigma_across=None):
    """(sigma_along, sigma_across) in pixels; None -> the EnMAP MTF requirement."""
    sa = mtf_to_sigma(ENMAP_MTF_NYQUIST['along']) if sigma_along is None else float(sigma_along)
    sc = mtf_to_sigma(ENMAP_MTF_NYQUIST['across']) if sigma_across is None else float(sigma_across)
    return sa, sc


def aperture_weights(sigma, offset, radius):
    """Share of a 1-D Gaussian point source at `offset` (pixels from the centre
    of pixel 0) recorded by pixels -radius..radius."""
    d = np.arange(-radius, radius + 1, dtype=float)
    return ndtr((d + 0.5 - offset) / sigma) - ndtr((d - 0.5 - offset) / sigma)


def psf_point_weights(sigma_along, sigma_across, u_along=0.0, u_across=0.0, radius=2):
    """Share of a point source at sub-pixel offset (u_along, u_across) from the
    centre of pixel (0, 0) recorded by pixel (i, j); i along track (rows), j
    across track (columns). Shape (2r+1, 2r+1)."""
    return np.outer(aperture_weights(sigma_along, u_along, radius),
                    aperture_weights(sigma_across, u_across, radius))


def _position_axis(sigma, radius, n_sub):
    """1-D weights of a point at n_sub uniformly spaced positions in the pixel."""
    u = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    return np.array([aperture_weights(sigma, x, radius) for x in u])


def psf_kernel(sigma_along=None, sigma_across=None, radius=2, n_sub=32):
    """PSF of a fire at a uniformly distributed position inside its pixel:
    psf_point_weights averaged over the pixel, normalised to sum 1.
    Defaults: the EnMAP MTF requirement."""
    sa, sc = default_sigmas(sigma_along, sigma_across)
    k = np.outer(_position_axis(sa, radius, n_sub).mean(0), _position_axis(sc, radius, n_sub).mean(0))
    return k / k.sum()


def psf_kernel_variance(sigma_along=None, sigma_across=None, radius=2, n_sub=32):
    """Variance, over a uniform sub-pixel position, of each kernel weight:
    Var_u[k_u(d)] for the separable point kernel k_u = a(u_a) x c(u_c)
    (E[a^2]E[c^2] - E[a]^2 E[c]^2). Shape (2r+1, 2r+1)."""
    sa, sc = default_sigmas(sigma_along, sigma_across)
    wa, wc = _position_axis(sa, radius, n_sub), _position_axis(sc, radius, n_sub)
    s = wa.mean(0).sum() * wc.mean(0).sum()          # the normalisation of psf_kernel()
    return (np.outer((wa ** 2).mean(0), (wc ** 2).mean(0)) - np.outer(wa.mean(0), wc.mean(0)) ** 2) / s ** 2


def kernel_otf(kernel, shape):
    """Eigenvalues (rfft2) of circular convolution with `kernel` (centred,
    odd size) on a grid of `shape`."""
    kh, kw = kernel.shape
    full = np.zeros(shape)
    full[:kh, :kw] = kernel
    full = np.roll(full, (-(kh // 2), -(kw // 2)), axis=(0, 1))
    return _fft.rfft2(full)


# --------------------------------------------------------------------------
# Fourier-domain Bayesian inversion
# --------------------------------------------------------------------------

def _full_spectrum_weights(n_cols_rfft, n_cols):
    """Weights that count every coefficient of the full 2-D spectrum once
    from an rfft2 array (interior columns stand for two)."""
    w = np.full(n_cols_rfft, 2.0)
    w[0] = 1.0
    if n_cols % 2 == 0:
        w[-1] = 1.0
    return w


def estimate_prior_variance(y, sigma2, valid, kernel, n_grid=241):
    """Type-II maximum-likelihood variance s^2 of the white prior, per band.

    y, sigma2, valid: (B, R, C) window data, noise variance, valid mask.
    s^2 maximises the marginal likelihood of the window's Fourier
    coefficients under the complete-data model (invalid samples filled with
    the median), Y(k) ~ N(0, |K(k)|^2 s^2 + v N) for k != 0:
        L(s^2) = - sum_k [ log(|K|^2 s^2 + v) + |Y|^2 / N / (|K|^2 s^2 + v) ]
    on a log grid of s^2 / v in [1e-4, 1e8], v = mean noise variance."""
    B, R, C = y.shape
    med = np.nan_to_num(np.nanmedian(np.where(valid, y, np.nan), axis=(1, 2)))
    z = np.where(valid, y, med[:, None, None]) - med[:, None, None]
    pw = np.abs(_fft.rfft2(z, axes=(1, 2), workers=-1)) ** 2 / (R * C)
    K2 = np.abs(kernel_otf(kernel, (R, C))) ** 2
    v = np.where(valid, sigma2, 0).sum((1, 2)) / np.maximum(valid.sum((1, 2)), 1)
    wts = np.broadcast_to(_full_spectrum_weights(K2.shape[1], C), K2.shape).copy()
    wts[0, 0] = 0.0                                                    # DC excluded
    ratio = np.logspace(-4, 8, n_grid)
    s2 = np.empty(B)
    for b in range(B):
        tot = K2[None] * (ratio[:, None, None] * v[b]) + v[b]           # (n_grid, R, C//2+1)
        ll = -np.sum(wts * (np.log(tot) + pw[b][None] / tot), axis=(1, 2))
        s2[b] = ratio[int(np.argmax(ll))] * v[b]
    return s2


def _pcg(apply_A, rhs, precond, tol, maxiter):
    """Preconditioned conjugate gradients batched over the leading axis
    (per-batch scalars). Returns (x, iterations, relative residuals)."""
    ax = tuple(range(1, rhs.ndim))
    shape = (-1,) + (1,) * (rhs.ndim - 1)
    x = np.zeros_like(rhs)
    r = rhs.copy()
    z = precond(r)
    p = z.copy()
    rz = np.sum(r * z, axis=ax)
    b_norm = np.sqrt(np.sum(rhs * rhs, axis=ax)) + 1e-300
    res = np.ones(len(rhs))
    n_it = 0
    for n_it in range(1, maxiter + 1):
        Ap = apply_A(p)
        pAp = np.sum(p * Ap, axis=ax)
        alpha = rz / np.where(pAp != 0, pAp, 1.0)
        x += alpha.reshape(shape) * p
        r -= alpha.reshape(shape) * Ap
        res = np.sqrt(np.sum(r * r, axis=ax)) / b_norm
        if np.all(res < tol):
            break
        z = precond(r)
        rz_new = np.sum(r * z, axis=ax)
        p = z + (rz_new / np.where(rz != 0, rz, 1.0)).reshape(shape) * p
        rz = rz_new
    return x, n_it, res


def bayes_deconvolve(y, sigma, valid, kernel=None, pad=None, n_samples=24, prior_var=None,
                     kernel_variance=None, tol=1e-8, maxiter=500, seed=0, workers=-1):
    """Posterior of the PSF-deconvolved radiance on one window, every band at once.

    y, sigma : (B, R, C) recorded radiance and its noise standard deviation.
    valid    : (B, R, C) or (R, C) bool, samples that are recorded and usable
               (False: saturated, flagged; they are left out of the data).
    kernel   : the PSF on the pixel grid (default psf_kernel()).
    pad      : rows/columns of unrecorded margin, whose radiance is also
               estimated (default 3 x kernel radius; it must be >= 2 x radius
               so the periodic FFT never wraps).
    prior_var: (B,) white-prior variances (default estimate_prior_variance()).
    kernel_variance : psf_kernel_variance() to add the sub-pixel position
               error to the noise from a first pass (two passes), or None.

    Returns dict: 'mean' (B, R, C) posterior mean, 'sd' posterior standard
    deviation (from n_samples exact posterior draws; None if n_samples = 0),
    'sd_data' (B,) the standard deviation of a complete window under the same
    prior (the data-determined reference), 'determined' = sd <= 3 sd_data,
    'prior_var', 'iterations', 'residual'."""
    kernel = psf_kernel() if kernel is None else np.asarray(kernel, float)
    y = np.asarray(y, float)
    B, R, C = y.shape
    sigma = np.broadcast_to(np.asarray(sigma, float), y.shape)
    valid = np.broadcast_to(np.asarray(valid, bool), y.shape) & np.isfinite(y) & (sigma > 0)
    kr = kernel.shape[0] // 2
    pad = 3 * kr if pad is None else int(pad)
    if pad < 2 * kr:
        raise ValueError(f'pad must be >= 2 x the kernel radius ({2 * kr})')
    shape = (_fft.next_fast_len(R + 2 * pad, real=True), _fft.next_fast_len(C + 2 * pad, real=True))
    sl = (slice(None), slice(pad, pad + R), slice(pad, pad + C))
    Kf = kernel_otf(kernel, shape)
    K2 = np.abs(Kf) ** 2
    var = np.where(valid, sigma, 1.0) ** 2
    if prior_var is None:
        prior_var = estimate_prior_variance(y, var, valid, kernel)
    s2 = np.asarray(prior_var, float).reshape(B)
    m = np.nan_to_num(np.nanmedian(np.where(valid, y, np.nan), axis=(1, 2)))
    obs = valid.astype(float)
    k2_sum = float(np.sum(kernel ** 2))

    def rf(a):
        return _fft.rfft2(a, axes=(1, 2), workers=workers)

    def irf(a):
        return _fft.irfft2(a, s=shape, axes=(1, 2), workers=workers)

    def embed(a):
        out = np.zeros((B,) + shape)
        out[sl] = a
        return out

    def solve(noise_var, rhs):
        # S v = rhs over the recorded samples, S = s^2 M H H^T M + diag(noise);
        # H H^T is the convolution with the kernel autocorrelation (|K|^2)
        def apply_S(v):
            hht = irf(rf(embed(v)) * K2)[sl]
            return obs * (s2[:, None, None] * hht) + noise_var * v
        diag = s2[:, None, None] * k2_sum + noise_var

        def precond(r):
            return obs * r / diag
        return _pcg(apply_S, obs * rhs, precond, tol, maxiter)

    def back(v):
        # x - m = s^2 H^T v, on the padded grid
        return s2[:, None, None] * irf(rf(embed(obs * v)) * np.conj(Kf))

    v, n_it, res = solve(var, y - m[:, None, None])           # H m = m: the kernel sums to 1
    x = m[:, None, None] + back(v)
    extra = 0.0
    if kernel_variance is not None:
        # sub-pixel position error of the averaged kernel: variance
        # sum_q v(p - q) (x_q - m)_+^2 of the fire excess
        excess = np.maximum(x - m[:, None, None], 0.0) ** 2
        extra = np.maximum(irf(rf(excess) * kernel_otf(kernel_variance, shape))[sl], 0.0)
        v, n_it2, res = solve(var + extra, y - m[:, None, None])
        x = m[:, None, None] + back(v)
        n_it += n_it2

    # data-determined reference: per-pixel posterior variance of a complete
    # window with the median weight, (1/N) sum_k 1/(|K|^2 w + 1/s^2)
    w_med = np.nan_to_num(np.nanmedian(np.where(valid, 1.0 / (var + extra), np.nan), axis=(1, 2)))
    post = 1.0 / (K2[None] * w_med[:, None, None] + 1.0 / s2[:, None, None])
    sd_data = np.sqrt((post * _full_spectrum_weights(K2.shape[1], shape[1])).sum((1, 2)) / (shape[0] * shape[1]))

    sd = determined = None
    if n_samples:
        # exact posterior draws (Matheron's rule): x0 ~ prior, e ~ noise,
        # x_s = x0 + s^2 H^T S^-1 (y + e - H x0); x_s - x_hat needs only the
        # perturbations
        rng = np.random.default_rng(seed)
        acc = np.zeros((B, R, C))
        for _ in range(n_samples):
            dx0 = np.sqrt(s2)[:, None, None] * rng.standard_normal((B,) + shape)
            e = rng.standard_normal((B, R, C)) * np.sqrt(var + extra)
            h_dx0 = irf(rf(dx0) * Kf)[sl]
            dv, _, _ = solve(var + extra, e - h_dx0)
            acc += (dx0 + back(dv))[sl] ** 2
        sd = np.sqrt(acc / n_samples)
        determined = sd <= 3.0 * sd_data[:, None, None]
    return {'mean': x[sl], 'sd': sd, 'sd_data': sd_data, 'determined': determined,
            'prior_var': s2, 'iterations': n_it, 'residual': res, 'pad': pad, 'grid': shape}


def deconvolve_cube(cube, sigma, valid, centres, half_window=12, kernel=None, n_jobs=-1, **kw):
    """bayes_deconvolve() on square windows around `centres` ((n, 2) row/col
    of fire clusters), in parallel over windows on every CPU core.

    cube, sigma, valid : (R, C, B) arrays (pixel-major, band last).
    Returns a list of (r0, r1, c0, c1, result) per window, result as
    bayes_deconvolve() but band-last (R_w, C_w, B)."""
    from joblib import Parallel, delayed
    R, C = cube.shape[:2]
    boxes = []
    for r, c in np.asarray(centres, int):
        r0, r1 = max(r - half_window, 0), min(r + half_window + 1, R)
        c0, c1 = max(c - half_window, 0), min(c + half_window + 1, C)
        boxes.append((r0, r1, c0, c1))

    def one(box):
        r0, r1, c0, c1 = box
        t = lambda a: np.moveaxis(np.asarray(a[r0:r1, c0:c1]), -1, 0)
        res = bayes_deconvolve(t(cube), t(sigma), t(valid), kernel=kernel, workers=1, **kw)
        for key in ('mean', 'sd', 'determined'):
            if res.get(key) is not None:
                res[key] = np.moveaxis(res[key], 0, -1)
        return box + (res,)

    if len(boxes) <= 1 or n_jobs == 1:
        return [one(b) for b in boxes]
    return Parallel(n_jobs=n_jobs)(delayed(one)(b) for b in boxes)

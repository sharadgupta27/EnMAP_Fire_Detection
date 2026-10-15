#!/usr/bin/env python
# coding: utf-8
"""
Convex multi-pixel retrieval of sub-pixel fire radiance with the instrument
point-spread function (PSF), reported as profile-likelihood BOUNDS on the
quantities the data identify.

WHY CONVEX. The parametric joint fit (psf_joint.py, VERIFICATION.md Section
25; removed 14 Oct 2026, in git abc269d) showed that the
neighbours of a saturated core carry the information to recover it
(started at the truth: 9 of 24 saturated synthetic cores at MAE 22 K) but
that the parametric joint model - one temperature, fraction and sub-pixel
position per pixel - has many local minima: from realistic starting values
the fit ends at ~3x the objective of the truth. Here every nonlinearity is
removed by discretising instead of estimating:

    y_p(l) = sum_j a_pj L_pj(l)
           + tau(l) e sum_q sum_k h_k(p - q) sum_t x_qkt B(l, T_t)            (1)

  T_t   : a fixed temperature grid (geometric, 450-2000 K by default);
  h_k   : the PSF of a point fire at one of a 3 x 3 grid of positions inside
          its pixel (psf.psf_point_weights); a fire spread over the
          pixel is a non-negative mix of them;
  x_qkt : non-negative area fraction of pixel q radiating at T_t from
          position k - a NON-PARAMETRIC, non-negative temperature
          distribution per pixel (and per sub-pixel position);
  a_pj  : background abundances, a >= 0, sum_j a_pj <= 1 (a zero 'shade').

(1) is linear in (x, a); with saturated bands as one-sided constraints the
weighted misfit

    J = sum_valid w (y - m)^2 + sum_censored w_c max(0, c - m)^2              (2)

is convex, and so is every problem below. They are solved by an interior-
point conic solver (Clarabel, through cvxpy), which is insensitive to the
near-collinearity of Planck curves at neighbouring grid temperatures that
stalls first-order and quasi-Newton methods on this problem.

WHAT IS REPORTED. The distribution x itself is not identifiable from SWIR
radiance (many distributions fit equally well; Keely et al. 2026 make the
same observation for their two-Gaussian p(T)), so it is never reported.
Reported instead, per pixel q, are BOUNDS on quantities LINEAR in x (or a
ratio of two):

    FRP_q   = sigma A sum_kt x_qkt T_t^4                fire radiative power
    P900_q  = sigma A sum_k sum_{T_t >= 900 K} x_qkt T_t^4   flaming power
    T_P,q   = sum x T^5 / sum x T^4                    power-weighted temperature

each as [min, max] over every (x, a) with J <= J* + Delta, Delta =
chi2_1(1 - alpha) x max(1, J*/n) - the profile-likelihood interval of the
functional, i.e. the 'strict bounds' construction (Backus 1970; Stark
1992). A linear functional is bounded by ONE conic problem per side
(maximise c.x subject to J <= J* + Delta); the ratio T_P by bisection on t
of max/min (N - t D) (Dinkelbach). A bound at the grid edge, or a lower FRP
bound of 0 (then T_P is undefined), is reported as OPEN: not identified.

LOCAL PROBLEMS. The bounds of pixel q use the data rows of the pixels
within Chebyshev radius r_data of q and leave FREE the fire of every pixel
whose PSF reaches those rows (radius r_data + kernel radius) and the
backgrounds of the data pixels. Dropping the other rows and freeing those
variables can only enlarge the admissible set, so the local bounds contain
the bounds of the whole-window problem: valid, possibly conservative, and
the cost per pixel does not grow with the scene.

MODEL ERROR. The kernel bank and the temperature grid are discretisations;
a relative error `rel_fire_error` x (recorded radiance above the local
background floor) is added in quadrature to the noise sigma of every band
(a fixed, data-based term, so (2) stays a quadratic).

DIFFERENCES FROM Keely et al. (2026, arXiv:2608.07580), stated because
both describe a sub-pixel temperature distribution:
  1. spatial: (1) couples pixels through the instrument PSF; Keely et al.
     fit each pixel independently;
  2. p(T): non-parametric and non-negative on a grid; Keely et al. use a
     two-component Gaussian mixture moment-matched to E[T] and sigma[T];
  3. saturation: saturated bands are one-sided constraints in (2); Keely
     et al. mask them per spectral window;
  4. optimisation: convex, solved to its global optimum by an interior-
     point method; Keely et al. minimise a non-convex loss with Adam;
  5. inference: profile-likelihood bounds on linear functionals, open where
     not identified; Keely et al. report point estimates of E[T], sigma[T];
  6. done by Keely et al. and not here: per-pixel water vapour, a real-time
     GPU implementation; their 900 K flaming/smouldering split is borrowed
     here for P900.

STATUS: experimental (VERIFICATION.md Section 26). Not used by the pipeline.
Runs in an isolated uv environment (cvxpy + clarabel); imports nothing from
the pipeline except the PSF model of psf.py.
"""

import time

import numpy as np

SIGMA_SB = 5.670374419e-8
PIXEL_AREA_M2 = 900.0
_C1 = 1e-3 * 1.191042 * 1e8          # = Fire_analysis_with_EnMAP._PLANCK_C1 (W m-2 sr-1 nm-1, lambda in um)
_C2 = 1.4387752 * 1e4                # = _PLANCK_C2 (um K)
_CHI2_1_95 = 3.841458820694124

__all__ = ['planck_nm', 'temperature_grid', 'position_kernels', 'local_bounds']


def planck_nm(wavelength_nm, T):
    """Planck radiance (W m-2 sr-1 nm-1), the same formula and constants as
    profile_retrieval._planck_nm (checked in the tests)."""
    x = np.asarray(wavelength_nm, float) * 1e-3
    return _C1 / x ** 5 / np.expm1(_C2 / (float(T) * x))


def temperature_grid(T_min=450.0, T_max=2000.0, n=12):
    return np.geomspace(T_min, T_max, n)


def position_kernels(sigma_along=None, sigma_across=None, positions=(-0.4, 0.0, 0.4), radius=2):
    """PSF of a point fire at each (u_along, u_across) in positions^2 inside its
    pixel, each normalised to sum 1: (n^2, 2r+1, 2r+1). Their non-negative
    mixes cover fires anywhere within +-0.4 px of the centre, point-like or
    spread."""
    from psf import psf_point_weights, default_sigmas
    sa, sc = default_sigmas(sigma_along, sigma_across)
    ks = []
    for ua in positions:
        for uc in positions:
            k = psf_point_weights(sa, sc, ua, uc, radius)
            ks.append(k / k.sum())
    return np.array(ks)


def _build(Y, valid, cens, lvl, libraries, sigma, wavelength, tau, fire_pixels, q, kernels, T_grid,
           r_data, emissivity, censor_weight, rel_fire_error):
    """Sparse whitened design of the local problem around q."""
    from scipy.sparse import coo_matrix
    R, C, K = Y.shape
    nk, kh, _ = kernels.shape
    kr = kh // 2
    i0, j0 = q
    member = np.array([[libraries[i][j] is not None for j in range(C)] for i in range(R)])
    data_px = [(i, j) for i in range(max(i0 - r_data, 0), min(i0 + r_data + 1, R))
               for j in range(max(j0 - r_data, 0), min(j0 + r_data + 1, C)) if member[i, j]]
    rf = r_data + kr
    fire_px = [(i, j) for i in range(max(i0 - rf, 0), min(i0 + rf + 1, R))
               for j in range(max(j0 - rf, 0), min(j0 + rf + 1, C)) if fire_pixels[i, j] and member[i, j]]
    nT = len(T_grid)
    B = np.stack([emissivity * tau * planck_nm(wavelength, t) for t in T_grid])           # (nT, K)
    s = np.sqrt(np.mean(B ** 2, axis=1))
    Bn = B / s[:, None]
    col_fire = {f: n * nk * nT for n, f in enumerate(fire_px)}
    n_fire_cols = len(fire_px) * nk * nT
    col_bkg, ncol = {}, n_fire_cols
    for p in data_px:
        col_bkg[p] = ncol
        ncol += libraries[p[0]][p[1]].shape[0]
    acc = {'v': ([], [], [], []), 'c': ([], [], [], [])}
    n_rows = {'v': 0, 'c': 0}
    for p in data_px:
        v, c = valid[p], cens[p]
        if not (v.any() or c.any()):
            continue
        L = np.asarray(libraries[p[0]][p[1]], float)
        bkg_floor = np.min(L, axis=0)
        sig2 = np.asarray(sigma[p], float) ** 2 + (rel_fire_error * np.maximum(Y[p] - bkg_floor, 0.0)) ** 2
        w = 1.0 / sig2
        w_med = float(np.median(w[v])) if v.any() else float(np.median(w))
        for kind, msk in (('v', v), ('c', c)):
            if not msk.any():
                continue
            kk = np.flatnonzero(msk)
            sw = np.sqrt(w[kk]) if kind == 'v' else np.full(len(kk), np.sqrt(censor_weight * w_med))
            rr = n_rows[kind] + np.arange(len(kk))
            R_, C_, V_, b_ = acc[kind]
            for jlib in range(L.shape[0]):
                R_.append(rr); C_.append(np.full(len(kk), col_bkg[p] + jlib)); V_.append(L[jlib, kk] * sw)
            for f in fire_px:
                di, dj = p[0] - f[0], p[1] - f[1]
                if abs(di) > kr or abs(dj) > kr:
                    continue
                for k in range(nk):
                    h = kernels[k, di + kr, dj + kr]
                    if h <= 0:
                        continue
                    base = col_fire[f] + k * nT
                    blk = h * Bn[:, kk] * sw[None, :]                       # (nT, n)
                    R_.append(np.tile(rr, nT)); C_.append(np.repeat(base + np.arange(nT), len(kk)))
                    V_.append(blk.ravel())
            b_.append((Y[p][kk] if kind == 'v' else lvl[p][kk]) * sw)
            n_rows[kind] += len(kk)
    cat = lambda x: np.concatenate(x) if x else np.zeros(0)
    mats = {}
    for kind in ('v', 'c'):
        R_, C_, V_, b_ = acc[kind]
        mats[kind] = (coo_matrix((cat(V_), (cat(R_).astype(int), cat(C_).astype(int))),
                                 shape=(n_rows[kind], ncol)).tocsr(), cat(b_))
    return dict(Av=mats['v'][0], bv=mats['v'][1], Ac=mats['c'][0], bc=mats['c'][1], ncol=ncol,
                col_fire=col_fire, col_bkg=col_bkg, data_px=data_px, fire_px=fire_px, nk=nk, nT=nT, s=s,
                T=np.asarray(T_grid, float), nv=n_rows['v'], nc=n_rows['c'])


def local_bounds(Y, valid, censored, level, libraries, sigma, wavelength, tau, fire_pixels, q,
                 kernels=None, T_grid=None, r_data=1, emissivity=1.0, censor_weight=100.0,
                 rel_fire_error=0.02, T_split=900.0, T_tol=20.0, solver='CLARABEL'):
    """95% bounds of FRP, P900 and T_P for pixel q = (i, j) from its local
    convex problem (module doc). libraries: (R, C) object array, None = not
    modelled. kernels: position_kernels() (with PSF) or np.ones((1, 1, 1))
    (no PSF; with r_data=0 a strictly per-pixel problem)."""
    import cvxpy as cp
    t_start = time.time()
    q = tuple(int(v) for v in q)
    Y = np.asarray(Y, float)
    valid = np.asarray(valid, bool)
    kernels = position_kernels() if kernels is None else np.asarray(kernels, float)
    T_grid = temperature_grid() if T_grid is None else np.asarray(T_grid, float)
    d = _build(Y, valid, np.asarray(censored, bool) & ~valid, np.nan_to_num(np.asarray(level, float)),
               libraries, sigma, np.asarray(wavelength, float), np.asarray(tau, float),
               np.asarray(fire_pixels, bool), q, kernels, T_grid, r_data, emissivity, censor_weight,
               rel_fire_error)
    out = {'q': q, 'n_rows': d['nv'], 'n_cens_rows': d['nc'], 'n_cols': d['ncol']}
    if q not in d['col_fire'] or d['nv'] == 0:
        out['status'] = 'not modelled'
        return out
    z = cp.Variable(d['ncol'], nonneg=True)
    J = cp.sum_squares(d['Av'] @ z - d['bv'])
    if d['nc']:
        J = J + cp.sum_squares(cp.pos(d['bc'] - d['Ac'] @ z))
    cons = [cp.sum(z[c0:c0 + libraries[p[0]][p[1]].shape[0]]) <= 1 for p, c0 in d['col_bkg'].items()]
    pr0 = cp.Problem(cp.Minimize(J), cons)
    pr0.solve(solver=solver)
    if pr0.status not in ('optimal', 'optimal_inaccurate'):
        out['status'] = f'fit {pr0.status}'
        return out
    J_star = float(pr0.value)
    scale = max(1.0, J_star / max(d['nv'], 1))
    Delta = _CHI2_1_95 * scale
    cvec = cp.Parameter(d['ncol'])
    prb = cp.Problem(cp.Maximize(cvec @ z), cons + [J <= J_star + Delta])
    base = d['col_fire'][q]
    nk, nT, s, T = d['nk'], d['nT'], d['s'], d['T']

    def vec(weight_T):
        c = np.zeros(d['ncol'])
        w = SIGMA_SB * PIXEL_AREA_M2 * T ** 4 * weight_T / s                 # x = z / s
        for k in range(nk):
            c[base + k * nT: base + (k + 1) * nT] = w
        return c

    def extreme(c, sense):
        cn = float(np.max(np.abs(c))) or 1.0
        cvec.value = sense * c / cn
        prb.solve(solver=solver)
        if prb.status not in ('optimal', 'optimal_inaccurate'):
            return np.nan
        return sense * float(prb.value) * cn

    out['FRP_lo'] = max(extreme(vec(np.ones(nT)), -1), 0.0)
    out['FRP_hi'] = extreme(vec(np.ones(nT)), +1)
    c9 = vec((T >= T_split).astype(float))
    out['P900_lo'] = max(extreme(c9, -1), 0.0)
    out['P900_hi'] = extreme(c9, +1)
    out.update(T_P_lo=np.nan, T_P_hi=np.nan, T_P_lo_open=True, T_P_hi_open=True)
    # T_P needs fire at q in EVERY admissible solution (otherwise the ratio is undefined)
    if np.isfinite(out['FRP_hi']) and out['FRP_lo'] > 1e-6 * out['FRP_hi']:
        g = lambda t, sense: extreme(vec(T - t), sense)            # N - t D
        lo_t, hi_t = np.log(T[0]), np.log(T[-1])
        if g(T[-1] * 0.999, +1) >= 0:
            out['T_P_hi'] = float(T[-1])
        else:
            a, b = lo_t, hi_t
            while np.exp(b) - np.exp(a) > T_tol:
                m_ = 0.5 * (a + b)
                (a, b) = (m_, b) if g(np.exp(m_), +1) >= 0 else (a, m_)
            out['T_P_hi'], out['T_P_hi_open'] = float(np.exp(b)), False
        if g(T[0] * 1.001, -1) <= 0:
            out['T_P_lo'] = float(T[0])
        else:
            a, b = lo_t, hi_t
            while np.exp(b) - np.exp(a) > T_tol:
                m_ = 0.5 * (a + b)
                (a, b) = (a, m_) if g(np.exp(m_), -1) <= 0 else (m_, b)
            out['T_P_lo'], out['T_P_lo_open'] = float(np.exp(a)), False
    out.update(status='ok', J_star=J_star, scale=scale, Delta=Delta, seconds=time.time() - t_start)
    return out

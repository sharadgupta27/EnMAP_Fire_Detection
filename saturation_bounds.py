#!/usr/bin/env python
# coding: utf-8
"""
Rigorous one-sided information from SATURATED fire pixels.

A saturated (censored) band says only that the at-sensor radiance is at
least the recorded ceiling, L_k >= c_k. For the brightest fire pixels that
is often nearly all there is (on tile t014, 128 of 757 pixels keep ~3
usable bands: ~29 censored, ~44 flagged by the L1B saturation-crosstalk
mask), so no temperature can be retrieved - but two quantities are still
bounded from below, WITHOUT assuming a single fire temperature:

1. Fire radiative power. Whatever the sub-pixel temperature distribution
   (components j with emissivity eps_j <= 1, area fractions f_j, sum <= 1),

       L_k - L_bkg,k = tau_k sum_j eps_j f_j B(lambda_k, T_j)
       FRP           = A sum_j eps_j f_j sigma T_j^4
                    >= A (L_k - L_bkg,k) / tau_k * R_k,
       R_k = min_T sigma T^4 / B(lambda_k, T),

   because every component's ratio sigma T_j^4 / B(lambda_k, T_j) is at
   least the minimum R_k. With L_k >= c_k, an upper bound on the background
   radiance and tau_k <= 1, the bound below is conservative in every
   factor:

       FRP >= FRP_min = A max_k (c_k - L_bkg,k^max) R_k        (tau_k := 1)

2. The temperature of the hottest component. With sum_j f_j eps_j <= 1,
   max_j B(lambda_k, T_j) >= sum_j eps_j f_j B(lambda_k, T_j) >= c_k - L_bkg,k,
   so the hottest component is at least the brightness temperature of the
   background-subtracted ceiling:

       T_hot >= T_b,min = max_k B^-1(lambda_k, c_k - L_bkg,k^max).

Both use only radiometric calibration, emissivity <= 1, the transmittance
<= 1 and an upper bound on the background radiance in the band
(L_bkg^max: here the maximum over the pixel's background library and the
scene background pool's 99.9th percentile - an assumption that the pixel's
background is not brighter than that, stated in every output). They are
lower bounds, not estimates: the true FRP of a saturated pixel can be much
larger. Summed with the retrieved FRP of the other pixels they make the
scene FRP a lower bound that no longer ignores the hottest pixels.

Unsaturated bands are not used here: under the single-temperature model
they are what the retrieval itself uses, and without a model they carry no
one-sided information.
"""

import numpy as np

SIGMA_SB = 5.670374419e-8
_H, _C, _KB = 6.62607015e-34, 2.99792458e8, 1.380649e-23
_T_GRID = np.exp(np.linspace(np.log(150.0), np.log(6000.0), 4000))


def planck_nm(wavelength_nm, T):
    """Blackbody spectral radiance, W m-2 sr-1 nm-1 (wavelength in nm)."""
    lam = np.asarray(wavelength_nm, float) * 1e-9
    T = np.asarray(T, float)
    with np.errstate(over='ignore'):
        return 2 * _H * _C ** 2 / lam ** 5 / np.expm1(_H * _C / (lam * _KB * T)) * 1e-9


def brightness_temperature_nm(wavelength_nm, radiance):
    """Inverse Planck: the T whose blackbody radiance at wavelength_nm is
    `radiance` (W m-2 sr-1 nm-1); NaN for radiance <= 0."""
    lam = np.asarray(wavelength_nm, float) * 1e-9
    L = np.asarray(radiance, float) * 1e9
    with np.errstate(invalid='ignore', divide='ignore'):
        T = _H * _C / (lam * _KB) / np.log1p(2 * _H * _C ** 2 / (lam ** 5 * L))
    return np.where(np.asarray(radiance) > 0, T, np.nan)


def frp_per_radiance_minimum(wavelength_nm):
    """R(lambda) = min over T of sigma T^4 / B(lambda, T)  [W m-2 per
    W m-2 sr-1 nm-1, i.e. sr nm], and the minimising T. Evaluated on a fine
    log grid 150-6000 K (the minimum lies well inside for SWIR wavelengths;
    it is checked)."""
    w = np.atleast_1d(np.asarray(wavelength_nm, float))
    B = planck_nm(w[:, None], _T_GRID[None, :])
    with np.errstate(divide='ignore'):
        ratio = SIGMA_SB * _T_GRID[None, :] ** 4 / B
    k = np.argmin(ratio, axis=1)
    if np.any((k == 0) | (k == len(_T_GRID) - 1)):
        raise ValueError('minimum of sigma T^4 / B at the edge of the temperature grid')
    return ratio[np.arange(len(w)), k], _T_GRID[k]


def saturated_pixel_bounds(wavelength_nm, censored, censor_level, background_max, pixel_area_m2=900.0):
    """Lower bounds for one pixel from its censored bands.

    wavelength_nm, censored (bool), censor_level (recorded radiance at the
    censored bands), background_max (upper bound on the background radiance
    per band) - all (n_bands,). Returns dict FRP_min_MW, T_hot_min_K, the
    band that sets each bound, and n_censored. NaN when no censored band has
    a ceiling above the background bound."""
    w = np.asarray(wavelength_nm, float)
    c = np.asarray(censored, bool) & np.isfinite(np.asarray(censor_level, float))
    out = {'FRP_min_MW': np.nan, 'T_hot_min_K': np.nan, 'band_FRP_nm': np.nan, 'band_T_nm': np.nan,
           'n_censored': int(c.sum())}
    if not c.any():
        return out
    excess = np.asarray(censor_level, float)[c] - np.asarray(background_max, float)[c]
    pos = excess > 0
    if not pos.any():
        return out
    wk, ex = w[c][pos], excess[pos]
    R, _ = frp_per_radiance_minimum(wk)
    frp = pixel_area_m2 * ex * R / 1e6
    tb = brightness_temperature_nm(wk, ex)
    i, j = int(np.argmax(frp)), int(np.nanargmax(tb))
    out.update(FRP_min_MW=float(frp[i]), band_FRP_nm=float(wk[i]), T_hot_min_K=float(tb[j]),
               band_T_nm=float(wk[j]))
    return out


# --------------------------------------------------------------------------
# 3. The joint bound: every band at once, by linear programming
# --------------------------------------------------------------------------
# FRP_min above uses the single most constraining band. All bands must hold
# SIMULTANEOUSLY for the same temperature distribution, so the sharpest
# statement is the linear programme over non-negative emitting areas x(T)
# (x = eps f, any distribution):
#
#     minimise   A sigma sum_j T_j^4 x_j
#     subject to sum_j B(lambda_k, T_j) x_j >= e_k   for every constrained band k
#                x >= 0
#
# with e_k = c_k - L_bkg,k^max (censored bands) or y_k - z sigma_k -
# L_bkg,k^max (linear, unflagged bands: their recorded radiance, less z
# noise standard deviations, is also a lower bound on the at-sensor
# radiance; tau_k <= 1 and eps <= 1 enter as in FRP_min). The single-band
# bound is this programme with one constraint, so the joint bound is never
# lower. Discretisation, made rigorous: T on a grid uniform in 1/T over
# [LP_T_MIN, LP_T_MAX]; a component between two nodes is replaced by the
# node above it (B larger in every band, same weight), which raises its FRP
# by at most kappa = max_j (T_j+1 / T_j)^4, so the grid optimum divided by
# kappa is a valid bound. A component outside the grid is replaced by the
# end node with its weight rescaled: sigma T^4 / B(lambda, T) decreases
# with T below and increases above its minimum T*(lambda) = c2 / (3.92
# lambda), which for every band >= LP_MIN_WAVELENGTH_NM lies inside the
# grid, so the replacement never raises the FRP. Only bands >=
# LP_MIN_WAVELENGTH_NM are used for that reason.
LP_T_MIN, LP_T_MAX, LP_N_T = 300.0, 3000.0, 800
LP_MIN_WAVELENGTH_NM = 1300.0
LP_NOISE_Z = 3.0
LINEAR_DN_FRACTION = 0.85     # above: detector compression (VERIFICATION.md S24.2); not a usable value
_LP_T = 1.0 / np.linspace(1.0 / LP_T_MIN, 1.0 / LP_T_MAX, LP_N_T)
LP_KAPPA = float(np.max((_LP_T[1:] / _LP_T[:-1]) ** 4))


def joint_frp_bound(wavelength_nm, excess_lower, pixel_area_m2=900.0):
    """The joint (linear-programming) FRP lower bound of one pixel.

    excess_lower: (n_bands,) lower bound on the fire term per band
    (radiance units), NaN or <= 0 where there is no constraint. Returns
    (FRP_min_MW, n_constraints); NaN when no band >= LP_MIN_WAVELENGTH_NM
    constrains the pixel."""
    from scipy.optimize import linprog
    w = np.asarray(wavelength_nm, float)
    e = np.asarray(excess_lower, float)
    k = np.isfinite(e) & (e > 0) & (w >= LP_MIN_WAVELENGTH_NM)
    if not k.any():
        return np.nan, 0
    B = planck_nm(w[k][:, None], _LP_T[None, :])
    # scale rows and columns for the solver; the optimum is unchanged
    s = 1.0 / e[k]
    cost = SIGMA_SB * _LP_T ** 4
    cs = cost.max()
    res = linprog(cost / cs, A_ub=-(B * s[:, None]), b_ub=-np.ones(int(k.sum())), bounds=(0, None),
                  method='highs')
    if res.status != 0:
        return np.nan, int(k.sum())
    return float(res.fun * cs * pixel_area_m2 / LP_KAPPA / 1e6), int(k.sum())


def bounds_for_pixels(wavelength_nm, censored, censor_levels, libraries, pool, pixel_area_m2=900.0,
                      pool_quantile=99.9, radiance=None, usable=None, noise_sigma=None):
    """saturated_pixel_bounds() for every pixel (rows of censored /
    censor_levels, full band axis). libraries: per-pixel background
    libraries (list of (J, n_bands) or list of lists of draws, full band
    axis); pool: (N, n_bands) scene background pool. Returns a DataFrame.

    With radiance (N, n_bands), usable (N, n_bands) bool (linear, unflagged,
    uncensored bands) and noise_sigma (N, n_bands) or (n_bands,), the joint
    bound FRP_min_joint_MW (joint_frp_bound) is added for every pixel with
    a censored band, from its censored AND its usable bands."""
    import pandas as pd
    pool = np.atleast_2d(pool)
    # one band at a time in float64 (the whole pool in float64 is ~1.8 GB)
    pool_hi = np.array([np.nanpercentile(np.asarray(pool[:, j], dtype=float), pool_quantile)
                        for j in range(pool.shape[1])])
    rows = []
    for i in range(len(censored)):
        L = libraries[i] if libraries is not None else None
        if isinstance(L, (list, tuple)):
            L = np.vstack([np.atleast_2d(x) for x in L])
        bmax = pool_hi if L is None else np.maximum(np.nanmax(np.atleast_2d(L), axis=0), pool_hi)
        row = saturated_pixel_bounds(wavelength_nm, censored[i], censor_levels[i], bmax, pixel_area_m2)
        if radiance is not None and np.asarray(censored[i], bool).any():
            c = np.asarray(censored[i], bool)
            e = np.where(c, np.asarray(censor_levels[i], float) - bmax, np.nan)
            if usable is not None:
                u = np.asarray(usable[i], bool) & ~c
                sg = 0.0 if noise_sigma is None else np.asarray(noise_sigma)[i] if np.ndim(noise_sigma) == 2 \
                    else np.asarray(noise_sigma)
                e = np.where(u, np.asarray(radiance[i], float) - LP_NOISE_Z * np.nan_to_num(sg) - bmax, e)
            fj, nj = joint_frp_bound(wavelength_nm, e, pixel_area_m2)
            # both are valid bounds; the single-band one may use a band < 1300 nm
            row['FRP_min_joint_MW'] = float(np.nanmax([fj, row['FRP_min_MW']])) \
                if np.isfinite(fj) or np.isfinite(row['FRP_min_MW']) else np.nan
            row['n_joint_constraints'] = nj
        else:
            row['FRP_min_joint_MW'], row['n_joint_constraints'] = np.nan, 0
        rows.append(row)
    return pd.DataFrame(rows)

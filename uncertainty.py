#!/usr/bin/env python
# coding: utf-8
"""
Uncertainty analysis of the retrievals (this work; VERIFICATION.md Section 30).

What each method already reports, and what this module adds:

  This work (v2)   per pixel: 95% profile-likelihood interval on T,
                   widened to cover two half-sample transmittance estimates
                   (profile_pipeline._merge_envelope) and five background
                   library draws. ADDED here: the matching interval on the
                   pixel's fire radiative power, FRP = eps sigma T^4 f A,
                   evaluated along the temperature profile (frp_interval).
  Amici (2022)     per pixel: mean and standard deviation of 15 trials and a
                   t-interval (published). Unchanged.
  Dozier (1981),   no uncertainty in the published methods. ADDED here: a
  Dennison (2006)  Monte Carlo propagation (method_monte_carlo) of the
                   instrument noise (per-band HySime estimate with the
                   shot-noise scaling of profile_retrieval) and, for
                   Dozier, of the background (drawn from the pixel's
                   neighbours instead of their median). The point estimates
                   stay those of the published procedures.
  Scene level      scene_budget(): the median temperature with a bootstrap
                   interval over pixels; the scene FRP with (i) the sum of
                   the per-pixel intervals under full correlation and (ii)
                   independent errors, (iii) the absolute radiometric
                   calibration as a common factor, (iv) the saturated-pixel
                   bounds/estimates as separate terms.

FRP interval. Along the profile T -> f_hat(T), FRP(T) = eps sigma T^4
f_hat(T) A. The profile-likelihood interval of FRP is the projection of the
joint 95% region onto FRP; evaluating FRP at the temperature interval's
edges (where f_at_ci_low/high are the profiled fractions) samples that
region along its ridge. Because f is a linear, well-determined parameter
at fixed T, the region is thin across the ridge, and the ridge end points
approximate the FRP extremes: against a brute-force (T, f) grid projection
of the joint region on synthetic pixels the ends agree to within 12% of
the interval width (algorithm_verification/test_uncertainty.py).

Radiometric calibration. A common gain error k scales every radiance, so f
(and FRP) by (1 + k) and leaves T unchanged to first order. Default 5%
(1 sigma would be smaller; the EnMAP mission requirement for absolute
radiometric accuracy is quoted as <= 5%); pass the product's value when it
is known.
"""
import numpy as np

SIGMA_SB = 5.670374419e-8
PIXEL_AREA_M2 = 900.0
CALIBRATION_REL_UNC = 0.05


def frp_mw(T, f, emissivity=1.0, area_m2=PIXEL_AREA_M2):
    """eps sigma T^4 f A, in MW (NaN where T or f is not finite)."""
    T = np.asarray(T, dtype=float)
    f = np.asarray(f, dtype=float)
    return emissivity * SIGMA_SB * T ** 4 * f * area_m2 / 1e6


def frp_interval(df, emissivity=1.0, area_m2=PIXEL_AREA_M2):
    """Per-pixel FRP and its interval from a ProfileML results frame
    (T_K, fire_fraction, T_ci_low, T_ci_high, f_at_ci_low, f_at_ci_high).
    Returns (frp, lo, hi) arrays in MW; lo/hi NaN where an interval edge is
    open (T_lower/upper_unbounded) or its fraction missing."""
    T = df['T_K'].values.astype(float)
    f = df['fire_fraction'].values.astype(float)
    p = frp_mw(T, f, emissivity, area_m2)
    a = frp_mw(df['T_ci_low'].values.astype(float), df['f_at_ci_low'].values.astype(float), emissivity, area_m2)
    b = frp_mw(df['T_ci_high'].values.astype(float), df['f_at_ci_high'].values.astype(float), emissivity, area_m2)
    stack = np.vstack([p, a, b])
    lo, hi = np.min(stack, axis=0), np.max(stack, axis=0)    # NaN wherever any of the three is
    open_ = np.zeros(len(T), bool)
    for c in ('T_lower_unbounded', 'T_upper_unbounded'):
        if c in df:
            open_ |= df[c].values.astype(bool)
    bad = open_ | ~np.isfinite(a) | ~np.isfinite(b)
    return p, np.where(bad, np.nan, lo), np.where(bad, np.nan, hi)


def add_frp_columns(df, emissivity=1.0, area_m2=PIXEL_AREA_M2):
    """FRP_MW, FRP_ci_low_MW, FRP_ci_high_MW on a ProfileML frame (in place)."""
    p, lo, hi = frp_interval(df, emissivity, area_m2)
    df['FRP_MW'] = p
    df['FRP_ci_low_MW'] = lo
    df['FRP_ci_high_MW'] = hi
    return df


def widen_frp(df, df_alt, emissivity=1.0, area_m2=PIXEL_AREA_M2):
    """Widen df's FRP interval to cover df_alt's (an alternative
    transmittance draw), as _merge_envelope does for T."""
    _, lo, hi = frp_interval(df_alt, emissivity, area_m2)
    ok = df_alt['converged'].values.astype(bool) & np.isfinite(df_alt['T_K'].values.astype(float))
    for c, other, fn in (('FRP_ci_low_MW', lo, np.fmin), ('FRP_ci_high_MW', hi, np.fmax)):
        cur = df[c].values.astype(float)
        merged = np.where(ok & np.isfinite(cur), fn(cur, other), cur)
        # an alternative draw with an open edge opens the merged one
        df[c] = np.where(ok & ~np.isfinite(other) & np.isfinite(cur), np.nan, merged)


# --------------------------------------------------------------------------
# Monte Carlo for methods that publish no uncertainty
# --------------------------------------------------------------------------

def _mc_one_pixel(func, y, w, background_draws, n_mc, sigma, seed, kwargs, needs):
    rng = np.random.default_rng(seed)
    T = np.full(n_mc, np.nan)
    for k in range(n_mc):
        yk = y + sigma * rng.standard_normal(len(y))
        if needs == 'background_spectrum':
            bk = background_draws[rng.integers(len(background_draws))] if background_draws is not None \
                else kwargs['_background']
            r = func(yk, w, bk, **{a: b for a, b in kwargs.items() if a != '_background'})
        else:
            r = func(yk, w, kwargs['_endmembers'], **{a: b for a, b in kwargs.items() if a != '_endmembers'})
        if r.get('converged') and not r.get('T_at_bound', False):
            T[k] = r['T_K']
    v = T[np.isfinite(T)]
    if len(v) < max(10, n_mc // 4):
        return {'T_mc_sd': np.nan, 'T_mc_p2_5': np.nan, 'T_mc_p97_5': np.nan, 'n_mc_valid': len(v)}
    return {'T_mc_sd': float(np.std(v, ddof=1)), 'T_mc_p2_5': float(np.percentile(v, 2.5)),
            'T_mc_p97_5': float(np.percentile(v, 97.5)), 'n_mc_valid': int(len(v))}


def method_monte_carlo(key, cube, wavelength, fire_points, sigma_inst, ref, n_mc=100, background=None,
                       local_backgrounds=None, method_kwargs=None, per_pixel_kwargs=None,
                       band_mask=None, n_jobs=-1, seed=0):
    """Monte Carlo temperature uncertainty of a published method that
    reports none (Dozier1981, Dennison2006). Per pixel, `n_mc` replicate
    spectra y + N(0, sigma) with sigma the two-regime instrument noise at y
    (sigma_inst, ref full band axis); Dozier additionally draws its
    background from `local_backgrounds[i]` (an (n, n_bands) array of the
    pixel's neighbours; else `background`). Replicates where the method
    has no solution or sits at its search bound are not counted.

    Returns a DataFrame with T_mc_sd, T_mc_p2_5, T_mc_p97_5 (the 95%
    Monte Carlo interval) and n_mc_valid, one row per fire pixel.
    """
    import pandas as pd
    import temperature_methods as tm
    spec = tm.METHOD_REGISTRY[key]
    w_full = np.asarray(wavelength, dtype=float)
    m = spec['channels'](w_full) if 'channels' in spec else (
        np.ones(len(w_full), bool) if band_mask is None else np.asarray(band_mask, bool))
    w = w_full[m]
    kw = {} if not method_kwargs or key not in method_kwargs else dict(method_kwargs[key])
    offset = kw.pop('subtract_spectrum', None)
    fixed = kw.pop('fixed_endmembers', None)
    kw.pop('background', None)
    for k_, v_ in list(kw.items()):
        if isinstance(v_, np.ndarray) and v_.shape[-1] == len(m):
            kw[k_] = v_[..., m]
    if offset is not None:
        offset = np.asarray(offset, float)[m]
    seeds = np.random.default_rng(seed).integers(0, 2 ** 31 - 1, size=len(fire_points))
    tasks = []
    for i, (r_, c_) in enumerate(np.asarray(fire_points)):
        y = np.asarray(cube[r_, :, c_], dtype=float)
        sig = sigma_inst * np.sqrt(np.maximum(np.abs(y) / np.maximum(ref, 1e-300), 1.0))
        sig = np.where(np.isfinite(sig), sig, 0.0)[m]
        y = y[m] - (offset if offset is not None else 0.0)
        kwi = dict(kw)
        if per_pixel_kwargs and key in per_pixel_kwargs:
            for k_, seq in per_pixel_kwargs[key].items():
                if k_ == 'background':
                    continue
                v_ = np.asarray(seq[i])
                kwi[k_] = v_[m] if v_.ndim == 1 and v_.shape[0] == len(m) else v_
        draws = None
        if spec['needs'] == 'background_spectrum':
            if local_backgrounds is not None and local_backgrounds[i] is not None and len(local_backgrounds[i]) >= 10:
                draws = np.asarray(local_backgrounds[i], dtype=float)[:, m]
            else:
                kwi['_background'] = np.asarray(background, dtype=float)[m]
        else:
            E = np.atleast_2d(np.asarray(fixed, dtype=float))[:, m]
            kwi['_endmembers'] = E
        tasks.append((spec['func'], y, w, draws, int(n_mc), sig, int(seeds[i]), kwi, spec['needs']))
    try:
        from joblib import Parallel, delayed
        rows = Parallel(n_jobs=n_jobs)(delayed(_mc_one_pixel)(*t) for t in tasks) if len(tasks) >= 8 \
            else [_mc_one_pixel(*t) for t in tasks]
    except ImportError:                                    # pragma: no cover
        rows = [_mc_one_pixel(*t) for t in tasks]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# scene level
# --------------------------------------------------------------------------

def bootstrap_median(x, n_boot=2000, seed=0, alpha=0.05):
    """Percentile-bootstrap interval of the median of x over its elements."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(0, len(x), size=(n_boot, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 100 * alpha / 2)), float(np.percentile(meds, 100 * (1 - alpha / 2)))


def scene_budget(df, emissivity=1.0, calibration_rel_unc=CALIBRATION_REL_UNC, saturated_bounds=None,
                 saturated_estimates=None, adequate_only=True):
    """Scene-level uncertainty budget of a ProfileML-v2 frame with the FRP
    columns. Returns a list of dict rows (quantity, value, low, high, unit,
    note), ready for a CSV and the report.

    saturated_bounds: per-pixel model-free FRP lower bounds (MW) of pixels
    without a retrieval (summed as a one-sided term). saturated_estimates:
    optional (frp, lo, hi) arrays of the absorption-band estimates of those
    pixels (absorption_window.py), summed as a separate two-sided term."""
    rows = []
    q = df['fit_quality'].values if 'fit_quality' in df else np.array(['ok'] * len(df))
    use = (q == 'ok') if adequate_only else np.isfinite(df['T_K'].values.astype(float))
    use &= np.isfinite(df['T_K'].values.astype(float))
    open_ = np.zeros(len(df), bool)
    for c in ('T_lower_unbounded', 'T_upper_unbounded'):
        if c in df:
            open_ |= df[c].values.astype(bool)
    T = df['T_K'].values.astype(float)
    med, lo, hi = bootstrap_median(T[use & ~open_])
    rows.append({'quantity': 'median temperature of adequate fits', 'value': med, 'low': lo, 'high': hi,
                 'unit': 'K', 'note': f'{int((use & ~open_).sum())} pixels; 95% percentile bootstrap over pixels '
                                      '(sampling only; per-pixel errors are in the pixel intervals)'})
    w = (df['T_ci_high'] - df['T_ci_low']).values.astype(float)
    rows.append({'quantity': 'median per-pixel 95% interval width', 'value': float(np.nanmedian(w[use & ~open_])),
                 'low': np.nan, 'high': np.nan, 'unit': 'K',
                 'note': 'profile likelihood, widened over transmittance and library draws'})
    if 'FRP_MW' in df:
        p = df['FRP_MW'].values.astype(float)
        a = df['FRP_ci_low_MW'].values.astype(float)
        b = df['FRP_ci_high_MW'].values.astype(float)
        u = use & np.isfinite(p)
        fin = u & np.isfinite(a) & np.isfinite(b)
        tot = float(np.nansum(p[u]))
        # pixels with an open interval: their point value counts, their
        # interval is open above, so the correlated upper sum is open too
        # pixels with an open interval enter at their point value on both
        # ends (stated in the note): their upper end is open
        n_open = int((u & ~fin).sum())
        corr_lo = float(np.nansum(a[fin]) + np.nansum(p[u & ~fin]))
        corr_hi = float(np.nansum(b[fin]) + np.nansum(p[u & ~fin]))
        sd = (b - a) / (2 * 1.96)
        ind = float(np.sqrt(np.nansum(sd[fin] ** 2)))
        rows.append({'quantity': 'FRP, retrieved pixels (point sum)', 'value': tot, 'low': np.nan, 'high': np.nan,
                     'unit': 'MW', 'note': f'{int(u.sum())} adequately fitted pixels'})
        rows.append({'quantity': 'FRP, retrieved pixels - errors fully correlated', 'value': tot,
                     'low': corr_lo, 'high': corr_hi, 'unit': 'MW',
                     'note': 'sum of the per-pixel 95% ends (shared transmittance / calibration: conservative); '
                             f'{n_open} pixel(s) with an open interval enter at their point value '
                             '(their upper end is unbounded)'})
        rows.append({'quantity': 'FRP, retrieved pixels - errors independent', 'value': tot,
                     'low': tot - 1.96 * ind, 'high': tot + 1.96 * ind, 'unit': 'MW',
                     'note': 'quadrature of per-pixel half-widths / 1.96 (lower limit of the scene uncertainty)'})
        rows.append({'quantity': 'FRP, radiometric calibration (common factor)', 'value': tot,
                     'low': tot * (1 - calibration_rel_unc), 'high': tot * (1 + calibration_rel_unc), 'unit': 'MW',
                     'note': f'+/-{100 * calibration_rel_unc:.0f}% gain error scales f and FRP, not T'})
        if saturated_bounds is not None:
            sbv = np.asarray(saturated_bounds, float)
            rows.append({'quantity': 'FRP, unretrieved saturated pixels - model-free lower bound', 'value': float(np.nansum(sbv)),
                         'low': float(np.nansum(sbv)), 'high': np.inf, 'unit': 'MW',
                         'note': f'{int(np.isfinite(sbv).sum())} pixels; valid for any sub-pixel temperature distribution'})
        if saturated_estimates is not None:
            fe, fl, fh = (np.asarray(x, float) for x in saturated_estimates)
            ok = np.isfinite(fe) & np.isfinite(fl) & np.isfinite(fh)
            rows.append({'quantity': 'FRP, saturated pixels - absorption-band estimate', 'value': float(np.sum(fe[ok])),
                         'low': float(np.sum(fl[ok])), 'high': float(np.sum(fh[ok])), 'unit': 'MW',
                         'note': f'{int(ok.sum())} pixels; sum of per-pixel 95% ends (correlated: shared tau)'})
    return rows

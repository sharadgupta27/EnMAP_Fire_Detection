#!/usr/bin/env python
# coding: utf-8
"""
Scene-adaptive estimation of the atmospheric transmittance on the fire term
from the scene's own fire pixels (this work).

--------------------------------------------------------------------------
1. THE GAP
--------------------------------------------------------------------------

Every retrieval in this repository, and every published VSWIR sub-pixel
retrieval that works on at-sensor (L1B) radiance, fits the fire term as an
UNATTENUATED Planck curve:

    L_obs(lambda) = f * eps * B(lambda, T) + sum_j q_j L_j(lambda)          (1)

The background endmembers L_j are measured at-sensor spectra and therefore
already contain the atmosphere; the fire term does not. What the residuals
of (1) look like on real EnMAP fire pixels is not noise but the SHAPE of the
atmospheric gas transmittance: the model overshoots on the wing of the
1.9 um water band (1960-1990 nm), across the 2.0-2.06 um CO2 bands, and in
the CH4/H2O roll-off beyond 2300 nm, with relative residuals of 15-50% in
those bands against 1-3% in the clean windows (VERIFICATION.md S11.3 masked
the two worst bands; S11.4 lists "no atmospheric transmittance term" as the
largest open item and "the honest ceiling on accuracy claims"). PRISMA work
(Amici et al. 2022) sidestepped this by using the L2B bottom-of-atmosphere
radiance product; EnMAP has no such product. The 2026 AVIRIS-3 retrieval
(arXiv:2608.07580) solves it with a per-scene 6S look-up table and a
retrieved column water vapour - a radiative-transfer code dependency.

--------------------------------------------------------------------------
2. METHOD: the fires as their own calibration sources
--------------------------------------------------------------------------

Write the physically correct model with a one-way (ground-to-sensor)
transmittance tau(lambda) on the emitted term:

    L_obs,i(lambda) = tau(lambda) f_i eps B(lambda, T_i) + sum_j q_ij L_j(lambda)   (2)

for every fire pixel i = 1..N of a scene. tau is COMMON to all fire pixels
of the scene (same atmosphere, same viewing geometry to within a few degrees
across a 30 km scene; a local smoke excess on top of it is the separate,
per-pixel smoke-transmittance correction in temperature_methods.py). Given N pixels at
different temperatures, (2) is a bilinear model in (tau) x (f_i, T_i, q_ij)
and is solved by ALTERNATING least squares:

  step P (per pixel, tau fixed): the profile-likelihood retrieval of
         profile_retrieval.py, with `tau` passed as the fixed multiplier -
         a global 1-D search per pixel;
  step T (per band, pixels fixed): with s_ik = f_i eps B(lambda_k, T_i) and
         g_ik = sum_j q_ij L_j(lambda_k),

             tau_k = [sum_i w_ik s_ik (y_ik - g_ik) + r] / [sum_i w_ik s_ik^2 + r]   (3)

         the weighted least-squares slope through the origin with a ridge
         r toward tau_k = 1 (r = `ridge` times the median over bands of the
         information sum_i w_ik s_ik^2), closed form, weights
         w_ik = 1/sigma_ik^2 from the same error model the retrieval uses,
         censored (saturated) bands excluded, and the result clipped to
         tau_k <= 1.

The two steps minimise the same weighted sum of squares (plus the ridge),
so the objective is non-increasing between them; because the covariance
is re-evaluated each round (it depends on the modelled radiance) the
sequence is not strictly monotone late in the iteration, and the round
with the lowest total chi2 is returned (8 rounds by default; the ridge
slows the deep dips, which is why 4 rounds are not enough).

2.1 What is, and is not, identifiable - and what is done about it.
In the Wien regime B(lambda, T) ~ exp(-c2 / (lambda T)), so multiplying tau
by a "Wien tilt" exp(-c / lambda) is exactly equivalent to shifting every
pixel's 1/T_i by c / c2, and multiplying tau by a constant is exactly
equivalent to rescaling every f_i: the smooth part of tau is NOT
identifiable from fire pixels alone. The first version of this module
therefore projected ln(tau) onto the complement of the smooth components
(a constant and a tilt, or a level per spectral window) on the argument
that whatever is removed is absorbed harmlessly by (f_i, T_i). Measured on
synthetic truth, that argument is WRONG once the background is a flexible
multi-endmember mixture over two spectral windows: applying any of four
such normalisations to the TRUE transmittance biased the retrieved
temperature by 15-25 K, whereas the raw true transmittance gave -1 K
(verify17_profile.py, section D). The temperature is sensitive to the
smooth part of tau at roughly 2 K per 1% of level difference between the
1.6 and 2.2 um windows and 6 K per 0.1 um of Wien tilt (same section).

The estimator therefore removes NOTHING and pins the weakly identifiable
smooth directions with two physical constraints instead:

  * a ridge toward tau = 1 (the fire term's continuum transmittance is
    unity; its true smooth part - aerosol, far wings, grey-body emissivity
    - is then carried by f_eff and by a temperature offset bounded by the
    sensitivity coefficients above, i.e. a few K for a realistic SWIR
    continuum of 0.90-0.97);
  * the hard bound tau <= 1 (a transmittance times an emissivity cannot
    exceed unity, and SWIR path radiance is negligible and additive).

The bound is load-bearing: without it the level of the 2.1-2.3 um window
drifted to 1.09 on the Portugal scene and to 1.04 on synthetic truth
(where it produced a +13 K bias at 1000-1200 K), driven by the background
library's inability to represent a burnt surface; with it the estimate is
independent of the library (scene-wide, local 11x11 or mixed libraries:
median |delta tau| = 0.003) and the continuum sits at 1.000 in both
windows, with dips only at the H2O (1470-1490, 1968-1990), CO2
(2050-2070) and CH4/H2O (2320-2395 nm) bands.

2.2 Validation without an atmosphere model. Three checks that need no
external radiative-transfer code (all in verify17_profile.py):
(a) synthetic - a known gas transmittance injected into 40 synthetic fire
    pixels on real EnMAP backgrounds is recovered (median error 0.04 in
    the absorption bands, 0.00 in the continuum) and the temperature bias
    it induces in every published method (-3 to -8 K median, but 50-60 K
    mean absolute error) is removed (chain: -0.4 K median, and the
    remaining error is the library's, see profile_retrieval.py);
(b) split-half - tau estimated on one random half of the Portugal scene's
    55 fire pixels and applied to the other half reduces the held-out
    misfit from 8-11% to 3% on 89-100% of the held-out pixels, and the two
    half-estimates agree to 0.004 (median absolute difference);
(c) physical - the recovered dips sit at the known gas bands, and the
    1968 nm water-wing dip (tau ~ 0.4) is DEEPER than the two-way
    absorption seen in the scene's background continuum, which is what a
    fire's own hot, moist plume gases would produce (hot-gas
    self-absorption; the background pixels do not look through a plume).

2.3 Where this sits. The result is a per-scene tau(lambda) that
run_pipeline() hands to profile_retrieval() (and that any method accepting
a fixed `tau`, e.g. smoke_corrected_nlls(), can take). It is estimated
from the scene alone: no radiative-transfer code, no column water vapour
retrieval, no look-up table - which is what makes it deployable in the
product. Its weaknesses are equally plain: it needs enough fire pixels
(three is the hard floor; a dozen at varied temperatures is comfortable),
its continuum is a prior rather than a measurement (2.1), and on the one
real scene checked it moves the median retrieved temperature by about
-40 K relative to the unattenuated fit - a large, physically expected
correction (unmodelled long-wave absorption makes a fire look hotter) that
has not yet been checked against an independent temperature measurement.

Units: radiance in W m^-2 sr^-1 nm^-1 and wavelength in nm throughout.
"""

import numpy as np

from profile_retrieval import (profile_retrieval, noise_sigma, _planck_nm, _design,
                               weighted_fcls, _many)

__all__ = ['estimate_fire_term_transmittance', 'remove_wien_tilt', 'split_half_validation']


def _window_indicators(wavelength, gap_nm=50.0):
    """Indicator columns for the contiguous spectral windows of the fit
    band set (a new window starts wherever the band spacing jumps above
    `gap_nm`, i.e. across an excluded atmospheric absorption band)."""
    lam = np.asarray(wavelength, dtype=float)
    order = np.argsort(lam)
    breaks = np.where(np.diff(lam[order]) > gap_nm)[0]
    labels = np.zeros(len(lam), dtype=int)
    start = 0
    for k, b in enumerate(breaks):
        labels[order[start:b + 1]] = k
        start = b + 1
    labels[order[start:]] = len(breaks)
    n_win = len(breaks) + 1
    return np.column_stack([(labels == k).astype(float) for k in range(n_win)])


def remove_wien_tilt(tau, wavelength, weights=None, mode='windows'):
    """Project ln(tau) onto the orthogonal complement of the SMOOTH
    components that the temperature/fraction parameters already
    represent, equation (4). Returns (tau_fine, coef, mode).

    mode='tilt'    : span{1, 1/lambda} - a constant (degenerate with f) and
                     a Wien tilt (degenerate with a common shift of 1/T).
    mode='windows' : the indicator of every contiguous spectral window plus
                     1/lambda - additionally a free LEVEL per window. The
                     level of one SWIR window against another is a 2-point
                     quantity that the fire pixels cannot separate from the
                     background library's own window-to-window shape
                     (measured: the 1.6-vs-2.2 um level of tau changed from
                     1.11 to 0.95 when the library was drawn from the fire
                     pixels' own neighbourhood instead of the whole scene,
                     while the gas features did not move), so it is not
                     attributed to the atmosphere. Default.
    """
    tau = np.asarray(tau, dtype=float)
    lam_um = np.asarray(wavelength, dtype=float) * 1e-3
    good = np.isfinite(tau) & (tau > 0)
    w = np.ones(len(tau)) if weights is None else np.asarray(weights, dtype=float)
    w = np.where(good, w, 0.0)
    lt = np.where(good, np.log(np.where(good, tau, 1.0)), 0.0)
    sw = np.sqrt(w)
    if mode in (None, 'none'):
        return np.where(good, tau, np.nan), None, 'none'
    if mode in ('runmed', 'runmed_uniform'):
        # per-window running median of ln(tau) over +-width_nm, i.e. a
        # high-pass that keeps only features narrower than the window
        width = _RUNMED_WIDTH_NM
        W = _window_indicators(wavelength)
        lam = np.asarray(wavelength, dtype=float)
        smooth = np.zeros(len(tau))
        for k in range(W.shape[1]):
            idx = np.where((W[:, k] > 0) & good)[0]
            for i in idx:
                near = idx[np.abs(lam[idx] - lam[i]) <= width]
                smooth[i] = np.median(lt[near])
        tau_fine = np.where(good, np.exp(lt - smooth), np.nan)
        return tau_fine, smooth, mode
    if mode == 'windows':
        X = np.column_stack([_window_indicators(wavelength), 1.0 / lam_um])
    elif mode == 'poly2':
        # per-window quadratic in 1/lambda: level, tilt and curvature of
        # each window are left to (f, T, background); only narrower
        # structure is kept
        W = _window_indicators(wavelength)
        cols = []
        for k in range(W.shape[1]):
            cols += [W[:, k], W[:, k] / lam_um, W[:, k] / lam_um ** 2]
        X = np.column_stack(cols)
    else:
        X = np.column_stack([np.ones(len(tau)), 1.0 / lam_um])
    coef, *_ = np.linalg.lstsq(X * sw[:, None], lt * sw, rcond=None)
    tau_fine = np.where(good, np.exp(lt - X @ coef), np.nan)
    return tau_fine, coef, mode


_RUNMED_WIDTH_NM = 60.0


ABSORBER_REGIONS = {'H2O 1.4um wing': (1461.0, 1520.0), 'H2O 1.9um wing': (1960.0, 2000.0),
                    'CO2 2.0um': (2000.0, 2100.0), 'CH4/H2O 2.3um': (2280.0, 2400.0)}
CONTINUUM_ANCHORS = ((1520.0, 1740.0), (2100.0, 2280.0))


def background_absorption_template(pool, wavelength, anchors=CONTINUUM_ANCHORS,
                                   regions=ABSORBER_REGIONS, n_sample=3000, rng=None):
    """The scene's gas-absorption signature from its own background pixels.

    Every background spectrum is solar irradiance times surface reflectance
    times the TWO-WAY atmospheric transmittance. Fitting a quadratic
    continuum in log radiance through the gas-free anchor windows and taking
    g = -ln(L / continuum) gives, band by band, the apparent two-way optical
    depth of the atmosphere at EnMAP's resolution; the median over a few
    thousand pixels removes the surface. Within the anchors g is zero by
    construction; outside the absorber regions it is not used.

    Returns dict(template=g (n_bands,), regions, anchor_mask, region_masks).
    """
    w = np.asarray(wavelength, float)
    X = np.atleast_2d(np.asarray(pool))
    X = np.asarray(X[np.all(np.isfinite(X), axis=1) & np.all(X > 0, axis=1)], dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)
    if len(X) > n_sample:
        X = X[rng.choice(len(X), n_sample, replace=False)]
    anchor = np.zeros(len(w), bool)
    for lo, hi in anchors:
        anchor |= (w >= lo) & (w <= hi)
    if anchor.sum() < 6:
        raise ValueError('background_absorption_template: too few anchor bands in the window')
    G = np.empty((len(X), len(w)))
    for i, L in enumerate(X):
        c = np.polyfit(w[anchor], np.log(L[anchor]), 2)
        G[i] = -np.log(L / np.exp(np.polyval(c, w)))
    g = np.maximum(np.median(G, axis=0), 0.0)
    masks = {k: (w >= lo) & (w <= hi) for k, (lo, hi) in regions.items()}
    return {'template': g, 'regions': dict(regions), 'anchor_mask': anchor, 'region_masks': masks,
            'wavelength': w}


def template_scale_factors(tau, prior_template, weights=None, min_template=1e-3):
    """Information-weighted least-squares scale a_k of ln tau on -g within
    each absorber region (a_k >= 0), and the template-shaped transmittance
    exp(-a_k g) (unity outside the regions). Returns dict(scales, tau_template,
    residual_ln = ln tau - ln tau_template)."""
    g = prior_template['template']
    tau = np.asarray(tau, float)
    lt = np.log(np.clip(tau, 1e-3, 1.0))
    wgt = np.ones(len(tau)) if weights is None else np.asarray(weights, float)
    tau_tpl = np.ones(len(tau))
    scales = {}
    for k, mk in prior_template['region_masks'].items():
        sel = mk & (g > min_template) & np.isfinite(lt) & (wgt > 0)
        if sel.sum() < 2:
            scales[k] = np.nan
            continue
        a = float(np.sum(wgt[sel] * (-lt[sel]) * g[sel]) / np.sum(wgt[sel] * g[sel] ** 2))
        a = max(a, 0.0)
        scales[k] = a
        tau_tpl[mk] = np.exp(-a * g[mk])
    return {'scales': scales, 'tau_template': tau_tpl, 'residual_ln': lt - np.log(tau_tpl)}


def estimate_fire_term_transmittance(fire_spectra, wavelength, endmember_libs, n_rounds=8,
                                     ridge=0.5, tau_bounds=(0.2, 1.0), emissivity=1.0,
                                     min_pixels=3, verbose=True, per_pixel_kwargs=None,
                                     smooth_mode='none', ridge_target='one', prior_template=None,
                                     device=None, initial_retrievals=None, adequacy_cap=None,
                                     **retrieval_kwargs):
    """Alternating estimation of the common fire-term transmittance
    tau_fine(lambda) across the fire pixels of one scene, equations (2)-(4).

    fire_spectra : (N, n_bands) over the fit window.
    endmember_libs : list of N libraries (J, n_bands), or one library.
    n_rounds : alternating rounds (each = one profile retrieval per pixel
        plus one closed-form band solve).
    ridge : strength of the pull of each band's tau toward the previous
        round's value, in units of the median per-band information
        sum_i w_ik s_ik^2 - so a band whose fire signal is far below the
        scene's typical keeps its previous value instead of being set by
        noise, while a well-lit band is unaffected.
    tau_bounds : hard clip on tau after each solve. The upper bound 1.0 is
        PHYSICAL - a transmittance (times a grey-body emissivity) cannot
        exceed unity, and SWIR path radiance is negligible and additive -
        and it is load-bearing: it stops the weakly identifiable
        window-to-window level of tau from drifting upward when the
        background library cannot represent a burnt surface (measured on
        the Portugal scene: 1.09 in the 2.1-2.3 um window without the
        bound; on synthetic truth the bound removed a +13 K bias at
        1000-1200 K).
    smooth_mode : 'none' (default), 'tilt', 'windows', 'poly2' or 'runmed'
        - which smooth components, if any, are removed from ln(tau) after
        every band solve (remove_wien_tilt()). Default is NONE: on
        synthetic truth every normalisation biased the temperature by
        15-25 K even when applied to the TRUE transmittance, because the
        smooth part of tau is not absorbed by (f, T) once the background is
        a flexible multi-endmember mixture over two spectral windows (see
        module docstring 2.1); the weakly identifiable smooth directions are
        instead anchored by the ridge.
    ridge_target : 'one' (default) - the ridge pulls every band toward
        tau = 1, a prior that the fire term's continuum transmittance is
        unity (its true smooth part - aerosol, far wings, grey-body
        emissivity - is then carried by f_eff and a small, stated T offset);
        'previous' - toward the previous round's value (the ridge then only
        damps the iteration and the smooth directions drift); or an array
        (n_bands,) of per-band target values.
    prior_template : optional dict from background_absorption_template().
        The ridge then pulls each absorber region toward the PHYSICAL shape
        of the scene's own gas absorption, exp(-a_k g(lambda)), with the
        scale a_k of every region re-estimated from the data each round
        (template_scale_factors()) and unity outside the regions. With many
        fire pixels the data dominate and the estimate is the free one;
        with a handful of pixels the prior carries the band shape and only
        the few scales a_k are left to the data. a_k = 0.5 is the pure
        one-way path through the same column the background sees (weak
        absorption, sun and view near nadir); a_k > 0.5 is excess absorption
        in the fire's own line of sight (plume H2O, CO2, CH4). The final
        a_k are returned ('template_scales').
    retrieval_kwargs : passed to profile_retrieval (snr, reference_radiance,
        rel_model_error, corr_length_nm, include_shade, T_bounds, ...).
    per_pixel_kwargs : optional list of N dicts passed to profile_retrieval
        for that pixel only (valid_mask, censored_mask, censor_level from
        enmap_quality.load_pixel_quality). Censored bands never enter the
        band solve (3): a saturated band says nothing about tau.
    device : where the per-pixel retrievals of every round run (all CPU
        cores by default; see profile_batch.resolve_device()).
    initial_retrievals : optional list of N profile_retrieval results with
        tau = 1 and these same libraries and keyword arguments - i.e. round
        0 of this estimator - to use instead of recomputing it. Round 0 does
        not depend on anything the rounds estimate, so the scene estimate,
        both split-half estimates and the split-half 'before' retrievals
        all compute exactly the same thing; split_half_validation() takes
        it from the scene estimate ('retrievals_round0').
    adequacy_cap : None (default: every converged pixel enters) or c > 0.
        The scene estimate then minimises the CAPPED (trimmed) objective

            L(tau) = sum_i min(chi2_i(tau), c * m0 * dof_i)

        with m0 the median reduced chi-square of the converged pixels in
        round 0 (tau = 1), fixed for the whole call. A pixel above the cap
        has zero gradient in L, so the band solve (3) uses only the pixels
        below it - exactly the pixels the mixture model describes to within
        c times the scene's typical misfit - and the best round is chosen
        by L itself, one objective throughout. With the error model
        calibrated so that the median reduced chi-square is 1, c = 5 is the
        'poor fit' threshold of run_profile_on_pixels(); m0 transfers it to
        the uncalibrated rounds. Pixels whose residuals the model cannot
        describe would otherwise put their structured misfit into tau.

    Returns dict: tau_fine (n_bands,), tau_raw (before tilt removal),
    tilt (a, b), history (per-round total weighted chi2 and median relative
    RMSE), retrievals (final per-pixel profile_retrieval results with tau
    applied), retrievals_round0 (the tau = 1 retrievals of round 0),
    n_pixels_used.
    """
    Y = np.atleast_2d(np.asarray(fire_spectra, dtype=float))
    wavelength = np.asarray(wavelength, dtype=float)
    N, K = Y.shape
    if not isinstance(endmember_libs, (list, tuple)):
        endmember_libs = [endmember_libs] * N
    if N < min_pixels:
        raise ValueError(f'estimate_fire_term_transmittance: need at least {min_pixels} fire '
                         f'pixels, got {N}')
    kw = dict(retrieval_kwargs)
    kw.setdefault('include_shade', True)
    include_shade = kw['include_shade']
    # the per-band step uses the DIAGONAL error model, whatever the retrieval uses
    diag_kw = {k: kw.get(k) for k in ('snr', 'sigma_instrument', 'reference_radiance') if k in kw}
    eps = kw.get('rel_model_error', 0.05)

    if per_pixel_kwargs is None:
        per_pixel_kwargs = [{}] * N
    tau = np.ones(K)
    history = []
    retrievals = None
    best = None                      # (chi2_total, tau, retrievals) of the best round
    retrievals_round0 = None
    cap_red = None                   # reduced-chi2 cap of the trimmed objective (adequacy_cap)
    n_used_rounds = []
    for rnd in range(n_rounds + 1):
        # ---- step P: per-pixel profile retrieval with tau fixed ----------
        if rnd == 0 and initial_retrievals is not None:
            if len(initial_retrievals) != N:
                raise ValueError(f'initial_retrievals has {len(initial_retrievals)} entries for {N} pixels')
            retrievals = list(initial_retrievals)
        else:
            retrievals = _many([Y[i] for i in range(N)], wavelength, list(endmember_libs),
                               per_pixel=list(per_pixel_kwargs), device=device, emissivity=emissivity,
                               tau=tau, **kw)
        if rnd == 0:
            retrievals_round0 = retrievals
        ok = [r for r in retrievals if r['converged']]
        if adequacy_cap is not None and cap_red is None:
            red0 = np.array([r['chi2_min'] / max(r['dof'], 1) for r in ok])
            red0 = red0[np.isfinite(red0)]
            cap_red = float(adequacy_cap) * float(np.median(red0)) if len(red0) else np.inf
        if cap_red is None:
            use = [r['converged'] for r in retrievals]
            chi2_total = float(np.nansum([r['chi2_min'] for r in ok]))
        else:
            use = [bool(r['converged'] and r['chi2_min'] <= cap_red * max(r['dof'], 1)) for r in retrievals]
            chi2_total = float(np.nansum([min(r['chi2_min'], cap_red * max(r['dof'], 1)) for r in ok]))
        n_used_rounds.append(int(np.sum(use)))
        rmse_rel = float(np.nanmedian([r['rmse'] / np.mean(np.abs(Y[i][r['_good']]))
                                       for i, r in enumerate(retrievals) if r['converged']]))
        history.append({'round': rnd, 'chi2_total': chi2_total, 'median_rmse_rel': rmse_rel,
                        'n_converged': len(ok), 'n_used': n_used_rounds[-1]})
        if verbose:
            print(f'  transmittance round {rnd}: total {"capped " if cap_red is not None else ""}chi2 '
                  f'{chi2_total:.4g}, median relative RMSE {100 * rmse_rel:.2f}%, {len(ok)}/{N} pixels'
                  + (f' ({n_used_rounds[-1]} below the adequacy cap)' if cap_red is not None else ''))
        if best is None or chi2_total < best[0]:
            best = (chi2_total, tau.copy(), retrievals, n_used_rounds[-1])
        if rnd == n_rounds:
            break
        # ---- step T: per-band weighted slope (3) ------------------------
        num = np.zeros(K)
        den = np.zeros(K)
        for i, r in enumerate(retrievals):
            if not use[i]:
                continue
            good = r['_good'].copy()
            if r.get('_censored') is not None:          # drop censored rows from the band solve
                good[np.where(good)[0][r['_censored']]] = False
            w_i = wavelength[good]
            a = r['abundances']
            lib = np.atleast_2d(endmember_libs[i])[:, good]
            # fire signal WITHOUT tau, and the background part of the model
            s = a[0] * emissivity * _planck_nm(w_i, r['T_K'])
            A_bkg = _design(w_i, r['T_K'], lib, emissivity, np.ones(good.sum()), include_shade)[:, 1:]
            g = A_bkg @ a[1:]
            m = r['model'][good]
            sig = noise_sigma(Y[i][good], m, rel_model_error=eps, snr=diag_kw.get('snr'),
                              **{k: (None if v is None else np.asarray(v, float)[good])
                                 for k, v in diag_kw.items() if k != 'snr'})
            wgt = 1.0 / sig ** 2
            num[good] += wgt * s * (Y[i][good] - g)
            den[good] += wgt * s * s
        info = np.median(den[den > 0]) if np.any(den > 0) else 1.0
        lam_r = ridge * info
        if prior_template is not None:
            # scale the background absorption template to the current free
            # estimate, region by region, and pull toward that curve
            free = np.clip(np.where(den > 0, num / np.maximum(den, 1e-300), 1.0), tau_bounds[0], tau_bounds[1])
            scales = template_scale_factors(free, prior_template, weights=den)
            target = scales['tau_template']
        elif isinstance(ridge_target, str):
            target = np.ones(K) if ridge_target == 'one' else tau
        else:
            target = np.asarray(ridge_target, dtype=float)
        tau_new = (num + lam_r * target) / (den + lam_r)
        tau_new = np.clip(tau_new, tau_bounds[0], tau_bounds[1])
        tau_fine, coef, _ = remove_wien_tilt(tau_new, wavelength, weights=den, mode=smooth_mode)
        tau = np.where(np.isfinite(tau_fine), tau_fine, 1.0)

    tau_raw = tau_new if n_rounds > 0 else np.ones(K)
    # The alternation is not an exact coordinate descent (the covariance is
    # re-evaluated each round and the smooth projection is not a
    # minimisation step), so the objective can rise slightly late in the
    # iteration; the round with the lowest total chi2 is the one returned.
    chi2_best, tau_best, retr_best, n_used_best = best
    return {'tau_fine': tau_best, 'tau_raw': tau_raw, 'tau_last_round': tau,
            'smooth_coef': coef if n_rounds > 0 else None, 'smooth_mode': smooth_mode,
            'history': history, 'retrievals': retr_best, 'retrievals_round0': retrievals_round0,
            'n_pixels_used': int(n_used_best), 'adequacy_cap_reduced_chi2': cap_red,
            'band_information': den, 'chi2_total_best': chi2_best,
            'template_scales': (template_scale_factors(tau_best, prior_template, weights=den)['scales']
                                if prior_template is not None else None)}


def split_half_validation(fire_spectra, wavelength, endmember_libs, rng=None, n_rounds=4,
                          emissivity=1.0, per_pixel_kwargs=None, estimator_kwargs=None,
                          device=None, round0_retrievals=None, **retrieval_kwargs):
    """Out-of-sample check of the transmittance estimate: tau_fine is
    estimated on a random half of the fire pixels and APPLIED to the other
    half, whose misfit with and without it is compared. Returns dict with
    the median relative RMSE of the held-out half before/after, the
    per-pixel temperature change, and the two half-estimates' agreement
    (median |tau_A - tau_B|). `estimator_kwargs` (ridge, ridge_target,
    smooth_mode, tau_bounds) go to estimate_fire_term_transmittance();
    everything else to profile_retrieval(). `round0_retrievals`: the tau = 1
    retrievals of all N pixels with the same libraries and arguments (the
    scene estimate's 'retrievals_round0'), reused for both halves' round 0
    and for the held-out 'before' fits - the same computation three times
    over otherwise."""
    Y = np.atleast_2d(np.asarray(fire_spectra, dtype=float))
    N = Y.shape[0]
    rng = rng if rng is not None else np.random.default_rng(0)
    if not isinstance(endmember_libs, (list, tuple)):
        endmember_libs = [endmember_libs] * N
    if per_pixel_kwargs is None:
        per_pixel_kwargs = [{}] * N
    estimator_kwargs = dict(estimator_kwargs or {})    # ridge, ridge_target, smooth_mode, tau_bounds
    perm = rng.permutation(N)
    A_idx, B_idx = perm[: N // 2], perm[N // 2:]
    out = {}
    taus = {}
    for name, fit_idx, test_idx in (('A', A_idx, B_idx), ('B', B_idx, A_idx)):
        est = estimate_fire_term_transmittance(Y[fit_idx], wavelength,
                                               [endmember_libs[i] for i in fit_idx],
                                               n_rounds=n_rounds, emissivity=emissivity,
                                               verbose=False, device=device,
                                               per_pixel_kwargs=[per_pixel_kwargs[i] for i in fit_idx],
                                               initial_retrievals=(None if round0_retrievals is None else
                                                                   [round0_retrievals[i] for i in fit_idx]),
                                               **estimator_kwargs, **retrieval_kwargs)
        taus[name] = est['tau_fine']
        before, after, dT = [], [], []
        Yt = [Y[i] for i in test_idx]; Lt = [endmember_libs[i] for i in test_idx]
        pkt = [per_pixel_kwargs[i] for i in test_idx]
        if round0_retrievals is not None:
            R0 = [round0_retrievals[i] for i in test_idx]
        else:
            R0 = _many(Yt, wavelength, Lt, per_pixel=pkt, device=device, emissivity=emissivity,
                       **retrieval_kwargs)
        R1 = _many(Yt, wavelength, Lt, per_pixel=pkt, device=device, emissivity=emissivity,
                   tau=est['tau_fine'], **retrieval_kwargs)
        for i, r0, r1 in zip(test_idx, R0, R1):
            if r0['converged'] and r1['converged']:
                scale = np.mean(np.abs(Y[i][r0['_good']]))
                before.append(r0['rmse'] / scale)
                after.append(r1['rmse'] / scale)
                dT.append(r1['T_K'] - r0['T_K'])
        out[name] = {'n_fit': len(fit_idx), 'n_test': len(test_idx),
                     'median_rmse_rel_before': float(np.median(before)),
                     'median_rmse_rel_after': float(np.median(after)),
                     'frac_pixels_improved': float(np.mean(np.array(after) < np.array(before))),
                     'median_dT_K': float(np.median(dT)), 'dT_K': np.asarray(dT)}
    both = np.isfinite(taus['A']) & np.isfinite(taus['B'])
    out['tau_agreement_median_abs_diff'] = float(np.median(np.abs(taus['A'][both] - taus['B'][both])))
    out['tau_A'], out['tau_B'] = taus['A'], taus['B']
    return out

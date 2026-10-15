#!/usr/bin/env python
# coding: utf-8
"""
Sub-pixel fire temperature retrieval from VSWIR imaging spectroscopy:
the published method families, implemented against one common interface so
they can be run side by side on the same EnMAP spectra.

All methods share the general spectral mixing model reviewed in
Veraverbeke et al. (2018), RSE 216:105-121, Eq. 2:

    L(lambda) = sum_i f_i * L_i(lambda) + residual                     (V2)

where L_i is endmember i's radiance and f_i its areal fraction. What
distinguishes the families is how the FIRE endmember is parameterised and
how many background endmembers accompany it.

  Dozier1981     dozier_bispectral()  Dozier (1981); Matson & Dozier (1981).
                                      Two channels, two unknowns (T, p),
                                      solved exactly; evaluated here on two
                                      SWIR channels with the measured
                                      background radiance in place of the
                                      background Planck term.
  Dennison2006   dennison_mesma()     Dennison et al. (2006), RSE 100:212.
                                      MESMA with one reflected endmember, one
                                      Planck emitted endmember from a
                                      500-1500 K library in 10 K steps (101
                                      endmembers) and shade; every
                                      (reflected, emitted) pair is solved by
                                      unconstrained least squares, models
                                      with 0 <= f_r + f_e <= 1 are kept and
                                      the lowest RMSE wins. Their channel
                                      subset and encoded-radiance weighting
                                      are reproduced.
  Amici2022      amici2022_multitrial() Amici et al. (2022), JGR Biogeosciences
                 planck_nlls()        127:e2022JG007055, and the code released
                                      with it (Zenodo 10.5281/zenodo.7273879):
                                      joint nonlinear fit of T and the
                                      abundances of m = 20 background spectra
                                      drawn from the image, over 1400-1900 nm,
                                      repeated N = 15 times with fresh draws;
                                      the temperature is the mean of the
                                      trials with a t-interval. planck_nlls()
                                      is one trial, amici2022_multitrial() the
                                      published per-pixel procedure.
  (joint_nlls()  is NOT a published configuration: it is the same joint
                 estimator on the caller's band window with wide bounds, used
                 only by the verification scripts to compare estimators on a
                 common window.)
  ProfileML      profile_retrieval()  This work (profile_retrieval.py):
                                      separable maximum-likelihood profile
                                      over T, global 1-D search, explicit
                                      covariance, profile-likelihood
                                      interval, censored saturated bands.
                                      Paired with the scene transmittance
                                      estimate of fire_transmittance.py
                                      in run_pipeline().

REMOVED. Two further families were implemented, tested and then withdrawn,
because on this data they were not supported by the residuals:

  * a two-thermal-component (flaming + smouldering) fit, after Eckmann et
    al. (2008, 2009) and Dennison & Matheson (2011);
  * a continuous temperature distribution p(T) integrated against Planck,
    after the full-physics VSWIR inversions.

Both are legitimate published methods. They were dropped here because they
carry more freedom than these spectra can constrain: against single-
component synthetic truth their mean absolute temperature bias was 19.2 K
and 16.4 K respectively, against 3.1 K for Amici2022 and 2.5 K for
Dennison2006, and on real EnMAP fire pixels the distribution fit collapsed
onto the single-component solution (sigma driven to its lower bound, median
difference 0.0 K) - i.e. it recovered the simpler model rather than adding
information. See algorithm_verification/VERIFICATION.md section 10. The
`n_sources` argument of planck_nlls() still accepts more than one component
for anyone who wants to re-test that on a scene with resolvable
flaming/smouldering structure; it is simply not offered as a routine method
HERE.

The two-component fit was not dropped everywhere, though - see
Fire_analysis_with_EnMAP.two_component_escalation() and VERIFICATION.md
section 15 for where it belongs instead: not as a routine method run over
every pixel (which is what made it over-parameterised above), but as a
targeted re-fit offered only to pixels whose single-component temperature
is pinned at the search bound (VERIFICATION.md section 14 - a pixel too
bright for one Planck curve to explain at all), and kept only when a
nested-model significance test confirms the second component earns its
place on that specific pixel. That is the advance this repository's own
retrieval makes beyond the published Amici2022 method it otherwise
implements faithfully.

EMISSIVITY. Every method takes an `emissivity` argument, and every one of
them is INVARIANT in temperature with respect to it, because a
wavelength-independent (grey-body) emissivity enters only as the product
f * eps:

    f * eps * B(lambda, T)

so eps and the fire fraction are exactly degenerate - see
`emissivity_degeneracy_demo()`, which verifies this numerically. The
consequence for reporting is that what these methods retrieve is the
EFFECTIVE fire fraction f_eff = f * eps, not f, and a paper should say so.
Breaking the degeneracy needs a spectrally varying eps(lambda) whose shape
differs from a Planck curve; the literature does not support assuming one
for flames (Veraverbeke et al. 2018 note that flame emissivity depends on
path length through the flame and soot content, citing Giglio & Kendall
2001 and Riggan et al. 2004, and advise treating the blackbody assumption
with caution, especially for cooler fires).

Units: radiance in W m^-2 sr^-1 nm^-1 and wavelength in nm throughout,
matching Fire_analysis_with_EnMAP.Planck_function().
"""

import numpy as np
from scipy import optimize

from Fire_analysis_with_EnMAP import Planck_function, _PLANCK_C1, _PLANCK_C2
from profile_retrieval import profile_retrieval as _profile_retrieval


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _planck_nm(wavelength_nm, T, emissivity=1.0):
    """Planck radiance, wavelength in nm, result in W m^-2 sr^-1 nm^-1."""
    return Planck_function(np.asarray(wavelength_nm, dtype=float) * 1e-3, float(T),
                           float(emissivity))


# --------------------------------------------------------------------------
# Dozier (1981) two-channel solution, evaluated on two SWIR channels
# --------------------------------------------------------------------------

def dozier_bispectral(radiance, wavelength, background, lam1=1600.0, lam2=2200.0,
                      emissivity=1.0, T_bounds=(350.0, 2500.0), saturated=None):
    """Two-channel exact solution for (T, f).

    Dozier (1981), RSE 11:221, Eqs. 1-2, writes the radiance of a pixel
    containing a hot target of temperature T_t over a fraction p of its area
    and a background of known temperature T_b over the rest as

        L(lam) = p * B(lam, T_t) + (1 - p) * B(lam, T_b)

    in each of two channels (3.7 and 11 um for AVHRR), and solves the pair
    for (T_t, p). Here the two channels are SWIR channels and the background
    term B(lam, T_b) is replaced by the measured background radiance of the
    scene, which in the SWIR is reflected sunlight rather than thermal
    emission; this is the substitution Dennison et al. (2006, Sect. 1)
    describe for applying the method to imaging-spectrometer radiance. With
    f = p and eps the fire emissivity the model per band is

        L_obs(lam) = f * eps * B(lam, T) + (1 - f) * L_bkg(lam)

    Rearranged, f cancels from the ratio of the two bands' fire excess:

        [L_obs(l1) - L_bkg(l1)] / [L_obs(l2) - L_bkg(l2)]
            = [eps*B(l1,T) - L_bkg(l1)] / [eps*B(l2,T) - L_bkg(l2)]      (D1)

    which is a single monotonic equation in T over the SWIR (the ratio of
    Planck curves at two wavelengths is strictly monotonic in T), so it has
    at most one physical root. Solved by Brent bracketing; f then follows
    from either band.

    This is the classical method and it is exact - no optimiser, no
    initialisation, no background library. Its weakness is that it uses two
    bands out of ~200 and therefore inherits their noise directly, and it
    needs a background estimate it cannot itself provide. The pipeline gives
    each pixel the median of its neighbouring non-fire pixels (the
    background-temperature estimate of the original AVHRR applications),
    falling back to the scene background.

    `saturated` (boolean mask over `wavelength`, e.g. the L1B saturation /
    quality flags): the two equations assume measured radiances, so a
    pixel whose either channel is saturated or flagged has no solution
    (a clipped radiance would return a wrong temperature, not a bound) and
    is returned as not converged with reason 'channel saturated'.

    Returns dict with T_K, fire_fraction, method, converged.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    radiance = np.asarray(radiance, dtype=float)
    background = np.asarray(background, dtype=float)

    i1 = int(np.argmin(np.abs(wavelength - lam1)))
    i2 = int(np.argmin(np.abs(wavelength - lam2)))
    l1, l2 = wavelength[i1], wavelength[i2]
    d1 = radiance[i1] - background[i1]
    d2 = radiance[i2] - background[i2]

    nan = {'T_K': np.nan, 'fire_fraction': np.nan, 'method': 'Dozier (1981) two-channel',
           'converged': False, 'bands_nm': (float(l1), float(l2))}
    if saturated is not None:
        sat = np.asarray(saturated, dtype=bool)
        if sat[i1] or sat[i2]:
            return dict(nan, reason='channel saturated')
    # The fire excess must be positive in both bands for a hot sub-pixel
    # source; if it is not, this pixel does not satisfy the model.
    if not (np.isfinite(d1) and np.isfinite(d2)) or d1 <= 0 or d2 <= 0:
        return nan

    ratio_obs = d1 / d2

    def g(T):
        b1 = emissivity * _planck_nm(l1, T) - background[i1]
        b2 = emissivity * _planck_nm(l2, T) - background[i2]
        # Below the temperature at which the hot component out-radiates the
        # background in the longer band, the model has no physical solution;
        # mark it rather than dividing through a sign change.
        if b2 <= 0:
            return np.nan
        return b1 / b2 - ratio_obs

    # Scan for a sign change instead of trusting the interval endpoints. The
    # denominator eps*B(l2,T) - L_bkg(l2) passes through zero somewhere
    # inside any generous T range, so g is undefined on the cold part of the
    # interval and brentq cannot be handed the raw bounds.
    grid = np.linspace(float(T_bounds[0]), float(T_bounds[1]), 400)
    vals = np.array([g(T) for T in grid])
    ok = np.isfinite(vals)
    T = None
    idx = np.where(ok[:-1] & ok[1:] & (np.sign(vals[:-1]) * np.sign(vals[1:]) < 0))[0]
    if idx.size:
        a, b = grid[idx[0]], grid[idx[0] + 1]
        try:
            T = optimize.brentq(g, a, b, xtol=1e-3, maxiter=200)
        except (ValueError, RuntimeError):
            T = None
    if T is None:
        return nan

    denom = emissivity * _planck_nm(l2, T) - background[i2]
    f = d2 / denom if denom > 0 else np.nan
    model = (f * emissivity * _planck_nm(wavelength, T) + (1.0 - f) * background
             if np.isfinite(f) else np.full_like(wavelength, np.nan))
    rmse = float(np.sqrt(np.mean((model - radiance) ** 2))) if np.isfinite(f) else np.nan
    return {'T_K': float(T), 'fire_fraction': float(f),
            'method': 'Dozier (1981) two-channel', 'converged': bool(np.isfinite(f)),
            'T_at_bound': _bound_flag(T, T_bounds[0], T_bounds[1]),
            'bands_nm': (float(l1), float(l2)), 'model': model, 'rmse': rmse}


def _bound_flag(T, lo, hi, rel_tol=1e-6, abs_tol=1e-6):
    """True when a fitted temperature is sitting on the edge of its own
    search interval.

    A bounded optimiser always returns a number, and when the data pull the
    solution past the far end of the interval that number is the bound
    itself - not an estimate of anything. Such a value must not be reported
    as a measurement: it says "at least this hot", and its true value is
    unconstrained above. It also silently biases any aggregate that includes
    it (a median, a bias against Landsat, a method-agreement statistic),
    always in the same direction, so it has to be visible in the output
    rather than left for the reader to spot.

    Measured on the two ordered scenes: 1 of 16 Yalu pixels and 2 of 4
    Mojave pixels come back pinned at the upper bound for both Dennison2006
    (its 1800 K grid top) and Amici2022 (its 2000 K NLLS bound)."""
    if not np.isfinite(T):
        return False
    span = max(abs(hi - lo), 1.0)
    return bool(T <= lo + max(abs_tol, rel_tol * span) or
                T >= hi - max(abs_tol, rel_tol * span))


# --------------------------------------------------------------------------
# Dennison et al. (2006): MESMA with a Planck emitted-radiance library
# --------------------------------------------------------------------------

# Emitted-radiance library of Dennison et al. (2006), Sect. 4: "a temperature
# range of 500-1500 K ... Emitted radiance was modeled at increments of 10 K
# between 500 and 1500 K ... creating a total of 101 emitted radiance
# endmembers."
DENNISON2006_T_GRID = np.arange(500.0, 1500.0 + 1e-6, 10.0)

# Channels retained by Dennison et al. (2006), Sect. 4. They omitted AVIRIS
# channels centred below 1200 nm, between 1320 and 1510 nm, between 1775 and
# 1975 nm, and above 2365 nm; everything else was modelled.
DENNISON2006_CHANNEL_WINDOWS_NM = ((1200.0, 1320.0), (1510.0, 1775.0), (1975.0, 2365.0))

# AVIRIS encoded radiance (Dennison et al. 2006, Sect. 4): radiance in
# uW cm^-2 nm^-1 sr^-1 multiplied by a gain of 500 for channels 1-160 and
# 1000 for channels 161-224 (the fourth AVIRIS spectrometer, from about
# 1.88 um). Model RMSE was computed in encoded units, so in the model
# selection every channel beyond the boundary carries twice the weight of a
# channel below it. The boundary lies inside the omitted 1775-1975 nm gap,
# so its exact placement does not matter.
DENNISON2006_ENCODED_GAIN = (500.0, 1000.0)
DENNISON2006_ENCODED_GAIN_BOUNDARY_NM = 1880.0
_W_M2_SR_NM_TO_UW_CM2_SR_NM = 100.0


def dennison_channel_mask(wavelength):
    """Boolean mask of the channels Dennison et al. (2006) modelled."""
    w = np.asarray(wavelength, dtype=float)
    m = np.zeros(w.shape, dtype=bool)
    for lo, hi in DENNISON2006_CHANNEL_WINDOWS_NM:
        m |= (w >= lo) & (w <= hi)
    return m


def dennison_mesma(radiance, wavelength, reflected_endmembers, T_grid=None, emissivity=1.0,
                   transmittance=None, saturated=None, apply_channel_subset=True,
                   encoded_weights=True):
    """Dennison et al. (2006), RSE 100:212-222, as published.

    Model (their Eq. 9), one reflected-radiance endmember L_r, one emitted
    endmember L_et and shade L_s = 0:

        L_m(lam) = f_r L_r(lam) + f_et L_et(lam) + f_s L_s + eps_err

    with f_r + f_et + f_s = 1. Because the shade radiance is zero the pair
    (f_r, f_et) is found by unconstrained least squares (their singular-value
    decomposition) and f_s = 1 - f_r - f_et. The emitted endmembers are
    Planck curves (their Eq. 3, unit emissivity) for T = 500, 510, ...,
    1500 K, 101 in all; they modelled these with MODTRAN at sensor level,
    i.e. multiplied by the one-way atmospheric transmittance of their
    scene, which `transmittance` reproduces when a spectrum is supplied
    (unit transmittance otherwise). Every (reflected, emitted) pair is
    tested (606 models for their six reflected endmembers); "only models
    with land cover fraction and fire fractional area summing to between 0
    and 1 were considered", and "the single model with the lowest RMSE was
    selected". RMSE and residuals were not otherwise constrained.

    Channels: their subset (dennison_channel_mask) is applied when
    apply_channel_subset is True, and "only unsaturated channels were
    modeled", which `saturated` (a boolean mask over `wavelength`)
    implements. RMSE was computed in AVIRIS encoded radiance, whose gain is
    twice as large beyond 1.88 um; encoded_weights=True reproduces that
    weighting in the model selection and returns the encoded RMSE alongside
    the physical one.

    `reflected_endmembers` plays the part of their EAR-selected library
    (six class endmembers); the caller decides what goes in it.

    Returns dict with T_K, fire_fraction (= f_et), reflected_fraction,
    shade_fraction, rmse (physical units over the fitted channels),
    rmse_encoded, endmember_index, n_bands_fit, model (full-length array,
    NaN outside the fitted channels), T_at_bound, converged.
    """
    w = np.asarray(wavelength, dtype=float)
    y = np.asarray(radiance, dtype=float)
    R = np.atleast_2d(np.asarray(reflected_endmembers, dtype=float))
    T_grid = DENNISON2006_T_GRID if T_grid is None else np.asarray(T_grid, dtype=float)
    method_name = 'Dennison et al. (2006) MESMA, Planck emitted-radiance library'

    m = np.isfinite(y) & np.all(np.isfinite(R), axis=0)
    if apply_channel_subset:
        m &= dennison_channel_mask(w)
    if saturated is not None:
        m &= ~np.asarray(saturated, dtype=bool)
    nan = {'T_K': np.nan, 'fire_fraction': np.nan, 'reflected_fraction': np.nan,
           'shade_fraction': np.nan, 'rmse': np.nan, 'rmse_encoded': np.nan,
           'endmember_index': -1, 'n_bands_fit': int(m.sum()), 'model': np.full(w.shape, np.nan),
           'method': method_name, 'T_at_bound': False, 'converged': False,
           'n_endmembers': int(R.shape[0]) + 2, 'n_models_tested': 0}
    if m.sum() < 3:
        return nan

    ww, yw, Rw = w[m], y[m], R[:, m]
    tau = np.ones(ww.shape) if transmittance is None else np.asarray(transmittance, dtype=float)[m]
    if encoded_weights:
        gain = np.where(ww < DENNISON2006_ENCODED_GAIN_BOUNDARY_NM,
                        DENNISON2006_ENCODED_GAIN[0], DENNISON2006_ENCODED_GAIN[1])
        gain = gain * _W_M2_SR_NM_TO_UW_CM2_SR_NM
    else:
        gain = np.ones(ww.shape)

    # Least squares in encoded units: residual_enc = gain * residual.
    Rg, yg = Rw * gain, yw * gain
    a = np.einsum('jn,jn->j', Rg, Rg)
    d = Rg @ yg
    best = None
    n_tested = 0
    for T in T_grid:
        e = emissivity * tau * _planck_nm(ww, T)
        eg = e * gain
        b = Rg @ eg
        c = float(eg @ eg)
        g = float(eg @ yg)
        det = a * c - b * b
        with np.errstate(divide='ignore', invalid='ignore'):
            f_r = (c * d - b * g) / det
            f_e = (a * g - b * d) / det
        tot = f_r + f_e
        ok = np.isfinite(tot) & (tot >= 0.0) & (tot <= 1.0)
        n_tested += len(tot)
        if not ok.any():
            continue
        resid = yg[None, :] - f_r[:, None] * Rg - f_e[:, None] * eg[None, :]
        sse = np.sum(resid ** 2, axis=1)
        sse[~ok] = np.inf
        j = int(np.argmin(sse))
        if best is None or sse[j] < best[0]:
            best = (float(sse[j]), float(T), float(f_r[j]), float(f_e[j]), j)
    if best is None:
        nan['n_models_tested'] = n_tested
        return nan

    sse, T, f_r, f_e, j = best
    model_w = f_r * Rw[j] + f_e * emissivity * tau * _planck_nm(ww, T)
    model = np.full(w.shape, np.nan)
    model[m] = model_w
    return {'T_K': T, 'fire_fraction': f_e, 'reflected_fraction': f_r,
            'shade_fraction': 1.0 - f_r - f_e,
            'rmse': float(np.sqrt(np.mean((model_w - yw) ** 2))),
            'rmse_encoded': float(np.sqrt(sse / len(yw))),
            'endmember_index': int(j), 'n_bands_fit': int(len(yw)), 'model': model,
            'method': method_name,
            'T_at_bound': _bound_flag(T, float(T_grid[0]), float(T_grid[-1])),
            'converged': True, 'n_endmembers': int(R.shape[0]) + 2,
            'n_models_tested': int(n_tested)}


# Name kept for older callers; the implementation is the published one.
dennison_library = dennison_mesma


# --------------------------------------------------------------------------
# Dennison et al. (2006): the three scene-dependent inputs of the method
# --------------------------------------------------------------------------
# Their Sect. 4 prepares the MESMA inputs in three steps that depend on the
# scene: (i) solar path radiance, modelled with MODTRAN, "fit to sampled path
# radiance found in deeply shadowed areas of the scene, and then subtracted
# from each spectrum"; (ii) six reflected endmembers, one per land-cover
# class, each selected by minimum Endmember Average RMSE (EAR; Dennison &
# Roberts 2003) from 200 image spectra of that class; (iii) emitted
# endmembers modelled with MODTRAN at sensor level, i.e. Planck radiance
# times the one-way transmittance of the scene's atmosphere. Without MODTRAN
# and analyst-drawn class polygons, the image-based equivalents below keep
# the structure of each step: (i) the path radiance is the median spectrum
# of the darkest 0.1% of the scene's background pixels (the 'deeply
# shadowed' sample; a dark-object estimate, which over-estimates the path
# term by whatever those pixels still reflect, small beyond 1200 nm); (ii)
# the six classes are k-means clusters of the background pool on the
# method's channels, 200 spectra drawn from each, EAR within each; (iii)
# the one-way transmittance is exp(-m g) with g the scene's own two-way
# gas optical depth (fire_transmittance.background_absorption_template) and
# m = sec(view zenith) / (sec(sun zenith) + sec(view zenith)) the one-way
# share of the two-way air mass. The method itself (dennison_mesma) is
# unchanged; these only supply its inputs.

DENNISON2006_N_CLASSES = 6        # Sect. 4: six land-cover classes
DENNISON2006_N_PER_CLASS = 200    # Sect. 4: 200 image spectra per class
DENNISON2006_DARK_FRACTION = 0.001


def _dennison_gain(wavelength):
    w = np.asarray(wavelength, dtype=float)
    return np.where(w < DENNISON2006_ENCODED_GAIN_BOUNDARY_NM, DENNISON2006_ENCODED_GAIN[0],
                    DENNISON2006_ENCODED_GAIN[1]) * _W_M2_SR_NM_TO_UW_CM2_SR_NM


def dennison_path_radiance(pool, wavelength, dark_fraction=DENNISON2006_DARK_FRACTION):
    """Image-based solar path radiance (full band axis): the median spectrum
    of the darkest `dark_fraction` of `pool` by mean radiance over the
    Dennison et al. (2006) channels. Never negative."""
    w = np.asarray(wavelength, dtype=float)
    X = np.atleast_2d(pool)
    m = dennison_channel_mask(w)
    bright = np.zeros(X.shape[0])
    for j in np.flatnonzero(m):          # band by band: the pool can be ~10^6 spectra
        bright += np.asarray(X[:, j], dtype=float)
    bright /= max(int(m.sum()), 1)
    ok = np.isfinite(bright)
    n = max(int(round(dark_fraction * ok.sum())), 10)
    idx = np.flatnonzero(ok)[np.argsort(bright[ok])[:n]]
    return np.maximum(np.median(np.asarray(X[idx], dtype=float), axis=0), 0.0)


def endmember_average_rmse(spectra, weights=None):
    """EAR (Dennison & Roberts 2003) of every spectrum in one class: the
    mean RMSE with which it models each OTHER spectrum of the class in a
    two-endmember model (the spectrum scaled by its fraction, plus shade at
    zero radiance; least-squares fraction). `weights` multiply the
    residuals (encoded-radiance gain). Returns (n,) array."""
    S = np.atleast_2d(np.asarray(spectra, dtype=float))
    g = np.ones(S.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    S = S * g
    n, b = S.shape
    G = S @ S.T                              # G[i, j] = s_i . s_j
    nn = np.diag(G)
    # residual^2 of j modelled by i: |s_j|^2 - (s_i.s_j)^2 / |s_i|^2
    with np.errstate(divide='ignore', invalid='ignore'):
        sse = nn[None, :] - G ** 2 / nn[:, None]
    rmse = np.sqrt(np.maximum(sse, 0.0) / b)
    np.fill_diagonal(rmse, 0.0)
    return rmse.sum(axis=1) / max(n - 1, 1)


def dennison_ear_endmembers(pool, wavelength, n_classes=DENNISON2006_N_CLASSES,
                            n_per_class=DENNISON2006_N_PER_CLASS, path_radiance=None, rng=None):
    """Six reflected endmembers, one per class, by minimum EAR (see the
    block comment above). Classes: k-means (k-means++ start, fixed seed) of
    `pool` on the method's channels in encoded radiance, after removing
    `path_radiance`. Returns dict(endmembers (k, n_bands) full axis, path
    radiance removed; class_sizes; ear (k,))."""
    from scipy.cluster.vq import kmeans2
    rng = rng if rng is not None else np.random.default_rng(0)
    w = np.asarray(wavelength, dtype=float)
    m = dennison_channel_mask(w)
    X = np.atleast_2d(pool)
    take = rng.choice(X.shape[0], size=min(20000, X.shape[0]), replace=False)
    X = np.asarray(X[np.sort(take)], dtype=float)
    if path_radiance is not None:
        X = X - np.asarray(path_radiance, dtype=float)[None, :]
    X = X[np.all(np.isfinite(X[:, m]), axis=1)]
    g = _dennison_gain(w[m])
    Z = X[:, m] * g
    _, lab = kmeans2(Z, int(n_classes), minit='++', seed=int(rng.integers(2 ** 31 - 1)))
    E, sizes, ears = [], [], []
    for c in range(int(n_classes)):
        idx = np.flatnonzero(lab == c)
        if len(idx) < 2:
            continue
        if len(idx) > n_per_class:
            idx = rng.choice(idx, size=n_per_class, replace=False)
        ear = endmember_average_rmse(X[idx][:, m], weights=g)
        j = int(np.argmin(ear))
        E.append(X[idx[j]]); sizes.append(int((lab == c).sum())); ears.append(float(ear[j]))
    return {'endmembers': np.array(E), 'class_sizes': np.array(sizes), 'ear': np.array(ears)}


def one_way_transmittance(pool, wavelength, sun_zenith_deg=None, view_zenith_deg=None):
    """One-way atmospheric transmittance of the scene (full band axis,
    unity outside the gas-absorption regions), exp(-m g): g the two-way
    gas optical depth of the scene's own background, m the one-way share
    of the two-way air mass (0.5 when the geometry is unknown).
    Returns (tau, m)."""
    import fire_transmittance as ft
    w = np.asarray(wavelength, dtype=float)
    if sun_zenith_deg is not None and view_zenith_deg is not None:
        ms = 1.0 / np.cos(np.radians(float(sun_zenith_deg)))
        mv = 1.0 / np.cos(np.radians(float(view_zenith_deg)))
        share = mv / (ms + mv)
    else:
        share = 0.5
    # only the anchor and absorber bands enter the template (the deep water
    # bands elsewhere can be <= 0, which the template's log would reject)
    sel = np.zeros(len(w), bool)
    for lo, hi in tuple(ft.CONTINUUM_ANCHORS) + tuple(ft.ABSORBER_REGIONS.values()):
        sel |= (w >= lo) & (w <= hi)
    tpl = ft.background_absorption_template(np.asarray(np.atleast_2d(pool)[:, sel], dtype=float), w[sel])
    in_region = np.zeros(int(sel.sum()), bool)
    for mk in tpl['region_masks'].values():
        in_region |= mk
    g = np.zeros(len(w))
    g[sel] = np.where(in_region, tpl['template'], 0.0)
    return np.exp(-share * g), float(share)


def dennison_scene_inputs(pool, wavelength, sun_zenith_deg=None, view_zenith_deg=None, rng=None):
    """The three image-based inputs above, ready for run_methods_on_pixels:
    dict(path_radiance, endmembers, transmittance, one_way_share,
    class_sizes, ear)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    Lp = dennison_path_radiance(pool, wavelength)
    ear = dennison_ear_endmembers(pool, wavelength, path_radiance=Lp, rng=rng)
    X = np.atleast_2d(pool)
    sub = np.asarray(X[rng.choice(X.shape[0], size=min(3000, X.shape[0]), replace=False)], dtype=float)
    tau, share = one_way_transmittance(sub, wavelength, sun_zenith_deg, view_zenith_deg)
    return {'path_radiance': Lp, 'endmembers': ear['endmembers'], 'transmittance': tau,
            'one_way_share': share, 'class_sizes': ear['class_sizes'], 'ear': ear['ear']}


# --------------------------------------------------------------------------
# Amici et al. (2022): joint nonlinear fit, m background draws, N trials
# --------------------------------------------------------------------------

# The published configuration. Values with a section reference are from
# Amici et al. (2022), JGR Biogeosciences 127:e2022JG007055; the rest are
# from the code released with the paper (Fire_analysis_with_PRISMA.py,
# Zenodo 10.5281/zenodo.7273879), which the paper's Data Availability
# statement names as the software used.
AMICI2022_WINDOW_NM = (1400.0, 1900.0)          # code: 1400 < lambda < 1900 (Sect. 3.2.3 quotes 1400-1800)
AMICI2022_ABSORPTION_BLANK = 0.003               # code: Planck term zeroed where L < 0.003 W m-2 sr-1 nm-1 (Sect. 3.2.2)
AMICI2022_T_BOUNDS = (100.0, 2000.0)             # code: T_min = 100, T_max = 2000
AMICI2022_FIRE_ABUNDANCE_BOUNDS = (0.05, 1.0)    # code: bounds (0.05, 1) on the unnormalised fire abundance
AMICI2022_BACKGROUND_ABUNDANCE_BOUNDS = (0.0, 1.0)
AMICI2022_N_BACKGROUND = 20                      # Sect. 4.2: m = 20
AMICI2022_N_TRIALS = 15                          # Sect. 4.2: N = 15
AMICI2022_TOL = 1e-8                             # code: gtol = ftol = xtol = 1e-8, method 'trf'


def amici_window_mask(wavelength, window_nm=AMICI2022_WINDOW_NM):
    """Boolean mask of the fit window of Amici et al. (2022), open interval."""
    w = np.asarray(wavelength, dtype=float)
    return (w > window_nm[0]) & (w < window_nm[1])


def planck_nlls(radiance, wavelength, background_endmembers, n_sources=1,
                emissivity=1.0, T_bounds=AMICI2022_T_BOUNDS, rng=None,
                window_nm=AMICI2022_WINDOW_NM,
                absorption_blank_threshold=AMICI2022_ABSORPTION_BLANK,
                fire_abundance_bounds=AMICI2022_FIRE_ABUNDANCE_BOUNDS,
                tol=AMICI2022_TOL, max_nfev=None, analytic_jacobian=False):
    """One trial of Amici et al. (2022): the joint nonlinear least-squares
    fit of their Eq. 4 to one spectrum with one draw of background spectra.

        L_LM(lam) = sum_i p_i,fire L_fire(T_i; lam) + sum_j p_j,bkg L_j,bkg(lam)

    with L_fire the Planck function of their Eq. 3 (emissivity eps) and the
    weights constrained to sum to one. As in the released code the sum-to-one
    constraint is imposed by normalising every abundance by their total
    inside the model, the fit runs over 1400 < lam < 1900 nm, the Planck term
    is set to zero in the channels where the measured radiance is below
    0.003 W m-2 sr-1 nm-1 (the water-vapour bands, Sect. 3.2.2: "set to zero
    the signal in correspondence of the SWIR atmospheric water absorption
    bands"), the starting point is a uniform random temperature in
    [T_min, T_max] with fire abundance 0.5 and background abundances
    1/(2m), the bounds are [100, 2000] K, [0.05, 1] for the fire abundance
    and [0, 1] for the background abundances, and scipy's trust-region
    reflective solver runs with gtol = ftol = xtol = 1e-8 and its default
    evaluation budget. A failed fit returns converged=False, which the
    published procedure answers by redrawing the background
    (amici2022_multitrial).

    The defaults reproduce the published configuration; the keyword
    arguments exist so the verification scripts can vary one element at a
    time. Returns dict with T_K (hottest), fire_fraction (normalised),
    all_T_K, all_fractions, rmse (over the fitted channels), model
    (full-length, NaN outside the fitted channels), T_at_bound, converged.
    """
    w = np.asarray(wavelength, dtype=float)
    y = np.asarray(radiance, dtype=float)
    B = np.atleast_2d(np.asarray(background_endmembers, dtype=float))
    n_bkg = B.shape[0]
    rng = rng if rng is not None else np.random.default_rng(0)
    method_name = (f'Amici et al. (2022) joint Planck-mixture NLLS (n={n_sources})' if n_sources == 1
                   else f'Amici et al. (2022) joint NLLS, {n_sources} thermal components')
    nan = {'T_K': np.nan, 'fire_fraction': np.nan, 'all_T_K': [], 'all_fractions': [],
           'rmse': np.nan, 'model': np.full(w.shape, np.nan), 'method': method_name,
           'T_at_bound': False, 'converged': False, 'n_bands_fit': 0}

    m = np.isfinite(y) & np.all(np.isfinite(B), axis=0)
    if window_nm is not None:
        m &= amici_window_mask(w, window_nm)
    if m.sum() == 0:
        return nan
    ww, yw, Bw = w[m], y[m], B[:, m]
    blank = (yw < absorption_blank_threshold) if absorption_blank_threshold is not None \
        else np.zeros(ww.shape, dtype=bool)
    nan['n_bands_fit'] = int(len(yw))

    # The parameter-independent factors of _planck_nm(), hoisted out of the
    # optimiser's inner loop. curve_fit evaluates model() once per residual
    # and once per parameter for every finite-difference Jacobian - 22
    # evaluations per step in the published configuration - always on the
    # same `ww`, so the micrometre conversion, the fifth power and the c1
    # factor are computed once per pixel instead of once per evaluation.
    # The arithmetic left inside is unchanged term for term and in the same
    # association order, so the modelled spectrum is bit-identical to
    # emissivity * _planck_nm(x, T) (checked in test_correctness.py).
    _x_um = ww * 1e-3
    _planck_scale = 1.0 * (_PLANCK_C1 / (_x_um ** 5))
    _blank_idx = np.flatnonzero(blank)
    if _blank_idx.size == 0:
        _blank_idx = None

    # The Planck term depends on the temperatures alone, and a
    # finite-difference Jacobian perturbs one parameter at a time: 21 of the
    # 22 evaluations that build it leave every temperature where it was. The
    # blanked Planck vector is therefore cached per temperature, so the
    # exponential runs once per Jacobian rather than 22 times. A cached
    # array is never written to after it is blanked (the multiply below
    # returns a new one), and the value returned is unchanged either way.
    _planck_cache = {}

    def model(x, *args):
        args = np.asarray(args, dtype=float)
        norm = args[n_sources:].sum()
        out = None
        hoisted = x is ww                     # the only grid curve_fit uses
        for i in range(n_sources):
            if hoisted:
                key = (i, float(args[i]))
                planck = _planck_cache.get(key)
                if planck is None:
                    planck = emissivity * (_planck_scale
                                           * (1 / (np.exp(_PLANCK_C2 / (args[i] * _x_um)) - 1)))
                    if _blank_idx is not None:
                        planck[_blank_idx] = 0.0
                    if len(_planck_cache) >= 2 * n_sources + 2:
                        _planck_cache.clear()
                    _planck_cache[key] = planck
            else:
                planck = emissivity * _planck_nm(x, args[i])
                planck[blank] = 0.0
            term = (args[n_sources + i] / norm) * planck
            out = term if out is None else out + term
        return out + (args[2 * n_sources:] / norm) @ Bw

    def jac(x, *args):
        """Exact derivative of model() - an alternative to the finite
        difference curve_fit computes by default, see `analytic_jacobian`.

        With g the unnormalised mixture, S the sum of the abundances and
        f = g / S the modelled spectrum,

            df/dT_i    = (p_i / S) dB(T_i)/dT,  dB/dT = B z e^z /((e^z - 1) T)
            df/dp_i    = (B_i - f) / S          (fire abundances)
            df/dp_j    = (b_j - f) / S          (background abundances)

        which costs three array operations against one full model
        evaluation per parameter."""
        args = np.asarray(args, dtype=float)
        norm = args[n_sources:].sum()
        planck = np.empty((n_sources, len(ww)))
        dplanck = np.empty((n_sources, len(ww)))
        for i in range(n_sources):
            z = _PLANCK_C2 / (args[i] * _x_um)
            ez = np.exp(z)
            b = emissivity * (_planck_scale * (1 / (ez - 1)))
            db = b * (z * ez / (ez - 1)) / args[i]
            if _blank_idx is not None:
                b[_blank_idx] = 0.0
                db[_blank_idx] = 0.0
            planck[i], dplanck[i] = b, db
        f = ((args[n_sources:2 * n_sources] / norm) @ planck
             + (args[2 * n_sources:] / norm) @ Bw)
        J = np.empty((len(ww), 2 * n_sources + n_bkg))
        J[:, :n_sources] = ((args[n_sources:2 * n_sources] / norm)[:, None]
                            * dplanck).T
        J[:, n_sources:2 * n_sources] = ((planck - f) / norm).T
        J[:, 2 * n_sources:] = ((Bw - f) / norm).T
        return J

    # One uniform draw, repeated for every source, as in the released code.
    T0 = float(rng.uniform(T_bounds[0], T_bounds[1]))
    x0 = [T0] * n_sources + [0.5] * n_sources + [1.0 / (2 * n_bkg)] * n_bkg
    lo = ([T_bounds[0]] * n_sources + [fire_abundance_bounds[0]] * n_sources
          + [AMICI2022_BACKGROUND_ABUNDANCE_BOUNDS[0]] * n_bkg)
    hi = ([T_bounds[1]] * n_sources + [fire_abundance_bounds[1]] * n_sources
          + [AMICI2022_BACKGROUND_ABUNDANCE_BOUNDS[1]] * n_bkg)
    try:
        popt, _ = optimize.curve_fit(model, ww, yw, p0=x0, bounds=(lo, hi), method='trf',
                                     gtol=tol, ftol=tol, xtol=tol, max_nfev=max_nfev,
                                     **({'jac': jac} if analytic_jacobian else {}))
    except Exception:
        return nan

    temps = np.asarray(popt[:n_sources], dtype=float)
    fr = np.asarray(popt[n_sources:], dtype=float)
    fr = fr / np.sum(fr)
    order = np.argsort(temps)[::-1]
    temps = temps[order]
    fire_fr = fr[:n_sources][order]
    fitted = model(ww, *popt)
    full = np.full(w.shape, np.nan)
    full[m] = fitted
    return {'T_K': float(temps[0]), 'fire_fraction': float(fire_fr[0]),
            'all_T_K': [float(t) for t in temps],
            'all_fractions': [float(v) for v in fire_fr],
            'rmse': float(np.sqrt(np.mean((fitted - yw) ** 2))), 'model': full,
            'method': method_name, 'n_bands_fit': int(len(yw)),
            'T_at_bound': _bound_flag(temps[0], T_bounds[0], T_bounds[1]),
            'converged': True}


def amici2022_multitrial(radiance, wavelength, background_pool, n_trials=AMICI2022_N_TRIALS,
                         n_background=AMICI2022_N_BACKGROUND, emissivity=1.0, rng=None,
                         max_redraws=5, **fit_kwargs):
    """The per-pixel procedure of Amici et al. (2022), Sect. 3.2.2 and 4.2:
    "we solve Equation 4 N times, selecting each time m random background
    points within those having the proper HFDI ... the temperature value is
    provided by averaging the N estimations ... an a-posteriori confidence
    interval", with n = 1 thermal source, m = 20 and N = 15.

    Each trial draws m spectra from `background_pool` with replacement
    (numpy randint in the released code) and runs planck_nlls(); a trial
    whose fit fails is redrawn up to max_redraws times, as in the code. The
    reported temperature and fire fraction are the means over the valid
    trials, with their standard deviation and the 95 % Student-t interval of
    the mean (scipy.stats.t.interval, as in the code). `model` and `rmse`
    are those of the trial whose temperature is closest to the mean.


    Two diagnostics that are NOT part of the published estimate are returned
    alongside it: T_median_K, the median of the trials, and n_trials_cold,
    the number of trials that ended below 300 K. A trial started at a cold
    random temperature (the code draws the start uniformly in
    [T_min, T_max] = [100, 2000] K) can stop in the flat region where the
    Planck term is negligible; the published mean is pulled by such trials
    while the median is not. Both are reported so that the effect can be
    quantified rather than hidden.

    Returns dict with T_K, fire_fraction, std_T_K, CI_T_low, CI_T_high,
    std_fire_fraction, n_trials_valid, trial_T_K, T_median_K, n_trials_cold,
    rmse, model, T_at_bound, converged.
    """
    from scipy import stats as st
    pool = np.atleast_2d(np.asarray(background_pool, dtype=float))
    rng = rng if rng is not None else np.random.default_rng(0)
    method_name = 'Amici et al. (2022), N-trial mean of the joint Planck-mixture NLLS'
    trials = []
    for _ in range(int(n_trials)):
        res = None
        counter = 0
        while res is None and counter < max_redraws:
            idx = rng.integers(pool.shape[0], size=int(n_background))
            r = planck_nlls(radiance, wavelength, pool[idx], emissivity=emissivity, rng=rng,
                            **fit_kwargs)
            if r['converged']:
                res = r
            counter += 1
        if res is not None:
            trials.append(res)
    w = np.asarray(wavelength, dtype=float)
    if not trials:
        return {'T_K': np.nan, 'fire_fraction': np.nan, 'std_T_K': np.nan, 'CI_T_low': np.nan,
                'CI_T_high': np.nan, 'std_fire_fraction': np.nan, 'n_trials_valid': 0,
                'trial_T_K': [], 'T_median_K': np.nan, 'n_trials_cold': 0, 'rmse': np.nan,
                'model': np.full(w.shape, np.nan), 'method': method_name, 'T_at_bound': False,
                'converged': False, 'n_bands_fit': 0}
    T = np.array([r['T_K'] for r in trials])
    P = np.array([r['fire_fraction'] for r in trials])
    T_mean, P_mean = float(np.mean(T)), float(np.mean(P))
    if len(T) > 1:
        ci = st.t.interval(0.95, len(T) - 1, loc=T_mean, scale=st.sem(T))
    else:
        ci = (np.nan, np.nan)
    k = int(np.argmin(np.abs(T - T_mean)))
    T_lo = fit_kwargs.get('T_bounds', AMICI2022_T_BOUNDS)
    return {'T_K': T_mean, 'fire_fraction': P_mean, 'std_T_K': float(np.std(T)),
            'CI_T_low': float(ci[0]), 'CI_T_high': float(ci[1]),
            'std_fire_fraction': float(np.std(P)), 'n_trials_valid': int(len(T)),
            'trial_T_K': [float(t) for t in T], 'T_median_K': float(np.median(T)),
            'n_trials_cold': int(np.sum(T < 300.0)), 'rmse': trials[k]['rmse'],
            'model': trials[k]['model'], 'method': method_name,
            'n_bands_fit': int(trials[k]['n_bands_fit']),
            'T_at_bound': _bound_flag(T_mean, T_lo[0], T_lo[1]), 'converged': True}


def joint_nlls(radiance, wavelength, background_endmembers, n_sources=1,
               emissivity=1.0, T_bounds=(300.0, 2000.0), rng=None):
    """The joint nonlinear least-squares estimator (one Planck term plus a
    linear background mixture, sum-to-one by normalisation) on the CALLER'S
    band window with wide bounds. This is not a published configuration: it
    is the estimator of Amici et al. (2022) stripped of their window, their
    blanking rule and their abundance floor, so that the verification
    scripts can compare estimators on one common window and isolate the
    effect of the fitting method from the effect of the window. Use
    planck_nlls() / amici2022_multitrial() for the published method.

    Returns the same schema as planck_nlls().
    """
    return planck_nlls(radiance, wavelength, background_endmembers, n_sources=n_sources,
                       emissivity=emissivity, T_bounds=T_bounds, rng=rng, window_nm=None,
                       absorption_blank_threshold=None, fire_abundance_bounds=(0.0, 1.0),
                       tol=1e-10, max_nfev=20000)


# --------------------------------------------------------------------------
# background estimation (needed by Dozier (1981), which cannot supply its own)
# --------------------------------------------------------------------------

def background_from_histogram(cube, wavelength, hotspot_index=None,
                              wl_lo=2060.0, wl_hi=2430.0, statistic='median'):
    """Scene background spectrum, from a histogram threshold on the hotspot index.

    Dozier (1981) needs an external estimate of what the pixel would have
    radiated without the fire, and the quality of that estimate goes
    straight into the retrieved temperature. Taking a plain scene median is
    wrong in a smoky or partly burnt scene, because the median is pulled by
    whatever fraction of the scene is already hot or charred.

    Instead, split the scene on its own hotspot-index histogram using Otsu's
    method (`otsu_threshold`, already used elsewhere in this pipeline for
    fire detection) and average only the cold class. Otsu picks the
    threshold maximising between-class variance of the 256-bin histogram, so
    it adapts to the scene rather than to a hard-coded radiance, and it is
    exactly the "threshold on the histogram" the background problem calls
    for.

    Returns (background_spectrum, info) where info records the threshold and
    how many pixels formed the estimate.
    """
    from Fire_analysis_with_EnMAP import compute_hotspot_index, otsu_threshold

    wavelength = np.asarray(wavelength, dtype=float)
    if hotspot_index is None:
        hotspot_index = compute_hotspot_index(np.asarray(cube, dtype=float), wavelength,
                                              wl_lo=wl_lo, wl_hi=wl_hi)

    thr = float(otsu_threshold(hotspot_index.ravel()))
    cold = np.isfinite(hotspot_index) & (hotspot_index <= thr)
    if cold.sum() < 10:
        raise ValueError('Otsu split left too few background pixels to average.')

    rows, cols = np.where(cold)
    # band by band: a (n_cold, n_bands) float64 gather is ~1 GB on a full
    # tile; the per-band statistic is the same number
    reducer = np.nanmedian if statistic == 'median' else np.nanmean
    background = np.array([reducer(np.asarray(cube[:, b, :][rows, cols], dtype=float))
                           for b in range(cube.shape[1])])
    return background, {'otsu_threshold': thr, 'n_background_px': int(cold.sum()),
                        'fraction_of_scene': float(cold.mean()), 'statistic': statistic}


# --------------------------------------------------------------------------
# emissivity
# --------------------------------------------------------------------------

def emissivity_degeneracy_demo(radiance, wavelength, background_endmembers,
                               emissivities=(1.0, 0.9, 0.8, 0.7)):
    """Demonstrate that a grey-body emissivity is exactly degenerate with the
    fire fraction, and therefore cannot be retrieved.

    The thermal term is f * eps * B(lambda, T). Any (f, eps) pair with the
    same product gives an identical spectrum, so the likelihood surface is
    flat along f*eps = const. Retrieved T should be unchanged and retrieved
    f should scale as 1/eps. Returns the table that shows it."""
    rows = []
    for eps in emissivities:
        r = planck_nlls(radiance, wavelength, background_endmembers,
                        emissivity=eps, rng=np.random.default_rng(0))
        rows.append({'emissivity': eps, 'T_K': r['T_K'],
                     'fire_fraction': r['fire_fraction'],
                     'f_times_eps': r['fire_fraction'] * eps if np.isfinite(r['fire_fraction']) else np.nan})
    return rows


# --------------------------------------------------------------------------
# Smoke-transmittance correction (this repository, candidate, unpublished) -
# local-atmosphere / smoke-transmittance
# correction for the fire Planck term
# --------------------------------------------------------------------------
#
# GAP THIS FILLS. None of the published methods above (Dozier 1981, Dennison et al. 2006,
# Amici et al. 2022) put an atmosphere between the fire and the sensor. Each
# reduces, over the fit window, to the additive mixture
#
#     L_obs(lambda) = sum_j q_j * L_bkg,j(lambda) + f * eps * B(lambda, T)      (V2)
#
# where the background endmembers L_bkg,j are REAL measured TOA spectra (so
# they already carry whatever atmosphere sits over wherever they were drawn
# from), while the fire term is an unattenuated Planck curve - implicitly
# assuming the atmosphere between the flame and the sensor is perfectly
# transparent. VERIFICATION.md Section 11.4 already names the missing
# atmospheric-transmittance term as "the largest open item and the honest
# ceiling on accuracy claims" for the retrieval as a whole; this module
# elsewhere treats smoke only qualitatively (_HAZE_RESISTANT_RGB_NM, the
# 'Smoke/haze' background exemplar in Fire_analysis_with_EnMAP.py) and never
# as a quantitative correction. Wildfire scenes routinely have exactly this
# atmosphere - their own smoke plume - sitting between the flame and EnMAP,
# and a plume is spatially compact (tens to hundreds of metres) against an
# EnMAP scene (30 km across), so its effect is local, not scene-wide, and a
# SCENE-WIDE background pool (as all the published methods draw from) cannot see it.
#
# DERIVATION. Let tau(lambda) be the one-way transmittance from ground to
# sensor along this pixel's line of sight and L_path(lambda) the radiance the
# atmosphere adds along it (independent of what is on the ground, the
# standard planar/non-adjacency approximation). For a pixel with fire
# fraction f, cold-surface fractions r_j (sum r_j = 1) drawn from endmembers
# with unattenuated surface radiance L_surf,j:
#
#     L_obs = L_path + tau * [ (1-f) sum_j r_j L_surf,j(lambda) + f eps B(lambda,T) ]   (S1)
#
# A real background-only pixel (f=0) measured under the SAME atmosphere gives
# L_bkg,j(lambda) = L_path(lambda) + tau(lambda) L_surf,j(lambda), i.e.
# tau*L_surf,j = L_bkg,j - L_path. Substituting into (S1) and using
# q_j = (1-f) r_j (so sum q_j = 1-f, matching (V2)'s own parameterisation):
#
#     L_obs = sum_j q_j L_bkg,j(lambda) + f * eps * tau(lambda) * B(lambda,T) + f * L_path(lambda)   (S2)
#
# (S2) reduces EXACTLY to (V2) when tau=1, L_path=0 - a nested generalisation,
# not a different model - and both new terms scale with f, so a pixel with no
# fire (f=0) is completely unaffected, and the correction is largest exactly
# where it matters: on pixels the model says already contain fire. This
# module implements BOTH halves of (S2):
#
#   1. local_background_pool()/local_background_quadrants() - q_j drawn from
#                                             this pixel's own neighbourhood
#                                             instead of a scene-wide pool,
#                                             so a localised plume's added
#                                             path radiance is already inside
#                                             q_j*L_bkg,j, and the quadrant
#                                             split supplies the spatial-
#                                             consistency gate below;
#   2. estimate_relative_transmittance()   - tau(lambda) for the remaining,
#                                             genuinely multiplicative,
#                                             attenuation of the fire's OWN
#                                             emission, AND delta_l_path (S4)
#                                             for the leftover additive term,
#                                             both estimated from the local-
#                                             versus-clear-reference contrast
#                                             behind THREE independent gates
#                                             (spectral shape, spatial
#                                             consistency, VNIR corroboration
#                                             - see that function's docstring
#                                             and VERIFICATION.md Section
#                                             16.5/16.6);
#   3. smoke_corrected_nlls()              - (S2)'s solver with tau(lambda)
#                                             and delta_l_path supplied as
#                                             FIXED inputs, never fitted
#                                             parameters, so it has exactly
#                                             the same degrees of freedom as
#                                             Amici2022 and cannot repeat the
#                                             over-parameterisation failure of
#                                             Section 10.1/15.2;
#   4. apply_local_smoke_correction()/run_smoke_corrected_pixels() - the
#                                             single-pixel orchestrator (with
#                                             graceful fall-back to the plain
#                                             single-Planck fit whenever the
#                                             local pool is too small or no
#                                             gate passes) and the per-scene
#                                             driver, respectively - the
#                                             finalised entry points of the correction.
#
# WHAT tau(lambda) HONESTLY IS. Radiance perturbations from aerosol/smoke are
# a sum of two competing effects - transmittance loss (darkens) and added
# scattered-sunlight path radiance (brightens) - and their net sign depends
# on surface albedo and geometry (Kaufman & Remer 1994; the same reason
# MODIS/imaging-spectrometer "dark target" AOD retrieval deliberately works
# over low-reflectance surfaces). This is not resolved in general; it is
# sidestepped by restricting tau(lambda) estimation to the SWIR fit window
# already used by the retrieval (>=1400 nm), where background reflectance is
# low-to-moderate and aerosol path radiance is already small, so the
# local-vs-clear contrast there is dominated by extinction (the dark-target
# regime), and by requiring an independent VNIR brightening signature before
# trusting that reading (Gate 3). tau(lambda) reported here is therefore a
# RELATIVE excess transmittance - this pixel's line of sight against the
# scene's own typical (already implicitly calibrated-for) condition - not an
# absolute atmospheric correction; exactly the quantity needed to remove a
# LOCALISED excess, and no more than that is claimed. See
# VERIFICATION.md Section 16.
# --------------------------------------------------------------------------

def local_background_pool(cube, wavelength, row, col, exclude_mask,
                          half_window=5, min_pixels=10):
    """Background spectra pool drawn from pixel (row, col)'s own spatial
    neighbourhood, rather than from a scene-wide draw (contrast the scene-wide draw of Dennison et al. (2006) and Amici et al. (2022),
    `bkg_pool`, and the scene-wide Otsu split of `background_from_histogram`).

    `exclude_mask` (n_rows, n_cols) bool, True = do not use this pixel as
    background - pass the scene's fire/hotspot detection mask so a local
    plume's OWN fire pixels, or neighbouring detections, cannot contaminate
    the local background estimate.

    Returns (n_local, n_bands) or None if fewer than `min_pixels` finite,
    non-excluded neighbours survive (edge of scene, or a hotspot cluster
    wider than the window) - callers should fall back to the scene-wide pool
    in that case, exactly as `run_methods_on_pixels` already falls back to
    `np.median(pool, axis=0)` when no background_spectrum is supplied.
    """
    # Only the window is converted to float: the scene cube is float32 and
    # np.asarray(cube, dtype=float) on the whole of it copied ~2 GB per
    # call, once per fire pixel (build_libraries() calls this for each).
    exclude_mask = np.asarray(exclude_mask, dtype=bool)
    n_rows, _n_bands, n_cols = cube.shape
    r0, r1 = max(row - half_window, 0), min(row + half_window + 1, n_rows)
    c0, c1 = max(col - half_window, 0), min(col + half_window + 1, n_cols)
    sub_exclude = exclude_mask[r0:r1, c0:c1]
    sub_cube = np.asarray(cube[r0:r1, :, c0:c1], dtype=float)
    rr, cc = np.where(~sub_exclude)
    if rr.size < min_pixels:
        return None
    spectra = sub_cube[rr, :, cc]
    spectra = spectra[np.all(np.isfinite(spectra), axis=1)]
    if spectra.shape[0] < min_pixels:
        return None
    return spectra


def local_background_quadrants(cube, wavelength, row, col, exclude_mask,
                               half_window=5, min_pixels=5):
    """Up to 4 spatially-separated sub-pools of the same neighbourhood
    `local_background_pool` draws from, split at the pixel's own row/col
    into NE/NW/SE/SW quadrants.

    Exists only to feed the spatial-consistency check in
    `estimate_relative_transmittance`: a genuine atmospheric perturbation
    (smoke sitting over the whole neighbourhood) should shift ALL FOUR
    quadrants the same way, while a local land-cover feature confined to one
    corner of the window (a rock outcrop, a different crop parcel) should
    not. Returns a list of up to 4 (n_local_q, n_bands) arrays (fewer if a
    quadrant has under `min_pixels` valid neighbours - e.g. at a scene edge).
    """
    exclude_mask = np.asarray(exclude_mask, dtype=bool)
    n_rows, _n_bands, n_cols = cube.shape
    r0, r1 = max(row - half_window, 0), min(row + half_window + 1, n_rows)
    c0, c1 = max(col - half_window, 0), min(col + half_window + 1, n_cols)
    quadrants = []
    for rr_slice, cc_slice in ((slice(r0, row + 1), slice(c0, col + 1)),
                               (slice(r0, row + 1), slice(col, c1)),
                               (slice(row, r1), slice(c0, col + 1)),
                               (slice(row, r1), slice(col, c1))):
        sub_exclude = exclude_mask[rr_slice, cc_slice]
        sub_cube = np.asarray(cube[rr_slice, :, cc_slice], dtype=float)   # window only, not the cube
        rr, cc = np.where(~sub_exclude)
        if rr.size < min_pixels:
            continue
        spectra = sub_cube[rr, :, cc]
        spectra = spectra[np.all(np.isfinite(spectra), axis=1)]
        if spectra.shape[0] >= min_pixels:
            quadrants.append(spectra)
    return quadrants


def _fit_log_ratio_slope(local_spectrum, clear_spectrum, wavelength, fit_mask, alpha, lambda_ref_nm):
    """The (S3) OLS-through-the-origin fit, factored out so the main
    estimate and the per-quadrant consistency check in
    `estimate_relative_transmittance` share one implementation. Returns
    (c, r2, t_stat, n_bands); c=0, r2=nan, t_stat=nan, n=0 if too few valid
    bands survive `fit_mask`."""
    m = (np.asarray(fit_mask, dtype=bool) & np.isfinite(local_spectrum)
         & np.isfinite(clear_spectrum) & (local_spectrum > 0) & (clear_spectrum > 0))
    if m.sum() < 8:
        return 0.0, np.nan, np.nan, int(m.sum())
    lam = wavelength[m]
    x = (lam / lambda_ref_nm) ** (-alpha)
    y = np.log(local_spectrum[m] / clear_spectrum[m])
    sxx = float(np.sum(x * x))
    c = float(np.sum(x * y) / sxx) if sxx > 0 else 0.0
    yhat = c * x
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = int(m.sum())
    resid_var = ss_res / max(n - 1, 1)
    se_c = float(np.sqrt(resid_var / sxx)) if sxx > 0 else np.inf
    t_stat = abs(c) / se_c if se_c > 0 else 0.0
    return c, r2, t_stat, n


def _vnir_haze_corroboration(local_spectrum, clear_spectrum, wavelength,
                             vnir_lo_nm=450.0, vnir_hi_nm=700.0, min_brightening=0.02):
    """Independent, spectrally distinct corroboration that a candidate SWIR
    excess-transmittance anomaly is atmospheric rather than a local land-
    cover change.

    Aerosol/smoke scattering brightens VNIR radiance (Rayleigh+Mie
    backscatter of downwelling sunlight into the sensor); a change in local
    ground cover has no particular reason to also brighten the VNIR by a
    comparable margin at the same location. Requiring both signatures at
    once - SWIR consistent with (S3), AND a real VNIR brightening - uses two
    spectrally separated bands governed by different physics (extinction in
    the SWIR fit window; backscatter here), so a coincidental land-cover
    match to both is far less likely than a match to either alone.

    Returns (corroborated: bool, vnir_ratio: float, n_vnir_bands: int).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    vmask = ((wavelength >= vnir_lo_nm) & (wavelength <= vnir_hi_nm)
             & np.isfinite(local_spectrum) & np.isfinite(clear_spectrum) & (clear_spectrum > 0))
    n_vnir = int(vmask.sum())
    if n_vnir < 5:
        return False, np.nan, n_vnir
    ratio = float(np.median(local_spectrum[vmask] / clear_spectrum[vmask]))
    return bool(ratio >= 1.0 + min_brightening), ratio, n_vnir


def estimate_relative_transmittance(local_bkg_spectrum, clear_bkg_spectrum, wavelength,
                                    fit_mask=None, alpha=1.6, lambda_ref_nm=1600.0,
                                    min_r2=0.4, min_t_stat=2.0,
                                    quadrant_spectra=None, min_quadrant_agree=0.6,
                                    require_vnir_corroboration=True,
                                    vnir_lo_nm=450.0, vnir_hi_nm=700.0, min_vnir_brightening=0.02,
                                    estimate_path_radiance=True):
    """Local excess-transmittance spectrum tau(lambda) - and, optionally, an
    additive local path-radiance term - from the contrast between a
    spatially local background pool and the scene's clear-sky reference
    background (e.g. `background_from_histogram`'s Otsu-cold class,
    additionally buffered away from every detected hotspot).

    Model (Angstrom-type power-law extinction, alpha fixed from the biomass-
    burning smoke literature - Reid et al. 2005; Eck et al. 2001 AERONET
    climatology - rather than fitted, so this stays a ONE-parameter
    regression):

        ln[local(lambda) / clear(lambda)] = c * (lambda / lambda_ref)^(-alpha)   (S3)

    fit for the single scalar c by ordinary least squares through the origin
    (closed form, no optimiser, cannot fail to converge; `_fit_log_ratio_slope`).
    tau(lambda) is then exp(c * (lambda/lambda_ref)^(-alpha)): c < 0 (local
    pool darker than the clear reference, more so at shorter SWIR
    wavelengths - the dark-target, extinction-dominated regime this is
    restricted to, see module header) gives tau < 1 and rising with
    wavelength, exactly the shape needed in (S2). c > 0 is also accepted
    (scattering-brightened local background) and yields tau > 1, i.e. the
    correction can raise OR lower the fitted temperature - it is not a
    one-directional fudge.

    THREE INDEPENDENT GATES, all required before (S3) is applied. A single
    loose gate lets fit noise masquerade as a real physical effect - Section
    10.1 and 15.2 of VERIFICATION.md both found exactly this failure mode for
    other methods in this module - and Section 16.5 measured it here too: a
    check against real (non-smoke) scene pixels found the shape gate alone
    fires on real land-cover contrast at a 17.5% (7/40) rate. Requiring all
    three together is the fix, verified in `verify16_smoke.py` Test C/C2/C3
    to bring that down substantially without losing the synthetic recovery
    of Test B:

      1. SPECTRAL SHAPE (always evaluated): R^2 >= min_r2 AND
         |c|/SE(c) >= min_t_stat on the SWIR log-ratio fit above.
      2. SPATIAL CONSISTENCY (evaluated whenever `quadrant_spectra` supplies
         at least 2 usable quadrants - see `local_background_quadrants`):
         at least `min_quadrant_agree` of the quadrants whose own (S3) fit
         also clears R^2 >= min_r2 must agree in SIGN with the pooled `c`.
         A real plume perturbs the whole neighbourhood; a corner-confined
         land-cover feature does not. Silently skipped (does not block) when
         fewer than 2 quadrants are supplied, so this degrades gracefully
         near scene edges rather than refusing to run.
      3. VNIR CORROBORATION (evaluated whenever `require_vnir_corroboration`
         and the SAME two input spectra span the VNIR - i.e. they are the
         full-cube spectra, not already sliced to the SWIR fit window):
         `_vnir_haze_corroboration` must also see the local pool brighter
         than the clear reference in the visible by at least
         `min_vnir_brightening`. Skipped (does not block) when the inputs do
         not cover the VNIR range at all, so a caller working only in the
         SWIR still gets gates 1-2.

    Otherwise tau = 1 everywhere is returned - exact fall-back to the
    uncorrected (V2)/(S2 with tau=1) model - never a spurious correction from
    an under-determined regression.

    ADDITIVE PATH-RADIANCE TERM. (S2) also has a `f * L_path(lambda)` term
    that a pure transmittance model does not capture. Rather than assume an
    external radiative-transfer value for it (untested against this data),
    it is estimated empirically, ONLY when the shape+spatial+VNIR gates
    above already passed, as the part of the local-clear contrast the fitted
    multiplicative model (S3) does not explain:

        delta_L_path = median[ local(lambda) - tau(lambda)*clear(lambda) ]     (S4)

    over the same fit-window bands - a data-driven residual, not a new
    physics assumption, consistent with this module's existing preference
    for scene-derived quantities over external LUTs (`background_from_histogram`
    is the precedent). Reported as a single scalar (the fit window is narrow
    enough that a smoothly-varying L_path is adequately approximated as flat
    across it); 0.0 when the gates do not pass or `estimate_path_radiance`
    is False.

    Returns dict with tau (array, len(wavelength)), delta_l_path (float), c,
    alpha, r2, t_stat, applied (bool), reason, and diagnostic sub-fields
    quadrant_agree_frac, n_quadrants_used, vnir_ratio, vnir_corroborated.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    local_bkg_spectrum = np.asarray(local_bkg_spectrum, dtype=float)
    clear_bkg_spectrum = np.asarray(clear_bkg_spectrum, dtype=float)
    if fit_mask is None:
        fit_mask = np.ones(len(wavelength), dtype=bool)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    ones = np.ones(len(wavelength), dtype=float)

    def _reject(reason, **extra):
        out = {'tau': ones, 'delta_l_path': 0.0, 'c': 0.0, 'alpha': float(alpha),
               'r2': np.nan, 't_stat': np.nan, 'applied': False, 'reason': reason,
               'quadrant_agree_frac': np.nan, 'n_quadrants_used': 0,
               'vnir_ratio': np.nan, 'vnir_corroborated': None}
        out.update(extra)
        return out

    c, r2, t_stat, n = _fit_log_ratio_slope(local_bkg_spectrum, clear_bkg_spectrum, wavelength,
                                            fit_mask, alpha, lambda_ref_nm)
    if n < 8:
        return _reject(f'too few valid bands ({n} < 8)')
    if not (r2 >= min_r2 and t_stat >= min_t_stat):
        return _reject(f'shape gate failed (r2={r2:.2f} < {min_r2} or t={t_stat:.2f} < {min_t_stat})',
                       c=c, r2=r2, t_stat=t_stat)

    # Gate 2 - spatial consistency across quadrants. A quadrant counts as
    # AGREEING only if its own fit both matches the pooled c's sign AND
    # itself shows a comparable shape (r2 at least half of min_r2) - a
    # quadrant with no perturbation at all (r2 near 0, the null case) must
    # count as DISAGREEING, not be dropped from the tally, or a corner-
    # confined anomaly (3 flat quadrants + 1 real one) would have only 1
    # quadrant "usable" and the check would be skipped as under-determined
    # rather than correctly seeing 1-of-4 agreement. Only a genuine data
    # shortage (nq < 8 valid bands) is excluded from the tally.
    quadrant_agree_frac, n_quad_used = np.nan, 0
    if quadrant_spectra:
        signs = []
        for q in quadrant_spectra:
            q_spec = np.nanmedian(np.atleast_2d(np.asarray(q, dtype=float)), axis=0)
            cq, r2q, _tq, nq = _fit_log_ratio_slope(q_spec, clear_bkg_spectrum, wavelength,
                                                    fit_mask, alpha, lambda_ref_nm)
            if nq < 8:
                continue
            agree = bool(cq != 0 and np.sign(cq) == np.sign(c) and r2q >= 0.5 * min_r2)
            signs.append(agree)
        n_quad_used = len(signs)
        if n_quad_used >= 2:
            quadrant_agree_frac = float(np.mean(signs))
            if quadrant_agree_frac < min_quadrant_agree:
                return _reject(f'spatial-consistency gate failed ({quadrant_agree_frac:.2f} < '
                               f'{min_quadrant_agree} of {n_quad_used} quadrants agree in sign)',
                               c=c, r2=r2, t_stat=t_stat,
                               quadrant_agree_frac=quadrant_agree_frac, n_quadrants_used=n_quad_used)

    # Gate 3 - VNIR haze corroboration (only meaningful if the inputs span the VNIR).
    vnir_corroborated, vnir_ratio = None, np.nan
    if require_vnir_corroboration:
        vnir_corroborated, vnir_ratio, n_vnir = _vnir_haze_corroboration(
            local_bkg_spectrum, clear_bkg_spectrum, wavelength, vnir_lo_nm, vnir_hi_nm,
            min_vnir_brightening)
        if n_vnir >= 5 and not vnir_corroborated:
            return _reject(f'VNIR corroboration failed (ratio={vnir_ratio:.3f} < '
                           f'{1.0 + min_vnir_brightening})',
                           c=c, r2=r2, t_stat=t_stat,
                           quadrant_agree_frac=quadrant_agree_frac, n_quadrants_used=n_quad_used,
                           vnir_ratio=vnir_ratio, vnir_corroborated=False)

    tau = np.exp(c * (wavelength / lambda_ref_nm) ** (-alpha))
    delta_l_path = 0.0
    if estimate_path_radiance:
        resid = local_bkg_spectrum[fit_mask] - tau[fit_mask] * clear_bkg_spectrum[fit_mask]
        finite = np.isfinite(resid)
        if finite.any():
            delta_l_path = float(np.median(resid[finite]))

    return {'tau': tau, 'delta_l_path': delta_l_path, 'c': c, 'alpha': float(alpha),
            'r2': float(r2), 't_stat': float(t_stat), 'applied': True, 'reason': 'ok',
            'quadrant_agree_frac': quadrant_agree_frac, 'n_quadrants_used': n_quad_used,
            'vnir_ratio': vnir_ratio, 'vnir_corroborated': vnir_corroborated}


def smoke_corrected_nlls(radiance, wavelength, background_endmembers, tau,
                         emissivity=1.0, T_bounds=(300.0, 2000.0), rng=None,
                         delta_l_path=0.0):
    """Single-component Planck NLLS with a FIXED transmittance spectrum (and,
    optionally, a fixed additive local path-radiance term) applied to the
    fire term - the solver for (S2) above.

        L_model(lambda) = sum_j q_j*L_bkg,j(lambda) + f*[eps*tau(lambda)*B(lambda,T) + delta_l_path]

    `tau` and `delta_l_path` are supplied already estimated
    (`estimate_relative_transmittance`), never fitted, so this has exactly
    the same free parameters as `joint_nlls(n_sources=1)` - (T, one fire
    abundance, one abundance per background endmember) - and
    `tau=np.ones_like(wavelength), delta_l_path=0.0` reproduces
    `joint_nlls(n_sources=1)` exactly (verified in
    `algorithm_verification/verify16_smoke.py`, test A). Structurally this is
    `joint_nlls` with `n_sources` fixed at 1 and two extra fixed quantities
    added to the Planck term, on the caller's band window; it is not the
    published configuration of Amici et al. (2022), which planck_nlls()
    and amici2022_multitrial() implement.

    Returns dict with T_K, fire_fraction, rmse, model, method, T_at_bound,
    converged - the same schema as the other single-component methods.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    radiance = np.asarray(radiance, dtype=float)
    tau = np.asarray(tau, dtype=float)
    B = np.atleast_2d(np.asarray(background_endmembers, dtype=float))
    n_bkg = B.shape[0]
    rng = rng if rng is not None else np.random.default_rng(0)

    def model(x, *args):
        args = np.asarray(args, dtype=float)
        T = args[0]
        a_fire = args[1]
        b = args[2:]
        norm = a_fire + np.sum(b)
        fire_term = (a_fire / norm) * (emissivity * tau * _planck_nm(x, T) + delta_l_path)
        bkg_term = (b / norm) @ B
        return fire_term + bkg_term

    x0 = [rng.uniform(T_bounds[0], T_bounds[1]), 0.5] + [1.0 / (2 * n_bkg)] * n_bkg
    lo = [T_bounds[0], 0.0] + [0.0] * n_bkg
    hi = [T_bounds[1], 1.0] + [1.0] * n_bkg
    method_name = 'smoke-transmittance-corrected NLLS (this repository, candidate, unpublished)'
    try:
        popt, _ = optimize.curve_fit(model, wavelength, radiance, p0=x0,
                                     bounds=(lo, hi), method='trf',
                                     ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=20000)
    except Exception:
        return {'T_K': np.nan, 'fire_fraction': np.nan, 'rmse': np.nan, 'model': None,
                'method': method_name, 'T_at_bound': False, 'converged': False}

    T = float(popt[0])
    fr = np.asarray(popt[1:], dtype=float)
    fr = fr / np.sum(fr)
    fitted = model(wavelength, *popt)
    rmse = float(np.sqrt(np.mean((fitted - radiance) ** 2)))
    return {'T_K': T, 'fire_fraction': float(fr[0]), 'rmse': rmse, 'model': fitted,
            'method': method_name, 'T_at_bound': _bound_flag(T, T_bounds[0], T_bounds[1]),
            'converged': True}


def apply_local_smoke_correction(cube, wavelength, row, col, exclude_mask,
                                 clear_bkg_spectrum, band_mask, emissivity=1.0,
                                 T_bounds=(300.0, 2000.0), half_window=5, min_pixels=10,
                                 n_endmembers=20, alpha=1.6, lambda_ref_nm=1600.0,
                                 min_r2=0.4, min_t_stat=2.0, min_quadrant_agree=0.6,
                                 require_vnir_corroboration=True, min_vnir_brightening=0.02,
                                 estimate_path_radiance=True, rng=None):
    """One pixel, end to end: local background, the three-gate tau/path-
    radiance estimate, and the corrected fit - falling back gracefully to
    the plain (uncorrected) single-Planck fit whenever the local pool is too
    small or none of the gates in `estimate_relative_transmittance` pass.

    This is the finalised entry point of the smoke-transmittance correction - the function a per-scene driver
    (`run_smoke_corrected_pixels`) or the GUI's pipeline call should use, so
    that "local background too small at the scene edge" and "no smoke
    detected here" are handled once, in one place, rather than by every
    caller re-implementing the same fall-back.

    Returns the `smoke_corrected_nlls`-style dict, plus `fallback` (bool),
    `fallback_reason` (str or None), and the raw `tau_diagnostics` dict from
    `estimate_relative_transmittance` (None on fall-back).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    band_mask = np.asarray(band_mask, dtype=bool)
    clear_bkg_spectrum = np.asarray(clear_bkg_spectrum, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)

    fire_spectrum = np.asarray(cube[row, :, col], dtype=float)
    pool = local_background_pool(cube, wavelength, row, col, exclude_mask,
                                 half_window=half_window, min_pixels=min_pixels)

    def _fallback(reason):
        lib_source = pool if pool is not None else np.atleast_2d(clear_bkg_spectrum)
        lib = lib_source[rng.integers(lib_source.shape[0], size=min(n_endmembers, lib_source.shape[0]))]
        res = joint_nlls(fire_spectrum[band_mask], wavelength[band_mask], lib[:, band_mask],
                          n_sources=1, emissivity=emissivity, T_bounds=T_bounds, rng=rng)
        res['method'] = 'smoke-transmittance-corrected NLLS (this repository, candidate, unpublished)'
        res['fallback'] = True
        res['fallback_reason'] = reason
        res['tau_diagnostics'] = None
        return res

    if pool is None:
        return _fallback(f'local background pool below min_pixels={min_pixels} (scene edge or dense hotspot cluster)')

    local_spectrum = np.nanmedian(pool, axis=0)
    if not np.all(np.isfinite(local_spectrum)) or np.any(local_spectrum <= 0):
        return _fallback('local background spectrum has non-finite or non-positive bands')

    quadrants = local_background_quadrants(cube, wavelength, row, col, exclude_mask,
                                           half_window=half_window, min_pixels=max(5, min_pixels // 2))
    tau_diag = estimate_relative_transmittance(
        local_spectrum, clear_bkg_spectrum, wavelength, fit_mask=band_mask, alpha=alpha,
        lambda_ref_nm=lambda_ref_nm, min_r2=min_r2, min_t_stat=min_t_stat,
        quadrant_spectra=quadrants, min_quadrant_agree=min_quadrant_agree,
        require_vnir_corroboration=require_vnir_corroboration, min_vnir_brightening=min_vnir_brightening,
        estimate_path_radiance=estimate_path_radiance)

    if not tau_diag['applied']:
        res = _fallback(f"gates not cleared: {tau_diag['reason']}")
        res['tau_diagnostics'] = tau_diag
        return res

    lib = pool[rng.integers(pool.shape[0], size=min(n_endmembers, pool.shape[0]))]
    res = smoke_corrected_nlls(fire_spectrum[band_mask], wavelength[band_mask], lib[:, band_mask],
                               tau_diag['tau'][band_mask], emissivity=emissivity, T_bounds=T_bounds,
                               rng=rng, delta_l_path=tau_diag['delta_l_path'])
    res['fallback'] = False
    res['fallback_reason'] = None
    res['tau_diagnostics'] = tau_diag
    return res


def _smoke_one_pixel(window, wavelength, row, col, exclude_window, clear_bkg_spectrum, band_mask,
                     seed, kwargs):
    """apply_local_smoke_correction() on a pixel's own neighbourhood window
    (module level, so joblib can ship it)."""
    return apply_local_smoke_correction(window, wavelength, row, col, exclude_window,
                                        clear_bkg_spectrum, band_mask,
                                        rng=np.random.default_rng(seed), **kwargs)


def run_smoke_corrected_pixels(cube, wavelength, fire_points, hotspot_mask, clear_bkg_spectrum,
                               band_mask, emissivity=1.0, T_bounds=(300.0, 2000.0),
                               half_window=5, min_pixels=10, n_endmembers=20, alpha=1.6,
                               rng=None, progress=None, n_jobs=-1, **gate_kwargs):
    """The smoke-transmittance correction over every fire pixel - the per-scene
    driver, parallel in role to `run_methods_on_pixels` for the published
    methods, on every CPU core. A pixel's correction reads only its
    (2 half_window + 1)^2 neighbourhood, so each worker is sent that window
    (clipped at the scene edge, exactly as the local-background functions
    clip it) rather than the cube. Each pixel has its own random stream,
    seeded from `rng` up front, so a result does not depend on the order or
    the split of the work.

    `hotspot_mask` (n_rows, n_cols) bool should be the scene's fire/hotspot
    detection mask (True = detected), so no fire pixel's own neighbours, or a
    neighbouring detection, contaminate its local background estimate - this
    IS `exclude_mask` for `local_background_pool`/`local_background_quadrants`,
    named for what it means at this call site.

    Returns a pandas.DataFrame with the same schema as
    `run_methods_on_pixels()['<key>']` (Pixel/ind_x/ind_y/T_K/fire_fraction/
    rmse/converged/T_at_bound/method/citation) plus the correction's own diagnostic
    columns: fallback, fallback_reason, tau_applied, tau_c, tau_r2, tau_t_stat,
    tau_delta_l_path, quadrant_agree_frac, n_quadrants_used, vnir_ratio,
    vnir_corroborated.
    """
    import pandas as pd

    wavelength = np.asarray(wavelength, dtype=float)
    hotspot_mask = np.asarray(hotspot_mask, dtype=bool)
    rng = rng if rng is not None else np.random.default_rng(0)
    citation = ('this repository, candidate (unpublished) - local-background / '
               'smoke-transmittance correction of the fire Planck term, VERIFICATION.md Section 16')

    fire_points = np.asarray(fire_points)
    n_rows, _n_bands, n_cols = cube.shape
    seeds = rng.integers(0, 2 ** 31 - 1, size=max(len(fire_points), 1))
    kw = dict(emissivity=emissivity, T_bounds=T_bounds, half_window=half_window,
              min_pixels=min_pixels, n_endmembers=n_endmembers, alpha=alpha, **gate_kwargs)

    def _args(i):
        r_, c_ = int(fire_points[i][0]), int(fire_points[i][1])
        r0, r1 = max(r_ - half_window, 0), min(r_ + half_window + 1, n_rows)
        c0, c1 = max(c_ - half_window, 0), min(c_ + half_window + 1, n_cols)
        return (cube[r0:r1, :, c0:c1], wavelength, r_ - r0, c_ - c0, hotspot_mask[r0:r1, c0:c1],
                clear_bkg_spectrum, band_mask, int(seeds[i]), kw)

    n_pix = len(fire_points)
    if n_jobs != 1 and n_pix >= 8:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=n_jobs)(delayed(_smoke_one_pixel)(*_args(i)) for i in range(n_pix))
    else:
        results = [_smoke_one_pixel(*_args(i)) for i in range(n_pix)]

    rows = []
    for i, ((r_, c_), res) in enumerate(zip(fire_points, results)):
        diag = res.get('tau_diagnostics') or {}
        rows.append({
            'Pixel': i, 'ind_x': int(r_), 'ind_y': int(c_),
            'T_K': res.get('T_K', np.nan), 'fire_fraction': res.get('fire_fraction', np.nan),
            'rmse': res.get('rmse', np.nan), 'converged': bool(res.get('converged', False)),
            'T_at_bound': bool(res.get('T_at_bound', False)), 'method': res.get('method'),
            'citation': citation, 'fallback': bool(res.get('fallback', True)),
            'fallback_reason': res.get('fallback_reason'),
            'tau_applied': bool(diag.get('applied', False)) if diag else False,
            'tau_c': diag.get('c', np.nan) if diag else np.nan,
            'tau_r2': diag.get('r2', np.nan) if diag else np.nan,
            'tau_t_stat': diag.get('t_stat', np.nan) if diag else np.nan,
            'tau_delta_l_path': diag.get('delta_l_path', np.nan) if diag else np.nan,
            'quadrant_agree_frac': diag.get('quadrant_agree_frac', np.nan) if diag else np.nan,
            'n_quadrants_used': diag.get('n_quadrants_used', 0) if diag else 0,
            'vnir_ratio': diag.get('vnir_ratio', np.nan) if diag else np.nan,
            'vnir_corroborated': diag.get('vnir_corroborated') if diag else None,
        })
        if progress is not None and (i % 10 == 0):
            progress('SmokeSTCR', i + 1, len(fire_points))

    columns = ['Pixel', 'ind_x', 'ind_y', 'T_K', 'fire_fraction', 'rmse', 'converged', 'T_at_bound',
              'method', 'citation', 'fallback', 'fallback_reason', 'tau_applied', 'tau_c', 'tau_r2',
              'tau_t_stat', 'tau_delta_l_path', 'quadrant_agree_frac', 'n_quadrants_used',
              'vnir_ratio', 'vnir_corroborated']
    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


# --------------------------------------------------------------------------
# registry + driver
# --------------------------------------------------------------------------

# Keys double as the tag written into output filenames, so every results file
# says which method produced it and in which year that method was published.
METHOD_REGISTRY = {
    'Dozier1981': {
        'func': dozier_bispectral,
        'citation': 'Dozier (1981), RSE 11:221; two SWIR channels, measured background',
        'needs': 'background_spectrum',
    },
    # 'channels': the method's own published channel set, taken from the FULL
    # spectrum - not from the scene window `band_mask`, which drops
    # 1200-1320 nm, 1449 nm and 1990-2040 nm that these papers fit.
    'Dennison2006': {
        'func': dennison_mesma,
        'citation': 'Dennison et al. (2006), RSE 100:212',
        'needs': 'endmembers',
        'channels': dennison_channel_mask,
    },
    'Amici2022': {
        'func': amici2022_multitrial,
        'citation': 'Amici et al. (2022), JGR Biogeosciences 127:e2022JG007055; code Zenodo 10.5281/zenodo.7273879',
        'needs': 'pool',
        'channels': amici_window_mask,
    },
    # This repository's own method: separable maximum-likelihood
    # profile retrieval - global 1-D temperature search, explicit noise +
    # model-error covariance, profile-likelihood interval, censored
    # (saturated) bands. See profile_retrieval.py. Through this registry it
    # runs with the SAME endmember draw and band window as the published
    # methods, so the comparison figures isolate the estimator; the full
    # chain with the scene transmittance and the L1B quality masks is
    # run_pipeline()'s ProfileML stage, not this entry.
    'ProfileML': {
        'func': _profile_retrieval,
        'citation': 'this work (profile_retrieval.py)',
        'needs': 'endmembers',
    },
}

DEFAULT_METHOD = 'Amici2022'

# Human-readable method names: authors and year, as they should appear in a
# log line or a figure label. The registry keys stay short because they are
# also the tags inside output filenames.
METHOD_DISPLAY_NAMES = {
    'Dozier1981': 'Dozier (1981)',
    'Dennison2006': 'Dennison et al. (2006)',
    'Amici2022': 'Amici et al. (2022)',
    'ProfileML': 'This work (2026), profile-likelihood retrieval',
    'SmokeSTCR': 'Smoke-transmittance correction (this repository, candidate)',
}

# Short forms for axis labels and panel titles, where the full name overflows.
METHOD_SHORT_NAMES = {
    'Dozier1981': 'Dozier (1981)',
    'Dennison2006': 'Dennison et al. (2006)',
    'Amici2022': 'Amici et al. (2022)',
    'ProfileML': 'This work',
    'SmokeSTCR': 'Smoke-corrected fit',
}


def method_display_name(key):
    """Author-year name for a registry key (the key itself if unknown)."""
    return METHOD_DISPLAY_NAMES.get(key, key)


def method_short_name(key):
    """Short name for axis labels and titles (the key itself if unknown)."""
    return METHOD_SHORT_NAMES.get(key, method_display_name(key))


_EXTRA_RESULT_KEYS = ('T_ci_low', 'T_ci_high', 'T_lower_unbounded', 'T_upper_unbounded',
                      'reduced_chi2', 'rho_lag1', 'fire_lrt', 'n_bands_invalid', 'n_bands_censored',
                      'std_T_K', 'CI_T_low', 'CI_T_high', 'n_trials_valid', 'T_median_K',
                      'n_trials_cold', 'reflected_fraction', 'shade_fraction', 'rmse_encoded',
                      'n_bands_fit')


def _run_method_one_pixel(spec, key, i, r_, c_, y, w, pool, bkg_single, n_endmembers, seed,
                          emissivity, method_kwargs, per_pixel_values, m_len, cols=None):
    """One (method, pixel) fit; a module-level function so joblib can ship it."""
    rng = np.random.default_rng(seed)
    kwargs = {'emissivity': emissivity}
    if method_kwargs and key in method_kwargs:
        kwargs.update(method_kwargs[key])
    for k_, v_ in (per_pixel_values or {}).items():
        kwargs[k_] = v_
    # driver-level inputs (see run_methods_on_pixels): a spectrum subtracted
    # from the pixel and the endmembers, a fixed endmember set, a per-pixel
    # background spectrum
    offset = kwargs.pop('subtract_spectrum', None)
    fixed = kwargs.pop('fixed_endmembers', None)
    bkg = kwargs.pop('background', bkg_single)
    if offset is not None:
        y = y - offset
    if spec['needs'] == 'background_spectrum':
        res = spec['func'](y, w, bkg, **kwargs)
    elif spec['needs'] == 'pool':
        kwargs.setdefault('rng', rng)
        kwargs.setdefault('n_background', n_endmembers)
        res = spec['func'](y, w, pool, **kwargs)
    else:
        if fixed is not None:
            lib = np.atleast_2d(fixed)
        else:
            lib = pool[rng.integers(pool.shape[0], size=min(n_endmembers, pool.shape[0]))]
            if cols is not None:
                lib = np.asarray(lib[:, cols], dtype=float)
            if offset is not None:
                lib = lib - offset
        res = spec['func'](y, w, lib, **kwargs)
    row = {'Pixel': i, 'ind_x': int(r_), 'ind_y': int(c_),
           'T_K': res.get('T_K', np.nan),
           'fire_fraction': res.get('fire_fraction', np.nan),
           'rmse': res.get('rmse', np.nan),
           'converged': bool(res.get('converged', False)),
           'T_at_bound': bool(res.get('T_at_bound', False)),
           'method': key, 'citation': spec['citation']}
    for k_ in _EXTRA_RESULT_KEYS:
        if k_ in res:
            row[k_] = res[k_]
    return row


def run_methods_on_pixels(cube, wavelength, fire_points, bkg_pool,
                          methods=('Amici2022',), band_mask=None,
                          background_spectrum=None, emissivity=1.0,
                          n_endmembers=20, rng=None, progress=None, method_kwargs=None,
                          per_pixel_kwargs=None, n_jobs=-1, precomputed=None):
    """Run one or more registered methods over every fire pixel.

    precomputed, if given, is {method_key: DataFrame} of results already
    computed for these pixels in this schema; those methods are not run
    again (see amici2022_comparison_frame()).

    Returns {method_key: pandas.DataFrame}, one row per fire pixel, with
    columns Pixel / ind_x / ind_y / T_K / fire_fraction / rmse / converged /
    T_at_bound / method / citation. Each frame is independently saveable,
    which is what lets a results file be named after the method that
    produced it.

    `T_at_bound` marks rows whose temperature is the edge of that method's
    own search interval rather than a fitted value (see _bound_flag()).
    Those rows are lower limits, not measurements; exclude them from any
    aggregate, or report them separately.

    background_spectrum is required by Dozier1981 and should come from
    background_from_histogram(); passing None makes that method fall back to
    the median of bkg_pool, which is weaker (see that function's docstring).

    method_kwargs, if given, is {method_key: {kwarg: value}} of extra
    keyword arguments for that method's function (e.g. a fixed `tau` for
    ProfileML). per_pixel_kwargs is the same shape but every value is a
    sequence indexed by fire pixel (e.g. the per-pixel `saturated` channel
    mask Dennison et al. (2006) need; the arrays are given over the full
    band axis and cut to band_mask here). Methods that return extra
    diagnostics (T_ci_low, reduced_chi2, std_T_K, ...) get those as
    additional columns; the ten base columns are identical for every
    method.

    Band windows: `band_mask` is the scene-level window, used by Dozier
    (1981) and this work. Dennison et al. (2006) and Amici et al. (2022) are
    given their own published channel sets from the full spectrum (registry
    'channels'), so they are fitted over exactly the channels their papers
    describe; the scene window would drop 1200-1320 nm, 1449 nm and
    1990-2040 nm from them.
    """
    import pandas as pd

    wavelength = np.asarray(wavelength, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)
    m_scene = np.ones(len(wavelength), dtype=bool) if band_mask is None else np.asarray(band_mask, bool)
    bkg_pool = np.atleast_2d(bkg_pool)
    # The pool can be ~10^6 spectra: it is never copied whole. Medians are
    # taken band by band and the endmember draws gather their rows first
    # (identical values, a fraction of the memory).
    if background_spectrum is None:
        from profile_retrieval import column_median
        bkg_single = column_median(bkg_pool, m_scene)
    else:
        bkg_single = np.asarray(background_spectrum, dtype=float)[m_scene]

    unknown = [k for k in methods if k not in METHOD_REGISTRY]
    if unknown:
        raise ValueError(f'Unknown method(s) {unknown}; available: {sorted(METHOD_REGISTRY)}')

    out = {}
    fire_points = np.asarray(fire_points)
    n_pix = len(fire_points)
    # One seed per (method, pixel), drawn up front from `rng`, so a pixel's
    # fit does not depend on how many pixels ran before it or on how the
    # work was split over processes.
    seeds = {key: rng.integers(0, 2 ** 31 - 1, size=max(n_pix, 1)) for key in methods}
    try:
        from joblib import Parallel, delayed
        have_joblib = True
    except ImportError:                                   # pragma: no cover
        have_joblib = False
    for key in methods:
        if precomputed and key in precomputed:
            # already run elsewhere with the published configuration (the
            # pipeline's stage 5 for Amici2022); its seeds above are still
            # drawn, so the other methods' draws do not depend on this
            out[key] = precomputed[key]
            continue
        spec = METHOD_REGISTRY[key]
        m = spec['channels'](wavelength) if 'channels' in spec else m_scene
        if spec['needs'] == 'pool':
            pool, cols = np.asarray(bkg_pool[:, m], dtype=float), None
        elif spec['needs'] == 'background_spectrum':
            pool, cols = None, None           # not used; keeps it out of the workers
        else:
            pool, cols = bkg_pool, m          # rows drawn first, then cut and converted
        w = wavelength[m]

        def _cut(v_):
            v_ = np.asarray(v_)
            if v_.ndim >= 1 and v_.shape[-1] == len(m):
                return v_[..., m]
            return v_

        # method_kwargs arrays given over the full band axis (a transmittance
        # spectrum, a fixed endmember set, a spectrum to subtract) are cut to
        # this method's channels like the per-pixel ones
        mk_key = None
        if method_kwargs and key in method_kwargs:
            mk_key = {k_: (_cut(v_) if isinstance(v_, np.ndarray) else v_)
                      for k_, v_ in method_kwargs[key].items()}
        mk_all = dict(method_kwargs or {})
        if mk_key is not None:
            mk_all[key] = mk_key

        def _per_pixel(i):
            vals = {}
            if per_pixel_kwargs and key in per_pixel_kwargs:
                for k_, seq in per_pixel_kwargs[key].items():
                    v_ = np.asarray(seq[i])
                    vals[k_] = v_[m] if v_.ndim == 1 and v_.shape[0] == len(m) else v_
            return vals

        def _args(i):
            r_, c_ = fire_points[i]
            return (spec, key, i, r_, c_, np.asarray(cube[r_, :, c_], dtype=float)[m], w, pool,
                    bkg_single, n_endmembers, int(seeds[key][i]), emissivity, mk_all,
                    _per_pixel(i), len(m), cols)

        if have_joblib and n_jobs != 1 and n_pix >= 8:
            if progress is not None:
                progress(key, 1, n_pix)
            rows = Parallel(n_jobs=n_jobs)(delayed(_run_method_one_pixel)(*_args(i)) for i in range(n_pix))
            if progress is not None:
                progress(key, n_pix, n_pix)
        else:
            rows = []
            for i in range(n_pix):
                rows.append(_run_method_one_pixel(*_args(i)))
                if progress is not None and (i % 50 == 0):
                    progress(key, i + 1, n_pix)
        if rows:
            out[key] = pd.DataFrame(rows)
        else:
            # pd.DataFrame([]) has no columns at all, so a zero-detection scene
            # produces a frame that breaks every consumer selecting a column -
            # plot_method_comparison() does exactly that on 'T_K'. Give the
            # empty case the documented schema so it flows through the figures
            # and the CSV writer unchanged.
            out[key] = pd.DataFrame(columns=[
                'Pixel', 'ind_x', 'ind_y', 'T_K', 'fire_fraction', 'rmse',
                'converged', 'T_at_bound', 'method', 'citation'])
    return out


def amici2022_comparison_frame(results):
    """The comparison-schema frame (run_methods_on_pixels) of Amici et al.
    (2022) from the pipeline's stage-5 results frame (main_temp_estimation,
    the published configuration), so the published method is fitted once
    per run and every output that names it reports the same numbers.
    `rmse` is the RMSE of the fit with the fire term from the stage-5
    significance test (rmse_with_fire)."""
    import pandas as pd
    r = results
    T = r['mean_temp_K'].astype(float).values
    n_valid = r['n_trials_valid'].values if 'n_trials_valid' in r else np.isfinite(T).astype(int)
    out = pd.DataFrame({
        'Pixel': r['Pixel'].values if 'Pixel' in r else np.arange(len(r)),
        'ind_x': r['ind_x'].astype(int).values, 'ind_y': r['ind_y'].astype(int).values,
        'T_K': T, 'fire_fraction': r['mean_P_coeff'].values,
        'rmse': r['rmse_with_fire'].values if 'rmse_with_fire' in r else np.nan,
        'converged': np.isfinite(T) & (n_valid > 0),
        'T_at_bound': r['T_at_bound'].astype(bool).values if 'T_at_bound' in r else False,
        'method': 'Amici2022', 'citation': METHOD_REGISTRY['Amici2022']['citation']})
    for src, dst in (('std_temp_K', 'std_T_K'), ('CI_temp_low_K', 'CI_T_low'), ('CI_temp_high_K', 'CI_T_high'),
                     ('n_trials_valid', 'n_trials_valid'), ('median_temp_K', 'T_median_K'),
                     ('n_bands_fit', 'n_bands_fit')):
        if src in r:
            out[dst] = r[src].values
    return out


ALL_METHODS = {k: v['func'] for k, v in METHOD_REGISTRY.items()}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _fit_one_pixel_all_methods(y_full, wavelength, m_scene, lib_full, bkg_single, emissivity, rng,
                               pool_full=None, method_kwargs=None, keys=None):
    """Every registered method on one full-axis spectrum, each over its own
    channels (registry 'channels', else the scene window m_scene); each
    result's 'model' is returned on the full band axis, NaN elsewhere."""
    out = {}
    for key, spec in METHOD_REGISTRY.items():
        if keys is not None and key not in keys:
            continue
        mk = spec['channels'](wavelength) if 'channels' in spec else m_scene
        y, w, lib = y_full[mk], wavelength[mk], lib_full[:, mk]
        kwargs = {'emissivity': emissivity}
        if method_kwargs and key in method_kwargs:
            # full-axis arrays cut to this method's channels, as in run_methods_on_pixels
            kwargs.update({k_: (np.asarray(v_)[..., mk] if isinstance(v_, np.ndarray)
                                and v_.shape[-1] == len(mk) else v_)
                           for k_, v_ in method_kwargs[key].items()})
        offset = kwargs.pop('subtract_spectrum', None)
        fixed = kwargs.pop('fixed_endmembers', None)
        kwargs.pop('background', None)
        if offset is not None:
            y = y - offset
            lib = lib - offset
        if fixed is not None:
            lib = np.atleast_2d(fixed)
        if spec['needs'] == 'background_spectrum':
            res = spec['func'](y, w, bkg_single, **kwargs)
        elif spec['needs'] == 'pool':
            pool = lib if pool_full is None else np.asarray(pool_full[:, mk], dtype=float)
            res = spec['func'](y, w, pool, rng=rng, **kwargs)
        else:
            res = spec['func'](y, w, lib, **kwargs)
        mdl = res.get('model')
        if mdl is not None and np.shape(mdl) == (int(mk.sum()),):
            full = np.full(len(wavelength), np.nan)
            full[mk] = mdl + (offset if offset is not None else 0.0)
            res = dict(res, model=full)
        out[key] = res
    return out


def plot_method_comparison(method_frames, cube, wavelength, band_mask, bkg_pool,
                           background_spectrum, scene, output_dir, scene_id,
                           date_tag, reference='Amici2022', emissivity=1.0,
                           n_endmembers=20, dpi=300, method_kwargs=None):
    """Three publication figures comparing every retrieval that was run.

    Returns the list of paths written. Figures are named with the scene and
    date, and every panel is labelled with the method key (which carries the
    publication year), so a figure is never ambiguous about its source.

      1  ..._MethodMaps_...      retrieved temperature per method, shared
                                 colour scale, so panels are directly
                                 comparable by eye.
      2  ..._MethodAgreement_... each method against the reference, with a
                                 1:1 line and a Bland-Altman difference
                                 plot. This is the figure that shows whether
                                 two independent formulations agree, which
                                 is the strongest internal check available
                                 without ground truth.
      3  ..._MethodSpectralFit_. the hottest pixel's measured spectrum with
                                 each method's modelled spectrum and
                                 residuals - the evidence that the forward
                                 model actually reproduces the data.
    """
    import os
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    keys = [k for k in METHOD_REGISTRY if k in method_frames]
    if not keys:
        return []
    paths = []
    lat, lon = scene['lat'], scene['lon']
    ref_key = reference if reference in keys else keys[0]

    # ---- 1. maps -----------------------------------------------------
    # Drawn with the same background/halo/inset treatment as the main
    # pipeline figures (_scatter_over_composite, _add_detection_inset): a
    # bare scatter of s=6 points on white showed only that detections exist
    # somewhere, which is not enough to compare one method's spatial pattern
    # against another's - the thing this figure exists to do.
    from Fire_analysis_with_EnMAP import (_build_rgb_composite, _scatter_over_composite,
                                          _add_detection_inset, _plot_agreement_axes)
    allT = np.concatenate([method_frames[k]['T_K'].values for k in keys])
    allT = allT[np.isfinite(allT)]
    vlo, vhi = (np.percentile(allT, 2), np.percentile(allT, 98)) if allT.size else (300, 1500)
    rgb = _build_rgb_composite(scene)
    aspect = 1.0 / max(np.cos(np.radians(np.nanmean(lat))), 1e-3)
    fig, axes = plt.subplots(1, len(keys), figsize=(5.6 * len(keys), 5.0), squeeze=False)
    for ax, k in zip(axes[0], keys):
        df = method_frames[k]
        good = np.isfinite(df['T_K'].values)
        px, py = df['ind_x'].values[good], df['ind_y'].values[good]
        vals = df['T_K'].values[good]
        sc_ = _scatter_over_composite(ax, lon, lat, rgb, px, py, vals, 'inferno', vlo, vhi)
        if sc_ is not None:
            fig.colorbar(sc_, ax=ax, label='Retrieved T (K)', shrink=0.85)
            _add_detection_inset(ax, lon, lat, rgb, px, py, vals, 'inferno', vlo, vhi,
                                 aspect, scene=scene)
        ax.set_title(f"{method_short_name(k)}\n{int(good.sum())}/{len(df)} converged", fontsize=10)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_aspect(aspect)
        ax.tick_params(axis='x', rotation=30)
    fig.suptitle(f'Sub-pixel fire temperature by method, {scene_id}', fontsize=11)
    fig.tight_layout()
    p = os.path.join(output_dir, f'Fig_MethodMaps_{scene_id}_{date_tag}.png')
    fig.savefig(p, dpi=dpi, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ---- 2. agreement ------------------------------------------------
    others = [k for k in keys if k != ref_key]
    if others:
        fig, axes = plt.subplots(2, len(others), figsize=(4.6 * len(others), 8), squeeze=False)
        ref = method_frames[ref_key]['T_K'].values

        # One shared x/y range across every panel, not just within each
        # panel's own pair - otherwise a method that converges on fewer
        # pixels (e.g. Dozier1981, which needs the hot component to
        # out-radiate the background) gets its own narrower axis, and panels
        # that are not directly comparable in range end up looking like they
        # are, each stretched to fill its own box.
        pooled = [ref[np.isfinite(ref)]]
        for k in others:
            v = method_frames[k]['T_K'].values
            pooled.append(v[np.isfinite(v)])
        pooled = np.concatenate(pooled) if pooled else np.array([])
        if pooled.size:
            lo, hi = float(np.min(pooled)), float(np.max(pooled))
            pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.05
            shared_limits = (lo - pad, hi + pad)
        else:
            shared_limits = None

        ref_bound = (method_frames[ref_key]['T_at_bound'].values.astype(bool)
                     if 'T_at_bound' in method_frames[ref_key] else np.zeros(len(ref), bool))
        for j, k in enumerate(others):
            a = method_frames[k]['T_K'].values
            good = np.isfinite(a) & np.isfinite(ref) & ~ref_bound
            # A temperature at a method's search bound is a lower limit, not
            # an estimate (run_methods_on_pixels); it stays out of the
            # agreement statistics.
            if 'T_at_bound' in method_frames[k] and len(method_frames[k]) == len(ref):
                good &= ~method_frames[k]['T_at_bound'].values.astype(bool)
            ax = axes[0][j]
            # Equal limits AND a square box, so the 1:1 line is at a true 45
            # degrees and the fitted line's departure from it can be read as
            # an angle. Previously the limits were stretched to [vlo, vhi]
            # from the shared colour scale and the box was left to fill the
            # subplot, so two methods differing by 10% could be drawn looking
            # like agreement or like a large bias depending only on how many
            # methods were in the figure. `limits=shared_limits` now also
            # makes the x-axis identical across panels (see above), so
            # panels can be compared side by side directly.
            _plot_agreement_axes(ax, ref[good], a[good], limits=shared_limits,
                                 xlabel=f'{method_short_name(ref_key)} T (K)',
                                 ylabel=f'{method_short_name(k)} T (K)')
            md = np.median(np.abs(a[good] - ref[good])) if good.any() else np.nan
            ax.set_title(f'{method_short_name(k)} vs {method_short_name(ref_key)}\n'
                         f'median |ΔT| = {md:.0f} K, n = {int(good.sum())} (bound-pinned excluded)', fontsize=9.5)
            ax.legend(fontsize=8, loc='lower right')
            ax2 = axes[1][j]
            mean_ = 0.5 * (a[good] + ref[good]); diff = a[good] - ref[good]
            ax2.scatter(mean_, diff, s=8, alpha=0.6, color='#3C6E8F')
            if good.any():
                bias = np.mean(diff); sd = np.std(diff)
                for yv, ls, lab in ((bias, '-', f'bias {bias:+.1f} K'),
                                    (bias + 1.96 * sd, ':', f'+1.96 SD ({bias + 1.96 * sd:+.1f})'),
                                    (bias - 1.96 * sd, ':', f'-1.96 SD ({bias - 1.96 * sd:+.1f})')):
                    ax2.axhline(yv, ls=ls, color='k', lw=1, label=lab)
                ax2.legend(fontsize=7)
            ax2.set_xlabel('mean of the two (K)'); ax2.set_ylabel('difference (K)')
            ax2.set_title(f'Bland-Altman: {method_short_name(k)} − {method_short_name(ref_key)}',
                          fontsize=9.5)
        fig.suptitle(f'Method agreement, {scene_id}', fontsize=11)
        fig.tight_layout()
        p = os.path.join(output_dir, f'Fig_MethodAgreement_{scene_id}_{date_tag}.png')
        fig.savefig(p, dpi=dpi, bbox_inches='tight'); plt.close(fig); paths.append(p)

    # ---- 3. spectral fit of the hottest well-fitted pixel -------------
    # The hottest pixel by raw temperature is routinely a degenerate fit: a
    # bright non-fire pixel that every method pushes to its upper bound
    # with a fire fraction at the floor. Such a pixel shows nothing about
    # the forward model. Take the hottest pixel whose temperature is not at
    # a bound in ANY method that converged on it, and, where this work's
    # frame carries them, whose reduced chi-square is within the adequate
    # range; fall back to the plain maximum only if no pixel qualifies.
    ref_df = method_frames[ref_key]
    if np.isfinite(ref_df['T_K'].values).any():
        ok = np.isfinite(ref_df['T_K'].values)
        if 'T_at_bound' in ref_df:
            ok &= ~ref_df['T_at_bound'].values.astype(bool)
        for k in keys:
            df_k = method_frames[k]
            if len(df_k) != len(ref_df) or 'T_at_bound' not in df_k:
                continue
            conv = df_k['converged'].values.astype(bool) & np.isfinite(df_k['T_K'].values)
            ok &= ~(conv & df_k['T_at_bound'].values.astype(bool))
        if 'ProfileML' in method_frames and 'reduced_chi2' in method_frames['ProfileML'] \
                and len(method_frames['ProfileML']) == len(ref_df):
            chi = method_frames['ProfileML']['reduced_chi2'].values
            ok &= ~(np.isfinite(chi) & (chi > 3.0))
        if not ok.any():
            ok = np.isfinite(ref_df['T_K'].values)
        T_sel = np.where(ok, ref_df['T_K'].values, -np.inf)
        i = int(np.argmax(T_sel))
        r_, c_ = int(ref_df['ind_x'].iloc[i]), int(ref_df['ind_y'].iloc[i])
        wl_full = np.asarray(wavelength, float)
        m_scene = np.asarray(band_mask, bool)
        # plotted: the scene window plus every published channel set in the figure
        m = m_scene.copy()
        for k in keys:
            if 'channels' in METHOD_REGISTRY.get(k, {}):
                m |= METHOD_REGISTRY[k]['channels'](wl_full)
        w = wl_full[m]
        y_full = np.asarray(cube[r_, :, c_], float)
        y = y_full[m]
        pool_all = np.atleast_2d(bkg_pool)
        rng = np.random.default_rng(0)
        lib_full = np.asarray(pool_all[rng.integers(pool_all.shape[0],
                                                    size=min(n_endmembers, pool_all.shape[0]))], float)
        bkg1 = (np.median(np.asarray(pool_all[:, m_scene], float), axis=0) if background_spectrum is None
                else np.asarray(background_spectrum, float)[m_scene])
        fits = _fit_one_pixel_all_methods(y_full, wl_full, m_scene, lib_full, bkg1, emissivity, rng,
                                          pool_full=pool_all, method_kwargs=method_kwargs, keys=keys)
        fits = {k: (dict(r, model=np.asarray(r['model'], float)[m])
                    if r.get('model') is not None and np.shape(r['model']) == (len(m),) else r)
                for k, r in fits.items()}

        # `band_mask` (from cfg_band_mask_for_methods) has already dropped the
        # opaque water-vapour/CO2 windows from `w`, so no band inside them is
        # plotted - but the surviving bands on either side of a dropped
        # window are still adjacent samples as far as matplotlib's line
        # drawer is concerned, so '-' joined them with a straight segment
        # spanning the gap: a line the instrument never measured, drawn
        # across the exact interval the fit excludes as unreliable. Inserting
        # a NaN wherever the wavelength step jumps well above the typical
        # band spacing breaks the line there instead, the same fix
        # _mask_atmospheric_bands() uses for Fig1.
        def _break_at_band_gaps(x, *ys, gap_factor=2.5):
            x = np.asarray(x, dtype=float)
            steps = np.diff(x)
            typical = np.median(steps) if steps.size else 0.0
            gaps = np.where(steps > gap_factor * typical)[0] if typical > 0 else np.array([], dtype=int)
            if gaps.size == 0:
                return (x,) + ys
            x_out = list(x)
            ys_out = [list(v) for v in ys]
            for offset, gi in enumerate(gaps):
                idx = int(gi) + 1 + offset
                x_out.insert(idx, 0.5 * (x[gi] + x[gi + 1]))
                for v_out in ys_out:
                    v_out.insert(idx, np.nan)
            return (np.array(x_out),) + tuple(np.array(v) for v in ys_out)

        w_plot, y_plot = _break_at_band_gaps(w, y)

        from Fire_analysis_with_EnMAP import _fire_enhanced_rgb
        fig = plt.figure(figsize=(10.5, 7))
        gs = fig.add_gridspec(2, 2, width_ratios=[3.2, 1], height_ratios=[3, 1], wspace=0.08, hspace=0.08)
        ax = fig.add_subplot(gs[0, 0]); axr = fig.add_subplot(gs[1, 0], sharex=ax)
        axz = fig.add_subplot(gs[0, 1])
        ax.plot(w_plot, y_plot, 'k.', ms=4, label='EnMAP L1B measured')
        colors = ['#B3552C', '#3C6E8F', '#6E8F3C', '#8F3C6E', '#2C2C2C']
        for (k, res), col in zip(fits.items(), colors):
            mdl = res.get('model')
            if mdl is None or not np.isfinite(np.asarray(mdl, float)).any():
                continue
            mdl = np.asarray(mdl, float)
            # Each published method fits its own channel subset; the model
            # is NaN outside it and the line is simply absent there.
            _, mdl_plot, resid_plot = _break_at_band_gaps(w, mdl, mdl - y)
            n_fit = int(res.get('n_bands_fit', np.isfinite(mdl).sum()))
            ax.plot(w_plot, mdl_plot, '-', lw=1.3, color=col,
                    label=f"{method_short_name(k)}: T = {res['T_K']:.0f} K, f = {res['fire_fraction']:.4f}, "
                          f"{n_fit} bands")
            axr.plot(w_plot, resid_plot, '-', lw=1, color=col)
        axr.axhline(0, color='k', lw=0.8)
        ax.set_ylabel(r'Radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
        ax.set_title(f'Pixel (row {r_}, col {c_}): forward models against the measurement', fontsize=11)
        ax.legend(fontsize=8)
        ax.tick_params(labelbottom=False)
        axr.set_xlabel('Wavelength (nm)'); axr.set_ylabel('model − data')

        # True-scale zoom of the pixel on a fire-enhanced composite.
        n_rows, n_cols = scene['lat'].shape
        hz = 15
        r0, r1 = max(0, r_ - hz), min(n_rows, r_ + hz + 1)
        c0, c1 = max(0, c_ - hz), min(n_cols, c_ + hz + 1)
        axz.imshow(_fire_enhanced_rgb(scene, r0, r1, c0, c1), interpolation='nearest',
                   extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5))
        axz.add_patch(plt.Rectangle((c_ - 0.5, r_ - 0.5), 1, 1, fill=False, ec='#00e5ff', lw=1.2))
        for k in keys:
            df_k = method_frames[k]
            if k != ref_key or 'ind_x' not in df_k:
                continue
            good_k = np.isfinite(df_k['T_K'].values)
            rr, cc = df_k['ind_x'].values[good_k], df_k['ind_y'].values[good_k]
            inside = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1) & ~((rr == r_) & (cc == c_))
            for rr_, cc_ in zip(rr[inside], cc[inside]):
                axz.add_patch(plt.Rectangle((cc_ - 0.5, rr_ - 0.5), 1, 1, fill=False, ec='white', lw=0.5))
        axz.set_xticks([]); axz.set_yticks([])
        axz.set_title(f'{(r1 - r0) * 30 / 1000:.2f} km box, true scale\ncyan: this pixel; white: other detections',
                      fontsize=7.5)
        fig.subplots_adjust(left=0.09, right=0.98, top=0.93, bottom=0.09)
        p = os.path.join(output_dir, f'Fig_MethodSpectralFit_{scene_id}_{date_tag}.png')
        fig.savefig(p, dpi=dpi, bbox_inches='tight'); plt.close(fig); paths.append(p)

    return paths

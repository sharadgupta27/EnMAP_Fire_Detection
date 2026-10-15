#!/usr/bin/env python
# coding: utf-8
"""
Temperatures of saturated fire pixels from the atmospheric absorption bands:
the atmosphere as a graded neutral-density filter (this work; VERIFICATION.md
Section 29).

Idea. Every retrieval in this repository - and Dozier (1981), Dennison et
al. (2006), Amici et al. (2022), Keely et al. (2026) - discards the bands
inside the strong water-vapour absorptions (1.34-1.46 um, 1.79-1.96 um) and
beyond 2.39 um, because there the atmosphere removes most of the signal.
For a pixel whose fire saturates the detector in the window bands, that is
exactly what makes these bands useful: the fire's radiance reaches the
sensor attenuated by a one-way transmittance tau(lambda) that falls
continuously from ~1 in the wings to ~0 in the band centres, so the
absorption bands act as a set of neutral-density filters of graded
strength. A fire bright enough to saturate every window band still leaves
the attenuated bands inside the linear range of the detector.

What is needed is tau(lambda) for the FIRE term in those bands - the path
from the flame to the sensor, which differs from the background's two-way
path and includes the plume. It is measured on the same scene: for the
adequately fitted, unsaturated fire pixels, (T, f) are known from the window
bands, so each such pixel gives tau_ij = (y_ij - b_ij) / (f_i B(lambda_j,
T_i)) in every absorption band j (b_ij its fitted background); the scene
estimate is the signal-weighted median over pixels and its pixel-to-pixel
spread (plume variability, background and temperature errors together) is
carried into the retrieval as a per-band model error. The retrieval of a
saturated pixel is then the profile-likelihood fit of profile_retrieval()
on its usable bands - the absorption bands plus whatever window bands
remain - with tau as the fire-term transmittance.

Validation (Section 29): the method is scored by HOLDING OUT the window
bands of adequately fitted unsaturated pixels (whose window-band v2
temperature is the reference), with tau estimated from the other half of
the pixels - the same yardstick as Section 24.7.

STATUS: NOT USED BY THE PIPELINE ON EnMAP (Section 29). The principle holds
- with the absorption bands alone, held-out pixels are retrieved without
bias (-14 K, MAE 64 K, 95% intervals ~420 K wide, coverage 0.97) - but the
EnMAP L1B product delivers no band between 1780 and 1968 nm, the bands
beyond 2.39 um of the saturated pixels are themselves at 92-97% of full
scale and flagged, and of the 1.4 um bands only 1343, 1355 and 1449 nm
carry enough fire signal in ordinary pixels for a transmittance estimate.
A saturated pixel therefore gains ~3 bands; degraded to that pattern,
held-out pixels come out with MAE 140 K and intervals that cover the
reference in 25%. Kept for sensors that record the 1.9 um wings
unsaturated (e.g. AVIRIS-type instruments) - to be re-validated there.

Caveats. (1) The flame-to-sensor column of a saturated pixel (the most
intense fires) can hold more plume water vapour than that of the pixels tau
was measured on; the pixel-to-pixel spread of tau enters the uncertainty,
but a systematically wetter plume would bias the result. (2) The detector
response compresses below the 63 000 DN censoring limit (Section 24.2), so
bands above LINEAR_DN_FRACTION of full scale are lower bounds, not values.
"""
import numpy as np

import profile_retrieval as pr

ABSORPTION_REGIONS_NM = (('H2O 1.4 um', 1340.0, 1460.0),
                         ('H2O 1.9 um', 1790.0, 1960.0),
                         ('H2O/CO2 > 2.39 um', 2393.0, 2500.0))
LINEAR_DN_FRACTION = 0.85     # above: a lower bound (detector compression, Section 24.2)
DN_FULL_SCALE = 65535
TAU_MIN = 0.03                # weaker bands carry too little fire signal to estimate tau
MIN_TAU_PIXELS = 10           # pixels needed for a band's tau estimate
MIN_USABLE_BANDS = 8


def absorption_mask(wavelength):
    """Bands inside the absorption regions."""
    w = np.asarray(wavelength, dtype=float)
    m = np.zeros(len(w), bool)
    for _, lo, hi in ABSORPTION_REGIONS_NM:
        m |= (w >= lo) & (w <= hi)
    return m


def _noise_sigma(y, sigma_inst, ref):
    """Instrument noise of the two-regime model of profile_retrieval."""
    return sigma_inst * np.sqrt(np.maximum(np.abs(y) / np.maximum(ref, 1e-300), 1.0))


def fixed_T_fit(y, w, lib, T, tau, sigma, emissivity=1.0, closure_weight=1e4):
    """Fire fraction and background abundances of one pixel at a FIXED
    temperature (the v2 estimate), on the bands given. Returns (f, a, chi2)
    with a the library abundances (shade excluded)."""
    fire = emissivity * tau * pr._planck_nm(w, T)
    A = np.column_stack([fire, lib.T, np.zeros(len(w))])
    a, chi2, _, _ = pr.weighted_fcls(A, y, sigma, closure_weight=closure_weight)
    return float(a[0]), a[1:-1], chi2


def estimate_absorption_tau(spectra, wavelength, libraries, T, window_mask, tau_window, usable,
                            sigma_inst, ref, emissivity=1.0, pixels=None):
    """Fire-term transmittance in the absorption bands from pixels whose
    temperature is known from the window bands.

    spectra (N, n_bands) full axis; libraries list of (J, n_bands) full axis;
    T (N,) window-band temperatures; window_mask (n_bands,) the bands of
    the v2 fit, tau_window (n_bands,) the v2 fire-term transmittance (used
    on window_mask); usable (N, n_bands) bands that are linear and
    unflagged; sigma_inst, ref (n_bands,) the noise model; pixels: indices
    to use (default all with finite T).

    Returns dict(tau (n_bands,) NaN outside the absorption bands / where
    not estimable, log_spread (n_bands,) robust sd of ln tau_ij over
    pixels, n_pixels (n_bands,), tau_ij (n_used, n_bands), pixels).
    """
    w = np.asarray(wavelength, dtype=float)
    X = np.asarray(spectra, dtype=float)
    am = absorption_mask(w)
    win = np.asarray(window_mask, bool)
    pixels = np.flatnonzero(np.isfinite(T)) if pixels is None else np.asarray(pixels)
    n_b = len(w)
    tij = np.full((len(pixels), n_b), np.nan)
    wij = np.zeros((len(pixels), n_b))
    for k, i in enumerate(pixels):
        y = X[i]
        L = np.asarray(libraries[i], dtype=float)
        sel = win & usable[i]
        if sel.sum() < MIN_USABLE_BANDS:
            continue
        sig = _noise_sigma(y[sel], sigma_inst[sel], ref[sel])
        f, a, _ = fixed_T_fit(y[sel], w[sel], L[:, sel], T[i], tau_window[sel], sig, emissivity)
        if not f > 0:
            continue
        bands = am & usable[i]
        fire = f * emissivity * pr._planck_nm(w[bands], T[i])
        b = a @ L[:, bands]
        tij[k, bands] = (y[bands] - b) / fire
        sn = _noise_sigma(y[bands], sigma_inst[bands], ref[bands])
        wij[k, bands] = (fire / sn) ** 2
    tau = np.full(n_b, np.nan)
    spread = np.full(n_b, np.nan)
    npx = np.zeros(n_b, int)
    for j in np.flatnonzero(am):
        ok = np.isfinite(tij[:, j]) & (wij[:, j] > 0)
        npx[j] = int(ok.sum())
        if npx[j] < MIN_TAU_PIXELS:
            continue
        t, wt = tij[ok, j], wij[ok, j]
        o = np.argsort(t)
        cw = np.cumsum(wt[o]) / wt.sum()
        tau[j] = float(t[o][np.searchsorted(cw, 0.5)])
        pos = t > 0
        if pos.sum() >= MIN_TAU_PIXELS:
            lt = np.log(t[pos])
            med = np.log(max(tau[j], 1e-6))
            spread[j] = float(1.4826 * np.median(np.abs(lt - med)))
    return {'tau': tau, 'log_spread': spread, 'n_pixels': npx, 'tau_ij': tij, 'pixels': pixels}


def retrieve_pixel(y, wavelength, library, tau, log_spread, use, lower_bound, sigma_inst, ref,
                   min_bands=MIN_USABLE_BANDS,
                   spread_scale=1.0, eps_floor=0.0, emissivity=1.0, T_bounds=(300.0, 2000.0),
                   n_grid=200):
    """Profile-likelihood retrieval of one pixel on the bands `use` (data)
    and `lower_bound` (censored: model >= recorded radiance), with the
    fire-term transmittance `tau` and a per-band model error
    spread_scale * log_spread * y (the tau spread, as a relative error) plus
    eps_floor * y, added in quadrature to the instrument noise.
    Returns the profile_retrieval result dict (n_bands_used added)."""
    w = np.asarray(wavelength, dtype=float)
    sel = (use | lower_bound) & np.isfinite(tau) & (tau > 0)
    if (use & sel).sum() < min_bands:
        out = pr._nan_result()
        out['reason'] = f'only {int((use & sel).sum())} usable bands'
        out['n_bands_used'] = int((use & sel).sum())
        return out
    ys = np.asarray(y, dtype=float)[sel]
    rel = np.sqrt((spread_scale * np.nan_to_num(log_spread[sel], nan=0.0)) ** 2 + eps_floor ** 2)
    sig = np.sqrt(_noise_sigma(ys, sigma_inst[sel], ref[sel]) ** 2 + (rel * np.abs(ys)) ** 2)
    cens = lower_bound[sel] & ~use[sel]
    res = pr.profile_retrieval(ys, w[sel], np.asarray(library, dtype=float)[:, sel], emissivity=emissivity,
                               sigma=sig, tau=tau[sel], T_bounds=T_bounds, n_grid=n_grid,
                               censored_mask=cens if cens.any() else None,
                               censor_level=np.where(cens, ys, np.nan) if cens.any() else None)
    res['n_bands_used'] = int((use & sel).sum())
    res['n_bands_lower_bound'] = int(cens.sum())
    return res


def band_classes(dn, valid, censored, wavelength, linear_fraction=LINEAR_DN_FRACTION):
    """Per pixel and band: 'use' (linear, unflagged, not censored) and
    'lower' (unflagged and above the linear range, or censored: the
    recorded radiance is a lower bound). Arrays (N, n_bands) bool."""
    dn = np.asarray(dn)
    lin = dn < linear_fraction * DN_FULL_SCALE
    use = np.asarray(valid, bool) & lin
    lower = (np.asarray(valid, bool) & ~lin) | np.asarray(censored, bool)
    return use, lower

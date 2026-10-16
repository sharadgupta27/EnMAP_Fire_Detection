#!/usr/bin/env python
# coding: utf-8
"""
Wildfire temperature estimation from EnMAP hyperspectral imagery.

Adapted from the PRISMA pipeline in
'Wildfires-Temperature-Estimation-with-PRISMA-main/Fire_analysis_with_PRISMA.py'
(Amici, INGV). The multi-temperature Planck-mixture fit (Planck_function,
fitClass, estimate_temperature, main_temp_estimation) is reused essentially
unchanged, since that part of the pipeline is sensor-agnostic physics. What
changes is everything upstream and downstream of it:

  * Ingestion: a single EnMAP ENVI/GeoTIFF radiance cube + its .hdr, read
    directly through GDAL, instead of PRISMA's HDF5 (.he5) VNIR/SWIR
    subdatasets. No ancillary shapefiles are required.
  * AOI selection: fire/hot pixels are found automatically from the image
    itself via a SWIR hotspot index + a robust MAD-based threshold, instead of
    prescribed ROI shapefiles.
  * Geolocation: EnMAP products carry an affine GeoTransform, so lat/lon
    per pixel is computed from that directly (no lat/lon subdatasets, no
    geopandas rotation-estimation hack). Note this is a ROTATED affine for
    L1B - the ENVI header's map info carries a rotation term (-10.61 deg on
    the product this was checked against), which GDAL reports as non-zero
    gt[2]/gt[4]. Both the forward grid construction in
    _load_single_enmap_raster() and the inverse lookup in
    _sample_raster_at_lonlat() use the full six-term affine, so the
    rotation is handled; code added here must not assume an axis-aligned
    north-up grid. L1B is radiometrically corrected but not orthorectified,
    so that affine is an approximation of the true per-pixel geolocation -
    adequate for matching against 70 m ECOSTRESS and 30-100 m Landsat,
    but not a substitute for L1C/L2A geometry if you need precise
    co-registration.
  * Validation: optional cross-checks against field-measured temperatures
    and against Landsat 8/9 Collection-2 Level-2 surface temperature (ST)
    scenes.

IMPORTANT - product level.
The temperature retrieval fits a Planck (blackbody) radiance model against
the pixel spectrum, so it needs at-sensor or at-surface SPECTRAL RADIANCE
in physical units (W/(m^2*sr*nm) or convertible to it) - exactly what
PRISMA's L1/L2B products provide. EnMAP's L2A product is Bottom-of-Atmosphere
REFLECTANCE (dimensionless), which cannot be fit against Planck's law
directly. Point this script at an EnMAP L1B (at-sensor radiance) or L1C
(orthorectified at-sensor radiance) product, not L2A. This is a physical
requirement, not a formatting detail - running it on L2A reflectance will
produce numbers that look like temperatures but are not.

Verification status.
The ingestion path has since been exercised against a real DLR EnMAP L1B
product (datatake DT0000009864, 2023-03-14, Myanmar/Mizoram):
  * a 1000 x 1024 x 224 float32 cube loads in ~6 s, with wavelengths read
    from the ENVI .hdr spanning 418.42-2445.30 nm, monotonic;
  * the per-band GainOfBand/OffsetOfBand from the product's METADATA.XML
    reproduce the merged cube's radiance to <1e-9 (spot-checked on several
    bands), giving a scene maximum of ~0.3 W/(m^2*sr*nm) - the same units
    Planck_function() returns, thanks to its 1e-3 factor;
  * the rotated L1B affine round-trips correctly through the lat/lon grid.
The ECOSTRESS and Landsat loaders and validate_temperature() have likewise
been run against real ECO_L2T_LSTE and Landsat C2 L2 ST granules.

One ordering subtlety found during that check, worth knowing before
touching the band bookkeeping: _merge_vnir_swir_scenes() sorts the merged
cube by wavelength, so VNIR and SWIR bands interleave through the detector
overlap near 900-1000 nm, whereas METADATA.XML lists all 224 bands as a
VNIR block followed by a SWIR block. The two orderings are NOT
interchangeable (metadata entry 91 is 993.34 nm; the merged cube's band 91
is 951.36 nm). Nothing currently mixes them, and
_warn_if_wavelengths_unordered() trips if something starts to.

The Planck-mixture physics core is unchanged from the PRISMA original and
has not been independently re-validated here.

Check the header field names your product actually uses (EnMAP's own
processor, Metaspectral, or a re-exported product may differ slightly) with
`inspect_envi_header()` before a full run.
"""

import os
import re
import sys
import json
import glob
import math
import time
import shutil
import warnings
import contextlib
import urllib.request
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from datetime import date, datetime
from random import seed, uniform

import numpy as np
import pandas as pd
import scipy.stats as st
import scipy.ndimage as ndi
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter
from tqdm import tqdm

# solar_time.py sits beside this file. Importing it by name works whenever
# this directory is on sys.path, but the verification scripts load this module
# by PATH (importlib.util.spec_from_file_location), which does not put its
# directory on sys.path - so bootstrap it rather than making how this file was
# loaded decide whether it imports.
try:
    from solar_time import hours_of_day, solar_hour_from_utc
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solar_time import hours_of_day, solar_hour_from_utc


seed(1)


class PipelineCancelled(Exception):
    """Raised by run_pipeline() when a caller-supplied cancel_event was set
    before the run finished. Distinct from a real failure so a caller (e.g.
    the GUI) can show "Cancelled" rather than "Error"."""


#%% ###########################################################################
#                       Reused physics core (from PRISMA script)
################################################################################


# The two radiation constants of Planck_function(), named so that
# fitClass can reuse exactly the same values (and so the same rounding).
_PLANCK_C1 = 1e-3 * 1.191042 * 1e8
_PLANCK_C2 = 1.4387752 * 1e4


def Planck_function(x, Temp, emissivity=1.0):
    """Spectral radiance of a blackbody. x in micrometres, Temp in Kelvin."""
    c1 = _PLANCK_C1
    c2 = _PLANCK_C2
    y_planck_T = emissivity * (c1 / (x**5)) * (1 / (np.exp(c2 / (Temp * x)) - 1))
    return y_planck_T


class fitClass:

    def __init__(self):
        self.bkg = []
        self.emissivity = []
        self.N_sources = []
        self.absorption_bands = []
        self._prep = None
        self._planck_cache = {}

    def _prepare(self, x):
        """The factors of the forward model that do not depend on the fitted
        parameters: the wavelengths in micrometres, Planck_function()'s
        emissivity * (c1 / x**5), and the blanked channels as indices.

        This is the inner loop of the whole Amici et al. (2022) stage.
        curve_fit evaluates linear_mixture_function() once per residual and
        once more per parameter for each finite-difference Jacobian - with
        one fire source and 20 background spectra that is 23 evaluations per
        optimiser step, every one of them on the same wavelength grid. The
        conversion to micrometres, the fifth power and the c1/emissivity
        factor are therefore hoisted out and computed once per fit instead
        of 23 times per step. What is left inside the call is the same
        arithmetic, term for term and in the same association order as
        Planck_function(), so the fitted spectrum is unchanged bit for bit
        (checked in test_correctness.py).

        The cache is keyed on the identity of the inputs rather than their
        values: curve_fit passes the same xdata object on every call, and
        estimate_temperature() builds a fresh fitClass per fit, so a miss
        only costs a recomputation and never returns a stale array."""
        p = self._prep
        if p is not None and p[0] is x and p[1] is self.absorption_bands \
                and p[2] is self.emissivity:
            return p[3], p[4], p[5]
        x_um = x * 1e-3
        planck_scale = self.emissivity * (_PLANCK_C1 / (x_um ** 5))
        ab = np.asarray(self.absorption_bands)
        blank = np.flatnonzero(ab) if ab.size else None
        if blank is not None and blank.size == 0:
            blank = None                      # nothing to zero: skip the write
        self._prep = (x, self.absorption_bands, self.emissivity, x_um, planck_scale, blank)
        self._planck_cache.clear()        # keyed on the temperature alone
        return x_um, planck_scale, blank

    def linear_mixture_function(self, x, *args):
        sum_fire_sources = 0
        bkg = self.bkg
        N_sources = self.N_sources
        x_um, planck_scale, blank = self._prepare(x)

        # One conversion of the parameter tuple instead of two tuple slices
        # and two conversions; the slices below are views of it. ndarray.sum()
        # rather than np.sum() is the same reduction without numpy's
        # dispatch wrapper, which is worth having at 31 elements a call.
        params = np.asarray(args, dtype=float)
        Norm_coeffs = params[N_sources:].sum()

        # The Planck term depends on the temperatures alone. A
        # finite-difference Jacobian perturbs one parameter at a time, so 21
        # of the 22 evaluations that build it leave every temperature where
        # it was: cache the blanked Planck vector per temperature and the
        # exponential is evaluated once per Jacobian instead of 22 times.
        # The cached array is never written to after it is blanked (the
        # multiply below returns a new array), so the caller cannot see a
        # mutated one, and the value returned is the same either way.
        cache = self._planck_cache
        for i in range(N_sources):
            Temp = params[i]
            p_fire = params[i + N_sources] / Norm_coeffs

            key = (i, float(Temp))
            y_planck_T = cache.get(key)
            if y_planck_T is None:
                y_planck_T = planck_scale * (1 / (np.exp(_PLANCK_C2 / (Temp * x_um)) - 1))
                if blank is not None:
                    y_planck_T[blank] = 0
                if len(cache) >= 2 * N_sources + 2:
                    cache.clear()
                cache[key] = y_planck_T

            sum_fire_sources += p_fire * y_planck_T

        p_bkg = params[N_sources * 2:].reshape(bkg.shape[0], 1) / Norm_coeffs
        sum_backgrounds = (bkg * p_bkg).sum(0)
        fitted_curve = sum_fire_sources + sum_backgrounds

        return fitted_curve

    def jacobian(self, x, *args):
        """Exact derivative of linear_mixture_function(), for curve_fit's
        `jac`. Off unless config['amici_analytic_jacobian'] is set - see
        DEFAULT_CONFIG for why the default must stay off.

        With g the unnormalised mixture, S the sum of the abundances and
        f = g / S the modelled spectrum,

            df/dT_i = (p_i / S) dB(T_i)/dT,  dB/dT = B z e^z / ((e^z - 1) T)
            df/dp_i = (B_i - f) / S         (fire abundances)
            df/da_j = (b_j - f) / S         (background abundances)
        """
        bkg = self.bkg
        N = self.N_sources
        x_um, planck_scale, blank = self._prepare(x)
        params = np.asarray(args, dtype=float)
        S = params[N:].sum()

        planck = np.empty((N, len(x_um)))
        dplanck = np.empty((N, len(x_um)))
        for i in range(N):
            z = _PLANCK_C2 / (params[i] * x_um)
            ez = np.exp(z)
            b = planck_scale * (1 / (ez - 1))
            db = b * (z * ez / (ez - 1)) / params[i]
            if blank is not None:
                b[blank] = 0.0
                db[blank] = 0.0
            planck[i], dplanck[i] = b, db

        f = (params[N:2 * N] / S) @ planck + (params[2 * N:] / S) @ bkg
        J = np.empty((len(x_um), 2 * N + bkg.shape[0]))
        J[:, :N] = ((params[N:2 * N] / S)[:, None] * dplanck).T
        J[:, N:2 * N] = ((planck - f) / S).T
        J[:, 2 * N:] = ((bkg - f) / S).T
        return J


def estimate_temperature(ydata, xdata, bkg, emissivity, N_sources,
                          plot, verbose, T_min, T_max, opt_method,
                          absorption_blank_threshold=None, fire_abundance_min=0.05,
                          rng=None, max_nfev=None, failure_log=None,
                          analytic_jacobian=False):
    """Fit a Temp + fire-fraction + background-mixture model to one pixel spectrum.

    absorption_blank_threshold restores the legacy behaviour of zeroing the
    Planck term wherever `ydata < threshold`. It defaults to None (disabled)
    because that rule is unsound: it is an ABSOLUTE radiance cut, so it is
    unit-dependent, and it removes the fire term while leaving the
    background term in place - an internally inconsistent model over
    whichever channels it touches. How much it touches depends entirely on
    the fit window: 1 of 28 bands over 1400-1900 nm, but 62 of 89 (70%)
    over the full SWIR. Measured effect of disabling it, with the full SWIR
    window, p=0.02, SNR=150: bias at 600 K improves from -223.1 K to
    -10.4 K, at 800 K from -21.4 K to -3.2 K, at 1000 K from -27.8 K to
    -5.3 K. Opaque channels are now dropped from the residual entirely in
    main_temp_estimation() instead, which is the physically consistent way
    to ignore them. Set a float here only to reproduce older results.

    fire_abundance_min is the lower bound on each UNNORMALISED fire
    abundance. Because the model depends only on abundance ratios, it
    imposes an implicit floor on the retrievable fire fraction of
    fire_abundance_min / (N_sources + n_background) - a tuning artefact, not
    a radiometric limit. See main_temp_estimation()'s reported
    `fire_fraction_floor`.

    rng, if given, is a numpy Generator used for the random restart, so that
    parallel workers do not all replay the same sequence (the module-level
    `seed(1)` is inherited identically by every joblib worker)."""

    draw = (rng.uniform if rng is not None else uniform)
    # One INDEPENDENT restart per source. The original repeated a single
    # draw N_sources times, which starts every component at the same
    # temperature - a symmetric point the optimiser has no gradient reason
    # to leave, since the model is invariant under permuting components.
    x0 = ([float(draw(T_min, T_max)) for _ in range(N_sources)]
          + [0.5] * N_sources
          + [1 / (bkg.shape[0] * 2)] * bkg.shape[0])
    x0 = np.array(x0)

    bounds = ([(T_min, T_max)] * N_sources + [(fire_abundance_min, 1)] * N_sources
              + [(0, 1)] * bkg.shape[0])
    bounds = np.array(bounds).T

    inst = fitClass()
    inst.bkg = bkg
    inst.emissivity = emissivity
    inst.N_sources = N_sources
    inst.absorption_bands = (np.zeros(len(ydata), dtype=bool)
                             if absorption_blank_threshold is None
                             else ydata < absorption_blank_threshold)

    try:
        popt = optimization.curve_fit(inst.linear_mixture_function, xdata, ydata,
                                       x0, bounds=bounds, gtol=1e-8, ftol=1e-8,
                                       xtol=1e-8, verbose=verbose, method=opt_method,
                                       max_nfev=max_nfev,
                                       **({'jac': inst.jacobian} if analytic_jacobian else {}))
    except Exception as exc:
        # Expected, already-handled noise, not a sign of a broken run:
        # curve_fit occasionally fails to converge for a particular random
        # background draw (the RuntimeError scipy raises when it can't find
        # a fit within max_nfev, or a ValueError from a degenerate draw).
        # main_temp_estimation() retries with a fresh random background draw
        # (up to 5x per trial) specifically to absorb this, and only a
        # pixel where every retry across every trial fails ends up NaN.
        #
        # So this is NOT warned about here any more. Warning per attempt
        # meant a single self-healed hiccup printed a scary message, while
        # Python's default "show each warning once" filter simultaneously
        # hid how often it really happened - the worst of both. The message
        # is handed to the caller via failure_log instead, and
        # main_temp_estimation() warns once per pixel only if a trial is
        # actually lost, reporting the count. `n_fit_failures` in the
        # per-pixel result records the rest.
        if failure_log is not None:
            failure_log.append(f'{exc.__class__.__name__}: {exc}')
        else:
            warnings.warn(f'The optimization process was not successful '
                          f'({exc.__class__.__name__}: {exc}).')
        popt = []

    if (plot is True) and (len(popt) > 1):
        _ = plt.figure()
        a = inst.linear_mixture_function(xdata, *popt[0])
        plt.style.use('ggplot')
        plt.plot(xdata, a, color='red', label='Linear mixture model')
        plt.plot(xdata, ydata, color='blue', label='Fire Signal')
        plt.legend(loc='upper right', fontsize=15)
        plt.xlabel('Wavelength ($nm$)', fontsize=15)
        plt.ylabel(r'Spectral radiance ($W/(m^2\cdot sr \cdot nm)$)', fontsize=15)

    if len(popt) > 0:
        # Gauge fixing. The model depends on the abundances only through
        # their ratios, so (a, b) -> c*(a, b) is an exact continuous
        # symmetry: verified to leave the modelled spectrum unchanged to
        # <=5.6e-17, and visible as a single zero singular value in the
        # Jacobian (3.5e-11 against a leading 2.19). The optimiser
        # therefore stops at an arbitrary point along that ray. Dividing by
        # Norm_coeffs picks the canonical representative with
        # sum(abundances) = 1, which is what makes the returned fractions
        # reproducible rather than gauge-dependent. Temperatures are
        # unaffected - the null direction is orthogonal to them.
        Norm_coeffs = np.sum(popt[0][N_sources:])
        temps = np.asarray(popt[0][:N_sources], dtype=float)
        fractions = np.asarray(popt[0][N_sources:], dtype=float) / Norm_coeffs
        # Components are exchangeable, so their fitted order is arbitrary.
        # Sort hottest-first, carrying each component's own fire fraction
        # with it, so that "component 0" means the same physical thing in
        # every trial and averaging across trials is meaningful.
        order = np.argsort(temps)[::-1]
        temps = temps[order]
        fractions = np.concatenate([fractions[:N_sources][order], fractions[N_sources:]])
        output = [list(temps), list(fractions)]
    else:
        output = []

    return output


# Wavelength ranges (nm) where the atmosphere is effectively opaque in the
# SWIR - the 1.4 um and 1.9 um water-vapour bands. At-sensor radiance there
# carries no surface information, so these channels are removed from the
# fit rather than being modelled. Fixed physical windows, deliberately not
# a threshold on the data itself: the retrieval must not depend on the
# radiometric units the cube happens to be in.
_ATMOSPHERIC_EXCLUSIONS_NM = ((1340.0, 1460.0), (1790.0, 1960.0),
                              (1990.0, 2040.0), (2395.0, 2450.0))

# The last two were added on measured evidence, not by inspection of a
# textbook band list. Fitting the full SWIR and looking at where the
# forward model misses most, the residual peaks at 2005-2014 nm (3.2-3.5x
# the median residual). Checking the scene's own background radiance there:
#
#     1980-2040 nm : 0.00036-0.00209   (CO2 2.01 um)
#     2400-2450 nm : 0.00034-0.00073
#     scene band median                 0.00211
#
# i.e. an ~83% drop - these channels are largely opaque, so at-sensor
# radiance carries little surface information and modelling them with an
# unattenuated Planck curve is wrong. (2090-2145 nm also misfits, but its
# background radiance is 0.0022-0.0023, i.e. NOT absorbed - that residual
# is genuine model inadequacy, most likely unmodelled atmospheric
# transmittance, and is deliberately left in rather than masked away.)

# Refuse to shrink a fit window below this many bands. With N_sources=1 and
# N_bkg_points=20 the model already has 22 free parameters; fitting that to
# fewer bands than this is not a fit.
_MIN_FIT_BANDS = 12


def _nnls_sum_to_one(A, y, weight=1e3):
    """Non-negative least squares for `A f = y` with sum(f)=1 imposed as a
    weighted row. Returns (fractions, sum of squared errors)."""
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    scale = weight * (np.linalg.norm(A) / max(A.shape[0], 1) + 1e-30)
    f, _ = optimization.nnls(np.vstack([A, np.full((1, A.shape[1]), scale)]),
                             np.concatenate([y, [scale]]))
    return f, float(np.sum((A @ f - y) ** 2))


def _nested_f_test(sse0, sse1, n, p0, p1, tiny=0.0):
    """Standard nested-model F statistic (H0, with p0 parameters, nested in
    H1, with p1 > p0), given each model's sum of squared errors over n
    points. Shared by fire_component_significance() (is a first Planck
    component needed at all?) and escalation_significance() (is a second
    one needed on top of the first?), so both apply the same degenerate-
    branch handling rather than two copies drifting apart.

    The degenerate branches resolve toward H0, not H1: a vanishing sse1
    makes the textbook F = [(sse0-sse1)/(p1-p0)] / [sse1/dof] a 0/0 or a
    division that reads as "infinitely significant" even when H0 already
    explains the data (sse0 at or below `tiny`, a floor on the data's own
    scale so this does not depend on the radiance units) or when H1 did not
    improve on H0 at all. F=inf is kept only for the one case that deserves
    it: H1 fits exactly and H0 does not. See fire_component_significance()
    for the reachable scenario (the background/first-component pool can
    contain a near-duplicate of the pixel itself) that makes this matter in
    practice rather than only in principle."""
    dof = max(n - p1, 1)
    improvement = sse0 - sse1
    if not (np.isfinite(sse0) and np.isfinite(sse1)):
        F = 0.0
    elif sse0 <= tiny:
        F = 0.0          # H0 alone already fits to numerical precision
    elif improvement <= 0:
        F = 0.0          # the added component made the fit no better
    elif sse1 <= 0:
        F = np.inf        # only H1 fits exactly - genuinely significant
    else:
        F = (improvement / max(p1 - p0, 1)) / (sse1 / dof)
    p_value = float(1.0 - st.f.cdf(F, p1 - p0, dof)) if np.isfinite(F) else 0.0
    return float(F), p_value, dof


def fire_component_significance(radiance, wavelength, background_endmembers, T_K,
                                emissivity=1.0):
    """Is the thermal component actually needed to explain this spectrum?

    The retrieval is bounded (T_min..T_max) and therefore ALWAYS returns a
    temperature, for a burning pixel and a bare-soil pixel alike. Nothing in
    the fit itself says whether the Planck term earned its place. Without
    that test a temperature map is not falsifiable, which is a problem for
    publication.

    So compare two nested models over the same bands, both solved by
    non-negative least squares with sum-to-one closure:

        H0: background endmembers only                    (p0 parameters)
        H1: background endmembers + eps*B(lambda, T)      (p0 + 1)

    and form the standard nested-model F statistic

        F = [(SSE0 - SSE1) / (p1 - p0)] / [SSE1 / (n - p1)]

    Returned alongside the fractional RMSE reduction, which is the more
    robust of the two numbers: F assumes independent, homoscedastic
    residuals, and hyperspectral residuals are correlated between adjacent
    bands, so F is optimistic in absolute terms. Use it to RANK pixels and
    to reject the clearly-insignificant ones, not as an exact p-value.

    Measured on this scene (DT0000009864), adding the thermal component cuts
    RMSE by 64-78% with F = 426-1226 across hot, middling and weak
    detections - so on that scene the component is strongly justified
    everywhere, and the residual 8-19% misfit is model inadequacy rather
    than absence of fire.

    Degenerate cases resolve toward H0: a spectrum the background already
    explains, or one the fire term does not improve, returns F = 0 and
    p = 1 rather than being declared significant. See the branch below.
    """
    radiance = np.asarray(radiance, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    B = np.atleast_2d(np.asarray(background_endmembers, dtype=float))
    # Shade endmember (identically zero) mirrors the MESMA formulations and
    # lets H0 absorb overall brightness, so H1 cannot win merely by scaling.
    cols = [B[j] for j in range(B.shape[0])] + [np.zeros_like(wavelength)]

    _, sse0 = _nnls_sum_to_one(np.column_stack(cols), radiance)
    fire = emissivity * Planck_function(wavelength * 1e-3, float(T_K), 1.0)
    _, sse1 = _nnls_sum_to_one(np.column_stack([fire] + cols), radiance)

    n, p0 = len(radiance), len(cols)
    p1 = p0 + 1
    # `tiny`, a numerical floor on the data's own scale rather than an
    # absolute epsilon, is what keeps the degenerate branches below reading
    # relative to the measurement instead of depending on the radiance
    # units - see _nested_f_test() for why they matter: the endmember pool
    # is drawn from the same scene, so it can reach the case where it
    # contains the pixel itself or a near-duplicate of it, which is not a
    # hypothetical.
    scale = float(np.sum(radiance ** 2))
    tiny = 1e-12 * scale if np.isfinite(scale) and scale > 0 else 0.0
    F, p_value, dof = _nested_f_test(sse0, sse1, n, p0, p1, tiny=tiny)
    rmse0, rmse1 = math.sqrt(sse0 / n), math.sqrt(sse1 / n)
    return {'rmse_background_only': rmse0,
            'rmse_with_fire': rmse1,
            'rmse_reduction_frac': (1.0 - rmse1 / rmse0) if rmse0 > 0 else np.nan,
            'F_statistic': F,
            'p_value': p_value,
            'n_bands': int(n)}


################################################################################
#     Adaptive two-component escalation for search-bound-pinned pixels
################################################################################
#
# VERIFICATION.md section 14 found that the hottest pixels can peg at the
# single-component search bound (T_max): the model is not failing to
# converge, it genuinely cannot reproduce the measurement with one Planck
# curve, because the pixel is bright enough that a single component would
# need an unphysical temperature (section 14.1's worked example needs
# ~2850 K, above any wildfire flame temperature) to explain both a short-
# and a long-wavelength excess at once. What such a pixel returns is a
# LOWER LIMIT, not a retrieval, and n_trials_at_T_bound / T_at_bound exist
# to flag that rather than let it silently bias an aggregate.
#
# Section 10.1 also found, separately, that fitting every pixel with a
# second (flaming + smouldering) Planck component is NOT supported by
# these data as a routine method: against single-component synthetic truth
# it added mean absolute bias of 19.2 K (worst case +99.5 K) relative to
# 2.5-3.1 K for the single-component fit, because the extra freedom
# absorbs noise on pixels that never needed it.
#
# Both findings are true at once, and the resolution is not to pick one:
# offer the second component ONLY to the pixels the first finding shows
# cannot be explained without it (T_at_bound), and only KEEP it when
# escalation_significance() below - the same nested-model logic
# fire_component_significance() already uses to justify the first
# component - confirms the second one earns its place on that specific
# pixel. A pixel whose escalated fit is not significantly better, or whose
# escalated fit is itself unconstrained, keeps its single-component result
# and its T_at_bound flag rather than being reported as a spurious
# two-temperature retrieval.


def escalation_significance(radiance, wavelength, background_endmembers, T1_K, T2_K,
                            emissivity=1.0):
    """Is a SECOND thermal component justified on top of a first, already-
    fitted one?

    The one-component analogue of this question is what
    fire_component_significance() answers (is any thermal component needed
    over the background alone?); this is the next nested comparison up,
    nested at T1_K rather than at the background alone:

        H0: background endmembers + shade + eps*B(lambda, T1)      (p0 params)
        H1: H0 + eps*B(lambda, T2)                                 (p0 + 1)

    T1_K and T2_K are both held fixed at their own independently-fitted
    values (from a joint two-component estimate_temperature() call) rather
    than being free parameters of this test, exactly as
    fire_component_significance() already fixes T at its NLLS point
    estimate before testing significance - the same accepted approximation,
    not a new one. Passing T1_K and T2_K from the SAME two-component fit
    (rather than pairing T2_K with a different single-component fit's T1)
    keeps H0 a literal sub-model of H1: dropping the eps*B(lambda, T2) term
    from H1 and nothing else recovers H0 exactly.

    Returns the same shape of dict as fire_component_significance() (RMSE
    for both models, the fractional reduction, F, p-value and band count),
    with keys named for one-vs-two components rather than none-vs-one. See
    _nested_f_test() for the degenerate-branch handling and why F should be
    used to rank/gate pixels rather than read as an exact p-value."""
    radiance = np.asarray(radiance, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)
    B = np.atleast_2d(np.asarray(background_endmembers, dtype=float))
    cols = [B[j] for j in range(B.shape[0])] + [np.zeros_like(wavelength)]

    fire1 = emissivity * Planck_function(wavelength * 1e-3, float(T1_K), 1.0)
    fire2 = emissivity * Planck_function(wavelength * 1e-3, float(T2_K), 1.0)

    _, sse0 = _nnls_sum_to_one(np.column_stack([fire1] + cols), radiance)
    _, sse1 = _nnls_sum_to_one(np.column_stack([fire1, fire2] + cols), radiance)

    n = len(radiance)
    p0 = len(cols) + 1
    p1 = p0 + 1
    scale = float(np.sum(radiance ** 2))
    tiny = 1e-12 * scale if np.isfinite(scale) and scale > 0 else 0.0
    F, p_value, dof = _nested_f_test(sse0, sse1, n, p0, p1, tiny=tiny)
    rmse0, rmse1 = math.sqrt(sse0 / n), math.sqrt(sse1 / n)
    return {'rmse_one_component': rmse0,
            'rmse_two_component': rmse1,
            'rmse_reduction_frac': (1.0 - rmse1 / rmse0) if rmse0 > 0 else np.nan,
            'F_statistic': F,
            'p_value': p_value,
            'n_bands': int(n)}


def two_component_escalation(fire_pixel, xdata, bkg, config, rng,
                             p_value_max=0.001, min_rmse_reduction_frac=0.5,
                             min_temperature_separation_K=150.0):
    """Re-fit one already-SWIR-windowed pixel spectrum with a two-Planck-
    component model, and decide via escalation_significance() whether the
    second component is worth keeping.

    fire_pixel/xdata/bkg carry exactly the shapes a single
    estimate_temperature() call over one background draw takes (already
    restricted to the fit window) - this function only answers, for a
    pixel the CALLER has already decided is worth asking (see the module
    note above: that means T_at_bound in main_temp_estimation()), whether
    escalating helps. It does not decide which pixels to ask.

    The default gates are deliberately far stricter than a textbook
    alpha=0.05/any-improvement pair, for a reason found empirically while
    validating this function (algorithm_verification/test_correctness.py
    sections 13-14): with two-Planck-component freedom, EVEN A PIXEL
    GENERATED FROM A SINGLE TRUE COMPONENT can show a formally
    "significant" improvement once a second free temperature is offered,
    because the bounded joint NLLS practically never converges to the
    exact true parameters, and a second flexible Planck curve can partly
    absorb that small, structured convergence residual - the same failure
    mode VERIFICATION.md section 10.1 documents for the original,
    un-gated two-component method (mean bias 19.2 K on single-component
    synthetic truth), just showing up in the significance test itself
    rather than in the point estimate. A 5% RMSE-reduction floor did not
    reject that failure mode in testing; a 50% floor (still well below the
    64-91% a genuine fire component achieves on real pixels per
    fire_component_significance()'s own measurement) did, so escalation
    now additionally requires the fitted components to be well separated
    in temperature (min_temperature_separation_K) - a spurious second
    component from NLLS imprecision sits close to the first, while a real
    flaming+smouldering pair does not.

    Returns None - meaning "keep the single-component result and its
    T_at_bound flag" - in four cases: the two-component fit does not
    converge; either fitted component itself lands on the search bound
    (which would only relocate the unresolved artefact from one component
    to two, not resolve it); the two fitted temperatures are not separated
    by at least min_temperature_separation_K; or
    escalation_significance() does not clear both the p-value and the
    minimum-effect-size gate. Otherwise returns a dict with both
    components' temperatures and fractions (hottest first, as
    estimate_temperature() already sorts them) plus the significance
    diagnostics, for the caller to report instead of the pinned single-
    component value."""
    two = estimate_temperature(fire_pixel, xdata, bkg, config['emissivity'], 2,
                               False, False, config['T_min'], config['T_max'],
                               config['method'],
                               absorption_blank_threshold=config.get('absorption_blank_threshold'),
                               fire_abundance_min=config.get('fire_abundance_min', 0.05),
                               rng=rng, max_nfev=config.get('max_nfev'),
                               analytic_jacobian=config.get('amici_analytic_jacobian', False))
    if not two:
        return None

    T_hot, T_cool = float(two[0][0]), float(two[0][1])
    f_hot, f_cool = float(two[1][0]), float(two[1][1])

    span = max(abs(float(config['T_max']) - float(config['T_min'])), 1.0)
    edge = max(1e-6, 1e-3 * span)
    if T_hot >= float(config['T_max']) - edge or T_cool >= float(config['T_max']) - edge:
        return None
    if abs(T_hot - T_cool) < min_temperature_separation_K:
        return None

    sig = escalation_significance(fire_pixel, xdata, bkg, T_hot, T_cool, config['emissivity'])
    if sig['p_value'] > p_value_max or not (sig['rmse_reduction_frac'] >= min_rmse_reduction_frac):
        return None

    return {'T_hot_K': T_hot, 'T_cool_K': T_cool, 'f_hot': f_hot, 'f_cool': f_cool,
            'F_statistic': sig['F_statistic'], 'p_value': sig['p_value'],
            'rmse_reduction_frac': sig['rmse_reduction_frac']}


def _swir_band_window(wavelength, config, spectrum_norm=None, warn=False):
    """The fit window `config` selects, as a boolean mask over `wavelength`.

    THE SINGLE SOURCE OF TRUTH for band selection. Both the native retrieval
    (main_temp_estimation) and the alternative published methods
    (cfg_band_mask_for_methods -> temperature_methods.py) come through here,
    so they cannot drift apart. That matters more than ordinary DRY: the fit
    window is the single largest lever on the retrieved temperature
    (VERIFICATION.md section 5), so two methods compared over different
    windows are not being compared at all - the difference would be the
    window, not the method.

    They HAD drifted. The two copies this replaces tested the flags
    differently - `config['only_SWIR'] is True` in one and a plain truthiness
    test in the other - so any truthy non-bool value made them disagree.
    `np.True_`, which is what every numpy boolean expression returns (e.g.
    `cfg['only_SWIR'] = (wl > 1400).any()`), gated the SWIR window for the
    comparison methods and silently did NOT gate it for the native
    retrieval: 78 bands against 180 on a 224-band EnMAP scene. Truthiness is
    the correct reading of an on/off flag and is what both use now, so
    `only_SWIR = 1` now means what it plainly says.

    Parameters
    ----------
    spectrum_norm : 1-D array, optional
        The pixel's own spectrum normalised by its maximum, needed only by
        the per-pixel `only_feasible_SWIR` branch. Scene-level callers pass
        None, which skips that branch - it depends on an individual
        spectrum, so there is no scene-wide mask for it.
    warn : bool
        Warn when the atmospheric exclusion is skipped for leaving too few
        bands. True for the retrieval (where it is a per-pixel event worth
        surfacing), False for the scene-level mask (where it would fire once
        per method per scene and say nothing new).
    """
    wavelength = np.asarray(wavelength, dtype=float)

    if config.get('only_SWIR', True):
        if config.get('only_short_SWIR', False):
            mask = (wavelength > 1400) & (wavelength < 1900)
        elif config.get('only_feasible_SWIR', False) and spectrum_norm is not None:
            # NOTE ON spectrum_norm: in the PRISMA original this argument is
            # the L2C REFLECTANCE cube, so ">0.8" tested for detector
            # saturation (reflectance above 0.8 being unphysical). This
            # pipeline passes each spectrum normalised by its own maximum
            # instead, so the same expression now asks "does the SWIR
            # contain at least 80% of this pixel's spectral peak?" - a
            # different question (where the peak sits, not whether the
            # detector saturated). The branch is off by default; if you
            # enable it, that is the criterion you are enabling.
            mask = wavelength > 1400
            sat = config.get('saturation_threshold', 0.8)
            vals = np.asarray(spectrum_norm)[mask]
            # An empty selection would make np.max raise; treat it as "no
            # saturation found" and fall through to the conservative window.
            if vals.size and np.max(vals) > sat:
                first = int(np.where(vals > sat)[0][0])
                upper_bound = max(1900.0, float(wavelength[mask][first]))
                mask = (wavelength > 1400) & (wavelength < upper_bound)
            else:
                mask = (wavelength > 1400) & (wavelength < 1900)
        else:
            mask = wavelength > 1400
    elif config.get('only_short_SWIR', False):
        mask = (wavelength > 1400) & (wavelength < 1900)
    else:
        mask = wavelength > 0

    # Drop opaque atmospheric channels from the residual ENTIRELY - data,
    # model and background alike - rather than zeroing just the Planck term
    # inside them (the legacy `ydata < 0.003` rule, see
    # estimate_temperature()). Removing a channel from the fit is the
    # consistent way to say "this channel carries no surface information";
    # deleting one component of the model while keeping the others is not,
    # and measurably biases the retrieval cold.
    if config.get('exclude_atmospheric_windows', True):
        transparent = np.ones(len(wavelength), dtype=bool)
        for lo, hi in config.get('atmospheric_exclusions', _ATMOSPHERIC_EXCLUSIONS_NM):
            transparent &= ~((wavelength >= lo) & (wavelength <= hi))
        n_keep = int((mask & transparent).sum())
        if n_keep >= _MIN_FIT_BANDS:
            mask = mask & transparent
        elif warn:
            warnings.warn(
                f'Excluding the atmospheric windows would leave only '
                f'{n_keep} bands (minimum {_MIN_FIT_BANDS}); '
                'keeping them for this pixel. Check config["atmospheric_exclusions"] against '
                'your fit window.')
    return mask


def cfg_band_mask_for_methods(wavelength, config):
    """The scene-level fit window, for the alternative retrievals in
    temperature_methods.py.

    A thin wrapper over _swir_band_window() - see there for the selection
    rules and for why both callers must share one implementation. The
    per-pixel `only_feasible_SWIR` branch is skipped (spectrum_norm=None):
    it depends on the individual spectrum, so it has no scene-level mask."""
    return _swir_band_window(wavelength, config, spectrum_norm=None, warn=False)


def main_temp_estimation(pixel_number, xdata, fire_pixel,
                          fire_pixel_2D, bkg_pool, config):
    """Run N_trials randomized background draws and aggregate the temperature estimate
    for a single fire pixel. bkg_pool is the (n_bkg_candidates, n_bands) array of
    candidate background spectra to sample from.

    Note: this intentionally does not take the full hyperspectral cube as an
    argument (the original PRISMA version did, as DATA_CUBE_2B, to support an
    alternate background-sampling path that this rewrite doesn't use). Passing
    it here would be a real cost under joblib.Parallel: the whole cube would
    get pickled and shipped to every worker for every single pixel call."""

    results = []
    wl_full = xdata

    # Shared with cfg_band_mask_for_methods(), so the alternative retrievals
    # in temperature_methods.py are fitted over exactly these channels - see
    # _swir_band_window() for why that has to be one implementation.
    index_SWIR = _swir_band_window(wl_full, config, spectrum_norm=fire_pixel_2D,
                                   warn=True)

    xdata = wl_full[index_SWIR]
    fire_pixel = fire_pixel[index_SWIR]

    # Per-pixel RNG. The module-level `seed(1)` is re-executed on import in
    # every joblib worker, so without this each worker replays an identical
    # sequence of restarts and background draws - the N_trials samples are
    # then not independent across workers, which is precisely what the
    # spread across trials is being used to quantify. Deriving the seed from
    # pixel_number keeps runs reproducible while making pixels independent.
    rng = np.random.default_rng((int(config.get('random_seed', 0)) << 32) + int(pixel_number))

    # The mixture model normalises by sum(abundances) (see
    # fitClass.linear_mixture_function), so a zero lower bound on the fire
    # abundances lets the optimiser drive that sum to zero and every modelled
    # spectrum to inf/nan. The default of 0.05 is what keeps the sum bounded
    # away from zero, which makes this bound load-bearing rather than pure
    # tuning - say so here instead of failing opaquely inside curve_fit.
    if float(config.get('fire_abundance_min', 0.05)) <= 0:
        raise ValueError(
            'config["fire_abundance_min"] must be > 0 (got '
            '{0!r}): the model depends on the abundances only through their '
            'ratios, so a zero floor makes the normalisation singular. The '
            'cost of this bound is reported as fire_fraction_floor.'.format(
                config.get('fire_abundance_min')))

    failure_log = []
    trials_lost = 0
    for _ in range(config['N_trials']):
        counter = 0
        result_i = []
        while (len(result_i) == 0) & (counter < 5):
            idx = rng.integers(bkg_pool.shape[0], size=config['N_bkg_points'])
            bkg = bkg_pool[idx, :]
            bkg = bkg[:, index_SWIR]

            result_i = estimate_temperature(fire_pixel, xdata, bkg,
                                              config['emissivity'], config['N_sources'],
                                              config['plot'], config['verbose'],
                                              config['T_min'], config['T_max'],
                                              config['method'],
                                              absorption_blank_threshold=config.get(
                                                  'absorption_blank_threshold'),
                                              fire_abundance_min=config.get(
                                                  'fire_abundance_min', 0.05),
                                              rng=rng,
                                              max_nfev=config.get('max_nfev'),
                                              failure_log=failure_log,
                                              analytic_jacobian=config.get(
                                                  'amici_analytic_jacobian', False))
            counter += 1

        if len(result_i) > 0:
            results.append(result_i)
        else:
            trials_lost += 1

    # Warn only when the retries did not save the trial - i.e. when a
    # sample the uncertainty estimate depends on was actually lost. A
    # failure that the next background draw recovered from is normal and is
    # reported through n_fit_failures instead of as a warning.
    if trials_lost:
        first = failure_log[0] if failure_log else 'unknown'
        warnings.warn(
            f'Pixel {pixel_number}: {trials_lost} of {config["N_trials"]} trials lost after '
            f'5 background re-draws each ({len(failure_log)} failed fits in total; first: '
            f'{first}). The reported spread for this pixel is based on '
            f'{len(results)} trials. Raise config["max_nfev"] if this is widespread.')

    Temps = []
    p_coeff = []
    all_temps = []      # (n_valid, N_sources), hottest-first within each trial
    all_fracs = []
    valid_estimates = len(results)

    if valid_estimates > 0:
        for i in range(valid_estimates):
            # estimate_temperature() sorts components hottest-first, so
            # index 0 is the hot component in every trial and averaging
            # across trials compares like with like.
            Temps.append(results[i][0][0])
            p_coeff.append(results[i][1][0])
            all_temps.append(results[i][0])
            all_fracs.append(results[i][1][:config['N_sources']])

        if valid_estimates > 1:
            conf95_temp = st.t.interval(0.95, len(Temps) - 1, loc=np.mean(Temps), scale=st.sem(Temps))
            conf95_p_coeff = st.t.interval(0.95, len(p_coeff) - 1, loc=np.mean(p_coeff), scale=st.sem(p_coeff))
        else:
            conf95_temp = [np.nan, np.nan]
            conf95_p_coeff = [np.nan, np.nan]
    else:
        # NaN, not 0. A pixel whose every trial failed has no retrieved fire
        # fraction, and reporting 0.0 states a measurement that was never made.
        # Unlike mean_temp (already NaN on this branch) a finite 0.0 survives
        # every isfinite() filter downstream - it is drawn as a real detection
        # in the fire-fraction panel of plot_temperature_maps_figure() and
        # pulls that panel's 98th-percentile colour scale, so the temperature
        # and fraction panels end up disagreeing about which pixels carry a
        # retrieval at all.
        p_coeff = np.nan
        conf95_p_coeff = [np.nan, np.nan]
        Temps = np.nan
        conf95_temp = [np.nan, np.nan]

    # Spread of the trial distribution itself. THIS is the quantity to quote
    # as a retrieval uncertainty, not CI_temp_low/high.
    #
    # CI_temp_* is st.t.interval(...scale=sem), i.e. a confidence interval
    # for the MEAN of the trials, so its width falls as 1/sqrt(N_trials)
    # while the actual scatter does not. Measured on one pixel at
    # T=900 K, p=0.02: std stays 140-240 K while the CI half-width goes
    # 8.4 -> 109.8 -> 73.4 -> 23.5 K for N_trials = 5, 15, 45, 135 (the
    # ratios match t/sqrt(N)*sqrt(N/(N-1)) to three digits). At N=135 that
    # understates the true spread by about 6x, and it can be driven to zero
    # just by running more trials. The percentile bounds below do not have
    # that property. CI_temp_* is retained unchanged so older outputs stay
    # comparable.
    def _pct(values, q):
        arr = np.atleast_1d(np.asarray(values, dtype=float))
        return float(np.nanpercentile(arr, q)) if np.isfinite(arr).any() else float('nan')

    temp_lo, temp_hi = _pct(Temps, 2.5), _pct(Temps, 97.5)
    pc_lo, pc_hi = _pct(p_coeff, 2.5), _pct(p_coeff, 97.5)

    # How many trials came back sitting on their own search bound, and the
    # median as an aggregate that survives them.
    #
    # WHEN THIS MATTERS. Each trial fits against a different random draw of
    # N_bkg_points spectra out of bkg_pool. When that draw spans the true
    # background the trials agree closely and mean and median coincide;
    # measured against a planted 900 K source at the shipped defaults
    # (N_trials=15, N_bkg_points=20) the retrieval gives mean 899.9 K, median
    # 899.7 K, and a 2.5-97.5 spread of 898.9-900.6 K. On the real scene the
    # pool holds ~10^6 candidate background pixels, so this is the normal case
    # and median_temp simply agrees with mean_temp.
    #
    # It stops being the normal case when the pool is small or spans the
    # background poorly. A draw that cannot reproduce the background lets the
    # optimiser explain the residual with a COLD, essentially fire-free Planck
    # term instead - a local optimum near T_min, not a cooler fire. The trial
    # distribution then has two modes and the MEAN sits between them,
    # describing neither, while the median stays on the real one. That is the
    # whole reason this column exists; it costs nothing when the trials agree.
    #
    # Such a pixel is ALSO already visible through fire_rmse_reduction, which
    # collapses toward zero when the thermal component stops doing any work -
    # that is what fire_component_significance() is for, and it remains the
    # column to filter a published temperature map on.
    #
    # n_trials_at_T_bound covers the other, unrelated failure mode:
    # temperature_methods._bound_flag() refuses to report a temperature
    # sitting on its own search interval as a measurement ("at least this hot,
    # true value unconstrained above"), having measured 1 of 16 Yalu and 2 of 4
    # Mojave pixels pinned at the UPPER bound on real scenes. The native
    # retrieval had no equivalent flag. It catches bound-pinning only; the cold
    # optimum described above is near T_min but not on it, so it is not counted
    # here.
    #
    # Both columns are ADDITIVE: mean_temp keeps its exact previous meaning, so
    # older outputs stay comparable.
    _t_arr = np.atleast_1d(np.asarray(Temps, dtype=float))
    _t_ok = _t_arr[np.isfinite(_t_arr)]
    _span = max(abs(float(config['T_max']) - float(config['T_min'])), 1.0)
    _edge = max(1e-6, 1e-3 * _span)
    n_at_bound = int(np.sum((_t_ok <= float(config['T_min']) + _edge) |
                            (_t_ok >= float(config['T_max']) - _edge)))
    median_temp = float(np.median(_t_ok)) if _t_ok.size else float('nan')
    # Pixel-level flag, as opposed to n_trials_at_T_bound's per-trial count:
    # is the REPORTED temperature itself a lower limit? Only the upper bound
    # matters here - a pixel pinned at T_min is the different, unrelated
    # "background draw could not reproduce the background" failure mode
    # median_temp already protects against (see the comment above), not a
    # pixel too bright for a single Planck curve. This is what gates the
    # two-component escalation below.
    T_at_bound = bool(np.isfinite(median_temp) and median_temp >= float(config['T_max']) - _edge)

    # Model selection: did the thermal component earn its place in this
    # pixel? Evaluated once, at the median retrieved temperature, against a
    # fresh background draw - see fire_component_significance(). Without
    # this the retrieval reports a temperature for every pixel it is handed,
    # including ones with no thermal signal at all, and a reader has no way
    # to tell the two apart.
    sig = {}
    if valid_estimates > 0:
        med_T = float(np.median(np.atleast_1d(Temps)))
        if np.isfinite(med_T):
            try:
                draw = rng.integers(bkg_pool.shape[0], size=config['N_bkg_points'])
                sig = fire_component_significance(
                    fire_pixel, xdata, bkg_pool[draw, :][:, index_SWIR],
                    med_T, config['emissivity'])
            except Exception as exc:
                warnings.warn(f'Pixel {pixel_number}: fire-significance test failed '
                              f'({exc.__class__.__name__}: {exc}).')

    # Adaptive two-component escalation (see the module note above
    # two_component_escalation()): offered ONLY to pixels whose reported
    # temperature is itself a lower limit (T_at_bound), and kept only when
    # escalation_significance() confirms the second component earns its
    # place on this specific pixel - every other pixel is completely
    # unaffected by this block. Off by config['escalate_bound_pinned_pixels']
    # = False for exact reproduction of pre-escalation results.
    escalation = None
    if T_at_bound and config.get('escalate_bound_pinned_pixels', True):
        try:
            draw2 = rng.integers(bkg_pool.shape[0], size=config['N_bkg_points'])
            escalation = two_component_escalation(
                fire_pixel, xdata, bkg_pool[draw2, :][:, index_SWIR], config, rng,
                p_value_max=config.get('escalation_p_value_max', 0.001),
                min_rmse_reduction_frac=config.get('escalation_min_rmse_reduction_frac', 0.5),
                min_temperature_separation_K=config.get(
                    'escalation_min_temperature_separation_K', 150.0))
        except Exception as exc:
            warnings.warn(f'Pixel {pixel_number}: two-component escalation failed '
                          f'({exc.__class__.__name__}: {exc}).')

    # Every value here is cast to a plain Python float, and the confidence
    # intervals are split into separate _low/_high columns rather than
    # stored as a single list-valued cell. Two reasons: (1) a cell holding
    # a Python list of numpy scalars gets written to CSV using numpy's
    # repr, which since NumPy 2.0 prints "np.float64(...)" around every
    # number instead of a plain decimal - float() strips that; (2) a
    # bracketed "[low, high]" string in one CSV cell isn't something a
    # spreadsheet or GIS tool can use as a number without re-parsing it -
    # two plain numeric columns are directly usable.
    n_sources = int(config['N_sources'])
    result = {'Pixel': pixel_number,
              'emissivity': config['emissivity'],
              'mean_P_coeff': float(np.mean(p_coeff)),
              'std_P_coeff': float(np.std(p_coeff)),
              'CI_P_coeff_low': float(conf95_p_coeff[0]),
              'CI_P_coeff_high': float(conf95_p_coeff[1]),
              'mean_temp': float(np.mean(Temps)),
              'std_temp': float(np.std(Temps)),
              # Robust to the bimodal trial distribution described above;
              # prefer this over mean_temp when any trial is at a bound.
              'median_temp': median_temp,
              # Trials that returned their own search bound rather than a
              # fitted value. Non-zero means mean_temp is mixing a retrieval
              # with a fire-free bound solution - filter on this before
              # publishing a temperature map.
              'n_trials_at_T_bound': n_at_bound,
              # Pixel-level companion to n_trials_at_T_bound: True means
              # median_temp (and mean_temp) is itself a lower limit, not a
              # retrieval - see the comment above T_at_bound's computation.
              'T_at_bound': T_at_bound,
              # Populated only when escalation to a two-component fit was
              # both attempted (T_at_bound) and found statistically
              # justified (escalation_significance); NaN/False otherwise.
              # mean_temp/median_temp above are UNCHANGED either way, so
              # existing columns keep their exact prior meaning - read
              # escalated_T_hot_K in place of median_temp for a pixel with
              # escalated_two_component=True, rather than treating the
              # bound-pinned single-component value as the retrieval.
              'escalated_two_component': bool(escalation is not None),
              'escalated_T_hot_K': float(escalation['T_hot_K']) if escalation else float('nan'),
              'escalated_T_cool_K': float(escalation['T_cool_K']) if escalation else float('nan'),
              'escalated_f_hot': float(escalation['f_hot']) if escalation else float('nan'),
              'escalated_f_cool': float(escalation['f_cool']) if escalation else float('nan'),
              'escalated_F_statistic': float(escalation['F_statistic']) if escalation else float('nan'),
              'escalated_p_value': float(escalation['p_value']) if escalation else float('nan'),
              'escalated_rmse_reduction': float(escalation['rmse_reduction_frac']) if escalation else float('nan'),
              'CI_temp_low': float(conf95_temp[0]),
              'CI_temp_high': float(conf95_temp[1]),
              # Preferred uncertainty: 95% spread of the trial distribution.
              'temp_p2_5': temp_lo,
              'temp_p97_5': temp_hi,
              'P_coeff_p2_5': pc_lo,
              'P_coeff_p97_5': pc_hi,
              'n_trials_valid': int(valid_estimates),
              'n_trials_lost': int(trials_lost),
              # Individual curve_fit failures, nearly all of which the
              # background re-draw recovered from. Non-zero here with
              # n_trials_lost == 0 means the retry logic did its job.
              'n_fit_failures': int(len(failure_log)),
              'n_bands_fit': int(len(xdata)),
              # Model-selection diagnostics (see fire_component_significance).
              # Filter a temperature map on fire_rmse_reduction before
              # publishing it: a pixel whose thermal component buys little
              # or no residual reduction has no retrievable fire signal,
              # whatever temperature the bounded fit reported.
              'fire_F_statistic': float(sig.get('F_statistic', np.nan)),
              'fire_p_value': float(sig.get('p_value', np.nan)),
              'fire_rmse_reduction': float(sig.get('rmse_reduction_frac', np.nan)),
              'rmse_with_fire': float(sig.get('rmse_with_fire', np.nan)),
              'rmse_background_only': float(sig.get('rmse_background_only', np.nan)),
              # Smallest fire fraction these bounds can express at all - a
              # tuning artefact of fire_abundance_min and N_bkg_points, not a
              # radiometric detection limit. Quote it whenever a retrieved
              # fraction approaches it.
              'fire_fraction_floor': float(config.get('fire_abundance_min', 0.05)
                                           / (n_sources + config['N_bkg_points']))}

    # Multi-component runs: report every component, not just the hottest, so
    # nothing is silently discarded. Components are ordered hottest-first.
    if n_sources > 1:
        temps_arr = np.asarray(all_temps, dtype=float) if all_temps else np.full((1, n_sources), np.nan)
        fracs_arr = np.asarray(all_fracs, dtype=float) if all_fracs else np.full((1, n_sources), np.nan)
        for k in range(n_sources):
            result[f'mean_temp_src{k}'] = float(np.nanmean(temps_arr[:, k]))
            result[f'std_temp_src{k}'] = float(np.nanstd(temps_arr[:, k]))
            result[f'mean_P_coeff_src{k}'] = float(np.nanmean(fracs_arr[:, k]))

    return result


def search_band(cube, cw, target_wavelength):
    """Return the band (rows, cols) closest to target_wavelength, and its index.
    cube must be (rows, bands, cols); cw is the per-band wavelength array."""
    if len(cw) != cube.shape[1]:
        raise ValueError('Dimensions do not match!')
    target_band_index = int(np.argmin(np.abs(cw - target_wavelength)))
    # No np.squeeze here: cube[:, target_band_index, :] is already exactly
    # (rows, cols) via basic indexing; squeezing would collapse a
    # single-row or single-column scene down to 1D and break every caller.
    return cube[:, target_band_index, :], target_band_index


#%% ###########################################################################
#                        EnMAP ingestion (new)
################################################################################

_ENVI_LIST_RE = re.compile(r'^\s*([A-Za-z0-9_ ]+?)\s*=\s*(.*)$', re.DOTALL)


def parse_envi_header(hdr_path):
    """Parse an ENVI .hdr text file into a {lowercase_key: value} dict.

    Pure-Python, no GDAL dependency, so it works even before GDAL/osgeo is
    installed and is trivially unit-testable. List-valued fields (wavelength,
    fwhm, data gain/offset values, ...) enclosed in `{ ... }` (which may span
    multiple lines) are returned as lists of stripped strings; scalar fields
    are returned as plain strings.
    """
    with open(hdr_path, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    # Join the file into logical lines: an opening '{' with no matching '}'
    # on the same line means the value continues on subsequent lines.
    raw_lines = text.splitlines()
    logical_lines = []
    buf = ''
    depth = 0
    for line in raw_lines:
        buf = line if depth == 0 else buf + '\n' + line
        depth += line.count('{') - line.count('}')
        if depth <= 0:
            logical_lines.append(buf)
            buf = ''
            depth = 0

    header = {}
    for line in logical_lines:
        if not line.strip() or line.strip().upper().startswith('ENVI'):
            continue
        m = _ENVI_LIST_RE.match(line)
        if not m:
            continue
        key = m.group(1).strip().lower()
        value = m.group(2).strip()
        if value.startswith('{'):
            inner = value[value.find('{') + 1: value.rfind('}')]
            items = [v.strip() for v in inner.split(',') if v.strip() != '']
            header[key] = items
        else:
            header[key] = value

    return header


def inspect_envi_header(hdr_path):
    """Print the parsed header so you can confirm your product's field names
    before pointing load_enmap_radiance() at it (different EnMAP processors
    have used slightly different key names historically)."""
    header = parse_envi_header(hdr_path)
    for k, v in header.items():
        preview = v if not isinstance(v, list) else f'[{len(v)} values] {v[:3]} ...'
        print(f'{k!r}: {preview}')
    return header


# EnMAP's published nominal detector edges (nm). Used only as the very last
# resort in load_enmap_radiance(), when no per-scene wavelength calibration
# can be found anywhere - not this scene's actual calibration, since
# dead/bad bands are commonly dropped from L1B products, which shifts real
# band positions away from a naive evenly-spaced grid.
_ENMAP_NOMINAL_RANGE_NM = {
    'vnir': (418.24, 998.31),
    'swir': (901.71, 2445.42),
}

# Two-tier: prefer a number with an explicit unit attached (so "Band 1
# (901.71 Nanometers)" matches "901.71 Nanometers", not the band index "1");
# only fall back to a bare decimal number when no unit is present anywhere.
_WAVELENGTH_WITH_UNIT_RE = re.compile(
    r'([-+]?\d*\.?\d+)\s*(nanometers?|nm|micrometers?|microns?|um|µm)\b', re.IGNORECASE)
_DECIMAL_NUMBER_RE = re.compile(r'([-+]?\d+\.\d+)')


def _wavelength_from_gdal_bands(ds, n_bands):
    """Look for a per-band wavelength directly on the raster itself (some
    GeoTIFF writers - including ENVI's own GeoTIFF export - embed it in the
    band description or band-level metadata). This is the most trustworthy
    fallback after a .hdr, since it comes from the file GDAL just opened
    rather than a sidecar that could belong to a different scene. Returns
    None (not a partial array) if any band can't be parsed - a wavelength
    array with only some entries populated would silently misalign the
    Planck fit for the rest."""
    values = []
    for b in range(1, n_bands + 1):
        band = ds.GetRasterBand(b)
        meta = band.GetMetadata() or {}
        text = ' '.join([band.GetDescription() or ''] +
                         [v for k, v in meta.items() if 'wavelength' in k.lower()])
        m = _WAVELENGTH_WITH_UNIT_RE.search(text)
        if m:
            val = float(m.group(1))
            unit = m.group(2).lower()
        else:
            m = _DECIMAL_NUMBER_RE.search(text)
            if not m:
                return None
            val = float(m.group(1))
            unit = 'nm'
        if unit.startswith(('micro', 'micron', 'um', 'µ')):
            val *= 1000.0
        values.append(val)
    return np.array(values)


def _find_enmap_metadata_xml(raster_path):
    """EnMAP's official L1B/L1C GeoTIFF product ships band calibration
    (wavelength, FWHM, gain, offset) in a *-METADATA.XML sidecar next to the
    *-SPECTRAL_IMAGE_VNIR.TIF / *-SPECTRAL_IMAGE_SWIR.TIF rasters, instead of
    an ENVI .hdr. Returns the sidecar path if found, else None."""
    directory = os.path.dirname(os.path.abspath(raster_path)) or '.'
    base = os.path.basename(raster_path)

    candidates = []
    m = re.match(r'(?i)^(.*)-SPECTRAL_IMAGE_(VNIR|SWIR)\.TIFF?$', base)
    if m:
        candidates.append(os.path.join(directory, m.group(1) + '-METADATA.XML'))
    candidates += sorted(glob.glob(os.path.join(directory, '*METADATA*.[Xx][Mm][Ll]')))

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def _xml_findall_by_keyword(root, keyword):
    """Case-insensitive, namespace-agnostic search for every element whose
    local tag name contains `keyword`, in document order."""
    keyword = keyword.lower()
    return [el for el in root.iter() if keyword in el.tag.rsplit('}', 1)[-1].lower()]


def _xml_first_int(root, keywords):
    """First element matching any of `keywords` (by _xml_findall_by_keyword)
    that parses as a single integer, e.g. <numberOfSWIRBands>133</...>."""
    for kw in keywords:
        for el in _xml_findall_by_keyword(root, kw):
            try:
                return int(el.text.strip())
            except (AttributeError, ValueError):
                continue
    return None


def _detector_from_filename(raster_path):
    """'vnir' / 'swir' / None, read off EnMAP's own product naming
    convention (*-SPECTRAL_IMAGE_VNIR.TIF / *_SWIR.TIF), which is reliable
    since it's how the vendor names the file, not a guess."""
    name = os.path.basename(raster_path).lower()
    if 'vnir' in name and 'swir' not in name:
        return 'vnir'
    if 'swir' in name and 'vnir' not in name:
        return 'swir'
    return None


_ENMAP_FILENAME_RE = re.compile(
    r'(?i)^ENMAP\d*-.*?-DT(\d+)_(\d{8}T\d{6}Z)_(\d+)_V\d+_\d{8}T\d{6}Z-SPECTRAL_IMAGE_(VNIR|SWIR)$')


def _short_scene_id(raster_path, max_len=40):
    """A short, meaningful identifier derived from the source image's own
    filename, for naming outputs. Not the full original name - EnMAP's own
    product filenames run 130+ characters, which makes for unusable output
    filenames - and not a fixed generic string either, which would silently
    let results from a second, different scene overwrite the first run's
    output in the same results directory.

    For EnMAP's official naming convention, pulls out just the parts that
    actually distinguish one acquisition from another (datatake ID,
    acquisition timestamp, scene counter, detector) and drops the
    boilerplate (mission tag, product level, processor version, processing
    timestamp). For any other filename, falls back to the name itself,
    truncated to max_len."""
    base = os.path.splitext(os.path.basename(raster_path))[0]

    m = _ENMAP_FILENAME_RE.match(base)
    if m:
        dt_id, timestamp, scene_num, detector = m.groups()
        short = f'DT{dt_id}_{timestamp}_{scene_num}_{detector}'
    else:
        short = base

    if len(short) > max_len:
        short = short[:max_len].rstrip('_-')
    return short


def _scene_folder_id(raster_path, max_len=40):
    """Name of the per-scene output folder: the acquisition itself (datatake
    ID, acquisition timestamp, scene counter), without the detector that
    _short_scene_id() keeps. A merged run covers VNIR and SWIR alike, and
    starting from the SWIR file, the VNIR file or the METADATA.XML of one
    scene must land in the same folder. Non-EnMAP names fall back to
    _short_scene_id()."""
    m = _ENMAP_FILENAME_RE.match(os.path.splitext(os.path.basename(raster_path))[0])
    if m:
        dt_id, timestamp, scene_num, _detector = m.groups()
        return f'DT{dt_id}_{timestamp}_{scene_num}'
    return _short_scene_id(raster_path, max_len=max_len)


def _read_enmap_metadata_xml(xml_path, n_bands, detector=None):
    """Best-effort parse of an EnMAP METADATA.XML sidecar for per-band
    wavelength/FWHM/gain/offset. EnMAP processor versions have varied their
    exact tag names over time, so this searches by keyword-in-tag-name
    rather than one fixed path.

    EnMAP's real METADATA.XML characterises ALL 224 bands (VNIR+SWIR) in one
    combined <bandCharacterisation> list, document-ordered VNIR bands first
    then SWIR - not split per detector file. So this tries, in order:
      (a) the full list already has exactly n_bands entries (some product
          variants may genuinely be per-detector), used as-is;
      (b) the full list splits cleanly into <numberOfVNIRBands> +
          <numberOfSWIRBands> (or similarly-named count tags) that sum to
          the full list's length, in which case the contiguous sub-range
          for the requested `detector` is sliced out - but ONLY if that
          sub-range's length equals n_bands exactly.
    Any other case returns None (for that field) rather than guessing -
    like _wavelength_from_gdal_bands, a misaligned array here would corrupt
    the physics silently, which is worse than falling through to the next,
    more conservative fallback tier.

    Returns None if no wavelength could be resolved at all, else a dict
    with wavelength (always present), wavelength_tag, and fwhm/gain/offset
    (each None if not found or not aligned).
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    def collect_all(keyword):
        elems = _xml_findall_by_keyword(root, keyword)
        if not elems:
            return None
        try:
            return np.array([float(el.text.strip()) for el in elems])
        except (AttributeError, ValueError):
            return None

    def first_match(keywords):
        for kw in keywords:
            vals = collect_all(kw)
            if vals is not None:
                return kw, vals
        return None, None

    wavelength_tag, wavelength_all = first_match(
        ('wavelengthcenterofband', 'wavelengthcenter', 'centralwavelength', 'centerwavelength', 'wavelength'))
    if wavelength_all is None:
        return None
    _, fwhm_all = first_match(('fwhmofband', 'fwhm'))
    _, gain_all = first_match(('gainofband', 'gain'))
    _, offset_all = first_match(('offsetofband', 'offset'))

    def field(arr, sl=slice(None)):
        return arr[sl] if (arr is not None and len(arr) == len(wavelength_all)) else None

    if len(wavelength_all) == n_bands:
        return {'wavelength': wavelength_all, 'wavelength_tag': wavelength_tag,
                'fwhm': field(fwhm_all), 'gain': field(gain_all), 'offset': field(offset_all)}

    if detector in ('vnir', 'swir'):
        n_vnir = _xml_first_int(root, ('numberofvnirbands', 'vnirbands'))
        n_swir = _xml_first_int(root, ('numberofswirbands', 'swirbands'))
        if n_vnir is not None and n_swir is not None and n_vnir + n_swir == len(wavelength_all):
            sl = slice(0, n_vnir) if detector == 'vnir' else slice(n_vnir, n_vnir + n_swir)
            if (sl.stop - sl.start) == n_bands:
                return {'wavelength': wavelength_all[sl], 'wavelength_tag': wavelength_tag + f' [{detector} slice]',
                        'fwhm': field(fwhm_all, sl), 'gain': field(gain_all, sl), 'offset': field(offset_all, sl)}

    return None


def _nominal_wavelength_grid(raster_path, n_bands):
    """Absolute last resort when load_enmap_radiance() finds no per-scene
    calibration anywhere: a linearly-spaced grid over EnMAP's published
    nominal detector range. The detector (VNIR vs SWIR) is read off the
    filename via _detector_from_filename(); if neither appears, the full
    VNIR+SWIR span is used."""
    detector = _detector_from_filename(raster_path)
    if detector is not None:
        lo, hi = _ENMAP_NOMINAL_RANGE_NM[detector]
    else:
        lo, hi = _ENMAP_NOMINAL_RANGE_NM['vnir'][0], _ENMAP_NOMINAL_RANGE_NM['swir'][1]
    return np.linspace(lo, hi, n_bands)


def _load_single_enmap_raster(raster_path, hdr_path=None, wavelength_key='wavelength',
                               fwhm_key='fwhm', gain_key='data gain values',
                               offset_key='data offset values',
                               ignore_key='data ignore value'):
    """Open one EnMAP radiance raster and hand it to _load_from_open_raster().

    This wrapper exists only to own the GDAL handle. Releasing it at the end
    of the reading code is not enough: when an exception propagates out of a
    function its frame stays alive through the traceback, and the frame holds
    the dataset - so a load that failed part-way used to leave the raster
    locked open for the life of the process. On Windows that makes the file
    un-deletable and un-reopenable, which the GUI hits the moment a user
    corrects a bad input and re-runs the same scene.

    See _load_from_open_raster() for what is actually read and returned.
    """
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(raster_path, gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f'GDAL could not open {raster_path}')
    try:
        return _load_from_open_raster(
            ds, raster_path, hdr_path=hdr_path, wavelength_key=wavelength_key,
            fwhm_key=fwhm_key, gain_key=gain_key, offset_key=offset_key,
            ignore_key=ignore_key)
    finally:
        ds = None          # released on every exit, exceptions included


def _load_from_open_raster(ds, raster_path, hdr_path=None, wavelength_key='wavelength',
                           fwhm_key='fwhm', gain_key='data gain values',
                           offset_key='data offset values',
                           ignore_key='data ignore value'):
    """Read an already-open EnMAP radiance raster (ENVI or GeoTIFF, whatever GDAL can
    open) - a single detector file (VNIR-only or SWIR-only), or a single
    file that's already a combined cube. This is the low-level worker;
    load_enmap_radiance() (below) is the public entry point and additionally
    auto-detects and merges a VNIR+SWIR sibling pair.

    Returns a dict with:
      cube        : (n_rows, n_bands, n_cols) float32 radiance, W/(m^2*sr*nm)
      wavelength  : (n_bands,) nm
      wavelength_source : human-readable description of where the wavelength
                    array came from - always check this once for a new
                    product; it tells you whether it's trustworthy per-scene
                    calibration or a guessed placeholder
      lat, lon    : (n_rows, n_cols) degrees (WGS84)
      geotransform, projection_wkt : from GDAL, for writing outputs back out
      nodata      : scalar or None

    Band calibration (wavelength, and DN->radiance gain/offset) is looked
    for in several places, in order of trust, and the first one that
    resolves ALL bands is used:
      1. An ENVI .hdr sidecar (classic ENVI/PRISMA-style products).
      2. Wavelength embedded directly on the raster's own GDAL band
         description/metadata (some GeoTIFF writers do this).
      3. An EnMAP *-METADATA.XML sidecar next to a *-SPECTRAL_IMAGE_VNIR/SWIR.TIF
         raster - this is the format EnMAP's official L1B/L1C GeoTIFF
         product actually ships instead of a .hdr, and also carries gain/
         offset for DN -> radiance conversion. The real file characterises
         all 224 VNIR+SWIR bands together in one list; this tier detects
         the <numberOfVNIRBands>/<numberOfSWIRBands> split and slices out
         just the 91 or 133 bands belonging to whichever detector file was
         opened (VNIR and SWIR genuinely overlap in nominal wavelength,
         ~902-993 nm - both detectors cover that band redundantly, so this
         is expected, not a sign the slice is wrong).
      4. A synthesized linearly-spaced grid over EnMAP's published nominal
         detector range (VNIR or SWIR, from the filename) - a last-resort
         placeholder, NOT this scene's real calibration. A loud warning is
         raised when this tier is used.
    Whichever tier is used, wavelength_source in the returned dict and the
    printed log say so explicitly - always check it for a new product.

    Radiance units: DN is converted via radiance = DN * gain + offset
    wherever gain/offset were found (tiers 1 and 3); tier 1 additionally
    divides by 1000 if the header's 'wavelength units' is 'Micrometers', to
    land in W/(m^2*sr*nm). If no gain/offset is found at all, the raster is
    assumed to already be physical-unit float radiance and used as-is -
    verify this against your product's documentation before trusting the
    retrieved temperatures.
    """
    from osgeo import osr

    header = {}
    if hdr_path is None:
        base, _ = os.path.splitext(raster_path)
        for candidate in (base + '.hdr', raster_path + '.hdr'):
            if os.path.exists(candidate):
                hdr_path = candidate
                break
    elif not os.path.exists(hdr_path):
        hdr_path = None
    if hdr_path is not None:
        header = parse_envi_header(hdr_path)

    arr = ds.ReadAsArray().astype('float32')  # (bands, rows, cols)
    if arr.ndim != 3:
        raise ValueError('Expected a multi-band cube; got shape ' + str(arr.shape))
    cube = np.transpose(arr, (1, 0, 2))  # -> (rows, bands, cols), matches PRISMA convention

    n_bands = cube.shape[1]
    units = str(header.get('wavelength units', 'nm')).strip().lower()
    is_micro_units = units.startswith('micro') or units in ('um', 'µm')

    gain_field = header.get(gain_key)
    offset_field = header.get(offset_key)
    # Check the length before reshaping, the same way the wavelength field is
    # checked below. A header listing a different number of gain values than
    # the cube has bands otherwise fails with an opaque "cannot reshape array
    # of size N into shape (1,M,1)", and silently applying a mismatched
    # calibration would be the same class of silent scaling error that
    # _ecostress_resolve_scale() exists to prevent on the ECOSTRESS side.
    if gain_field is not None and len(gain_field) != n_bands:
        warnings.warn(f'{hdr_path} lists {len(gain_field)} gain values but the cube has '
                      f'{n_bands} bands; ignoring the header calibration and treating the '
                      'raster as already being in physical radiance units. Verify this '
                      'against your product documentation before trusting the retrieved '
                      'temperatures.')
        gain_field = offset_field = None
    if offset_field is not None and len(offset_field) != n_bands:
        warnings.warn(f'{hdr_path} lists {len(offset_field)} offset values but the cube has '
                      f'{n_bands} bands; ignoring the header offset.')
        offset_field = None
    gain_arr = np.array([float(v) for v in gain_field]).reshape(1, n_bands, 1) if gain_field is not None else None
    offset_arr = (np.array([float(v) for v in offset_field]).reshape(1, n_bands, 1)
                  if (gain_arr is not None and offset_field is not None) else (0.0 if gain_arr is not None else None))
    gain_needs_micro_fix = is_micro_units

    wl_field = header.get(wavelength_key)
    if wl_field is not None and len(wl_field) == n_bands:
        wavelength = np.array([float(v) for v in wl_field])
        if is_micro_units:
            wavelength = wavelength * 1000.0
        wavelength_source = f'.hdr file ({hdr_path})'
    else:
        if wl_field is not None:
            warnings.warn(f"{hdr_path} lists {len(wl_field)} wavelengths but the cube "
                           f"has {n_bands} bands; ignoring it and auto-detecting instead.")
        print(f'No usable .hdr wavelength field for {raster_path}; auto-detecting band wavelengths...')

        wavelength = _wavelength_from_gdal_bands(ds, n_bands)
        if wavelength is not None:
            wavelength_source = 'GDAL band description/metadata (read directly from the raster)'
        else:
            xml_path = _find_enmap_metadata_xml(raster_path)
            xml_result = (_read_enmap_metadata_xml(xml_path, n_bands, detector=_detector_from_filename(raster_path))
                          if xml_path else None)
            if xml_result is not None:
                wavelength = xml_result['wavelength']
                wavelength_source = f"EnMAP metadata XML sidecar ({xml_path}, tag <{xml_result['wavelength_tag']}>)"
                if gain_arr is None and xml_result['gain'] is not None:
                    gain_arr = xml_result['gain'].reshape(1, n_bands, 1)
                    offset_arr = (xml_result['offset'].reshape(1, n_bands, 1)
                                  if xml_result['offset'] is not None else 0.0)
                    gain_needs_micro_fix = False  # units from the XML are used as-is; verify against your product docs
                    print(f"Using DN gain/offset from {xml_path} to convert to radiance "
                          "(units as given in the XML - verify against your product documentation).")
            else:
                wavelength = _nominal_wavelength_grid(raster_path, n_bands)
                wavelength_source = ("SYNTHESIZED nominal EnMAP detector range - "
                                      "NOT this scene's calibrated wavelengths")
                warnings.warn(
                    f'No wavelength calibration found for {raster_path}: no usable .hdr, no '
                    'GDAL band metadata, and no EnMAP METADATA.XML sidecar next to it. Falling '
                    "back to a linearly-spaced grid over EnMAP's published nominal detector "
                    'range. This is a placeholder, not a per-scene calibration - dead/bad bands '
                    'are often dropped from L1B products, which shifts real band positions away '
                    'from a naive linear grid. Verify against your product documentation before '
                    'trusting the retrieved temperatures.')

    print(f'Wavelength source: {wavelength_source} '
          f'({wavelength.min():.1f}-{wavelength.max():.1f} nm, {n_bands} bands)')
    print(f'First 3 band wavelengths (nm): {np.round(wavelength[:3], 2).tolist()}')

    if gain_arr is not None:
        cube = cube * gain_arr + offset_arr
        if gain_needs_micro_fix:
            cube = cube / 1000.0  # W/(m^2 sr um) -> W/(m^2 sr nm)

    nodata_val = header.get(ignore_key)
    nodata = float(nodata_val) if nodata_val is not None else ds.GetRasterBand(1).GetNoDataValue()
    if nodata is not None:
        cube = np.where(cube == nodata, np.nan, cube)

    geotransform = ds.GetGeoTransform()
    projection_wkt = ds.GetProjection()

    n_rows, _, n_cols = cube.shape
    cols_idx, rows_idx = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    # Pixel-center coordinates in the raster's native projected CRS
    x = geotransform[0] + (cols_idx + 0.5) * geotransform[1] + (rows_idx + 0.5) * geotransform[2]
    y = geotransform[3] + (cols_idx + 0.5) * geotransform[4] + (rows_idx + 0.5) * geotransform[5]

    # A raster with no coordinate reference system is a real input, not a
    # corrupt one - EnMAP L1B is explicitly not map-projected - so an empty or
    # unparseable WKT must not take the whole load down with OGR's opaque
    # "Corrupt data". The cube and its wavelengths are all the retrieval
    # itself needs; only attach_pixel_geolocation() and the validation against
    # other sensors want lat/lon. Warn loudly, because a silently
    # un-georeferenced scene would otherwise surface much later as an
    # inexplicably empty validation table.
    lat = lon = None
    if projection_wkt:
        try:
            src_srs = osr.SpatialReference()
            src_srs.ImportFromWkt(projection_wkt)
            dst_srs = osr.SpatialReference()
            dst_srs.ImportFromEPSG(4326)
            # Force (x, y) = (lon, lat) / (easting, northing) order regardless
            # of the authority-defined axis order GDAL 3+ otherwise applies.
            src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
            transform = osr.CoordinateTransformation(src_srs, dst_srs)
            pts = transform.TransformPoints(list(zip(x.ravel(), y.ravel())))
            lon = np.array([p[0] for p in pts]).reshape(n_rows, n_cols)
            lat = np.array([p[1] for p in pts]).reshape(n_rows, n_cols)
        except RuntimeError as exc:
            warnings.warn(f'{raster_path} carries a projection GDAL could not interpret '
                          f'({exc}); lat/lon come back as NaN. The temperature retrieval '
                          'still works; validation against Landsat/ECOSTRESS/VIIRS does not.')
    else:
        warnings.warn(f'{raster_path} carries no coordinate reference system; lat/lon come '
                      'back as NaN. The temperature retrieval still works; validation '
                      'against Landsat/ECOSTRESS/VIIRS does not.')
    if lat is None:
        lat = np.full((n_rows, n_cols), np.nan)
        lon = np.full((n_rows, n_cols), np.nan)

    return {
        'cube': cube,
        'wavelength': wavelength,
        'wavelength_source': wavelength_source,
        'lat': lat,
        'lon': lon,
        'geotransform': geotransform,
        'projection_wkt': projection_wkt,
        'nodata': nodata,
    }


_DLR_DETECTOR_FILENAME_RE = re.compile(r'(?i)^(.*-SPECTRAL_IMAGE_)(VNIR|SWIR)(\.TIFF?)$')


def _parse_dlr_detector_filename(raster_path):
    """(prefix, detector, ext) parsed from a stock DLR EnMAP detector
    filename (*-SPECTRAL_IMAGE_VNIR/SWIR.TIF), or None if raster_path
    doesn't match that convention (e.g. a custom or already-merged cube).
    Shared by _sibling_detector_path() and _merged_cube_path()."""
    m = _DLR_DETECTOR_FILENAME_RE.match(os.path.basename(raster_path))
    return m.groups() if m else None


def _sibling_detector_path(raster_path):
    """Given one EnMAP *-SPECTRAL_IMAGE_VNIR/SWIR.TIF path (the stock naming
    from a DLR EnMAP download, which ships VNIR and SWIR as two separate
    files), return the path of its sibling detector file (SWIR<->VNIR) if
    one exists next to it in the same directory, else None. This is how
    load_enmap_radiance() auto-detects and merges the VNIR+SWIR pair
    without the user having to point at both files themselves."""
    parsed = _parse_dlr_detector_filename(raster_path)
    if parsed is None:
        return None
    prefix, detector, ext = parsed
    directory = os.path.dirname(os.path.abspath(raster_path)) or '.'
    other = 'SWIR' if detector.upper() == 'VNIR' else 'VNIR'
    candidate = os.path.join(directory, prefix + other + ext)
    return candidate if os.path.exists(candidate) else None


def _merged_cube_path(raster_path):
    """Path where load_enmap_radiance() saves/looks for an already-merged
    VNIR+SWIR ENVI cube for this scene: same directory as raster_path,
    filename derived from the shared DLR naming prefix with 'MERGED' in
    place of 'VNIR'/'SWIR'. None if raster_path doesn't match that naming
    convention at all."""
    parsed = _parse_dlr_detector_filename(raster_path)
    if parsed is None:
        return None
    prefix, _detector, _ext = parsed
    directory = os.path.dirname(os.path.abspath(raster_path)) or '.'
    return os.path.join(directory, prefix + 'MERGED.dat')


_ENMAP_RASTER_EXTENSIONS = ('.TIF', '.TIFF', '.GEOTIFF', '.BSQ', '.BIL', '.BIP', '.JP2', '.JPEG2000')


def _find_detector_raster_next_to_metadata(xml_path, detector):
    """Locate the VNIR or SWIR spectral-image raster next to an EnMAP
    *-METADATA.XML sidecar, trying every raster extension EnMAP has shipped
    a product in (the same lookup, and the same extension list, as the
    open-source EnMAP-Box's own L1B importer - enmapboxprocessing/algorithm/
    importenmapl1balgorithm.py's findSpectralImageFilename()) - exact
    filename match first, then a case/suffix-insensitive glob (covers e.g.
    a lowercase '.tif' or a '_COG' variant). Returns the first match found
    on disk, or None."""
    directory = os.path.dirname(os.path.abspath(xml_path)) or '.'
    base = re.sub(r'-METADATA\.xml$', '', os.path.basename(xml_path), flags=re.IGNORECASE)
    stem = f'{base}-SPECTRAL_IMAGE_{detector.upper()}'
    for ext in _ENMAP_RASTER_EXTENSIONS:
        candidate = os.path.join(directory, stem + ext)
        if os.path.exists(candidate):
            return candidate
    matches = sorted(m for m in glob.glob(os.path.join(directory, stem + '*')) if os.path.isfile(m))
    return matches[0] if matches else None


def resolve_enmap_product_files(input_path, other_detector_path=None):
    """Resolve whatever the user points at into the (primary_raster_path,
    sibling_raster_path) pair load_enmap_radiance() actually needs, so a
    user never has to hunt down and separately pick the two big VNIR/SWIR
    raster files from a DLR EnMAP download themselves - just point this (and
    the GUI/CLI, which call it first) at the *-METADATA.XML sidecar that
    ships right alongside them:

      - input_path is a *-METADATA.XML sidecar: both the VNIR and SWIR
        spectral-image rasters are located next to it by filename
        (_find_detector_raster_next_to_metadata()). primary_raster_path is
        whichever one was found (VNIR preferred if both exist - order
        doesn't matter to the eventual merge, which sorts by wavelength
        regardless); the other becomes sibling_raster_path. Raises
        FileNotFoundError if NEITHER raster is found next to the metadata
        file (e.g. it was moved away from its product folder).
      - input_path is already a raster file (VNIR, SWIR, or a
        pre-merged/custom cube someone built themselves): returned as
        primary_raster_path unchanged, with sibling_raster_path=None -
        load_enmap_radiance()'s own _sibling_detector_path() auto-detection
        already handles finding a VNIR/SWIR sibling for a raster input, so
        this function only adds value for the METADATA.XML case above.

    other_detector_path, if given, always overrides auto-detection (in
    either case) and is returned as-is for sibling_raster_path - it's an
    explicit user choice, not a guess this function should second-guess."""
    is_metadata = os.path.basename(input_path).lower().endswith('-metadata.xml')
    if not is_metadata:
        return input_path, other_detector_path

    vnir = _find_detector_raster_next_to_metadata(input_path, 'vnir')
    swir = _find_detector_raster_next_to_metadata(input_path, 'swir')
    primary = vnir or swir
    if primary is None:
        raise FileNotFoundError(
            f'No VNIR or SWIR spectral-image raster found next to {input_path} - expected a '
            "*-SPECTRAL_IMAGE_VNIR/SWIR file in the same directory, in one of these formats: "
            + ', '.join(_ENMAP_RASTER_EXTENSIONS))

    if other_detector_path is not None:
        return primary, other_detector_path
    sibling = swir if primary is vnir else None
    return primary, sibling


def _write_envi_cube(path, cube, wavelength, geotransform, projection_wkt):
    """Write `cube` (rows, bands, cols) out as an ENVI binary + .hdr pair,
    via GDAL's own ENVI driver so it correctly generates the projection/
    map-info section of the header for whatever CRS this scene is in
    (UTM, geographic, ...) rather than this module hand-rolling ENVI's
    per-projection map-info syntax.

    Wavelength is written through the dataset's 'ENVI' metadata domain.
    Verified against GDAL's real ENVI driver source
    (frmts/raw/envidataset.cpp): metadata set in that domain is written
    into the .hdr VERBATIM as 'key = value' - GDAL does NOT brace-wrap list
    values itself, so the value passed here must already look like
    '{500.0, 510.0, ...}' to come back out as the bracketed list
    parse_envi_header() (elsewhere in this module) expects.

    Every pixel already carrying NaN for missing data (set during the
    original load) is marked as such via SetNoDataValue(nan), matching
    what's actually in the array by this point - not the source files'
    original raw-DN nodata sentinel, which no longer applies after the
    gain/offset conversion to physical radiance.

    Also writes ENVI's 'default bands' hint (again verified against the
    real GDAL source: it's only auto-generated from per-band
    SetColorInterpretation() calls, which this function doesn't make, so
    without writing it explicitly the header ends up with no RGB hint at
    all). Without it, QGIS (and ENVI itself) has no idea which 3 of a
    224-band cube are meant for a color display and falls back to bands
    1/2/3 - for this module's wavelength-sorted merge, three nearly
    identical ~418-429 nm violet bands, which is very likely why a merged
    cube opened in QGIS didn't look like a sensible RGB image. The bands
    picked here are the exact same ones _build_rgb_composite() uses for
    the in-app figures/preview, so QGIS's own default view matches."""
    from osgeo import gdal
    n_rows, n_bands, n_cols = cube.shape
    driver = gdal.GetDriverByName('ENVI')
    ds = driver.Create(path, n_cols, n_rows, n_bands, gdal.GDT_Float32)
    ds.SetGeoTransform(geotransform)
    ds.SetProjection(projection_wkt)

    wl_str = '{' + ', '.join(f'{w:.4f}' for w in wavelength) + '}'
    ds.SetMetadataItem('wavelength', wl_str, 'ENVI')
    ds.SetMetadataItem('wavelength units', 'Nanometers', 'ENVI')

    r_idx, g_idx, b_idx = _pick_rgb_band_indices(wavelength)
    ds.SetMetadataItem('default bands', f'{{{r_idx}, {g_idx}, {b_idx}}}', 'ENVI')

    for b in range(n_bands):
        band = ds.GetRasterBand(b + 1)
        band.WriteArray(cube[:, b, :].astype('float32'))
        band.SetDescription(f'Band {b + 1} ({wavelength[b]:.2f} Nanometers)')
        band.SetNoDataValue(float('nan'))

    # EnMAP's own geotransform (read straight off the source product and
    # passed through here unmodified) is a best-fit affine approximation of
    # a pushbroom sensor's true per-scanline geolocation, not a perfect
    # similarity transform - its two axes generally imply very slightly
    # different rotation angles. ENVI's .hdr 'rotation=' field can only
    # express ONE clean angle (verified against GDAL's real ENVI driver
    # source, frmts/raw/envidataset.cpp WriteProjectionInfo(): it averages
    # the two axes' angles and emits "Geotransform matrix has non
    # rotational terms" whenever they disagree). That averaging is harmless
    # here - every lat/lon this pipeline actually computes
    # (_load_single_enmap_raster, _resample_raster_to_lonlat, ...) uses the
    # full affine geotransform directly, never ENVI's single-angle
    # approximation - so this warning is suppressed rather than left to
    # alarm users about a real geolocation problem that doesn't exist. Any
    # other GDAL message still prints normally.
    def _quiet_envi_rotation_warning(err_class, err_no, msg):
        if 'non rotational terms' not in msg:
            gdal.CPLDefaultErrorHandler(err_class, err_no, msg)

    gdal.PushErrorHandler(_quiet_envi_rotation_warning)
    try:
        ds.FlushCache()
        ds = None
    finally:
        gdal.PopErrorHandler()


def _merge_vnir_swir_scenes(scene_a, scene_b):
    """Concatenate two already-loaded EnMAP detector scenes (VNIR + SWIR, in
    either order) into one combined cube spanning the full ~400-2500 nm
    range, the same way EnMAP's own combined METADATA.XML describes the
    full instrument (91 VNIR + 133 SWIR = 224 bands, with the two detectors
    genuinely overlapping in nominal wavelength around 902-993 nm - both
    are kept, not deduplicated, matching how the vendor's own metadata
    models it).

    Both detectors are orthorectified onto the same grid in EnMAP's
    official L1C/L2A products, so they're expected to share row/col
    dimensions - this raises a clear error rather than silently
    concatenating misaligned pixels if that assumption doesn't hold (e.g.
    a reprocessed or cropped file where only one detector was touched)."""
    cube_a, cube_b = scene_a['cube'], scene_b['cube']
    if cube_a.shape[0] != cube_b.shape[0] or cube_a.shape[2] != cube_b.shape[2]:
        raise ValueError(
            f'The two detector rasters have different row/col dimensions '
            f'({cube_a.shape[0]}x{cube_a.shape[2]} vs {cube_b.shape[0]}x{cube_b.shape[2]}) - '
            "they don't look like a co-registered VNIR+SWIR pair from the same EnMAP "
            'product. Merge only matching detector files from the same acquisition, '
            'or pass merge_vnir_swir=False to process one detector file alone.')

    gt_a, gt_b = scene_a['geotransform'], scene_b['geotransform']
    if max(abs(a - b) for a, b in zip(gt_a, gt_b)) > 1e-6:
        warnings.warn('The VNIR and SWIR geotransforms differ slightly; using the '
                       'first-loaded file\'s geotransform for the merged cube. Verify '
                       'the two files are really from the same acquisition/processing run.')

    cube = np.concatenate([cube_a, cube_b], axis=1)
    wavelength = np.concatenate([scene_a['wavelength'], scene_b['wavelength']])
    # Sort into a monotonic wavelength axis - besides being the natural
    # order, plot_spectra_figure() draws a plain line plot against
    # wavelength, which would zigzag if VNIR and SWIR bands were left in
    # file-concatenation order instead.
    # Stable, so bands with identical centres in the ~902-993 nm VNIR/SWIR
    # detector overlap keep VNIR-before-SWIR order deterministically rather
    # than however quicksort happens to break the tie on a given NumPy build.
    # This is exactly the band-index/wavelength alignment that
    # _warn_if_wavelengths_unordered() exists to protect.
    order = np.argsort(wavelength, kind='stable')

    return {
        'cube': cube[:, order, :],
        'wavelength': wavelength[order],
        'wavelength_source': (f"VNIR+SWIR merge ({scene_a['wavelength_source']} "
                               f"| {scene_b['wavelength_source']})"),
        'lat': scene_a['lat'],
        'lon': scene_a['lon'],
        'geotransform': gt_a,
        'projection_wkt': scene_a['projection_wkt'],
        'nodata': scene_a['nodata'],
    }


def load_enmap_radiance(raster_path, hdr_path=None, other_detector_path=None,
                         merge_vnir_swir=True, save_merged_cube=True, **kwargs):
    """Load an EnMAP radiance product, automatically combining a VNIR+SWIR
    pair into one cube spanning the full ~400-2500 nm range where possible.

    A stock EnMAP download from DLR ships VNIR and SWIR as two separate
    files (*-SPECTRAL_IMAGE_VNIR.TIF and *-SPECTRAL_IMAGE_SWIR.TIF). Point
    this at EITHER one:
      - merge_vnir_swir=True (the default): the sibling file is looked for
        automatically next to raster_path (same directory, same naming
        convention with VNIR/SWIR swapped) and merged in if found - no
        extra action needed. If no sibling is found (or raster_path doesn't
        match that naming convention at all - e.g. it's already a combined
        cube you built yourself), raster_path alone is loaded and used as-is.
      - other_detector_path: give this explicitly to merge with a specific
        file (different name or directory) instead of relying on
        auto-detection; this always takes precedence over auto-detection.
      - merge_vnir_swir=False: never merge, even if a sibling file sits
        right next to raster_path - use exactly the one file given.

    save_merged_cube=True (the default): the FIRST time a VNIR+SWIR merge
    actually happens for a given pair, the result is written once as an
    ENVI binary + .hdr next to the two source files (see
    _merged_cube_path() for the exact name) via _write_envi_cube(). Every
    call after that - for this function and for _sibling_detector_path()
    based auto-detection - finds that file FIRST and loads it directly,
    skipping the VNIR+SWIR reload-and-concatenate step entirely. Set False
    to skip writing it (e.g. a read-only input folder); merging still
    happens each time in that case, just isn't cached to disk. A failure
    to write it (permissions, disk space, ...) is a warning, not a fatal
    error - the run continues using the merged cube already in memory.

    To process an EnMAP scene you've already combined into your own single
    VNIR+SWIR cube file, just point raster_path at it - since that file
    won't match the *-SPECTRAL_IMAGE_VNIR/SWIR.TIF naming convention, it's
    loaded as-is with no merge attempted.

    hdr_path and any extra keyword arguments are forwarded to each detector
    file's own load (see _load_single_enmap_raster() for what they mean);
    with two files, hdr_path (if given) is assumed to apply to raster_path
    only - the sibling file's own .hdr (or other calibration source) is
    auto-detected independently.

    Returns the same dict shape as _load_single_enmap_raster() (cube,
    wavelength, wavelength_source, lat, lon, geotransform, projection_wkt,
    nodata) whether one or two files were loaded.
    """
    # The cached merge is reused whether or not the caller named the sibling
    # file explicitly: run_pipeline() always resolves the sibling first, and
    # the earlier `other_detector_path is None` guard made every pipeline run
    # re-merge and rewrite the 0.9 GB cube (ten minutes per scene).
    if merge_vnir_swir:
        existing_merged = _merged_cube_path(raster_path)
        if existing_merged and os.path.exists(existing_merged):
            print(f'Found previously-merged VNIR+SWIR cube: {existing_merged}')
            cached = _load_single_enmap_raster(existing_merged)
            _warn_if_wavelengths_unordered(cached)
            return cached

    primary = _load_single_enmap_raster(raster_path, hdr_path=hdr_path, **kwargs)

    sibling_path = other_detector_path
    if sibling_path is None and merge_vnir_swir:
        sibling_path = _sibling_detector_path(raster_path)
        if sibling_path:
            print(f'Found matching detector file: {sibling_path}')

    if sibling_path is None:
        _warn_if_wavelengths_unordered(primary)
        return primary

    print(f'Loading second detector file {sibling_path} to build a combined VNIR+SWIR cube...')
    secondary = _load_single_enmap_raster(sibling_path, **kwargs)
    merged = _merge_vnir_swir_scenes(primary, secondary)
    # The two per-file "Wavelength source" / "First 3 band wavelengths" lines
    # just above describe each detector's OWN range on its own (VNIR ~418 nm,
    # SWIR ~900 nm) - print an explicit summary of the actual MERGED result
    # right here too, so it doesn't have to be inferred from the last of
    # those two per-file lines.
    print(f'Merged VNIR+SWIR cube: {merged["cube"].shape[1]} bands, '
          f'{merged["wavelength"].min():.1f}-{merged["wavelength"].max():.1f} nm '
          f'(first 3: {np.round(merged["wavelength"][:3], 2).tolist()}, '
          f'last 3: {np.round(merged["wavelength"][-3:], 2).tolist()})')

    if save_merged_cube:
        merged_path = _merged_cube_path(raster_path)
        if merged_path:
            try:
                _write_envi_cube(merged_path, merged['cube'], merged['wavelength'],
                                  merged['geotransform'], merged['projection_wkt'])
                print(f'Saved merged cube for reuse: {merged_path} '
                      '(future runs on this VNIR/SWIR pair will load it directly '
                      'instead of re-merging)')
            except Exception as exc:
                warnings.warn(f'Could not save the merged VNIR+SWIR cube to {merged_path} '
                               f'({exc.__class__.__name__}: {exc}); continuing with the merged '
                               'cube already in memory for this run - future runs will just '
                               're-merge from the VNIR/SWIR files again.')

    _warn_if_wavelengths_unordered(merged)
    return merged


def _warn_if_wavelengths_unordered(scene):
    """Warn when a scene's wavelength array is not monotonically increasing.

    Every downstream band lookup - search_band(), compute_hotspot_index()'s
    SWIR window, the plotting code's RGB picks - assumes entry i of
    `wavelength` describes band i of `cube`. A non-monotonic array is the
    signature of that alignment having broken, and the specific way it
    breaks here is easy to hit: _merge_vnir_swir_scenes() sorts the merged
    cube by wavelength, so its VNIR and SWIR bands interleave through the
    ~900-1000 nm detector overlap, whereas EnMAP's METADATA.XML lists all
    224 bands VNIR-block-then-SWIR-block. Applying the metadata list to an
    already-sorted cube therefore mislabels every band in that overlap.
    (Verified on a real L1B product: metadata entry 91 is 993.34 nm while
    the merged cube's band 91 is 951.36 nm.)

    Nothing in the pipeline is currently wired to make that mistake - a
    merged cube carries its own .hdr, and _find_enmap_metadata_xml() only
    matches the per-detector VNIR/SWIR TIFs - so this is a tripwire for
    future changes and for hand-assembled products, not a known failure."""
    wl = np.asarray(scene.get('wavelength', ()), dtype=float)
    if wl.size < 2 or np.all(np.diff(wl) > 0):
        return
    n_bad = int(np.sum(np.diff(wl) <= 0))
    warnings.warn(
        f'Wavelength array is not monotonically increasing ({n_bad} descending step(s); '
        f'source: {scene.get("wavelength_source", "unknown")}). Band i of the cube may not '
        'correspond to wavelength[i] - check whether a VNIR-then-SWIR band list has been '
        'applied to a wavelength-sorted merged cube. Retrieved temperatures depend on this '
        'mapping being right.')


#%% ###########################################################################
#                Automatic hotspot (fire pixel) detection (new)
################################################################################


def otsu_threshold(values):
    """Manual Otsu threshold (no scikit-image dependency). values: 1D array,
    NaNs are dropped. Returns the threshold that maximizes between-class
    variance of a 256-bin histogram."""
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError('No finite values to threshold.')

    hist, bin_edges = np.histogram(values, bins=256)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]

    with np.errstate(invalid='ignore', divide='ignore'):
        mean1 = np.cumsum(hist * bin_centers) / weight1
        mean2 = (np.cumsum((hist * bin_centers)[::-1])[::-1]) / weight2

    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    if not np.isfinite(variance12).any():
        # Every candidate split leaves one class empty, so there is no
        # between-class variance to maximise and no meaningful threshold. This
        # is the constant-field case - all values land in one histogram bin, so
        # every entry is 0 * nan and np.nanargmax raises on the all-NaN slice.
        return float(bin_centers[0])
    idx = int(np.nanargmax(variance12))
    return bin_centers[idx]


def compute_hotspot_index(cube, wavelength, wl_lo=2060.0, wl_hi=2430.0):
    """SWIR hotspot index: (L_hi - L_lo) / (L_hi + L_lo), using the bands
    nearest wl_lo/wl_hi nm. A pixel actively emitting thermal radiance shows
    a disproportionate excess at the longer wavelength relative to a
    reflected-solar-only background pixel, so fire pixels score HIGHER on
    this index than background. This is the HFDI of Dennison & Roberts
    (2009) as used by Amici et al. (2022), Eq. 2, with the same sign
    convention: in their released code the background pool is HFDI <= -0.2
    and the pixels kept as fire are those with HFDI >= the threshold."""
    band_lo, _ = search_band(cube, wavelength, wl_lo)
    band_hi, _ = search_band(cube, wavelength, wl_hi)
    denom = band_hi + band_lo
    with np.errstate(invalid='ignore', divide='ignore'):
        index = np.where(denom > 0, (band_hi - band_lo) / denom, np.nan)
    return index


def swir_radiance_magnitude(cube, wavelength, wl_lo=2060.0, wl_hi=2430.0):
    """Total measured SWIR radiance L(wl_lo) + L(wl_hi) - exactly the
    DENOMINATOR of compute_hotspot_index(). Returned as its own map because
    the hotspot index alone cannot be thresholded safely: see
    detect_fire_pixels()'s `swir_radiance` argument for why."""
    band_lo, _ = search_band(cube, wavelength, wl_lo)
    band_hi, _ = search_band(cube, wavelength, wl_hi)
    return band_lo + band_hi


def robust_background_threshold(values, k=6.0):
    """median + k * robust-sigma (1.4826 * MAD) of `values`. This is the
    standard contextual-fire-algorithm style threshold (e.g. Giglio et al.'s
    MODIS/VIIRS active fire tests use a similar mean/MAD-based background
    test): appropriate when fire pixels are a rare-outlier tail on an
    overwhelmingly uniform background, which is the realistic case for a
    wildfire scene. Otsu, by contrast, assumes two comparably-sized
    populations and is unreliable here - with <1% of pixels on fire, Otsu
    tends to just split the background noise roughly in half instead of
    isolating the true outliers."""
    values = values[np.isfinite(values)]
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    return median + k * 1.4826 * mad


def detect_fire_pixels(hotspot_index, threshold=None, method='robust', k=6.0,
                        margin_px=5, min_cluster_size=1,
                        swir_radiance=None, radiance_threshold=None, radiance_k=6.0):
    """Threshold the hotspot index and return:
      fire_points      : (N, 2) int array of [row, col] for every fire pixel
      background_mask  : bool array, True where a pixel is safe to use as
                          background (i.e. not fire, and not within
                          margin_px of any fire pixel)
      threshold_used

    method='robust' (default) uses robust_background_threshold() - correct
    for the realistic case of rare fire pixels in an overwhelmingly uniform
    background. method='otsu' is offered as an alternative for scenes with
    a genuinely large, comparably-sized burned/burning area, where a
    balanced bimodal split is more appropriate.

    Uses scipy.ndimage for connected-component labeling and dilation instead
    of OpenCV, and dilation instead of a manual nested-loop margin (which in
    the original PRISMA script silently clipped rather than dropped
    out-of-bounds margin pixels at image edges).

    swir_radiance (swir_radiance_magnitude(), i.e. L_lo + L_hi) adds the
    SECOND, magnitude half of the detection test, and passing it is strongly
    recommended - without it this function has a large, systematic false-
    positive mode.

    Why. The hotspot index is a ratio, HI = (L_hi - L_lo) / (L_hi + L_lo).
    Write each measured band as L = L_true + eps with detector noise
    eps ~ N(0, sigma^2). Propagating to first order,

        Var(HI) ~= 2 * sigma^2 * (1 - HI^2)^2 / (L_hi + L_lo)^2 ,

    so the index's variance diverges as the denominator goes to zero. HI is
    therefore strongly HETEROSCEDASTIC across a scene, while
    robust_background_threshold() (median + k*1.4826*MAD) is a
    homoscedastic test: it estimates one spread from the whole image and
    applies it everywhere. In the dark tail - deep terrain shadow, water,
    cloud shadow, where L_hi + L_lo approaches the noise floor - HI is
    essentially a ratio of two noise samples and wanders over [-1, 1],
    clearing any global threshold purely by chance. Measured directly on
    the two scenes in this repository, 70% (Yalu River) and 87% (Mojave) of
    index-only detections sat BELOW the scene's 1st percentile of SWIR
    radiance: they were the darkest pixels in the image, not the hottest.

    The magnitude test is also the physically correct closure. A subpixel
    fire ADDS emitted radiance at wl_hi; it cannot make a pixel darker than
    the background at 2.4 um. So a genuine fire pixel must satisfy both

        (shape)     HI > threshold
        (magnitude) L_lo + L_hi > radiance_threshold

    and requiring both is what operational contextual fire algorithms do
    (Giglio et al. 2003, 2016 pair a ratio test with an absolute brightness
    test for exactly this reason; Dennison & Roberts 2009 and Murphy et al.
    2016 likewise gate their SWIR ratio on an absolute radiance level).
    Conjoining them is what removes the dark-pixel mode: on the two scenes
    here it cuts 57 -> 16 and 340 -> 4 detections, and every survivor gives
    a physically sensible model-free Wien-slope temperature (730-1260 K),
    which none of the discarded dark pixels do.

    radiance_threshold defaults to robust_background_threshold(
    swir_radiance, k=radiance_k). Pass swir_radiance=None to restore the
    old index-only behaviour (kept only so existing callers and the
    verification scripts still run)."""
    if threshold is None:
        if method == 'otsu':
            threshold = otsu_threshold(hotspot_index)
        elif method == 'robust':
            threshold = robust_background_threshold(hotspot_index, k=k)
        else:
            raise ValueError("method must be 'robust' or 'otsu'")

    fire_mask = np.nan_to_num(hotspot_index, nan=-np.inf) > threshold

    if swir_radiance is not None:
        if swir_radiance.shape != hotspot_index.shape:
            raise ValueError(f'swir_radiance has shape {swir_radiance.shape}, expected '
                             f'{hotspot_index.shape} to match hotspot_index.')
        if radiance_threshold is None:
            radiance_threshold = robust_background_threshold(swir_radiance, k=radiance_k)
        fire_mask &= np.nan_to_num(swir_radiance, nan=-np.inf) > radiance_threshold

    labels, n_clusters = ndi.label(fire_mask)
    if min_cluster_size > 1 and n_clusters > 0:
        sizes = ndi.sum_labels(fire_mask, labels, index=np.arange(1, n_clusters + 1))
        small = np.where(sizes < min_cluster_size)[0] + 1
        fire_mask = fire_mask & ~np.isin(labels, small)

    fire_points = np.argwhere(fire_mask)

    buffered_fire = ndi.binary_dilation(fire_mask, iterations=margin_px)
    background_mask = (~buffered_fire) & np.isfinite(hotspot_index)

    return fire_points, background_mask, threshold


#%% ###########################################################################
#                        Validation utilities (new)
################################################################################


def load_field_data(csv_path, lat_col='lat', lon_col='lon', temp_col='temperature_K'):
    """Field measurements CSV with at least lat/lon/temperature columns."""
    df = pd.read_csv(csv_path)
    for col in (lat_col, lon_col, temp_col):
        if col not in df.columns:
            raise ValueError(f"Expected column '{col}' in {csv_path}, found {list(df.columns)}")
    return df.rename(columns={lat_col: 'lat', lon_col: 'lon', temp_col: 'temperature_K'})


# Landsat 8/9 Collection-2 Level-2 surface temperature is delivered as
# uint16 DN with this fixed USGS scale/offset, and 0 reserved as fill.
_LANDSAT_ST_SCALE, _LANDSAT_ST_OFFSET, _LANDSAT_ST_FILL = 0.00341802, 149.0, 0

# The ST product's valid retrieval range tops out around 372 K; a genuine
# flame core is far hotter, so any comparison pixel at or near this ceiling
# is reporting the band's limit rather than the surface (see
# validate_temperature(), which reports how many such pixels it found).
_LANDSAT_ST_CEILING_K = 372.0


def load_landsat_lst(tif_path, apply_quality_mask=True, qa_pixel_path=None):
    """Landsat 8/9 Collection-2 Level-2 ST_B10 GeoTIFF -> Kelvin, using the
    USGS scale/offset (DN * 0.00341802 + 149.0).

    Fill is masked to NaN. USGS ships these rasters with nodata=0 declared,
    but a clipped/re-exported copy frequently loses that tag, and without it
    every fill pixel silently becomes 149.0 K - a large block of physically
    impossible 'surface temperature' that then drags the validation
    statistics in validate_temperature(). So fall back to the product's
    documented fill value when the tag is missing rather than trusting it
    unconditionally.

    apply_quality_mask (default True) additionally drops cloud, cloud
    shadow and dilated-cloud pixels using the granule's companion
    '..._QA_PIXEL.TIF' when it sits next to the ST band - a cloud-top
    temperature compared against an EnMAP fire pixel is not a validation
    point. Pass qa_pixel_path= to point elsewhere, or
    apply_quality_mask=False to keep the raw retrieval."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(tif_path), gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f'GDAL could not open {tif_path}')
    dn = ds.ReadAsArray().astype('float64')
    nodata = ds.GetRasterBand(1).GetNoDataValue()
    if nodata is None or not np.isfinite(nodata):
        nodata = _LANDSAT_ST_FILL
    lst_k = np.where(dn == nodata, np.nan, dn * _LANDSAT_ST_SCALE + _LANDSAT_ST_OFFSET)

    if apply_quality_mask:
        qa_file = qa_pixel_path
        if qa_file is None:
            m = re.match(r'(?i)^(.*)_ST_B\d+\.tif$', os.path.basename(str(tif_path)))
            if m:
                candidate = os.path.join(os.path.dirname(str(tif_path)),
                                         f'{m.group(1)}_QA_PIXEL.TIF')
                qa_file = candidate if os.path.isfile(candidate) else None
        if qa_file:
            qa = gdal.Open(str(qa_file), gdal.GA_ReadOnly).ReadAsArray().astype('uint32')
            if qa.shape == lst_k.shape:
                # Collection-2 QA_PIXEL bits: 0 fill, 1 dilated cloud,
                # 3 cloud, 4 cloud shadow. Cirrus (bit 2) is left in - it
                # barely perturbs the thermal retrieval.
                bad = (qa & ((1 << 0) | (1 << 1) | (1 << 3) | (1 << 4))) != 0
                n_before = int(np.isfinite(lst_k).sum())
                lst_k = np.where(bad, np.nan, lst_k)
                print(f'[Landsat] QA_PIXEL masking removed '
                      f'{n_before - int(np.isfinite(lst_k).sum())} of {n_before} valid pixels')
            else:
                warnings.warn(f'Landsat QA_PIXEL {qa_file} has shape {qa.shape}, '
                              f'expected {lst_k.shape} - not applied.')

    return {
        'lst_K': lst_k,
        'geotransform': ds.GetGeoTransform(),
        'projection_wkt': ds.GetProjection(),
    }


# Physically plausible land-surface-temperature window, used only to decide
# whether an ECOSTRESS raster is already in Kelvin or still in packed
# integer DN (see _ecostress_resolve_scale()). Deliberately generous: the
# coldest Antarctic surface ever recorded is ~180 K and an active fire core
# seen by a coarse thermal sensor tops out well under 400 K (the band
# saturates long before the true flame temperature), so anything landing
# inside this window is a temperature and anything far outside it is not.
_LST_PLAUSIBLE_K = (150.0, 400.0)

# Approximate upper end of the ECOSTRESS L2 LSTE retrieval. The product is
# produced over roughly 200-400 K, but its 8-12 um channels are designed
# for ambient land surfaces and depart from linearity well before a flaming
# front; a retrieval pinned up here is reporting the instrument's limit
# rather than the ground. Used only to flag saturated comparison pixels in
# validate_temperature(), never to mask data.
_ECOSTRESS_CEILING_K = 380.0


def _ecostress_resolve_scale(band, values, explicit_scale, explicit_offset):
    """Work out how to turn an ECOSTRESS LST raster's stored values into
    Kelvin, returning (scale, offset, why).

    This exists because the same 'ECOSTRESS LST' can reach the pipeline in
    two quite different encodings, and applying the wrong one is silent:

      * The native HDF5 ECO2LSTE / ECO_L2_LSTE 'SDS/LST' layer (and an
        AppEEARS export of it) stores uint16 DN with a documented x0.02
        scale factor and no offset - the conversion Amici et al. (2022)
        use, and what this function's `scale_factor=0.02` default used to
        hard-code.
      * The tiled Collection-2 ECO_L2T_LSTE product that LP DAAC and
        download_ecostress_lst() actually deliver ships one float32
        Cloud-Optimized GeoTIFF per band, ALREADY IN KELVIN, with the
        scale factor applied during production.

    Multiplying an already-Kelvin L2T tile by 0.02 turns a real 300 K
    surface into 6 K, which then flows all the way through
    validate_temperature() into the results CSV as a plausible-looking
    'ecostress_LST_K' column of ~5.8 - wrong by a factor of 50, with
    nothing in the output flagging it. So resolve this from the file
    itself rather than from a default that only suits one of the two.

    Order of precedence: an explicit caller override wins; then GDAL's own
    band scale/offset metadata if the producer set one; then the
    encoding inferred from the dtype and the actual value range."""
    if explicit_scale is not None:
        return float(explicit_scale), float(explicit_offset or 0.0), 'explicit scale_factor argument'

    # GDAL exposes a band-level scale/offset when the producer wrote one.
    # GetScale() returns 1.0 (not None) when unset, so only trust it when
    # it actually differs from the identity transform.
    gdal_scale, gdal_offset = band.GetScale(), band.GetOffset()
    if gdal_scale is not None and gdal_offset is not None and \
            not (abs(gdal_scale - 1.0) < 1e-12 and abs(gdal_offset) < 1e-12):
        return float(gdal_scale), float(gdal_offset), 'GDAL band scale/offset metadata'

    finite = values[np.isfinite(values)]
    finite = finite[finite != 0]
    if finite.size == 0:
        # Nothing to judge from - fall back to the identity so an all-fill
        # tile stays all-NaN downstream instead of being scaled into a
        # confidently wrong number.
        return 1.0, float(explicit_offset or 0.0), 'empty raster, assumed already Kelvin'

    median = float(np.median(finite))
    lo, hi = _LST_PLAUSIBLE_K
    if lo <= median <= hi:
        return 1.0, float(explicit_offset or 0.0), \
            f'values already in Kelvin (median {median:.1f} K)'
    if lo <= median * 0.02 <= hi:
        return 0.02, float(explicit_offset or 0.0), \
            f'packed DN, applying the documented x0.02 scale (median {median:.0f} DN -> {median * 0.02:.1f} K)'

    # Neither encoding explains the numbers. Don't guess - a wrong scale
    # here is exactly the failure mode this function exists to prevent.
    raise ValueError(
        f'Cannot determine the ECOSTRESS LST scaling: median stored value is {median:g}, '
        f'which is neither a plausible Kelvin temperature nor a x0.02-scaled one '
        f'(plausible range {lo:g}-{hi:g} K). Pass an explicit scale_factor= '
        f'(and add_offset= if needed) for this product.')


def _ecostress_sibling(path, suffix):
    """Path to a sibling band of the same ECO_L2T_LSTE granule, e.g. the
    '..._QC.tif' or '..._cloud.tif' next to '..._LST.tif'. LP DAAC splits
    one granule across several single-band files sharing a common stem, so
    the companion masks are found by swapping the trailing band name.
    Returns None when the file isn't there (an AppEEARS export or a
    hand-clipped LST band often travels alone)."""
    m = re.match(r'(?i)^(.*)_LST\.tif$', os.path.basename(str(path)))
    if not m:
        return None
    candidate = os.path.join(os.path.dirname(str(path)), f'{m.group(1)}_{suffix}.tif')
    return candidate if os.path.isfile(candidate) else None


def load_ecostress_lst(path, scale_factor=None, add_offset=None,
                       apply_quality_mask=True, qc_path=None, cloud_path=None):
    """Load an ECOSTRESS L2(T)_LSTE surface-temperature product and return
    its land-surface temperature in Kelvin.

    Handles both encodings this product ships in - the float32 Kelvin
    Collection-2 L2T tiles that download_ecostress_lst() fetches, and the
    x0.02-scaled integer DN of the native HDF5/AppEEARS layer - by
    inspecting the file instead of assuming one. See
    _ecostress_resolve_scale() for why that matters (guessing wrong is a
    silent factor-of-50 error). Pass scale_factor=/add_offset= explicitly
    to override the detection for an unusual export.

    apply_quality_mask (default True) additionally blanks pixels that the
    granule's own companion masks flag as unusable, when those files are
    present next to the LST band:
      * '..._cloud.tif' - non-zero means cloud, whose retrieved LST is the
        cloud top, not the ground, and would otherwise be silently compared
        against an EnMAP fire pixel.
      * '..._QC.tif' - bits 0-1 are the mandatory QA flags, where 3 means
        'pixel not produced'. Those pixels carry fill values that are not
        temperatures at all.
    Quality flags 0 (best) and 1 (nominal) are kept, as is 2 (produced,
    cloud detected) when no separate cloud band is available - 2 is left in
    deliberately, since the cloud band is the more reliable of the two and
    dropping every cloud-adjacent pixel can empty a small scene entirely.
    Pass explicit qc_path=/cloud_path= to point at masks stored elsewhere,
    or apply_quality_mask=False to keep the raw retrieval."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError(f'GDAL could not open {path}')
    band = ds.GetRasterBand(1)
    raw = ds.ReadAsArray().astype('float64')
    nodata = band.GetNoDataValue()

    # Drop fill BEFORE inferring the scale, so a large block of sentinel
    # values (Landsat-style 0, or ECOSTRESS's own NaN) cannot drag the
    # median used for the Kelvin-vs-DN decision away from the real data.
    valid = np.isfinite(raw)
    if nodata is not None and np.isfinite(nodata):
        valid &= (raw != nodata)

    scale, offset, why = _ecostress_resolve_scale(band, np.where(valid, raw, np.nan),
                                                  scale_factor, add_offset)
    lst_k = np.where(valid, raw * scale + offset, np.nan)
    print(f'[ECOSTRESS] {os.path.basename(str(path))}: {why}'
          f' -> scale={scale:g} offset={offset:g}')

    if apply_quality_mask:
        n_before = int(np.isfinite(lst_k).sum())
        cloud_file = cloud_path or _ecostress_sibling(path, 'cloud')
        if cloud_file:
            cloud = gdal.Open(str(cloud_file), gdal.GA_ReadOnly).ReadAsArray()
            if cloud.shape == lst_k.shape:
                lst_k = np.where(cloud != 0, np.nan, lst_k)
            else:
                warnings.warn(f'ECOSTRESS cloud mask {cloud_file} has shape {cloud.shape}, '
                              f'expected {lst_k.shape} - not applied.')
        qc_file = qc_path or _ecostress_sibling(path, 'QC')
        if qc_file:
            qc = gdal.Open(str(qc_file), gdal.GA_ReadOnly).ReadAsArray()
            if qc.shape == lst_k.shape:
                # Bits 0-1 == 3 -> 'pixel not produced due to missing/bad data'.
                lst_k = np.where((qc.astype('uint32') & 0x3) == 3, np.nan, lst_k)
            else:
                warnings.warn(f'ECOSTRESS QC mask {qc_file} has shape {qc.shape}, '
                              f'expected {lst_k.shape} - not applied.')
        n_after = int(np.isfinite(lst_k).sum())
        if cloud_file or qc_file:
            print(f'[ECOSTRESS] quality masking removed {n_before - n_after} of {n_before} '
                  f'valid pixels ({n_after} left)')

    return {
        'lst_K': lst_k,
        'geotransform': ds.GetGeoTransform(),
        'projection_wkt': ds.GetProjection(),
    }


_ISO_DATETIME_RE = re.compile(r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?)')


def _parse_iso_datetime_text(text):
    """First ISO-8601-shaped 'YYYY-MM-DDTHH:MM:SS[.ffffff]' substring in
    `text` parsed as a naive UTC datetime, or None if none is found/valid.
    EnMAP's own timestamps are already UTC (their filename convention ends
    the timestamp in 'Z'), so no timezone conversion is needed here - this
    just extracts and parses the digits."""
    if not text:
        return None
    m = _ISO_DATETIME_RE.search(text.strip())
    if not m:
        return None
    raw = m.group(1)
    for fmt in ('%Y-%m-%dT%H:%M:%S.%f', '%Y-%m-%dT%H:%M:%S'):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _acquisition_datetime_from_metadata_xml(xml_path):
    """Best-effort UTC acquisition datetime parsed from an EnMAP
    METADATA.XML sidecar, tried first (in _scene_acquisition_datetime()
    below) since it's the product's own recorded acquisition time rather
    than a naming-convention artifact. Unlike bandCharacterisation (whose
    tag names are confirmed against the open-source EnMAP-Box's own
    importer - see _read_enmap_metadata_xml()), no single confirmed tag name
    is available for this, so this searches several plausible keywords, in
    order of how directly each one names the acquisition instant, and skips
    any element whose tag also suggests it's a processing/generation
    timestamp instead (e.g. '...productionDateTime') rather than the actual
    sensing time. Returns None if nothing plausible parses, in which case
    _scene_acquisition_datetime() falls back to the filename convention."""
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return None

    keywords = ('sensingstart', 'sensingtime', 'starttime', 'acquisitiondate',
                'acquisitiontime', 'centertime', 'datetime', 'date')
    for keyword in keywords:
        for el in _xml_findall_by_keyword(root, keyword):
            tag_lower = el.tag.rsplit('}', 1)[-1].lower()
            if any(bad in tag_lower for bad in ('processing', 'generation', 'production', 'stop')):
                continue
            dt = _parse_iso_datetime_text(el.text)
            if dt is not None:
                return dt
    return None


def _scene_acquisition_datetime(raster_path):
    """Best-effort UTC acquisition datetime for this scene. Tries, in order:
    (1) an EnMAP METADATA.XML sidecar via
    _acquisition_datetime_from_metadata_xml() - the product's own recorded
    metadata, preferred when present; (2) EnMAP's own filename convention
    (...-DT<id>_<YYYYMMDDTHHMMSSZ>_<scene>_V...), which reliably encodes the
    same acquisition timestamp (confirmed against a real DLR sample filename
    quoted in the open-source EnMAP-Box's own L1B importer) and so is a
    solid fallback when no metadata sidecar is found (e.g. a renamed file,
    or a custom cube built without one). Used to search for a temporally-
    close ECOSTRESS/Landsat granule in download_ecostress_lst()/
    download_landsat_lst(). Returns None if neither resolves; pass the
    datetime explicitly in that case."""
    xml_path = _find_enmap_metadata_xml(raster_path)
    if xml_path is not None:
        dt = _acquisition_datetime_from_metadata_xml(xml_path)
        if dt is not None:
            return dt

    base = os.path.splitext(os.path.basename(raster_path))[0]
    m = _ENMAP_FILENAME_RE.match(base)
    if not m:
        return None
    _, timestamp, _, _ = m.groups()
    try:
        return datetime.strptime(timestamp, '%Y%m%dT%H%M%SZ')
    except ValueError:
        return None


def _ecostress_scene_datetime(path):
    """Best-effort UTC datetime parsed straight from an ECOSTRESS L2T_LSTE
    filename (LP DAAC's convention embeds it directly, e.g.
    'ECOv002_L2T_LSTE_26529_002_46QEL_20230311T134257_0710_01_LST.tif') -
    used to label the ECOSTRESS panel in plot_multi_sensor_comparison_figure()
    with the date it ACTUALLS shows, which is not necessarily anywhere near
    the EnMAP scene's own date (see download_ecostress_lst()'s time-window
    widening). Works whether ecostress_path was auto-fetched or supplied by
    the user directly, since it only depends on the file's standard name.
    Returns None if the filename doesn't match."""
    m = re.search(r'(\d{8}T\d{6})', os.path.basename(str(path)))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), '%Y%m%dT%H%M%S')
    except ValueError:
        return None


# Landsat Collection-2 scene-identifier prefix -> the spacecraft it names.
# Only 8 and 9 are ever downloaded here (download_landsat_lst() filters the
# STAC search to those two platforms), but the older sensors are listed too so
# that a user-supplied file is labelled honestly instead of being mislabelled
# as 8/9.
_LANDSAT_PLATFORM_NAMES = {
    'LC08': 'Landsat 8', 'LO08': 'Landsat 8', 'LT08': 'Landsat 8',
    'LC09': 'Landsat 9', 'LO09': 'Landsat 9', 'LT09': 'Landsat 9',
    'LE07': 'Landsat 7', 'LT05': 'Landsat 5', 'LT04': 'Landsat 4',
}

_LANDSAT_FILENAME_RE = re.compile(
    r'(?i)^(L[COTE]\d{2})_(L\d\w{2})_(\d{3})(\d{3})_(\d{8})_(\d{8})_(\d{2})_(\w{2})')

_LANDSAT_MTL_TIME_RE = re.compile(
    r'(?i)SCENE_CENTER_TIME\s*[=>]\s*"?(\d{2}:\d{2}:\d{2}(?:\.\d+)?)Z?')
_LANDSAT_MTL_DATE_RE = re.compile(r'(?i)DATE_ACQUIRED\s*[=>]\s*"?(\d{4}-\d{2}-\d{2})')
_LANDSAT_MTL_CRAFT_RE = re.compile(r'(?i)SPACECRAFT_ID\s*[=>]\s*"?(LANDSAT_\d)')


def _landsat_mtl_path(path):
    """The '..._MTL.txt'/'..._MTL.xml' metadata file belonging to a Landsat
    Collection-2 band raster, if one sits next to it. A band filename is the
    product identifier plus a band suffix, and the MTL file is that same
    identifier plus '_MTL', so this is a direct sibling lookup rather than a
    directory scan."""
    base = os.path.basename(str(path))
    m = re.match(r'(?i)^(.*?)_(?:ST|SR|QA|B)[_A-Z0-9]*\.(?:tif|tiff)$', base)
    if not m:
        return None
    directory = os.path.dirname(str(path))
    for ext in ('_MTL.txt', '_MTL.TXT', '_MTL.xml', '_MTL.XML'):
        candidate = os.path.join(directory, m.group(1) + ext)
        if os.path.isfile(candidate):
            return candidate
    return None


def _landsat_scene_info(path):
    """Everything the figures and the validation report need in order to
    describe a Landsat scene honestly: WHICH spacecraft it is and WHEN it was
    acquired.

    Returns a dict with 'platform' ('Landsat 8' / 'Landsat 9' / None),
    'datetime' (a UTC datetime when the time of day could be established,
    otherwise a plain date, otherwise None), 'wrs_path', 'wrs_row' and
    'time_source' (which of the three routes below supplied the time).

    A Landsat filename carries the acquisition DATE but no time of day, so
    the time is looked for in this order:

      1. the '..._acquisition.json' sidecar download_landsat_lst() writes
         beside the band it fetches, holding the exact acquisition instant
         and the platform straight from the STAC item - authoritative, and
         needs no parsing;
      2. a sibling '..._MTL.txt'/'..._MTL.xml' (SCENE_CENTER_TIME plus
         DATE_ACQUIRED), which is what a user who pulled a full scene from
         EarthExplorer will already have on disk;
      3. the filename's own 8-digit acquisition date, with no time.

    Route 3 is why this returns a date rather than a datetime in that case:
    _has_time_of_day() downstream distinguishes the two, so a date-only match
    is never silently treated as midnight UTC - which would invent a night
    overpass and a ~10 h offset from the EnMAP acquisition that did not
    happen."""
    base = os.path.basename(str(path))
    m = _LANDSAT_FILENAME_RE.match(base)
    if not m:
        return {'platform': None, 'datetime': None, 'wrs_path': None,
                'wrs_row': None, 'time_source': None}

    info = {
        'platform': _LANDSAT_PLATFORM_NAMES.get(m.group(1).upper()),
        'wrs_path': int(m.group(3)),
        'wrs_row': int(m.group(4)),
        'datetime': None,
        'time_source': None,
    }
    try:
        acq_date = datetime.strptime(m.group(5), '%Y%m%d').date()
    except ValueError:
        acq_date = None

    # 1. sidecar written by download_landsat_lst()
    sidecar = os.path.splitext(str(path))[0] + '_acquisition.json'
    if os.path.isfile(sidecar):
        try:
            with open(sidecar, encoding='utf-8') as fh:
                meta = json.load(fh)
            ts = pd.Timestamp(meta['datetime'])
            ts = ts.tz_convert('UTC') if ts.tzinfo is not None else ts
            info['datetime'] = ts.to_pydatetime().replace(tzinfo=None)
            info['time_source'] = 'STAC sidecar'
            platform = meta.get('platform')
            if platform:
                info['platform'] = str(platform).replace('landsat-', 'Landsat ')
            return info
        except Exception as exc:
            warnings.warn(f'Could not read Landsat acquisition sidecar {sidecar}: {exc}')

    # 2. MTL metadata file
    mtl = _landsat_mtl_path(path)
    if mtl:
        try:
            with open(mtl, encoding='utf-8', errors='replace') as fh:
                text = fh.read()
            craft = _LANDSAT_MTL_CRAFT_RE.search(text)
            if craft:
                info['platform'] = craft.group(1).replace('_', ' ').title()
            t_match = _LANDSAT_MTL_TIME_RE.search(text)
            d_match = _LANDSAT_MTL_DATE_RE.search(text)
            if t_match and (d_match or acq_date):
                day = (datetime.strptime(d_match.group(1), '%Y-%m-%d').date()
                       if d_match else acq_date)
                clock = t_match.group(1).split('.')[0]
                info['datetime'] = datetime.combine(
                    day, datetime.strptime(clock, '%H:%M:%S').time())
                info['time_source'] = os.path.basename(mtl)
                return info
        except Exception as exc:
            warnings.warn(f'Could not parse Landsat MTL {mtl}: {exc}')

    # 3. acquisition date only
    info['datetime'] = acq_date
    info['time_source'] = 'filename (date only, no time of day)' if acq_date else None
    return info


_PLANETARY_COMPUTER_STAC_ITEM_URL = (
    'https://planetarycomputer.microsoft.com/api/stac/v1/collections/landsat-c2-l2/items/{id}')


def fetch_landsat_acquisition_time(path, timeout=30):
    """Recover the exact overpass time of an already-downloaded Landsat
    Collection-2 band by asking Planetary Computer for the STAC item whose
    id is embedded in the filename, and cache it in the same
    '..._acquisition.json' sidecar download_landsat_lst() writes. Returns
    the datetime on success, or None (no exception) if the file isn't a
    Landsat product, already has a time, or the lookup fails.

    This exists for the common case of a Landsat scene that reached the
    pipeline WITHOUT its metadata: a Collection-2 filename gives the
    acquisition date but no time of day, and a user who exported just the
    ST_B10 band from EarthExplorer, or downloaded it before this sidecar
    existed, has no MTL file either. Rather than label such a scene by date
    alone - which also caps the reported EnMAP-Landsat offset at
    day resolution - this fills the gap once and writes it to disk, so it
    costs one HTTP request the first time and nothing afterwards.

    Deliberately best-effort and non-fatal: the pipeline must still run with
    no network, and a date-only label is a correct label, just a coarser
    one."""
    info = _landsat_scene_info(path)
    if info['platform'] is None or isinstance(info['datetime'], datetime):
        return info['datetime'] if isinstance(info['datetime'], datetime) else None

    m = _LANDSAT_FILENAME_RE.match(os.path.basename(str(path)))
    if not m:
        return None
    # A Planetary Computer STAC item id is the USGS product identifier with
    # the PROCESSING date dropped - 'LC09_L2SP_040036_20241019_02_T1', not
    # 'LC09_L2SP_040036_20241019_20241022_02_T1' (verified directly against
    # the API: the full form returns 404). The filename on disk carries both
    # dates, so the second one has to be removed to address the item.
    item_id = '_'.join((m.group(1), m.group(2), m.group(3) + m.group(4),
                        m.group(5), m.group(7), m.group(8)))

    try:
        req = urllib.request.Request(
            _PLANETARY_COMPUTER_STAC_ITEM_URL.format(id=item_id),
            headers={'User-Agent': 'Fire_analysis_with_EnMAP'})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            item = json.load(resp)
        props = item['properties']
        platform = _LANDSAT_PLATFORM_NAMES.get(item_id[:4].upper(),
                                                str(props.get('platform', 'Landsat')))
        sidecar = os.path.splitext(str(path))[0] + '_acquisition.json'
        with open(sidecar, 'w', encoding='utf-8') as fh:
            json.dump({'scene_id': item_id,
                       'datetime': props['datetime'],
                       'platform': platform,
                       'eo:cloud_cover': props.get('eo:cloud_cover'),
                       'source': 'Microsoft Planetary Computer STAC (landsat-c2-l2), '
                                 'looked up after download'},
                      fh, indent=2)
        print(f'Recovered {platform} overpass time for {item_id}: {props["datetime"]} '
              f'(cached in {os.path.basename(sidecar)})')
        return _landsat_scene_info(path)['datetime']
    except Exception as exc:
        print(f'Could not look up the Landsat overpass time for {item_id} ({exc}); '
              'the Landsat panel will be labelled by acquisition date only.')
        return None


def _landsat_platform_label(path, default='Landsat 8/9'):
    """'Landsat 8' or 'Landsat 9' for a figure title, falling back to the
    generic 'Landsat 8/9' only when the filename doesn't identify which
    spacecraft the scene came from."""
    if not path:
        return default
    return _landsat_scene_info(path)['platform'] or default


def _landsat_scene_date(path):
    """UTC acquisition time of a Landsat Collection-2 product, used to label
    the Landsat panel in plot_multi_sensor_comparison_figure() and to report
    the EnMAP-Landsat offset in validate_temperature(). Thin wrapper over
    _landsat_scene_info(): a full datetime when the time of day could be
    recovered from the STAC sidecar or an MTL file, a plain date when only
    the filename was available, or None if the filename isn't a Landsat scene
    identifier at all. Works whether landsat_lst_path was auto-fetched or
    supplied by the user directly."""
    return _landsat_scene_info(path)['datetime']


def _earthdata_netrc_path():
    """The exact netrc path earthaccess itself reads credentials from -
    verified directly against its real source (earthaccess/auth.py's own
    netrc_path()): the NETRC environment variable if set, else '_netrc'
    (NOT '.netrc') in the home directory on Windows, '.netrc' everywhere
    else. Writing to plain '.netrc' unconditionally (an earlier version of
    this module did) silently misses on Windows - earthaccess's own netrc
    strategy would never find what was written there, only ever seeing
    "no credentials" with no hint why. Shared by
    earthdata_credentials_available() and set_earthdata_credentials() so
    both always agree with earthaccess about which file is the real one."""
    env_netrc = os.environ.get('NETRC')
    if env_netrc:
        return env_netrc
    name = '_netrc' if os.name == 'nt' else '.netrc'
    return os.path.join(os.path.expanduser('~'), name)


def earthdata_credentials_available():
    """True if NASA Earthdata credentials are already reachable via the
    EARTHDATA_USERNAME/EARTHDATA_PASSWORD environment variables or a netrc
    entry for urs.earthdata.nasa.gov at earthaccess's own expected path
    (_earthdata_netrc_path()) - the two non-interactive strategies
    download_ecostress_lst() tries. Lets a caller (the GUI, in particular)
    check before starting a run whether it needs to collect credentials
    itself, since download_ecostress_lst() will never prompt. Only checks
    that a matching entry EXISTS, not that the password in it is still
    correct - an invalid one is only revealed by an actual login attempt
    inside download_ecostress_lst()."""
    if os.environ.get('EARTHDATA_USERNAME') and os.environ.get('EARTHDATA_PASSWORD'):
        return True
    netrc_path = _earthdata_netrc_path()
    if not os.path.exists(netrc_path):
        return False
    try:
        import netrc
        return netrc.netrc(netrc_path).authenticators('urs.earthdata.nasa.gov') is not None
    except Exception:
        return False


def set_earthdata_credentials(username, password, persist=False):
    """Make NASA Earthdata credentials available to download_ecostress_lst()
    for the rest of this process, without ever going through earthaccess's
    interactive login prompt (the input()/getpass() call that hangs when run
    from a GUI's background thread with no console attached). Sets
    EARTHDATA_USERNAME/EARTHDATA_PASSWORD in this process's environment; if
    persist=True, also writes a 'machine urs.earthdata.nasa.gov' entry to
    earthaccess's own expected netrc path (_earthdata_netrc_path() - '_netrc'
    on Windows, '.netrc' elsewhere; replacing any existing entry for that
    machine) so future runs - including the CLI - pick it up automatically
    via earthaccess's own 'netrc' login strategy."""
    os.environ['EARTHDATA_USERNAME'] = username
    os.environ['EARTHDATA_PASSWORD'] = password
    if not persist:
        return

    netrc_path = _earthdata_netrc_path()
    existing = ''
    if os.path.exists(netrc_path):
        with open(netrc_path, 'r', encoding='utf-8') as f:
            existing = f.read()
    existing = re.sub(
        r'\s*machine\s+urs\.earthdata\.nasa\.gov\s+login\s+\S+\s+password\s+\S+',
        '', existing).strip()
    entry = f'machine urs.earthdata.nasa.gov\nlogin {username}\npassword {password}\n'
    with open(netrc_path, 'w', encoding='utf-8') as f:
        f.write((existing + '\n\n' if existing else '') + entry)
    try:
        os.chmod(netrc_path, 0o600)
    except OSError:
        pass


def download_ecostress_lst(scene, acquisition_datetime, out_dir, max_time_diff_hours=24,
                            widen_search_days=1, short_name='ECO_L2T_LSTE', version='002',
                            fallback_versions=('003',),
                            prefer_daytime=True, night_penalty_hours=24.0):
    """Search NASA Earthdata for the ECOSTRESS LST granule closest in time to
    `acquisition_datetime` that intersects this EnMAP scene's footprint,
    download it, and return its local path (or None if nothing was found at
    all within widen_search_days).

    The search window is +/-max_time_diff_hours, defaulting to **24 h**.

    This is deliberately strict, and it will often return nothing. ECOSTRESS
    flies on the ISS rather than a sun-synchronous orbit, so its revisit
    over any given point is irregular and frequently exceeds a day - a real
    scene checked against NASA's CMR during development had its closest
    granule ~30 h away, which this default now rejects. That is the
    intended behaviour: a granule from a different day is not validation of
    a fire that has since moved, grown or gone out, and the day/night
    mismatch it usually brings makes the background land-surface
    temperatures incomparable as well (see _describe_time_offset()). Better
    no comparison than a misleading one.

    widen_search_days likewise defaults to 1, so the fallback search is the
    same width as the primary one and is skipped entirely. Raise both only
    if you deliberately want a looser comparison, and if you do, report the
    actual time offset alongside the result - validate_temperature() prints
    it.

    This is the piece that actually reaches out to NASA's servers - it needs
    two things this pipeline can't provide on its own:
      1. `pip install earthaccess` (NASA's own Python client; not a hard
         dependency of this module, since most runs don't need it).
      2. A free NASA Earthdata Login account (register at
         https://urs.earthdata.nasa.gov), with credentials already cached
         in a .netrc file or the EARTHDATA_USERNAME/EARTHDATA_PASSWORD
         environment variables BEFORE calling this function.

    This deliberately never falls back to earthaccess's interactive login
    prompt (a plain input() call) - this function can run inside a GUI's
    background worker thread with no real console attached, where an
    interactive prompt just hangs forever waiting on stdin that can never
    arrive (confirmed the hard way). If no netrc/environment credentials
    are found, it raises a clear error instead. Set credentials up once
    from an actual terminal first:
        python -c "import earthaccess; earthaccess.login(persist=True)"
    (that one-time run is interactive and fine - it's a real terminal) -
    or set EARTHDATA_USERNAME/EARTHDATA_PASSWORD.

    short_name='ECO_L2T_LSTE' is the tiled, geolocated ECOSTRESS Collection-2
    LST product; pass a different short_name/version if you need the
    swath-gridded L2_LSTE product instead. fallback_versions is tried in
    order whenever the requested `version` returns nothing for a window -
    '002' and '003' coexist in CMR and do not cover identical date ranges,
    so a miss in one is not evidence the location was never imaged. Pass
    fallback_versions=() to search only the requested version.

    prefer_daytime (default True) makes the granule choice day/night aware
    rather than purely nearest-in-time; see the ranking comment below for
    why that matters here specifically."""
    try:
        import earthaccess
    except ImportError as exc:
        raise ImportError(
            "download_ecostress_lst() needs the 'earthaccess' package: "
            "pip install earthaccess") from exc

    # Broad except is deliberate (both a missing/unreadable netrc and a
    # rejected password surface as generic exceptions from earthaccess's
    # login() - see download_ecostress_lst's own docstring for why this
    # never falls back to an interactive prompt to sort that out) - but
    # each strategy's actual exception is kept, not discarded, so the error
    # below can distinguish "no credentials configured at all" from "the
    # cached ones were tried and rejected", instead of blaming "not found"
    # every time regardless of which it actually was.
    auth = None
    strategy_errors = {}
    for strategy in ('netrc', 'environment'):
        try:
            candidate = earthaccess.login(strategy=strategy)
        except Exception as exc:
            strategy_errors[strategy] = f'{exc.__class__.__name__}: {exc}'
            continue
        if getattr(candidate, 'authenticated', False):
            auth = candidate
            break
        strategy_errors[strategy] = 'login() returned without authenticating (no exception raised)'
    if auth is None:
        details = '\n'.join(f'  - {strategy}: {msg}' for strategy, msg in strategy_errors.items())
        raise RuntimeError(
            'Could not authenticate with NASA Earthdata via either non-interactive strategy '
            "download_ecostress_lst() tries (it never prompts interactively - see this "
            f"function's docstring for why):\n{details}\n"
            "If both say credentials are missing/unavailable, set them up via the GUI's "
            'login prompt, or run '
            '`python -c "import earthaccess; earthaccess.login(persist=True)"` once from a '
            'real terminal. If instead a strategy says the login attempt itself failed, the '
            'cached username/password is likely wrong or expired - re-enter it.')

    lat, lon = scene['lat'], scene['lon']
    bbox = (float(np.nanmin(lon)), float(np.nanmin(lat)),
            float(np.nanmax(lon)), float(np.nanmax(lat)))

    # CMR's granule timestamps (e.g. '2023-03-11T13:42:57.320Z') parse as
    # tz-AWARE (UTC) Timestamps; acquisition_datetime is normally a naive
    # datetime (from _scene_acquisition_datetime()). Subtracting a naive
    # Timestamp from an aware one raises TypeError - which _time_gap()
    # below used to swallow via its broad except, silently returning
    # Timedelta.max for every single granule (so "closest" was actually
    # arbitrary, confirmed the hard way: a real run logged a granule
    # matched as "106751 days" away, i.e. pd.Timedelta.max printed
    # verbatim). Forcing UTC here up front fixes the comparison for real
    # instead of just failing quietly.
    acquisition_ts = pd.Timestamp(acquisition_datetime)
    if acquisition_ts.tzinfo is None:
        acquisition_ts = acquisition_ts.tz_localize('UTC')

    def _time_gap(granule):
        try:
            begin = granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime']
            return abs(pd.Timestamp(begin) - acquisition_ts)
        except (KeyError, TypeError, ValueError):
            return pd.Timedelta.max

    # LP DAAC publishes ECO_L2T_LSTE under more than one collection version
    # at once ('002' and '003' both resolve in CMR). A given date can be
    # present in one and absent from the other, so an empty '002' result is
    # not evidence that ECOSTRESS never imaged the scene - it may just mean
    # that period is only carried in '003'. Try the requested version
    # first, then the fallbacks, rather than reporting "no granule" on the
    # strength of a single collection.
    searched_versions = [version] + [v for v in (fallback_versions or ()) if v != version]

    def _search(half_window):
        t0 = acquisition_ts - half_window
        t1 = acquisition_ts + half_window
        for candidate_version in searched_versions:
            found = earthaccess.search_data(
                short_name=short_name, version=candidate_version,
                bounding_box=bbox, temporal=(t0.isoformat(), t1.isoformat()))
            if found:
                if candidate_version != version:
                    print(f'No {short_name} v{version} granule in this window; '
                          f'using v{candidate_version} instead.')
                return found
        return []

    results = _search(pd.Timedelta(hours=max_time_diff_hours))
    if not results:
        # ECOSTRESS flies on the ISS, not a sun-synchronous orbit - its
        # revisit interval over any given point is irregular and can easily
        # exceed 24-48 h, so an empty result here does not mean the earlier
        # "no login"/search-parameter bugs are back; it can just as well
        # mean there genuinely is no close-in-time granule. Widen the
        # search once, sized to actually find the closest one that exists
        # (rather than silently giving up), and say clearly how far away in
        # time it is so the caller can judge whether it's still useful for
        # validation.
        # At the default widen_search_days=1 the "wide" window is no wider
        # than the primary one, so re-querying would just repeat the same
        # request. Skip it and report the miss honestly.
        if pd.Timedelta(days=widen_search_days) <= pd.Timedelta(hours=max_time_diff_hours):
            print(f'No {short_name} granule within +/-{max_time_diff_hours} h of '
                  f'{acquisition_ts.isoformat()} UTC over this scene. ECOSTRESS (ISS orbit, '
                  'irregular revisit) often has no same-day pass; the run continues without '
                  'ECOSTRESS validation rather than substituting a granule from another day.')
            return None
        wide_results = _search(pd.Timedelta(days=widen_search_days))
        if not wide_results:
            print(f'No {short_name} granule found within {widen_search_days} days of '
                  f'{acquisition_ts.isoformat()} UTC over this scene - ECOSTRESS (ISS orbit, '
                  'irregular revisit) may simply not have imaged this location/date at all.')
            return None
        nearest_gap = min(_time_gap(g) for g in wide_results)
        print(f'No {short_name} granule within {max_time_diff_hours} h; using the closest '
              f'one available instead ({nearest_gap} away, from a {widen_search_days}-day '
              'search) - treat the comparison accordingly.')
        results = wide_results

    # Ranking deliberately is NOT purely "closest in time". ECOSTRESS's ISS
    # orbit drifts through local solar time, so the nearest granule to a
    # ~11:00-local EnMAP overpass is very often a pre-dawn one - and a
    # night land-surface temperature is not comparable to a daytime scene
    # even over unburnt ground, let alone as fire validation. (Observed on
    # a real run: the closest granule to a 2023-03-14 05:01 UTC EnMAP scene
    # over Myanmar was 30 h away at 04:15 local - a night overpass whose
    # peak LST was 299.5 K, i.e. carrying no fire signal at all.)
    #
    # So a daytime granule wins over a night one until it is
    # night_penalty_hours further away in time, turning "closest" into
    # "closest that is actually comparable". Set prefer_daytime=False to
    # get the old pure-time-gap behaviour (night overpasses are genuinely
    # useful for detecting thermal anomalies against a cold background, so
    # this is a preference, not a prohibition).
    scene_centre_lon = float(0.5 * (bbox[0] + bbox[2]))
    enmap_solar = _local_solar_hour(acquisition_ts.to_pydatetime(), scene_centre_lon)
    enmap_is_day = enmap_solar is None or 6.0 <= enmap_solar <= 18.0

    def _granule_begin(granule):
        try:
            return pd.Timestamp(granule['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        except (KeyError, TypeError, ValueError):
            return None

    def _is_daytime(granule):
        begin = _granule_begin(granule)
        solar = _local_solar_hour(begin.to_pydatetime(), scene_centre_lon) if begin is not None else None
        return None if solar is None else (6.0 <= solar <= 18.0)

    def _rank(granule):
        gap = _time_gap(granule)
        penalty = pd.Timedelta(0)
        if prefer_daytime:
            day = _is_daytime(granule)
            if day is not None and day != enmap_is_day:
                penalty = pd.Timedelta(hours=night_penalty_hours)
        return gap + penalty

    results = sorted(results, key=_rank)
    best_gap = _time_gap(results[0])
    try:
        best_id = results[0]['meta']['native-id']
    except (KeyError, TypeError):
        best_id = '?'
    best_is_day = _is_daytime(results[0])
    phase = '' if best_is_day is None else (' daytime' if best_is_day else ' NIGHT')
    print(f'Matched ECOSTRESS granule: {best_id} ({best_gap} from the EnMAP acquisition time,'
          f'{phase or " unknown-phase"} overpass)')
    if prefer_daytime and best_is_day is not None and best_is_day != enmap_is_day:
        same_phase = [g for g in results if _is_daytime(g) == enmap_is_day]
        if same_phase:
            print(f'  ! Chose an opposite-phase granule anyway: the nearest same-phase one is '
                  f'{min(_time_gap(g) for g in same_phase)} away, further than the '
                  f'{night_penalty_hours:g} h night_penalty_hours allows. Raise that argument to '
                  'prefer it, at the cost of a longer time gap.')
        else:
            print('  ! No same-phase granule was available at all - this comparison spans day '
                  'and night, so background LST differences will dominate any fire signal.')
    elif prefer_daytime and best_is_day is not None:
        closest_gap = min(_time_gap(g) for g in results)
        if closest_gap < best_gap:
            print(f'  (a granule {closest_gap} away existed but was the opposite day/night '
                  'phase, so this slightly more distant same-phase one was preferred)')

    os.makedirs(out_dir, exist_ok=True)
    downloaded = earthaccess.download(results[:1], out_dir)
    if not downloaded:
        return None

    # A single ECO_L2T_LSTE granule downloads as several separate band files
    # (LST, LST_err, QC, cloud, water, height, view_zenith, ...), not one -
    # pick the LST band specifically rather than assuming downloaded[0] is
    # it. '_LST.tif' (not '_LST_err.tif') is LP DAAC's own naming
    # convention for this product. earthaccess.download() returns plain
    # strings on some versions and pathlib.Path objects on others (a 1.0+
    # install raised 'expected string or bytes-like object, got
    # WindowsPath' here before this str() conversion was added) - always
    # normalize to str before regex-matching.
    lst_files = [str(f) for f in downloaded if re.search(r'_LST\.tif$', str(f), re.IGNORECASE)]
    if not lst_files:
        warnings.warn(f'Downloaded {len(downloaded)} file(s) for the matched ECOSTRESS '
                       f"granule but none looked like the LST band (expected a "
                       f"'..._LST.tif' file): {downloaded}")
        return None
    return lst_files[0]


_PLANETARY_COMPUTER_STAC_SEARCH_URL = 'https://planetarycomputer.microsoft.com/api/stac/v1/search'
_PLANETARY_COMPUTER_SAS_TOKEN_URL = 'https://planetarycomputer.microsoft.com/api/sas/v1/token'


def _planetary_computer_sign(url):
    """Append a short-lived, free, anonymous read token to a Microsoft
    Planetary Computer blob URL, via its public SAS-token endpoint - no
    account or API key needed at all (unlike download_ecostress_lst()'s NASA
    Earthdata Login). Verified directly against the real service: the raw
    blob URL alone 403s ('Public access is not permitted on this storage
    account'), but a plain unauthenticated GET of
    {_PLANETARY_COMPUTER_SAS_TOKEN_URL}/<storage-account>/<container>
    (both parsed straight out of url's own hostname/path) returns a token
    query string that makes that exact same URL downloadable."""
    parsed = urlparse(url)
    account = parsed.netloc.split('.')[0]
    container = parsed.path.lstrip('/').split('/', 1)[0]
    token_url = f'{_PLANETARY_COMPUTER_SAS_TOKEN_URL}/{account}/{container}'
    req = urllib.request.Request(token_url, headers={'User-Agent': 'Fire_analysis_with_EnMAP'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        token = json.load(resp)['token']
    sep = '&' if '?' in url else '?'
    return url + sep + token


def download_landsat_lst(scene, acquisition_datetime, out_dir, max_time_diff_hours=24,
                          max_cloud_cover=10.0, widen_search_days=1):
    """Search Microsoft's Planetary Computer STAC catalog for the closest-
    in-time Landsat 8/9 Collection-2 Level-2 scene, AT OR UNDER
    max_cloud_cover percent cloud cover, that intersects this EnMAP scene's
    footprint, download its ST_B10 (surface temperature) band, and return
    its local path - ready to hand straight to load_landsat_lst(), which
    already expects exactly this file. Returns None if no scene meeting the
    cloud-cover requirement exists within widen_search_days, however far
    from acquisition_datetime - a hazy/smoke-obscured scene is useless for
    a temperature comparison regardless of how close in time it is, so this
    never falls back to returning one just because nothing clearer turned
    up (unlike download_ecostress_lst()'s time-window widening, which has
    no equivalent quality gate to enforce).

    Real Landsat Collection-2 archives are NOT actually granule-searchable
    through NASA's own CMR/earthaccess despite being listed there at the
    collection level - confirmed directly against CMR: a granule search for
    either the Surface Reflectance or Surface Temperature Collection-2
    collection returns zero hits worldwide for any date, because USGS never
    populated per-scene granules there. Planetary Computer is a genuine,
    queryable, and (via _planetary_computer_sign(), also verified directly)
    fully public mirror of the same USGS archive instead, reached here with
    plain HTTP requests - no pip package or account of any kind needed,
    unlike ECOSTRESS's NASA Earthdata Login requirement.

    Landsat 7 (ETM+, with its scan-line-corrector-off gaps) is deliberately
    excluded from candidates; only Landsat 8/9 (OLI/TIRS) scenes are
    considered. max_time_diff_hours defaults to 24 h - same day as the
    EnMAP acquisition - and widen_search_days now defaults to 1, so the
    fallback window is no wider than the primary one and is skipped
    entirely.

    That is a deliberate tightening. The previous default widened to
    +/-16 days (Landsat 8+9's combined revisit), which reliably found a
    scene but not a useful one: an active fire moves, grows or goes out
    within hours, so a Landsat scene two weeks either side validates
    nothing about the fire EnMAP saw, however clear it is. With Landsat 8
    and 9 combined the repeat is 8 days, so most EnMAP dates simply have no
    same-day overpass and this will return None - which is the correct
    answer, not a failure. Raise widen_search_days deliberately if you want
    a looser comparison, and report the actual offset when you do
    (validate_temperature() prints it, and flags multi-day gaps)."""
    lat, lon = scene['lat'], scene['lon']
    bbox = [float(np.nanmin(lon)), float(np.nanmin(lat)),
            float(np.nanmax(lon)), float(np.nanmax(lat))]

    acquisition_ts = pd.Timestamp(acquisition_datetime)
    if acquisition_ts.tzinfo is None:
        acquisition_ts = acquisition_ts.tz_localize('UTC')

    def _gap_hours(feature):
        dt = pd.Timestamp(feature['properties']['datetime'])
        return abs((dt - acquisition_ts).total_seconds()) / 3600.0

    def _search(half_window):
        t0 = acquisition_ts - half_window
        t1 = acquisition_ts + half_window
        # Planetary Computer's STAC search (unlike CMR/earthaccess in
        # download_ecostress_lst()) strictly rejects a timezone-naive
        # datetime string (confirmed directly: '...T05:01:52' with no
        # offset 400s with "Input should have timezone info") -
        # acquisition_ts was forced to UTC above so isoformat() always
        # emits one.
        payload = {
            'collections': ['landsat-c2-l2'],
            'bbox': bbox,
            'datetime': f'{t0.isoformat()}/{t1.isoformat()}',
            'limit': 100,
        }
        req = urllib.request.Request(
            _PLANETARY_COMPUTER_STAC_SEARCH_URL, data=json.dumps(payload).encode('utf-8'),
            headers={'Content-Type': 'application/json', 'User-Agent': 'Fire_analysis_with_EnMAP'})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.load(resp)
        except Exception as exc:
            raise RuntimeError(f'Landsat search via Planetary Computer failed: {exc}') from exc
        return [f for f in result.get('features', [])
                if f.get('properties', {}).get('platform', '') in ('landsat-8', 'landsat-9')]

    features = _search(pd.Timedelta(hours=max_time_diff_hours))
    clear = [f for f in features if f['properties'].get('eo:cloud_cover', 100.0) <= max_cloud_cover]

    if not clear:
        # At the default widen_search_days=1 the "wide" window is no wider
        # than the primary one, so widening would repeat the same query.
        # Report the miss instead of substituting a scene from another day.
        if pd.Timedelta(days=widen_search_days) <= pd.Timedelta(hours=max_time_diff_hours):
            if features:
                best_cloud = min(f['properties'].get('eo:cloud_cover', 100.0) for f in features)
                print(f'{len(features)} Landsat 8/9 scene(s) within +/-{max_time_diff_hours} h, but '
                      f'all exceeded {max_cloud_cover:.0f}% cloud cover (clearest '
                      f'{best_cloud:.0f}%); skipping Landsat validation.')
            else:
                print(f'No Landsat 8/9 scene within +/-{max_time_diff_hours} h of '
                      f'{acquisition_ts.isoformat()} UTC over this scene. With Landsat 8 and 9 '
                      'combined the repeat is 8 days, so most dates have no same-day overpass; '
                      'the run continues without Landsat validation.')
            return None
        if features:
            best_cloud = min(f['properties'].get('eo:cloud_cover', 100.0) for f in features)
            print(f'{len(features)} Landsat 8/9 scene(s) found within {max_time_diff_hours} h, but '
                  f'all exceeded {max_cloud_cover:.0f}% cloud cover (clearest was {best_cloud:.0f}%); '
                  f'widening search to +/-{widen_search_days} days for a clearer one...')
        wide_features = _search(pd.Timedelta(days=widen_search_days))
        clear = [f for f in wide_features if f['properties'].get('eo:cloud_cover', 100.0) <= max_cloud_cover]
        if not clear:
            if wide_features:
                best_cloud = min(f['properties'].get('eo:cloud_cover', 100.0) for f in wide_features)
                print(f'No Landsat 8/9 scene at or under {max_cloud_cover:.0f}% cloud cover found '
                      f'within {widen_search_days} days either (clearest was {best_cloud:.0f}%); '
                      'skipping Landsat validation rather than using a cloud/haze-obscured scene.')
            else:
                print(f'No Landsat 8/9 scene found within {widen_search_days} days of '
                      f'{acquisition_ts.isoformat()} UTC over this scene at all.')
            return None

    clear.sort(key=_gap_hours)
    best = clear[0]
    props = best['properties']
    cloud = props.get('eo:cloud_cover', float('nan'))
    platform = _LANDSAT_PLATFORM_NAMES.get(best['id'][:4].upper(),
                                            str(props.get('platform', 'Landsat')))
    print(f"Matched {platform} scene: {best['id']} ({props['datetime']}, "
          f'{_gap_hours(best):.1f} h from the EnMAP acquisition time, {cloud:.0f}% cloud cover)')

    asset = best['assets'].get('lwir11')
    if asset is None:
        warnings.warn(f"Matched Landsat scene {best['id']} has no surface-temperature "
                       "('lwir11'/ST_B10) asset; skipping.")
        return None

    signed_url = _planetary_computer_sign(asset['href'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.basename(asset['href']))
    dl_req = urllib.request.Request(signed_url, headers={'User-Agent': 'Fire_analysis_with_EnMAP'})
    with urllib.request.urlopen(dl_req, timeout=180) as resp, open(out_path, 'wb') as f:
        shutil.copyfileobj(resp, f)

    # Persist the acquisition instant and the spacecraft next to the band.
    # A Landsat filename carries only the acquisition DATE, so without this
    # the figures could name the day but not the overpass time, and the
    # EnMAP-Landsat offset could only ever be reported to the nearest day.
    # The STAC item already holds both exactly, so write them out rather
    # than re-deriving them later - _landsat_scene_info() reads this file
    # first, which also keeps every later run offline.
    sidecar = os.path.splitext(out_path)[0] + '_acquisition.json'
    try:
        with open(sidecar, 'w', encoding='utf-8') as fh:
            json.dump({'scene_id': best['id'],
                       'datetime': props['datetime'],
                       'platform': platform,
                       'eo:cloud_cover': cloud,
                       'gap_hours_from_enmap': round(_gap_hours(best), 4),
                       'source': 'Microsoft Planetary Computer STAC (landsat-c2-l2)'},
                      fh, indent=2)
    except OSError as exc:
        warnings.warn(f'Could not write Landsat acquisition sidecar {sidecar}: {exc}. '
                      'The Landsat panel will be labelled by date only.')
    return out_path


def _sample_raster_at_lonlat(raster, geotransform, projection_wkt, lons, lats):
    """Nearest-pixel sample of `raster` at WGS84 (lons, lats)."""
    from osgeo import osr
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(projection_wkt)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(src_srs, dst_srs)
    pts = transform.TransformPoints(list(zip(np.atleast_1d(lons), np.atleast_1d(lats))))

    gt = geotransform
    det = gt[1] * gt[5] - gt[2] * gt[4]
    if abs(det) < 1e-30:
        raise ValueError('degenerate geotransform %r - every sample would be '
                         'mis-placed rather than merely imprecise' % (geotransform,))
    values = np.full(len(pts), np.nan)
    for i, (px, py, _) in enumerate(pts):
        col = ((px - gt[0]) * gt[5] - (py - gt[3]) * gt[2]) / det
        row = ((py - gt[3]) * gt[1] - (px - gt[0]) * gt[4]) / det
        col_i, row_i = int(round(col)), int(round(row))
        if 0 <= row_i < raster.shape[0] and 0 <= col_i < raster.shape[1]:
            values[i] = raster[row_i, col_i]
    return values


def _resample_raster_to_lonlat(raster, geotransform, projection_wkt, lon, lat):
    """Vectorized nearest-pixel resample of `raster` (in its own CRS/
    geotransform) onto an arbitrary WGS84 lon/lat grid - used to overlay a
    lower-resolution reference raster (e.g. ECOSTRESS at ~70 m) on an
    EnMAP scene's own lon/lat mesh for a directly comparable side-by-side
    plot. Same per-point math as _sample_raster_at_lonlat(), vectorized over
    a full 2D grid instead of a point list."""
    from osgeo import osr
    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromWkt(projection_wkt)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    shape = lon.shape
    pts = transform.TransformPoints(list(zip(lon.ravel(), lat.ravel())))
    px = np.array([p[0] for p in pts])
    py = np.array([p[1] for p in pts])

    gt = geotransform
    det = gt[1] * gt[5] - gt[2] * gt[4]
    if abs(det) < 1e-30:
        raise ValueError('degenerate geotransform %r - every sample would be '
                         'mis-placed rather than merely imprecise' % (geotransform,))
    col_i = np.round(((px - gt[0]) * gt[5] - (py - gt[3]) * gt[2]) / det).astype(int)
    row_i = np.round(((py - gt[3]) * gt[1] - (px - gt[0]) * gt[4]) / det).astype(int)

    out = np.full(px.shape, np.nan)
    in_bounds = (row_i >= 0) & (row_i < raster.shape[0]) & (col_i >= 0) & (col_i < raster.shape[1])
    out[in_bounds] = raster[row_i[in_bounds], col_i[in_bounds]]
    return out.reshape(shape)


def attach_pixel_geolocation(results_df, scene):
    """Attach WGS84 lat/lon and UTM easting/northing/zone for every row's
    pixel location (from its ind_x, ind_y against scene['lat']/scene['lon']).

    The UTM zone is chosen once from the scene's own center coordinate - a
    single EnMAP scene (~30x30 km) never spans more than one UTM zone - and
    used for every row, so eastings/northings stay directly comparable to
    each other rather than each pixel silently reprojecting into whichever
    zone it happens to fall in right at a boundary."""
    from osgeo import osr
    lat_grid, lon_grid = scene['lat'], scene['lon']

    out = results_df.copy()
    out['lat_wgs84'] = lat_grid[out['ind_x'].values, out['ind_y'].values]
    out['lon_wgs84'] = lon_grid[out['ind_x'].values, out['ind_y'].values]
    # After co-registration (coregistration.py) lat/lon are the corrected
    # positions; the product's own L1B positions are kept beside them.
    if 'lat_nominal' in scene:
        out['lat_wgs84_nominal'] = scene['lat_nominal'][out['ind_x'].values, out['ind_y'].values]
        out['lon_wgs84_nominal'] = scene['lon_nominal'][out['ind_x'].values, out['ind_y'].values]

    center_lat = float(np.nanmean(lat_grid))
    center_lon = float(np.nanmean(lon_grid))
    zone = int((center_lon + 180) // 6) + 1
    zone = min(max(zone, 1), 60)
    hemisphere = 'N' if center_lat >= 0 else 'S'
    utm_epsg = (32600 if center_lat >= 0 else 32700) + zone  # WGS84/UTM EPSG convention

    src_srs = osr.SpatialReference()
    src_srs.ImportFromEPSG(4326)
    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(utm_epsg)
    src_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    pts = transform.TransformPoints(list(zip(out['lon_wgs84'].values, out['lat_wgs84'].values)))
    out['utm_easting_m'] = [float(p[0]) for p in pts]
    out['utm_northing_m'] = [float(p[1]) for p in pts]
    out['utm_zone'] = f'{zone}{hemisphere}'

    return out


def _has_time_of_day(value):
    """True when `value` carries a real time-of-day. datetime is a subclass
    of date, so the isinstance order matters: _landsat_scene_date() can
    only return a plain date (Landsat's filename encodes no time), and
    treating that as midnight would both crash _local_solar_hour() and
    fabricate a 'night overpass' warning for a scene actually acquired at
    ~10:30 local."""
    return isinstance(value, datetime) or isinstance(value, pd.Timestamp)


def _local_solar_hour(dt_utc, lon_deg):
    """Approximate local solar time-of-day (hours, 0-24) at longitude
    lon_deg for a UTC datetime - just the UTC hour shifted by 15 deg per
    hour. Good to a few minutes, which is far finer than the day/night
    question it is used for in _describe_time_offset(). Returns None when
    the time-of-day isn't actually known."""
    if dt_utc is None or lon_deg is None or not np.isfinite(lon_deg):
        return None
    if not _has_time_of_day(dt_utc):
        return None
    # Arithmetic shared with thermal_partners.local_solar_hour() via
    # solar_time.py; the guards above are what distinguish this variant.
    return solar_hour_from_utc(hours_of_day(pd.Timestamp(dt_utc)), lon_deg)


def _describe_time_offset(label, enmap_dt, ref_dt, scene_lon):
    """One-line summary of how far a reference acquisition sits from the
    EnMAP one, with the warnings that gap deserves.

    Worth stating explicitly in the run log because both failure modes are
    invisible in the numbers alone. A multi-day gap means the fire has
    moved, grown or gone out between the two acquisitions, so a
    pixel-by-pixel bias/RMSE is comparing two different fires. And a
    day-vs-night mismatch is worse than a mere time gap: ECOSTRESS flies on
    the ISS and images a given point at a drifting local time, so its
    'closest' granule is often a pre-dawn overpass whose land-surface
    temperature is the nocturnal minimum - tens of Kelvin below the
    daytime EnMAP scene even over unburnt ground, before any fire signal is
    considered."""
    if enmap_dt is None or ref_dt is None:
        return f'  {label}: acquisition time unknown - time offset not checked'
    gap_h = (pd.Timestamp(ref_dt) - pd.Timestamp(enmap_dt)).total_seconds() / 3600.0
    # A date-only reference (Landsat) pins the gap no finer than the day it
    # was acquired, so quote it in days and don't imply spurious precision.
    if _has_time_of_day(ref_dt):
        line = f'  {label}: {gap_h:+.2f} h from the EnMAP acquisition'
    else:
        line = (f'  {label}: acquired {pd.Timestamp(ref_dt).date()} '
                f'({gap_h / 24.0:+.1f} days; filename carries no time of day)')
    notes = []
    if abs(gap_h) > 24.0:
        notes.append(f'{abs(gap_h) / 24.0:.1f} days apart - the fire itself will have changed; '
                     'treat per-pixel agreement as indicative only')
    elif abs(gap_h) > 3.0 and _has_time_of_day(ref_dt):
        notes.append('several hours apart - diurnal heating alone shifts background LST')
    enmap_solar = _local_solar_hour(enmap_dt, scene_lon)
    ref_solar = _local_solar_hour(ref_dt, scene_lon)
    if enmap_solar is not None and ref_solar is not None:
        line += f' (local solar {ref_solar:04.1f} h vs EnMAP {enmap_solar:04.1f} h)'
        ref_night = not (6.0 <= ref_solar <= 18.0)
        enmap_night = not (6.0 <= enmap_solar <= 18.0)
        if ref_night != enmap_night:
            notes.append('DAY/NIGHT MISMATCH - one acquisition is a night overpass; '
                         'background LST is not comparable')
    for note in notes:
        line += f'\n      ! {note}'
    return line


def _reference_stats(label, model_K, ref_K, ceiling_K, comparable_max_K):
    """Print bias/RMSE of the retrieved temperature against a reference
    LST, plus the two caveats that decide whether those numbers mean
    anything.

    Both Landsat's and ECOSTRESS's thermal channels saturate far below a
    flame core, so a reference pixel sitting at the band ceiling is
    reporting its own limit, not the ground - counted and reported
    separately rather than quietly averaged in. And because the retrieval
    deliberately targets sub-pixel fire, a pixel fit at 1500 K cannot
    agree with a 60-90 m sensor averaging that fire over its whole
    footprint; the comparable subset (retrieved below comparable_max_K,
    i.e. cooler/smouldering and mixed pixels) is where agreement is
    actually expected, so it is reported on its own line."""
    finite = np.isfinite(model_K) & np.isfinite(ref_K)
    n = int(finite.sum())
    if n == 0:
        print(f'[{label}] no overlapping valid pixels')
        return
    model, ref = model_K[finite], ref_K[finite]
    bias = float(np.mean(model - ref))
    rmse = float(np.sqrt(np.mean((model - ref) ** 2)))
    print(f'[{label}] n={n}  bias={bias:.1f} K  RMSE={rmse:.1f} K'
          f'  (reference {np.min(ref):.1f}-{np.max(ref):.1f} K)')

    at_ceiling = int(np.sum(ref >= ceiling_K - 1.0))
    if at_ceiling:
        print(f'  {at_ceiling}/{n} reference pixels at/above the ~{ceiling_K:.0f} K band '
              'ceiling - saturated, not a usable comparison')

    sub = model < comparable_max_K
    if sub.any() and int(sub.sum()) != n:
        b = float(np.mean(model[sub] - ref[sub]))
        r = float(np.sqrt(np.mean((model[sub] - ref[sub]) ** 2)))
        print(f'  comparable subset (retrieved < {comparable_max_K:.0f} K): '
              f'n={int(sub.sum())}  bias={b:.1f} K  RMSE={r:.1f} K')
    elif not sub.any():
        print(f'  every retrieved pixel is >= {comparable_max_K:.0f} K - all far above what a '
              '60-90 m thermal sensor can resolve, so no comparable subset exists')


def validate_temperature(results_df, landsat=None, field_df=None, ecostress=None,
                          field_max_dist_deg=0.01, enmap_dt=None, landsat_dt=None,
                          ecostress_dt=None, comparable_max_K=600.0,
                          landsat_label='Landsat', extra_references=None):
    """Attach Landsat LST, ECOSTRESS LST, and/or nearest-field-measurement
    comparisons to results_df, which must already carry 'lat_wgs84'/
    'lon_wgs84' (from attach_pixel_geolocation()) plus 'mean_temp_K'.
    Returns the augmented DataFrame; prints bias/RMSE against each
    reference where available. Everything here is Kelvin: Landsat's own
    USGS Collection-2 formula and ECOSTRESS's resolved scaling (see
    load_landsat_lst()/load_ecostress_lst()) both output Kelvin, and
    field_df['temperature_K'] is the documented input contract for
    load_field_data() - no conversion needed. Both Landsat's and
    ECOSTRESS's thermal channels saturate well below typical active fire
    temperatures, so agreement there is only meaningful for background/
    smouldering pixels, not for the hottest fire pixels - this is a sanity
    check, not a ground truth for the fire core. _reference_stats() makes
    that split explicit instead of leaving it to the docstring.

    enmap_dt/landsat_dt/ecostress_dt, if given, are the three acquisition
    times; the offsets between them are reported up front, with warnings
    for multi-day gaps and for day/night mismatches (see
    _describe_time_offset()). Passing them is strongly recommended - a
    validation number computed against a granule from a different day, or
    from a pre-dawn overpass, is not the number it appears to be.

    landsat_label names the spacecraft in the printed report ('Landsat 8' or
    'Landsat 9' - see _landsat_platform_label()). The two carry the same
    OLI/TIRS design but are separate instruments with their own calibration
    and their own overpass time, so a bias quoted against an unnamed
    'Landsat' is not reproducible.

    extra_references extends the same treatment to any further thermal
    sensor without this function needing to know about it - it is how the
    VIIRS and Sentinel-3 SLSTR partners from thermal_partners.py reach the
    report. Pass a sequence of dicts:

        {'label': 'VIIRS-LST (Suomi-NPP)',      # names the printed row
         'data': {...},                          # a loader's dict, needs
                                                 # 'lst_K'/'geotransform'/
                                                 # 'projection_wkt'
         'datetime': datetime | None,            # for the time-offset line
         'ceiling_K': 380.0,                     # band/encoding saturation
         'column': 'viirs_LST_K'}                # optional column name

    Each one gets its own sampled column, its own bias/RMSE line through
    _reference_stats(), and its own saturation caveat. `ceiling_K` matters
    and differs between sensors - SLSTR's int16 packing tops out at 355.5 K,
    BELOW ECOSTRESS's ~380 K - so it is taken per reference rather than
    assumed. A reference whose 'data' is None is skipped silently, which is
    what lets a caller pass a fixed-length list of optional partners."""
    out = results_df.copy()
    scene_lon = float(np.nanmean(out['lon_wgs84'].values)) if 'lon_wgs84' in out else None

    extras = [r for r in (extra_references or []) if r and r.get('data') is not None]

    if (landsat is not None and landsat_dt is not None) or \
            (ecostress is not None and ecostress_dt is not None) or \
            any(r.get('datetime') is not None for r in extras):
        print('Acquisition-time context:')
        if landsat is not None:
            print(_describe_time_offset(landsat_label, enmap_dt, landsat_dt, scene_lon))
        if ecostress is not None:
            print(_describe_time_offset('ECOSTRESS', enmap_dt, ecostress_dt, scene_lon))
        for ref in extras:
            if ref.get('datetime') is not None:
                print(_describe_time_offset(ref.get('label', 'extra reference'),
                                            enmap_dt, ref['datetime'], scene_lon))

    if landsat is not None:
        out['landsat_LST_K'] = _sample_raster_at_lonlat(
            landsat['lst_K'], landsat['geotransform'], landsat['projection_wkt'],
            out['lon_wgs84'].values, out['lat_wgs84'].values)
        _reference_stats(f'{landsat_label} LST', out['mean_temp_K'].values,
                         out['landsat_LST_K'].values, _LANDSAT_ST_CEILING_K, comparable_max_K)

    if ecostress is not None:
        out['ecostress_LST_K'] = _sample_raster_at_lonlat(
            ecostress['lst_K'], ecostress['geotransform'], ecostress['projection_wkt'],
            out['lon_wgs84'].values, out['lat_wgs84'].values)
        _reference_stats('ECOSTRESS LST', out['mean_temp_K'].values,
                         out['ecostress_LST_K'].values, _ECOSTRESS_CEILING_K, comparable_max_K)

    for ref in extras:
        label = ref.get('label', 'extra reference')
        column = ref.get('column') or ('%s_LST_K' % re.sub(r'[^A-Za-z0-9]+', '_',
                                                           label).strip('_').lower())
        data = ref['data']
        out[column] = _sample_raster_at_lonlat(
            data['lst_K'], data['geotransform'], data['projection_wkt'],
            out['lon_wgs84'].values, out['lat_wgs84'].values)
        _reference_stats(label, out['mean_temp_K'].values, out[column].values,
                         float(ref.get('ceiling_K', 380.0)), comparable_max_K)

    # Co-registered run: the same references sampled at the L1B positions
    # too, so the effect of the co-registration on the comparison is visible.
    if 'lon_wgs84_nominal' in out:
        lo_n, la_n = out['lon_wgs84_nominal'].values, out['lat_wgs84_nominal'].values
        nominal = [('landsat_LST_K', landsat), ('ecostress_LST_K', ecostress)]
        nominal += [(r.get('column') or ('%s_LST_K' % re.sub(r'[^A-Za-z0-9]+', '_', r.get('label', 'extra reference'))
                                         .strip('_').lower()), r['data']) for r in extras]
        for column, data in nominal:
            if data is not None and column in out:
                out[column + '_nominal'] = _sample_raster_at_lonlat(
                    data['lst_K'], data['geotransform'], data['projection_wkt'], lo_n, la_n)

    if field_df is not None and len(field_df) == 0:
        warnings.warn('field_df has no rows; skipping the field-measurement comparison.')
    elif field_df is not None:
        matched_temp = []
        for lo, la in zip(out['lon_wgs84'].values, out['lat_wgs84'].values):
            # Scale the longitude difference by cos(latitude) so this is an
            # honest distance-on-the-ground comparison, not raw degrees -
            # at 45 deg latitude a degree of longitude is ~30% shorter than
            # a degree of latitude, and the gap widens toward the poles.
            dlon = (field_df['lon'].values - lo) * math.cos(math.radians(la))
            dlat = field_df['lat'].values - la
            d2 = dlon ** 2 + dlat ** 2
            j = int(np.argmin(d2))
            matched_temp.append(field_df['temperature_K'].values[j] if math.sqrt(d2[j]) <= field_max_dist_deg else np.nan)
        out['field_temperature_K'] = matched_temp
        valid = out[['mean_temp_K', 'field_temperature_K']].dropna()
        if len(valid) > 0:
            bias = np.mean(valid['mean_temp_K'] - valid['field_temperature_K'])
            rmse = np.sqrt(np.mean((valid['mean_temp_K'] - valid['field_temperature_K']) ** 2))
            print(f'[Field data]  n={len(valid)}  bias={bias:.1f} K  RMSE={rmse:.1f} K')

    return out


def geoTIFF_creation(numRows, numCols, numChannels, channels, geotransform,
                      projection_wkt, dtype, name, unit_type=None):
    """Write a results raster using the source EnMAP scene's own GeoTransform
    and projection directly - unlike the PRISMA script's version, no
    rotation-angle estimation via geopandas reprojection is needed, because
    an EnMAP L1C/L2A product is already a properly georeferenced raster.

    unit_type, if given (e.g. 'degC'), is written to each band's standard
    GDAL unit-type tag so GIS software (QGIS, ArcGIS, ...) shows the
    correct units without the user having to guess or check documentation."""
    from osgeo import gdal
    gdal_dtype = gdal.GDT_UInt32 if dtype == 'uint32' else gdal.GDT_Float32
    dst_ds = gdal.GetDriverByName('GTiff').Create(name, numCols, numRows, numChannels, gdal_dtype)
    dst_ds.SetGeoTransform(geotransform)
    dst_ds.SetProjection(projection_wkt)
    for i in range(numChannels):
        band_data = channels[i].astype('uint32' if dtype == 'uint32' else 'float32')
        band = dst_ds.GetRasterBand(i + 1)
        band.WriteArray(band_data)
        if unit_type is not None:
            band.SetUnitType(unit_type)
    dst_ds.FlushCache()
    dst_ds = None


#%% ###########################################################################
#            Publication-quality figures (styled after Amici et al. 2022,
#            "Wildfires Temperature Estimation by Complementary Use of
#            Hyperspectral PRISMA and Thermal (ECOSTRESS & L8)", JGR
#            Biogeosciences, Figures 3, 5/6 and 8)
################################################################################

# The PRISMA paper's Figure 3 compares Fire/Vegetation/Smoke/Burned-fresh
# spectra, drawn from manually labeled ROIs. This pipeline has no such
# labels - only "fire" (from detect_fire_pixels) vs. "background" (the
# candidate pool main_temp_estimation() actually samples from) directly.
# _select_surface_feature_spectra() below adds an unsupervised analogue of
# the paper's other classes (Vegetation/Water/Bare ground/Smoke), picked by
# simple spectral-index heuristics from the same background pool, since the
# pipeline has no classifier to draw real labeled ROIs from.

_PUBLICATION_RC = {
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'axes.grid': False,
}


def _format_acquisition_dt(dt):
    """'2023-03-14 05:01 UTC' from a datetime, or '2023-03-14' from a plain
    date, or '' if dt is None - shared date-label formatting for the
    map-based figures below, each of which can show data from a different
    sensor's own overpass time (EnMAP vs. a Landsat scene found 13 days
    away, say), so every map panel gets its own date/time label rather than
    leaving the reader to guess whether "the date" on one panel applies to
    the others too."""
    if dt is None:
        return ''
    if isinstance(dt, datetime):
        return dt.strftime('%Y-%m-%d %H:%M UTC')
    return dt.strftime('%Y-%m-%d')


def _lonlat_formatter(is_lat):
    """Tick formatter rendering e.g. 42.678 as '42.678°N' / -121.03 as
    '121.030°W', matching the axis style of Figures 7-9 in the paper."""
    hemisphere = ('N', 'S') if is_lat else ('E', 'W')

    def fmt(value, _pos):
        hemi = hemisphere[0] if value >= 0 else hemisphere[1]
        return f'{abs(value):.3f}°{hemi}'

    return FuncFormatter(fmt)


def _pick_rgb_band_targets(wavelength, target_rgb_nm=(660.0, 560.0, 480.0)):
    """Best-available R/G/B target wavelengths for a color composite/preview
    of this scene: true color (660/560/480 nm, the target_rgb_nm default)
    when the scene's own wavelength range covers all three; for a SWIR-only
    detector file (no VNIR bands present), a true-color request would be
    meaningless, so this falls back to three bands spread evenly across
    whatever range the scene actually covers. Shared by
    _build_rgb_composite() (the in-app preview/figures) and
    _pick_rgb_band_indices() (the 'default bands' hint written into a saved
    ENVI file, so a GIS viewer like QGIS renders the same bands as RGB
    automatically instead of defaulting to the first three bands)."""
    wl_min, wl_max = float(np.nanmin(wavelength)), float(np.nanmax(wavelength))
    if wl_min <= target_rgb_nm[2] and wl_max >= target_rgb_nm[0]:
        return target_rgb_nm
    return tuple(wl_min + f * (wl_max - wl_min) for f in (0.85, 0.5, 0.15))


def _pick_rgb_band_indices(wavelength, target_rgb_nm=(660.0, 560.0, 480.0)):
    """1-based band indices nearest each of _pick_rgb_band_targets()'s
    target wavelengths - the form ENVI's 'default bands' header field (and
    similar GIS-viewer RGB hints) expect."""
    targets = _pick_rgb_band_targets(wavelength, target_rgb_nm)
    return tuple(int(np.argmin(np.abs(wavelength - t))) + 1 for t in targets)


_HAZE_RESISTANT_RGB_NM = (2200.0, 1650.0, 860.0)
"""R/G/B target wavelengths for a SWIR2-SWIR1-NIR false-color composite.
Used as _build_rgb_composite()'s default instead of true color: smoke and
haze are made of particles small enough to strongly scatter visible light
(the true-color 660/560/480 nm bands) but are largely transparent at these
longer wavelengths, so a fire scene's true-color background washes out
under smoke while this composite still shows the ground through it - the
same reasoning behind Landsat/MODIS 'SWIR' or fire/haze-penetration
composites. Vegetation renders orange/brown, burn scars dark red/black, and
water near-black; it is not a natural-color image."""


def _build_rgb_composite(scene, target_rgb_nm=_HAZE_RESISTANT_RGB_NM, percentile=(2.0, 98.0),
                          midtone=0.45):
    """Best-available color composite of the scene, for background context in
    the map figures. Defaults to a haze/smoke-resistant SWIR2-SWIR1-NIR
    false color (_HAZE_RESISTANT_RGB_NM - see its docstring) rather than true
    color, since these figures exist to show fire detections against their
    surroundings and a true-color background is exactly what smoke and haze
    obscure. Falls back to three bands spread evenly across whatever range
    the scene actually covers when it doesn't span the requested targets
    (e.g. a SWIR-only detector file with no VNIR bands). Pass
    target_rgb_nm=(660.0, 560.0, 480.0) explicitly for true color instead.
    This is a visualization stretch, not radiometrically calibrated
    reflectance.

    Two steps, not one. Each channel is first linearly stretched between its
    own `percentile` endpoints, then gamma-corrected so that its MEDIAN
    lands on `midtone`.

    The second step is what makes the result usable. An endpoint-only
    stretch fixes where each channel's 2nd and 98th percentiles land but
    says nothing about where the bulk of the pixels end up, and the three
    bands here have very differently shaped histograms: SWIR2 over forest is
    a narrow dark peak with a long bright tail (sand, bare drainage), while
    NIR is broad and centred. Measured on the Yalu scene, the endpoint-only
    stretch put the channel medians at R=0.11, G=0.30, B=0.59 - so the image
    came out both dark (median luminance 0.28) and cast strongly blue, and
    anything drawn over it in a dark colour was invisible. Equalising the
    medians removes the cast by construction and centres the tonal range,
    leaving colour to mean "this pixel differs from the scene's typical
    value in this band" - which is the only thing a false-colour composite
    can honestly mean anyway.

    Set midtone=None to keep the plain endpoint stretch."""
    cube, wavelength = scene['cube'], scene['wavelength']
    targets = _pick_rgb_band_targets(wavelength, target_rgb_nm)

    rgb = np.stack([search_band(cube, wavelength, t)[0] for t in targets], axis=-1).astype('float64')

    lo, hi = percentile
    for i in range(3):
        chan = rgb[..., i]
        finite = chan[np.isfinite(chan)]
        if finite.size == 0:
            continue
        p_lo, p_hi = np.nanpercentile(finite, [lo, hi])
        if p_hi <= p_lo:
            p_hi = p_lo + 1e-6
        stretched = np.clip((chan - p_lo) / (p_hi - p_lo), 0.0, 1.0)

        if midtone is not None:
            med = float(np.nanmedian(stretched))
            # gamma solving med**gamma == midtone; guard the degenerate
            # medians (0 or 1) where no gamma exists, and cap the exponent so
            # a near-black channel is not stretched into pure noise.
            if 1e-3 < med < 1.0 - 1e-3:
                gamma = float(np.clip(math.log(midtone) / math.log(med), 0.25, 4.0))
                stretched = stretched ** gamma
        rgb[..., i] = stretched

    return np.nan_to_num(rgb, nan=0.0)


def _adaptive_marker_size(n_points, s_min=6.0, s_max=55.0, target_total_area=3200.0):
    """Marker size (in points^2, matplotlib scatter's `s`) that shrinks as
    the point count grows, holding the combined marker area roughly
    constant instead of using one fixed size regardless of count. A scene
    with a handful of fire pixels wants large, clearly visible markers; a
    scene with 1000+ densely-clustered ones (a real wildfire front, not the
    sparse-handful case this was first tuned on) needs much smaller ones or
    they merge into a single blob that hides the image and each other.

    The area budget is deliberately generous. A 30 m EnMAP pixel inside a
    30 km scene is ~1/1000 of the frame; at any printable figure size that
    is smaller than a full stop, so a marker scaled anywhere near the true
    footprint is invisible and the panel reads as an empty image. These
    markers are symbols locating a detection, exactly as operational
    active-fire products draw them, not footprints - so they are sized to
    be legible rather than to scale."""
    if n_points <= 0:
        return s_min
    return float(np.clip(target_total_area / n_points, s_min, s_max))


def _muted_background(rgb, desaturate=0.8, lighten=0.42):
    """Push a false-color composite toward a pale grey version of itself, for
    use UNDER the colored detection markers.

    The composite and the markers otherwise compete for the same visual
    channel: the SWIR false color is fully saturated orange/red/black, and a
    colored point drawn on it - especially a dark one from the low end of a
    temperature ramp - simply disappears. Desaturating to luminance and then
    lifting toward white keeps every landscape feature legible as structure
    (ridges, drainages, burn scars, the fire's own smoke plume) while
    vacating the color axis entirely, so the only saturated color left in
    the panel is the data. Panels whose subject IS the image - the false-
    color context panel in the multi-sensor figure - keep the unmuted
    composite."""
    luminance = rgb @ np.array([0.2126, 0.7152, 0.0722])
    grey = np.repeat(luminance[..., None], 3, axis=-1)
    blended = (1.0 - desaturate) * rgb + desaturate * grey
    return np.clip(blended * (1.0 - lighten) + lighten, 0.0, 1.0)


def _fire_enhanced_rgb(scene, r0, r1, c0, c1, target_rgb_nm=_HAZE_RESISTANT_RGB_NM,
                       percentile=(1.0, 99.7), midtone=0.32, saturation=1.5):
    """SWIR2-SWIR1-NIR false colour of the window rows r0:r1, cols c0:c1,
    stretched on the window's own percentiles rather than the scene's, for
    the zoom insets.

    The scene composite is stretched over the whole tile and then muted
    under the markers, which is right for the main panel and wrong for a
    zoom: at true scale the reader wants to SEE the fire, and a flaming
    pixel is the brightest thing in the 2200 nm band by a wide margin. A
    local stretch with a high upper percentile keeps that contrast, the
    mild gamma lifts the terrain so the fire has context, and the
    saturation boost makes the emitted-radiance excess read as
    red/yellow against the cool tones of vegetation and soil. Each channel
    is gamma-corrected so that its median lands on `midtone`: without
    that, the NIR channel (bright over any vegetation) floods the window
    blue and the fire reads as a pale spot instead of a red one."""
    cube, wavelength = scene['cube'], scene['wavelength']
    targets = _pick_rgb_band_targets(wavelength, target_rgb_nm)
    win = np.stack([search_band(cube, wavelength, t)[0][r0:r1, c0:c1] for t in targets],
                   axis=-1).astype('float64')
    out = np.zeros_like(win)
    for i in range(3):
        chan = win[..., i]
        finite = chan[np.isfinite(chan)]
        if finite.size == 0:
            continue
        lo, hi = np.nanpercentile(finite, percentile)
        if hi <= lo:
            hi = lo + 1e-6
        stretched = np.clip((chan - lo) / (hi - lo), 0.0, 1.0)
        med = float(np.nanmedian(stretched))
        if midtone is not None and 1e-3 < med < 1.0 - 1e-3:
            stretched = stretched ** float(np.clip(math.log(midtone) / math.log(med), 0.25, 4.0))
        out[..., i] = stretched
    out = np.nan_to_num(out, nan=0.0)
    lum = out @ np.array([0.2126, 0.7152, 0.0722])
    out = np.clip(lum[..., None] + saturation * (out - lum[..., None]), 0.0, 1.0)
    return out


def _inset_corner(ax_frac_x, ax_frac_y, width, height, margin=0.02):
    """Pick the corner of the axes where an inset of the given size (axes
    fraction) covers the fewest of the points at (ax_frac_x, ax_frac_y).
    Preference order on ties: top-right, bottom-right, bottom-left,
    top-left, because the top-left carries the pixel-count label."""
    corners = [('top-right', 1.0 - width - margin, 1.0 - height - margin),
               ('bottom-right', 1.0 - width - margin, margin),
               ('bottom-left', margin, margin),
               ('top-left', margin, 1.0 - height - margin)]
    best = None
    for name, x0, y0 in corners:
        inside = ((ax_frac_x >= x0) & (ax_frac_x <= x0 + width)
                  & (ax_frac_y >= y0) & (ax_frac_y <= y0 + height))
        # The top-left corner carries the pixel-count label, so it has to
        # be clearly better, not merely tied, to be chosen.
        n = int(inside.sum()) + (1 if name == 'top-left' else 0)
        if best is None or n < best[0]:
            best = (n, name, x0, y0)
    return best[1], best[2], best[3]


def _scatter_over_composite(ax, lon, lat, rgb, ind_x, ind_y, values, cmap, vmin, vmax, s=None):
    """Draw the RGB composite as background, then each sparse per-pixel
    value as a colored marker on top - not its true ~30 m pixel footprint.
    detect_fire_pixels() finds a sparse set of pixels (from a handful to
    well over a thousand) scattered across a scene that can be tens of
    kilometers wide; at that zoom a single true-resolution pixel is a
    handful of screen pixels and effectively invisible, which is why
    plotting real footprints made every panel look identical at a glance.
    Fixed-size markers are the same convention operational active-fire
    products use for exactly this reason (e.g. FIRMS/VIIRS web maps render
    detections as icons, not literal pixel footprints) - s=None (the
    default) sizes them adaptively via _adaptive_marker_size() so a dense
    scene doesn't drown in overlapping circles; pass an explicit s to
    override.

    The background is muted (_muted_background()) and each marker is drawn
    over a white halo, because the alternative - saturated markers on a
    saturated SWIR composite - is exactly what makes a detection map
    unreadable: a dark-end marker on dark terrain and a bright-end marker on
    a bright ridge are both invisible, so the panel appears to show nothing
    wherever it matters most. The halo guarantees a fixed local contrast
    against whatever the marker happens to land on, so legibility no longer
    depends on where the fire is.

    rgb=None draws the markers only, over a background the caller has
    already drawn (the GUI's Sentinel-2 cloudless basemap)."""
    if rgb is not None:
        ax.pcolormesh(lon, lat, _muted_background(rgb), shading='auto', zorder=0)
    if len(ind_x) == 0:
        return None
    if s is None:
        s = _adaptive_marker_size(len(ind_x))
    # Paint in ascending value order so the highest-valued marker ends up on
    # TOP. Adjacent fire pixels are 30 m apart in a ~30 km scene, so at
    # full-scene zoom a cluster's markers overlap almost exactly; in input
    # order the last pixel written wins, which routinely buried the hottest
    # pixel of a fire front under a cooler neighbour and made the panel
    # understate the fire by hundreds of kelvin.
    order = np.argsort(np.nan_to_num(np.asarray(values, dtype='float64'), nan=-np.inf))
    ind_x, ind_y = np.asarray(ind_x)[order], np.asarray(ind_y)[order]
    values = np.asarray(values)[order]
    px_lon = lon[ind_x, ind_y]
    px_lat = lat[ind_x, ind_y]
    # White halo first, then the data-colored marker on top of it.
    ax.scatter(px_lon, px_lat, c='white', s=s * 1.9, linewidths=0.0, zorder=2)
    linewidth = 0.6 if s >= 25 else 0.3
    return ax.scatter(px_lon, px_lat, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                       s=s, edgecolors='#222222', linewidths=linewidth, zorder=3)


def _add_detection_inset(ax, lon, lat, rgb, ind_x, ind_y, values, cmap, vmin, vmax,
                          aspect, span_px=70, width=0.40, scene=None, outline_color='#00e5ff',
                          fill=True, fill_alpha=0.75):
    """Draw a zoomed inset on `ax` around the strongest detection, with a
    connector box showing where in the scene it came from.

    An EnMAP tile is ~30 x 30 km and an active fire is typically a handful
    of 30 m pixels, so the full-scene panel can only ever show WHERE the
    detections are, never their arrangement - which pixels are contiguous,
    whether a front is linear or scattered, how the hottest core sits inside
    it. That is the part a reader actually needs to judge a retrieval, and
    at full-scene zoom it is physically unresolvable. The inset restores it
    without giving up the geographic context of the main panel.

    Centred on the densest cluster of detections (the fire front on a real
    tile) rather than on the centroid of all of them, which for scattered
    detections lands on empty ground between clusters, or on the single
    highest value, which is often a bound-pinned candidate.

    The inset background is the fire-enhanced local composite of
    _fire_enhanced_rgb() when `scene` is given (a local percentile stretch
    of the SWIR2-SWIR1-NIR bands, so the flaming pixels glow red/yellow at
    true 30 m scale), else a local re-stretch of `rgb`. Detected pixels are
    drawn at their true footprints: filled with the value colormap at
    `fill_alpha` so the glow of the composite still reads through, and
    outlined in `outline_color` (cyan, absent from both the composite and
    the heat ramps) so a detection is identifiable even where its colour
    happens to match the ground. With fill=False only the outlines are
    drawn. The inset goes to whichever corner covers the fewest
    detections (_inset_corner). Returns the inset axes, or None when there
    is nothing to show."""
    if len(ind_x) == 0:
        return None
    ind_x, ind_y = np.asarray(ind_x), np.asarray(ind_y)
    finite = np.nan_to_num(np.asarray(values, dtype='float64'), nan=-np.inf)
    n_rows, n_cols = lon.shape

    # Centre on the DENSEST cluster of detections, not on the single
    # highest value. The highest retrieved temperature on a real tile is
    # routinely a pixel pinned at the search bound (a bright roof flagged
    # as a candidate), and a zoom on it shows a village; the fire front is
    # where the detections crowd together. Among the detections with the
    # most neighbours within a quarter of the span, the highest value wins.
    if len(ind_x) > 1:
        dx = np.abs(ind_x[:, None] - ind_x[None, :]) <= span_px // 4
        dy = np.abs(ind_y[:, None] - ind_y[None, :]) <= span_px // 4
        density = (dx & dy).sum(axis=1)
        cand = np.where(density == density.max())[0]
        peak = int(cand[np.argmax(finite[cand])])
    else:
        peak = 0

    # Size the window to the cluster the peak belongs to, not to a fixed
    # box: take every detection within span_px of the peak, fit the window
    # to their extent, and keep a floor so a single isolated pixel still
    # gets visible surroundings.
    near = (np.abs(ind_x - ind_x[peak]) <= span_px) & (np.abs(ind_y - ind_y[peak]) <= span_px)
    half = max(12, int(np.max([np.ptp(ind_x[near]), np.ptp(ind_y[near])])) + 6)
    half = min(half, span_px)
    r0, r1 = max(0, ind_x[peak] - half), min(n_rows, ind_x[peak] + half + 1)
    c0, c1 = max(0, ind_y[peak] - half), min(n_cols, ind_y[peak] + half + 1)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    sub_lon, sub_lat = lon[r0:r1, c0:c1], lat[r0:r1, c0:c1]
    x_lo, x_hi = float(np.nanmin(sub_lon)), float(np.nanmax(sub_lon))
    y_lo, y_hi = float(np.nanmin(sub_lat)), float(np.nanmax(sub_lat))

    # Where to put it: the corner with the fewest detections under it.
    height = width * aspect * 0.62
    x_all, y_all = lon[ind_x, ind_y], lat[ind_x, ind_y]
    xl, yl = ax.get_xlim(), ax.get_ylim()
    fx = (x_all - xl[0]) / (xl[1] - xl[0]) if xl[1] != xl[0] else np.zeros(len(x_all))
    fy = (y_all - yl[0]) / (yl[1] - yl[0]) if yl[1] != yl[0] else np.zeros(len(y_all))
    _, ix, iy = _inset_corner(fx, fy, width, height)

    axins = ax.inset_axes([ix, iy, width, height], xlim=(x_lo, x_hi), ylim=(y_lo, y_hi),
                           xticks=[], yticks=[])
    if scene is not None:
        local = _fire_enhanced_rgb(scene, r0, r1, c0, c1)
    else:
        local = np.asarray(rgb[r0:r1, c0:c1], dtype='float64').copy()
        for i in range(3):
            chan = local[..., i]
            lo_, hi_ = np.nanpercentile(chan, [1.0, 99.5])
            local[..., i] = np.clip((chan - lo_) / max(hi_ - lo_, 1e-6), 0.0, 1.0)
    axins.pcolormesh(sub_lon, sub_lat, local, shading='auto', zorder=0, rasterized=True)
    inside = (ind_x >= r0) & (ind_x < r1) & (ind_y >= c0) & (ind_y < c1)
    if inside.any():
        import matplotlib
        grid = np.full((r1 - r0, c1 - c0), np.nan)
        vals_arr = np.asarray(values, dtype='float64')
        grid[ind_x[inside] - r0, ind_y[inside] - c0] = vals_arr[inside]
        base = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
        shaded = base.with_extremes(bad=(0.0, 0.0, 0.0, 0.0))
        if fill:
            axins.pcolormesh(sub_lon, sub_lat, np.ma.masked_invalid(grid), cmap=shaded,
                              vmin=vmin, vmax=vmax, shading='auto', zorder=3, alpha=fill_alpha)
        # Outline the detected footprints: the 0.5 contour of the detection
        # mask on the pixel-centre grid, which at true scale traces the
        # boundary of each detected cluster.
        det = np.zeros((r1 - r0, c1 - c0), dtype=float)
        det[ind_x[inside] - r0, ind_y[inside] - c0] = 1.0
        if det.shape[0] >= 2 and det.shape[1] >= 2:
            axins.contour(sub_lon, sub_lat, det, levels=[0.5], colors=[outline_color], linewidths=0.6, zorder=4)
    for spine in axins.spines.values():
        spine.set_edgecolor('#111111')
        spine.set_linewidth(1.2)
    axins.set_aspect(aspect)
    axins.set_title(f'{int(inside.sum())} px at true scale, {(r1 - r0) * 30 / 1000:.1f} km', fontsize=6, pad=2)
    ax.indicate_inset_zoom(axins, edgecolor='#111111', linewidth=1.0, alpha=0.9)
    return axins


_SURFACE_FEATURE_COLORS = {
    'Vegetation': '#55a630',
    'Water/shadow': '#264653',
    'Bare ground/soil': '#bb8f5a',
    'Smoke/haze': '#9b5de5',
}


def _select_surface_feature_spectra(scene, background_mask):
    """Pick one representative background pixel for each of a few broad
    surface types (Vegetation/Water/Bare ground/Smoke), by simple spectral-
    index heuristics rather than a real classifier - the closest
    unsupervised analogue to the paper's manually labeled ROIs. NDVI
    (NIR~850 nm vs. Red~660 nm) separates vegetation (high NDVI) from water
    (strongly NIR-absorbing, low/negative NDVI); the most "typical"
    (median-NDVI) background pixel stands in for bare ground; the brightest
    pixel among the non-vegetated half stands in for smoke/haze (bright and
    spectrally flat is smoke's/cloud's signature, not vegetation's). Returns
    {label: (row, col)}, or {} if background_mask is empty."""
    cube, wavelength = scene['cube'], scene['wavelength']
    red = search_band(cube, wavelength, 660.0)[0]
    nir = search_band(cube, wavelength, 850.0)[0]
    ndvi = (nir - red) / (nir + red + 1e-9)

    rows, cols = np.where(background_mask)
    if rows.size == 0:
        return {}

    ndvi_bg = ndvi[rows, cols]
    with warnings.catch_warnings():
        # An all-NaN pixel is an expected input here, not an anomaly - it is
        # exactly the case the usable mask below exists to drop - so nanmean's
        # "Mean of empty slice" is noise. It is a warnings-module warning, not
        # a floating-point one, so errstate does not suppress it.
        warnings.simplefilter('ignore', RuntimeWarning)
        # band by band: cube[rows, :, cols] is a (n_bg, n_bands) copy of
        # ~0.9 GB on a full tile, and nanmean copies it again
        _s = np.zeros(rows.size); _n = np.zeros(rows.size)
        for _b in range(cube.shape[1]):
            _v = np.asarray(cube[:, _b, :][rows, cols], dtype=float)
            _f = np.isfinite(_v)
            _s[_f] += _v[_f]; _n += _f
        brightness_bg = np.where(_n > 0, _s / np.maximum(_n, 1), np.nan)

    # The pool can be non-empty and still carry no finite spectra (a fully
    # masked or nodata-filled scene). Restrict to usable pixels here rather
    # than letting each nanargmax below raise on an all-NaN slice: this is a
    # figure helper, and it must not be able to abort a completed retrieval.
    usable = np.isfinite(ndvi_bg) & np.isfinite(brightness_bg)
    if not usable.any():
        return {}
    rows, cols = rows[usable], cols[usable]
    ndvi_bg, brightness_bg = ndvi_bg[usable], brightness_bg[usable]

    picks = {}
    veg_i = int(np.nanargmax(ndvi_bg))
    picks['Vegetation'] = (rows[veg_i], cols[veg_i])

    water_i = int(np.nanargmin(ndvi_bg))
    picks['Water/shadow'] = (rows[water_i], cols[water_i])

    median_ndvi = np.nanmedian(ndvi_bg)
    soil_i = int(np.nanargmin(np.abs(ndvi_bg - median_ndvi)))
    picks['Bare ground/soil'] = (rows[soil_i], cols[soil_i])

    non_veg = np.where(ndvi_bg < median_ndvi)[0]
    pool = non_veg if non_veg.size else np.arange(len(ndvi_bg))
    smoke_i = int(pool[np.nanargmax(brightness_bg[pool])])
    picks['Smoke/haze'] = (rows[smoke_i], cols[smoke_i])

    return picks


# ---- Agreement plots: 1:1 at a true 45 degrees, with the fitted equation --

def _linear_fit(x, y):
    """Ordinary least squares y = slope*x + intercept, written longhand,
    plus R^2 and the RMSE about the fitted line.

    Returns None rather than a number when there is nothing to fit: fewer
    than two finite pairs, or an x with no spread at all. That second case
    is the one worth guarding - a vertical point cloud has no least-squares
    slope, and np.polyfit answers with a RankWarning and a meaningless
    coefficient instead of an error, which would then be printed on the
    figure as though it were a measurement."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 2:
        return None
    dx, dy = x - x.mean(), y - y.mean()
    sxx = float(np.dot(dx, dx))
    if sxx <= 0.0:
        return None
    slope = float(np.dot(dx, dy) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (slope * x + intercept)
    syy = float(np.dot(dy, dy))
    return {
        'slope': slope,
        'intercept': intercept,
        'n': int(x.size),
        'r2': float(1.0 - float(np.dot(resid, resid)) / syy) if syy > 0 else float('nan'),
        'rmse': float(np.sqrt(np.mean(resid ** 2))),
        'bias': float(np.mean(y - x)),
    }


def _equation_text(fit, unit='K'):
    """The fitted relationship written the way it would be read aloud -
    'y = 1.023x - 14.3 K' - with the sign folded into the operator rather
    than left as the '+ -14.3' that a naive format string produces."""
    b = fit['intercept']
    unit_suffix = (' ' + unit) if unit else ''
    return 'y = %.3fx %s %.1f%s' % (fit['slope'], '-' if b < 0 else '+',
                                    abs(b), unit_suffix)


def _plot_agreement_axes(ax, x, y, xlabel, ylabel, unit='K', color='#B3552C',
                          s=12, limits=None, show_fit=True, annotate=True,
                          fit_color='#1F4E79', annotate_loc=(0.04, 0.96)):
    """Scatter y against x on axes where a 1:1 relationship really is a
    45-degree line, and label the fitted relationship with its equation.
    Returns the _linear_fit() dict, or None when nothing could be fitted.

    TWO things are needed for the 45 degrees, and only having both makes it
    true. Identical x and y limits, because a 1:1 line drawn across unequal
    ranges is the diagonal of the data box rather than of the plane; and
    set_aspect('equal'), because otherwise matplotlib stretches the box to
    fill whatever space the subplot grid gave it and the line lands at
    whatever angle that produces. Getting only the first is the common case,
    and it is exactly what makes a scatter look like agreement when it is
    not: the eye reads distance from the diagonal, and the diagonal has been
    moved.

    The fitted line is ordinary least squares of y ON x, which is
    deliberately NOT symmetric - it minimises vertical residuals and so
    treats x as error-free. In a sensor-against-sensor comparison neither
    axis is error-free, so swapping the axes would give a different slope.
    The equation is therefore a description of this point cloud, not a
    calibration; the 1:1 line, not the fit, stays the reference the eye
    should measure against, and both are drawn for that reason."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[good], y[good], s=s, alpha=0.65, color=color,
               edgecolors='none', zorder=3)

    if limits is not None:
        lo, hi = float(limits[0]), float(limits[1])
    elif good.any():
        lo = float(min(np.min(x[good]), np.min(y[good])))
        hi = float(max(np.max(x[good]), np.max(y[good])))
        pad = 0.05 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.05
        lo, hi = lo - pad, hi + pad
    else:
        lo, hi = 0.0, 1.0

    ax.plot([lo, hi], [lo, hi], 'k--', lw=1.0, zorder=2, label='1:1')

    fit = _linear_fit(x, y) if show_fit else None
    if fit is not None:
        # Only across the span of x the fit was computed from, clipped to the
        # axis. A least-squares line says nothing outside the range of its own
        # x, and drawing it to the axis ends is how a fit over a narrow range
        # comes to look like a relationship that holds everywhere. Measured on
        # the Yalu scene: VIIRS LST spans 301.9-308.9 K against a retrieval
        # spanning 600-860 K, giving slope -1.844 at R^2 = 0.012 - no
        # relationship at all - which extended to the axis ends drew a bold
        # line straight across a 600 K plot. Clipped, it is the short mark it
        # should be, and the equation and R^2 in the annotation still report
        # exactly what was computed.
        ends = np.array([max(float(np.min(x[good])), lo),
                         min(float(np.max(x[good])), hi)], dtype=float)
        ax.plot(ends, fit['slope'] * ends + fit['intercept'], '-',
                color=fit_color, lw=1.4, zorder=4, label='Least-squares fit')

    # After the fit line, so autoscaling from it cannot widen the box back
    # out and quietly break the equal limits this whole function is about.
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if annotate and fit is not None:
        unit_suffix = (' ' + unit) if unit else ''
        ax.text(annotate_loc[0], annotate_loc[1],
                '%s\n$R^2$ = %.3f\nRMSE = %.1f%s\nn = %d'
                % (_equation_text(fit, unit), fit['r2'], fit['rmse'],
                   unit_suffix, fit['n']),
                transform=ax.transAxes, va='top', ha='left', fontsize=8,
                zorder=5,
                bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                          edgecolor='0.7', alpha=0.88))
    return fit


def _mask_atmospheric_bands(wavelength, spectrum, exclusions=_ATMOSPHERIC_EXCLUSIONS_NM):
    """A copy of `spectrum` with the opaque atmospheric windows set to NaN.

    NaN rather than deletion on purpose. Dropping the channels shortens the
    array and matplotlib then joins the two surviving neighbours with a
    straight segment straight across the gap - a line the instrument never
    measured, drawn at full weight, in the one part of the spectrum where
    the reader most needs to see that there is no data. A NaN breaks the
    line instead.

    `exclusions` defaults to the same _ATMOSPHERIC_EXCLUSIONS_NM the
    retrieval removes from the fit, so a plotted curve is the curve the fit
    actually saw. Pass () to disable."""
    wl = np.asarray(wavelength, dtype=float)
    out = np.array(spectrum, dtype=float, copy=True)
    for lo, hi in (exclusions or ()):
        out[(wl >= lo) & (wl <= hi)] = np.nan
    return out



def plot_spectra_figure(scene, results_pd, background_mask, wavelength,
                         hotspot_wl_lo=2060.0, hotspot_wl_hi=2430.0,
                         output_path=None, dpi=300, show_surface_features=True,
                         mask_water_bands=True,
                         atmospheric_exclusions=_ATMOSPHERIC_EXCLUSIONS_NM):
    """Two-panel spectral figure in the style of Figure 3 of the paper:
    absolute at-sensor radiance (a) and radiance normalized to each
    spectrum's own peak (b, the closest honest analogue to the paper's
    reflectance panel, since this pipeline never derives reflectance). Plots
    the hottest detected fire pixel and the mean +/-1 SD of the background
    pool used for its temperature fit, plus (when show_surface_features) one
    representative spectrum per heuristically-picked surface type from
    _select_surface_feature_spectra() - the paper's Vegetation/Water/Bare-
    ground/Smoke comparison, without a real classifier behind it. The two
    SWIR bands used by compute_hotspot_index() are shaded for reference.

    mask_water_bands (default True) removes the opaque atmospheric windows -
    the 1.4 and 1.9 um water-vapour bands, plus the 2.01 um CO2 band and the
    2.4 um tail, i.e. exactly the `atmospheric_exclusions` the retrieval
    itself drops - from every curve before anything is plotted or
    normalised. There are two reasons and only the second is about looks.

    The cosmetic one: at-sensor radiance in those windows falls to a few
    percent of the scene median (measured: an ~83% drop, see
    _ATMOSPHERIC_EXCLUSIONS_NM), so every curve on the panel collapses into
    the same near-zero notch and a third of the axis width is spent drawing
    atmosphere that all the spectra share.

    The one that matters: panel (b) divides each spectrum by its own peak
    and then asks the reader to compare SHAPES. The peak is in the SWIR and
    is unaffected, so the notch does not change the scaling - it just plants
    a deep trough in the middle of every shape being compared, at a
    wavelength where the surface contributed nothing to it. Removing the
    channels the fit already ignores makes the panel show the same spectrum
    the retrieval was performed on.

    The removed intervals are shaded and named in the legend, so the gaps
    read as deliberate rather than as missing data."""
    cube = scene['cube']
    exclusions = tuple(atmospheric_exclusions or ()) if mask_water_bands else ()

    def shown(spectrum):
        return _mask_atmospheric_bands(wavelength, spectrum, exclusions)

    hottest_idx = results_pd['mean_temp_K'].idxmax()
    hx = int(results_pd.loc[hottest_idx, 'ind_x'])
    hy = int(results_pd.loc[hottest_idx, 'ind_y'])
    fire_spectrum = shown(cube[hx, :, hy])
    fire_temp = results_pd.loc[hottest_idx, 'mean_temp_K']

    bkg_rows, bkg_cols = np.where(background_mask)
    bkg_spectra = cube[bkg_rows, :, bkg_cols]
    # Masked after the statistics, not before: nanmean/nanstd over the
    # excluded channels is well defined, it is simply not worth showing.
    bkg_mean = shown(np.nanmean(bkg_spectra, axis=0))
    bkg_std = shown(np.nanstd(bkg_spectra, axis=0))

    surface_picks = _select_surface_feature_spectra(scene, background_mask) if show_surface_features else {}

    with plt.rc_context(_PUBLICATION_RC):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        for ax in (ax1, ax2):
            for lo, hi in exclusions:
                ax.axvspan(lo, hi, color='#D9E7F2', zorder=0, linewidth=0)
            ax.axvspan(hotspot_wl_lo - 5, hotspot_wl_lo + 5, color='0.9', zorder=0)
            ax.axvspan(hotspot_wl_hi - 5, hotspot_wl_hi + 5, color='0.9', zorder=0)

        ax1.fill_between(wavelength, bkg_mean - bkg_std, bkg_mean + bkg_std,
                          color='#4c72b0', alpha=0.25, label='Background ±1 SD')
        ax1.plot(wavelength, bkg_mean, color='#4c72b0', linewidth=1.2, label='Background mean')
        ax1.plot(wavelength, fire_spectrum, color='#d1495b', marker='o', markersize=2.5,
                  linewidth=1, label=f'Hottest fire pixel ({fire_temp:.0f} K)')
        for label, (r, c) in surface_picks.items():
            ax1.plot(wavelength, shown(cube[r, :, c]), color=_SURFACE_FEATURE_COLORS[label],
                      linewidth=1.0, alpha=0.85, label=label)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel(r'Spectral radiance (W m$^{-2}$ sr$^{-1}$ nm$^{-1}$)')
        ax1.set_title('(a) At-sensor radiance')

        # np.nanmax over an all-NaN slice warns and returns NaN; these
        # spectra always retain the SWIR peak, but a caller passing a
        # narrower band subset should get a blank curve, not a warning.
        def normalized(spectrum):
            peak = np.nanmax(spectrum) if np.isfinite(spectrum).any() else np.nan
            return spectrum / peak if np.isfinite(peak) and peak > 0 else spectrum * np.nan

        ax2.plot(wavelength, normalized(bkg_mean), color='#4c72b0', linewidth=1.2,
                  label='Background mean')
        ax2.plot(wavelength, normalized(fire_spectrum), color='#d1495b', marker='o',
                  markersize=2.5, linewidth=1, label=f'Hottest fire pixel ({fire_temp:.0f} K)')
        for label, (r, c) in surface_picks.items():
            ax2.plot(wavelength, normalized(shown(cube[r, :, c])),
                      color=_SURFACE_FEATURE_COLORS[label], linewidth=1.0, alpha=0.85,
                      label=label)
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Radiance normalized to peak (–)')
        ax2.set_title('(b) Normalized radiance')

        handles, labels = ax1.get_legend_handles_labels()
        if exclusions:
            handles.append(mpatches.Patch(facecolor='#D9E7F2', edgecolor='none'))
            labels.append('Opaque H$_2$O/CO$_2$ bands (removed)')
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout(rect=(0, 0.06, 1, 1))

        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig


def plot_temperature_maps_figure(results_pd, scene, temp_map, hot_threshold=600.0,
                                  acquisition_dt=None, output_path=None, dpi=300):
    """Four-panel figure in the style of Figure 8 of the paper: (a) the full
    retrieved temperature map, (b) the same map masked to pixels at or above
    hot_threshold (the ~600 K reference for likely-flaming pixels used in
    the paper, after Dennison & Roberts 2009), sharing panel (a)'s color
    scale so the two are directly comparable, (c) the mean fire-fraction
    (p_fire) coefficient from the linear-mixture fit, and (d) the
    coefficient of variation (std/mean) of the temperature estimate, a
    per-pixel precision indicator. All four panels are drawn over a
    best-available color composite of the scene (see _build_rgb_composite())
    so the sparse detected/retrieved pixels are shown in their real
    surroundings rather than floating on blank space. acquisition_dt, if
    given (the EnMAP scene's own acquisition datetime - see
    _scene_acquisition_datetime()), is shown once for the whole figure,
    since all four panels are the same EnMAP image."""
    lat, lon = scene['lat'], scene['lon']
    ind_x, ind_y = results_pd['ind_x'].values, results_pd['ind_y'].values
    temp_values = results_pd['mean_temp_K'].values
    p_values = results_pd['mean_P_coeff'].values
    cv_values = (results_pd['std_temp_K'] / results_pd['mean_temp_K']).values
    rgb = _build_rgb_composite(scene)

    hot_mask = np.nan_to_num(temp_values, nan=-np.inf) >= hot_threshold

    vmin = float(np.nanmin(temp_values))
    vmax = float(np.nanmax(temp_values))
    cv_finite = cv_values[np.isfinite(cv_values)]
    cv_vmax = float(np.nanpercentile(cv_finite, 95)) if cv_finite.size else 1.0
    # p_fire is a SUBPIXEL area fraction: a 30 m EnMAP pixel containing a
    # few square metres of flame gives p_fire ~ 1e-3 to 1e-1. Stretching the
    # colour ramp over the full [0, 1] range it could in principle occupy
    # therefore paints every real detection the same colour at the bottom of
    # the scale, showing nothing. Scale to the data instead.
    p_finite = p_values[np.isfinite(p_values)]
    p_vmax = float(np.nanpercentile(p_finite, 98)) if p_finite.size else 1.0
    if not np.isfinite(p_vmax) or p_vmax <= 0:
        p_vmax = 1.0
    aspect = 1.0 / max(np.cos(np.radians(np.nanmean(lat))), 1e-3)
    lon_fmt, lat_fmt = _lonlat_formatter(is_lat=False), _lonlat_formatter(is_lat=True)

    # 'inferno' rather than 'viridis' for the two temperature panels: on the
    # muted background of _scatter_over_composite() a heat ramp reads as
    # temperature without a legend, and its bright end separates the flaming
    # pixels from everything else far more strongly than viridis's yellow-
    # green does. p_fire and CV keep their own ramps - they are not
    # temperatures and should not look like one.
    panels = [
        (ind_x, ind_y, temp_values, 'inferno', vmin, vmax,
         '(a) Retrieved temperature', 'Temperature (K)', True),
        (ind_x[hot_mask], ind_y[hot_mask], temp_values[hot_mask], 'inferno', vmin, vmax,
         f'(b) Temperature ≥ {hot_threshold:.0f} K (likely flaming)', 'Temperature (K)', True),
        (ind_x, ind_y, p_values, 'plasma', 0.0, p_vmax,
         '(c) Fire-fraction coefficient $p_{fire}$', '$p_{fire}$ (–)', False),
        (ind_x, ind_y, cv_values, 'magma', 0.0, cv_vmax,
         '(d) Temperature precision (std/mean)', 'CV (–)', False),
    ]

    with plt.rc_context(_PUBLICATION_RC):
        fig, axes = plt.subplots(2, 2, figsize=(11, 10))

        for ax, (px, py, values, cmap, lo, hi, title, cbar_label, inset) in zip(axes.ravel(), panels):
            pcm = _scatter_over_composite(ax, lon, lat, rgb, px, py, values, cmap, lo, hi)
            if pcm is not None:
                fig.colorbar(pcm, ax=ax, label=cbar_label, shrink=0.85)
            if inset and len(px):
                _add_detection_inset(ax, lon, lat, rgb, px, py, values, cmap, lo, hi,
                                      aspect, scene=scene)
            ax.set_title(title)
            ax.text(0.02, 0.98, f'n = {len(px)} pixels', transform=ax.transAxes,
                     ha='left', va='top', fontsize=8, color='white',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.55, pad=0.3))
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.xaxis.set_major_formatter(lon_fmt)
            ax.yaxis.set_major_formatter(lat_fmt)
            ax.tick_params(axis='x', rotation=30)
            ax.grid(True, color='white', linewidth=0.4, alpha=0.5)
            ax.set_aspect(aspect)

        if acquisition_dt is not None:
            fig.text(0.5, 0.995, f'EnMAP acquisition: {_format_acquisition_dt(acquisition_dt)}',
                     ha='center', va='top', fontsize=9, style='italic')
            fig.tight_layout(rect=(0, 0, 1, 0.97))
        else:
            fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig


def plot_hotspot_index_map(hotspot_index, scene, threshold_used, fire_points,
                            hotspot_wl_lo=2060.0, hotspot_wl_hi=2430.0,
                            acquisition_dt=None, output_path=None, dpi=300,
                            detection_info=None):
    """The detection figure.

    With `detection_info` (the GLRT detector of stage 2, detect_fire_pixels_glrt)
    it has two panels: (a) the detector statistic over the tile with the
    detections marked, and (b) the SWIR hotspot index (the HFDI of Dennison
    & Roberts 2009 as used by Amici et al. 2022) with the same detections
    marked. Without it (index thresholding, method='robust'/'otsu') it is
    the index panel alone, with `threshold_used` an index threshold.

    Both quantities are drawn as images under a robust percentile stretch
    of the whole tile, so the terrain is visible, and the detections are
    drawn on top as halo markers rather than left to the image: a detected
    fire is one or a few 30 m pixels in a 30 km tile, physically smaller
    than a printed dot, so an image alone shows nothing at the fire even
    with a colour scale centred on the threshold (that earlier version was
    a uniform blue tile). The colourbar carries the threshold as a line;
    a zoom inset shows the strongest cluster at true scale on a
    fire-enhanced composite."""
    lat, lon = scene['lat'], scene['lon']
    aspect = 1.0 / max(np.cos(np.radians(np.nanmean(lat))), 1e-3)
    lon_fmt, lat_fmt = _lonlat_formatter(is_lat=False), _lonlat_formatter(is_lat=True)
    fire_points = np.asarray(fire_points).reshape(-1, 2) if len(fire_points) else np.zeros((0, 2), int)
    px, py = fire_points[:, 0], fire_points[:, 1]
    date_txt = _format_acquisition_dt(acquisition_dt) if acquisition_dt is not None else ''
    glrt = detection_info is not None and detection_info.get('lam') is not None

    def _robust_limits(arr, lo=1.0, hi=99.5):
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            return 0.0, 1.0
        a, b = np.nanpercentile(finite, [lo, hi])
        return (float(a), float(b)) if b > a else (float(a), float(a) + 1e-6)

    def _finish(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.xaxis.set_major_formatter(lon_fmt); ax.yaxis.set_major_formatter(lat_fmt)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, color='white', linewidth=0.4, alpha=0.4)
        ax.set_aspect(aspect)

    def _markers(ax, values, cmap, vmin, vmax):
        if len(px) == 0:
            return None
        s_ = _adaptive_marker_size(len(px), s_min=8.0, s_max=40.0)
        order = np.argsort(np.nan_to_num(values, nan=-np.inf))
        xx, yy = lon[px[order], py[order]], lat[px[order], py[order]]
        ax.scatter(xx, yy, c='white', s=s_ * 1.9, linewidths=0, zorder=2)
        return ax.scatter(xx, yy, c=values[order], cmap=cmap, vmin=vmin, vmax=vmax, s=s_,
                          edgecolors='#222222', linewidths=0.4, zorder=3)

    with plt.rc_context(_PUBLICATION_RC):
        if glrt:
            fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 6.6))
        else:
            fig, ax_b = plt.subplots(figsize=(8, 7.2))
            ax_a = None

        if glrt:
            lam = np.asarray(detection_info['lam'], dtype='float64')
            thr = float(detection_info.get('threshold', threshold_used))
            with np.errstate(divide='ignore', invalid='ignore'):
                lam_img = np.log10(np.clip(lam, 1e-3, None))
            lo_a, hi_a = _robust_limits(lam_img, 1.0, 99.9)
            hi_a = max(hi_a, np.log10(max(thr, 1e-3)) + 0.5)
            pcm = ax_a.pcolormesh(lon, lat, lam_img, cmap='cividis', vmin=lo_a, vmax=hi_a,
                                  shading='auto', rasterized=True)
            cls = np.asarray(detection_info.get('point_class', np.ones(len(px), int)))
            lam_pts = np.asarray(detection_info.get('point_lambda', lam[px, py]), dtype='float64')
            lam_pts_img = np.log10(np.clip(lam_pts, 1e-3, None))
            sc = _markers(ax_a, lam_pts_img, 'autumn_r', lo_a, max(hi_a, np.nanmax(lam_pts_img) if lam_pts.size else hi_a))
            cbar = fig.colorbar(pcm, ax=ax_a, shrink=0.8, pad=0.03,
                                label='detector statistic  log$_{10}$ Λ')
            cbar.ax.axhline(np.log10(max(thr, 1e-3)), color='#d62728', linewidth=1.2)
            cbar.ax.text(1.15, np.log10(max(thr, 1e-3)), 'threshold', color='#d62728', fontsize=7,
                         va='center', ha='left', transform=cbar.ax.get_yaxis_transform())
            n_fire = int(detection_info.get('n_fire', (cls == 1).sum()))
            n_cand = int(detection_info.get('n_candidate', (cls == 2).sum()))
            p_fa = detection_info.get('p_fa', None)
            p_txt = f', $p_{{fa}}$ = {p_fa:g}' if p_fa is not None else ''
            _finish(ax_a, f'(a) Detector statistic Λ (threshold {thr:.1f}{p_txt})\n'
                          f'{n_fire} fire + {n_cand} candidate pixels')
            if len(px):
                _add_detection_inset(ax_a, lon, lat, _build_rgb_composite(scene), px, py, lam_pts_img,
                                     'autumn_r', lo_a, hi_a, aspect, scene=scene, fill=False)

        # (b) the hotspot index, robustly stretched, detections on top.
        idx = np.asarray(hotspot_index, dtype='float64')
        lo_b, hi_b = _robust_limits(idx, 1.0, 99.0)
        pcm_b = ax_b.pcolormesh(lon, lat, idx, cmap='Greys_r', vmin=lo_b, vmax=hi_b,
                                shading='auto', rasterized=True)
        cbar_b = fig.colorbar(pcm_b, ax=ax_b, shrink=0.8, pad=0.03,
                              label=f'hotspot index  (L$_{{{hotspot_wl_hi:.0f}}}$ − L$_{{{hotspot_wl_lo:.0f}}}$) / '
                                    f'(L$_{{{hotspot_wl_hi:.0f}}}$ + L$_{{{hotspot_wl_lo:.0f}}}$)')
        idx_pts = idx[px, py] if len(px) else np.zeros(0)
        if len(px):
            v_lo = float(np.nanmin(idx_pts)) if np.isfinite(idx_pts).any() else lo_b
            v_hi = float(np.nanmax(idx_pts)) if np.isfinite(idx_pts).any() else hi_b
            if v_hi <= v_lo:
                v_hi = v_lo + 1e-6
            sc_b = _markers(ax_b, idx_pts, 'RdYlBu_r', v_lo, v_hi)
            cb2 = fig.colorbar(sc_b, ax=ax_b, shrink=0.8, pad=0.02,
                               label='index of the detected pixels (markers)')
            cb2.ax.tick_params(labelsize=7)
        if not glrt and np.isfinite(threshold_used):
            cbar_b.ax.axhline(float(threshold_used), color='#d62728', linewidth=1.2)
            title_b = (f'SWIR hotspot index, threshold {float(threshold_used):.3f}\n'
                       f'{len(px)} pixels detected')
        else:
            title_b = f'(b) SWIR hotspot index at the {len(px)} detected pixels'
        _finish(ax_b, title_b)
        if len(px) and not glrt:
            _add_detection_inset(ax_b, lon, lat, _build_rgb_composite(scene), px, py, idx_pts,
                                 'RdYlBu_r', float(np.nanmin(idx_pts)), float(np.nanmax(idx_pts)) + 1e-6,
                                 aspect, scene=scene, fill=False)

        if date_txt:
            fig.text(0.5, 0.995, f'EnMAP acquisition: {date_txt}', ha='center', va='top',
                     fontsize=9, style='italic')
            fig.tight_layout(rect=(0, 0, 1, 0.97))
        else:
            fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig


def _annotate_hottest_pixels(ax, lon, lat, ind_x, ind_y, values, n=5, color='white'):
    """Ring the n hottest detected pixels, over an otherwise unaltered
    false-color image.

    The purpose of this panel is to call out the handful of most significant
    detections against the real imagery - the plume, the burn scar, the
    terrain - rather than to plot every detection, which panel (b) already
    does. It therefore must not paint over the image, which is why filled
    markers are not used here.

    The original annotation, following the paper's own Figure 9(c), was a
    plain black arrow per pixel. That does not survive contact with a real
    SWIR false-color composite: over dark forest or water a black arrow is
    invisible, which is exactly what happened on the Yalu scene. An open
    ring drawn with a DOUBLE stroke - a thick dark halo under a thin bright
    core - is legible on any background, light or dark, because whichever
    stroke matches the background the other one contrasts with it. The ring
    is open, so the pixel it marks is still visible inside it, and it is
    numbered so the panel can be discussed pixel by pixel in a caption.
    """
    if len(ind_x) == 0:
        return
    import matplotlib.patheffects as pe

    n = min(n, len(ind_x))
    order = np.argsort(np.nan_to_num(np.asarray(values, dtype='float64'),
                                     nan=-np.inf))[::-1][:n]
    px_lon = lon[np.asarray(ind_x)[order], np.asarray(ind_y)[order]]
    px_lat = lat[np.asarray(ind_x)[order], np.asarray(ind_y)[order]]

    # Open rings: dark halo first, bright core over it.
    ax.scatter(px_lon, px_lat, s=260, facecolors='none', edgecolors='#101010',
               linewidths=3.0, zorder=3)
    ax.scatter(px_lon, px_lat, s=260, facecolors='none', edgecolors=color,
               linewidths=1.3, zorder=4)

    lat_span = float(np.nanmax(lat) - np.nanmin(lat))
    for rank, (x, y) in enumerate(zip(px_lon, px_lat), start=1):
        ax.annotate(str(rank), xy=(x, y), xytext=(0, 13), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8, fontweight='bold',
                    color=color, zorder=5,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground='#101010')])
    del lat_span



def plot_multi_sensor_comparison_figure(results_pd, scene, ecostress=None, landsat=None,
                                         enmap_dt=None, ecostress_dt=None, landsat_dt=None,
                                         landsat_label='Landsat 8/9', extra_sources=None,
                                         output_path=None, dpi=300, n_arrows=5):
    """Multi-panel figure in the style of Figure 9 of the paper: every
    thermal sensor that was found for this scene, side by side with EnMAP's
    own retrieval.

    EnMAP's own retrieved fire temperature and a haze/smoke-resistant
    false-color composite (_build_rgb_composite(), with black arrows at the
    n_arrows hottest detected pixels - the paper's own panel (c) is a
    pansharpened RGB with arrows at the >1000 K pixels; see
    _annotate_hottest_pixels()) are always drawn. ECOSTRESS and Landsat LST
    each get a panel only when that sensor was actually found for this
    scene: a sensor with no data is dropped from the figure entirely rather
    than drawn as a blank "not available" placeholder, and the grid is
    resized to fit however many panels remain, so a scene with fewer
    available references gets a smaller, fully-populated figure instead of
    one padded out with empty boxes. This does mean a panel's letter
    (a), (b), ... is not tied to a fixed sensor across scenes - read it from
    the panel's own title, not its position.

    extra_sources appends one panel per further thermal sensor, in the order
    given: a sequence of {'label': str, 'data': loader-dict or None,
    'datetime': datetime or None}, where the loader dict is any of the
    'lst_K'/'geotransform'/'projection_wkt' dicts the loaders return. This
    is how the VIIRS and Sentinel-3 SLSTR partners resolved by
    _load_thermal_partners() reach a figure at all - they were being
    downloaded, sampled per pixel and written into the results CSV, but the
    only picture of the scene's thermal context stopped at the two sensors
    this function originally hard-coded. A source whose 'data' is None is
    skipped entirely, the same as a missing ECOSTRESS or Landsat panel now.

    enmap_dt/ecostress_dt/landsat_dt and each extra source's 'datetime', if
    given, are shown in that panel's own title - deliberately per-panel
    rather than one shared figure-level date, since these sensors were very
    likely NOT acquired at the same time (the auto-fetch helpers may match a
    granule days apart from the EnMAP acquisition when nothing closer passed
    the time-window/cloud-cover requirements), so one date for the whole
    figure would misrepresent every panel but one."""
    lat, lon = scene['lat'], scene['lon']
    ind_x, ind_y = results_pd['ind_x'].values, results_pd['ind_y'].values
    temp_values = results_pd['mean_temp_K'].values
    rgb = _build_rgb_composite(scene)
    aspect = 1.0 / max(np.cos(np.radians(np.nanmean(lat))), 1e-3)
    lon_fmt, lat_fmt = _lonlat_formatter(is_lat=False), _lonlat_formatter(is_lat=True)

    def _style_axes(ax, title):
        ax.set_title(title)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.xaxis.set_major_formatter(lon_fmt)
        ax.yaxis.set_major_formatter(lat_fmt)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, color='white', linewidth=0.4, alpha=0.4)
        ax.set_aspect(aspect)

    def _resampled_lst_panel(ax, lst_source, title, cbar_label):
        grid = _resample_raster_to_lonlat(
            lst_source['lst_K'], lst_source['geotransform'], lst_source['projection_wkt'], lon, lat)
        finite = grid[np.isfinite(grid)]
        vlo = float(np.nanpercentile(finite, 2)) if finite.size else 250.0
        vhi = float(np.nanpercentile(finite, 99)) if finite.size else 320.0
        # NaN cells (fill value, cloud/QC-masked, or outside the reference
        # raster's coverage) otherwise render as matplotlib's default "bad"
        # color, which is fully transparent - indistinguishable from this
        # figure's own white background/facecolor, so a masked-out patch
        # reads as a hole in the data rather than as the deliberately-
        # excluded pixels it is. An explicit mid-grey makes that visible.
        import matplotlib
        cmap = matplotlib.colormaps['inferno'].with_extremes(bad='#c8c8c8')
        pcm = ax.pcolormesh(lon, lat, grid, cmap=cmap, vmin=vlo, vmax=vhi, shading='auto')
        fig.colorbar(pcm, ax=ax, label=cbar_label, shrink=0.85)
        _style_axes(ax, title)

    def _titled(label, dt):
        suffix = _format_acquisition_dt(dt)
        return f'{label}\n{suffix}' if suffix else label

    # Only the sensors actually found for this scene get a panel - EnMAP and
    # the false-color context are always drawn; ECOSTRESS/Landsat/partners
    # are each included only when their data is not None, so a scene with
    # fewer available references gets a smaller figure rather than one
    # padded out with "not available" boxes. Panels are lettered from this
    # list rather than hard-coded, so a missing sensor cannot leave a gap in
    # the lettering or two panels both called '(d)'.
    panels = []
    if ecostress is not None:
        panels.append(('lst', ecostress, 'ECOSTRESS LST', ecostress_dt))
    panels.append(('enmap', None, 'EnMAP retrieved temperature', enmap_dt))
    panels.append(('rgb', None, 'False color (SWIR2/SWIR1/NIR)', enmap_dt))
    if landsat is not None:
        panels.append(('lst', landsat, f'{landsat_label} LST', landsat_dt))
    for src in (extra_sources or []):
        if src and src.get('data') is not None:
            panels.append(('lst', src['data'], src.get('label', 'LST'),
                           src.get('datetime')))

    # Column count adapts to how many panels survived, rather than a fixed
    # 2, so a 2- or 3-panel figure (most references missing) is one compact
    # row instead of a mostly-empty 2-column grid.
    n_panels = len(panels)
    if n_panels <= 2:
        n_cols = max(n_panels, 1)
    elif n_panels == 3:
        n_cols = 3
    else:
        n_cols = 2
    n_rows = int(np.ceil(n_panels / n_cols))

    with plt.rc_context(_PUBLICATION_RC):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 5.0 * n_rows),
                                 squeeze=False)
        flat = [ax for row in axes for ax in row]

        for i, (kind, data, label, dt) in enumerate(panels):
            ax = flat[i]
            letter = '(%s)' % chr(ord('a') + i)
            if kind == 'lst':
                _resampled_lst_panel(ax, data, _titled(f'{letter} {label}', dt),
                                      f'{label} (K)')
            elif kind == 'enmap':
                t_lo, t_hi = float(np.nanmin(temp_values)), float(np.nanmax(temp_values))
                pcm_temp = _scatter_over_composite(ax, lon, lat, rgb, ind_x, ind_y,
                                                    temp_values, 'inferno', t_lo, t_hi)
                if pcm_temp is not None:
                    fig.colorbar(pcm_temp, ax=ax, label='Temperature (K)', shrink=0.85)
                _style_axes(ax, _titled(f'{letter} {label} (n={len(ind_x)})', dt))
                _add_detection_inset(ax, lon, lat, rgb, ind_x, ind_y, temp_values,
                                      'inferno', t_lo, t_hi, aspect)
            else:
                ax.pcolormesh(lon, lat, rgb, shading='auto', zorder=0)
                _annotate_hottest_pixels(ax, lon, lat, ind_x, ind_y, temp_values, n=n_arrows)
                _style_axes(ax, _titled(
                    f'{letter} {label} + {min(n_arrows, len(ind_x))} hottest pixels ringed', dt))

        # A trailing odd panel would otherwise be drawn as an empty box with
        # ticks, which reads as a sensor that returned nothing.
        for ax in flat[len(panels):]:
            ax.set_visible(False)

        fig.tight_layout()

        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return None
        return fig


# Reference columns validate_temperature() may attach, in the order they
# should appear on a figure. Fixed here rather than taken from the
# DataFrame's column order, which depends on which partners were fetched and
# would silently reorder panels between runs of the same scene.
_REFERENCE_COLUMN_LABELS = (
    ('ecostress_LST_K', 'ECOSTRESS LST'),
    ('landsat_LST_K', 'Landsat LST'),
    ('viirs_LST_K', 'VIIRS LST'),
    ('slstr_LST_K', 'Sentinel-3 SLSTR LST'),
    ('field_temperature_K', 'Field measurement'),
)


def _discover_reference_columns(results_pd):
    """The (column, label) pairs in results_pd that hold a reference
    temperature with at least one finite value.

    The known columns come first, in _REFERENCE_COLUMN_LABELS order; any
    further '*_LST_K' column is then appended alphabetically. That tail
    matters because validate_temperature() derives a column name from the
    reference's label when the caller does not supply one, so a partner
    added to thermal_partners.py later reaches this figure without anything
    here needing to be updated."""
    known = [c for c, _ in _REFERENCE_COLUMN_LABELS]
    pairs = list(_REFERENCE_COLUMN_LABELS)
    pairs += [(c, c[:-6].replace('_', ' ').strip() + ' LST')
              for c in sorted(results_pd.columns)
              if c.endswith('_LST_K') and c not in known]

    out = []
    for col, label in pairs:
        if col not in results_pd.columns:
            continue
        values = pd.to_numeric(results_pd[col], errors='coerce').values
        if np.isfinite(values).any():
            out.append((col, label))
    return out


def plot_reference_agreement_figure(results_pd, references=None, output_path=None,
                                     dpi=300, comparable_max_K=600.0, unit='K'):
    """One scatter panel per thermal reference that reached the results
    table: EnMAP's retrieved temperature against that sensor's LST, with the
    1:1 line at a true 45 degrees and the least-squares relationship written
    out as an equation.

    This is the figure form of the bias/RMSE lines _reference_stats() already
    prints, and it carries exactly the same caveat - which is worth repeating
    here because a scatter with a 1:1 line on it invites the opposite
    reading. The retrieval reports the temperature of the SUB-PIXEL fire,
    while ECOSTRESS, Landsat, VIIRS and SLSTR each report one 60-1000 m
    footprint averaged together. For a hot pixel those are not the same
    quantity, 1:1 is not the expected relationship, and a fit far from it is
    not by itself an error. Where the equation does mean something is the
    comparable subset - pixels retrieved below comparable_max_K, cool enough
    that the footprint average is measuring roughly the same thing - so those
    points are drawn in their own colour and fitted separately.

    The panels deliberately keep identical x and y limits spanning BOTH
    quantities, which is what puts the 1:1 line at 45 degrees. On a scene
    with real fire that leaves the reference points bunched into one corner.
    That is not a defect of the plot; it is the measurement - a 900 K
    sub-pixel flame and a 320 K footprint mean genuinely are that far apart,
    and an axis rescaled to hide it would be the misleading version.

    Saturated reference pixels are not marked here; _reference_stats() counts
    them per sensor in the printed report, where the band ceiling for each
    sensor is already known.

    references, if given, is a sequence of (column, label) pairs; by default
    every reference column validate_temperature() may have written is picked
    up by _discover_reference_columns().

    Returns the path written, or None when the results table carries no
    reference column at all - in which case nothing is written, because an
    empty figure on disk is worse than no figure."""
    refs = list(references) if references is not None else _discover_reference_columns(results_pd)
    if not refs:
        return None

    model = pd.to_numeric(results_pd['mean_temp_K'], errors='coerce').values

    with plt.rc_context(_PUBLICATION_RC):
        fig, axes = plt.subplots(1, len(refs), figsize=(4.8 * len(refs), 5.0),
                                 squeeze=False)
        for ax, (col, label) in zip(axes[0], refs):
            ref = pd.to_numeric(results_pd[col], errors='coerce').values
            good = np.isfinite(ref) & np.isfinite(model)
            _plot_agreement_axes(ax, ref, model, xlabel=f'{label} ({unit})',
                                  ylabel=f'EnMAP retrieved T ({unit})', unit=unit)

            # The comparable subset, in its own colour and with its own fit,
            # drawn over the full-cloud fit rather than instead of it: the
            # two together are the point of the panel.
            sub = good & (model < comparable_max_K)
            if int(sub.sum()) >= 2:
                ax.scatter(ref[sub], model[sub], s=14, alpha=0.85, color='#3C6E8F',
                           edgecolors='none', zorder=6,
                           label=f'retrieved < {comparable_max_K:.0f} {unit}')
                sub_fit = _linear_fit(ref[sub], model[sub])
                if sub_fit is not None:
                    lo, hi = ax.get_xlim()
                    # Same rule as _plot_agreement_axes: across this subset's
                    # own x span only, never extrapolated to the axis ends.
                    ends = np.array([max(float(np.min(ref[sub])), lo),
                                     min(float(np.max(ref[sub])), hi)], dtype=float)
                    ax.plot(ends, sub_fit['slope'] * ends + sub_fit['intercept'],
                            '-', color='#3C6E8F', lw=1.3, zorder=6,
                            label='comparable subset: ' + _equation_text(sub_fit, unit))
                    ax.set_xlim(lo, hi)
                    ax.set_ylim(lo, hi)

            ax.set_title(f'EnMAP vs {label}\nn={int(good.sum())}', fontsize=10)
            ax.legend(fontsize=7, loc='lower right', framealpha=0.9)

        fig.suptitle('Retrieved sub-pixel fire temperature against each thermal '
                     'reference\n(1:1 shown for scale - coarse-footprint LST is not '
                     'expected to equal a sub-pixel flame temperature)', fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        if output_path:
            fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            return output_path
        return fig


def generate_publication_figures(results_pd, scene, temp_map, hotspot_index, threshold_used,
                                  fire_points, background_mask, wavelength, results_dir, scene_id,
                                  hotspot_wl_lo=2060.0, hotspot_wl_hi=2430.0,
                                  hot_threshold=600.0, acquisition_dt=None, dpi=300,
                                  detection_info=None):
    """Generate and save the three figures above as PNGs in results_dir.
    acquisition_dt (the EnMAP scene's own acquisition datetime - see
    _scene_acquisition_datetime()), if given, is stamped onto the two
    map-based figures (temperature maps, hotspot index); plot_spectra_figure()
    isn't map-based so it's skipped there. Returns the list of saved file
    paths."""
    paths = [
        os.path.join(results_dir, f'Fig1_spectra_{scene_id}.png'),
        os.path.join(results_dir, f'Fig2_temperature_maps_{scene_id}.png'),
        os.path.join(results_dir, f'Fig3_hotspot_index_{scene_id}.png'),
    ]

    plot_spectra_figure(scene, results_pd, background_mask, wavelength,
                         hotspot_wl_lo=hotspot_wl_lo, hotspot_wl_hi=hotspot_wl_hi,
                         output_path=paths[0], dpi=dpi)
    plot_temperature_maps_figure(results_pd, scene, temp_map, hot_threshold=hot_threshold,
                                  acquisition_dt=acquisition_dt, output_path=paths[1], dpi=dpi)
    plot_hotspot_index_map(hotspot_index, scene, threshold_used, fire_points,
                            hotspot_wl_lo=hotspot_wl_lo, hotspot_wl_hi=hotspot_wl_hi,
                            acquisition_dt=acquisition_dt, output_path=paths[2], dpi=dpi,
                            detection_info=detection_info)

    for p in paths:
        print(f'Saved {p}')
    return paths


#%% ###########################################################################
#                                   main()
################################################################################

@contextlib.contextmanager
def _joblib_progress(total, callback):
    """Make joblib.Parallel report real per-task COMPLETION progress via
    callback(n_done, total), instead of the naive approach of wrapping the
    input generator in tqdm (which only reflects how many tasks have been
    *dispatched* - under joblib's eager dispatch that reaches ~n_jobs tasks
    almost instantly, not actual completion). Patches joblib's internal
    BatchCompletionCallBack for the duration of the `with` block only; the
    callback always runs in the main process's dispatch thread regardless of
    backend (loky/multiprocessing included), so no extra IPC is needed."""
    import joblib.parallel

    class _Callback(joblib.parallel.BatchCompletionCallBack):
        _completed = 0

        def __call__(self, *args, **kwargs):
            _Callback._completed += self.batch_size
            callback(_Callback._completed, total)
            return super().__call__(*args, **kwargs)

    original = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _Callback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = original


# The configuration of Amici et al. (2022) exactly as published (JGR
# Biogeosciences 127:e2022JG007055, Sect. 3.2 and 4.2) and as coded in the
# software released with the paper (Fire_analysis_with_PRISMA.py, Zenodo
# 10.5281/zenodo.7273879). Stage 5 of run_pipeline() overrides the scene
# configuration with these values so that the reference retrieval is the
# published method and not a variant of it:
#   fit window 1400 < lambda < 1900 nm (code; the paper text quotes
#   1400-1800 nm), Planck term set to zero where the measured radiance is
#   below 0.003 W m-2 sr-1 nm-1 (the water-vapour bands), one thermal
#   source, m = 20 background spectra per trial, N = 15 trials, emissivity
#   1, T in [100, 2000] K, unnormalised fire abundance in [0.05, 1],
#   background abundances in [0, 1], trust-region reflective solver with
#   gtol = ftol = xtol = 1e-8 and scipy's default evaluation budget, and
#   the mean of the trials with a Student-t interval as the estimate.
# The full-SWIR window with the opaque channels removed (DEFAULT_CONFIG
# below) remains the scene window for the detector and for this work's
# retrieval; the measured benefit of that window for the joint fit is a
# finding about the window, not part of the published method, and is
# reported as such (VERIFICATION.md section 5).
AMICI2022_PUBLISHED_CONFIG = {
    'only_SWIR': True,
    'only_short_SWIR': True,
    'only_feasible_SWIR': False,
    'exclude_atmospheric_windows': False,
    'absorption_blank_threshold': 0.003,
    'N_sources': 1,
    'method': 'trf',
    'fire_abundance_min': 0.05,
    'max_nfev': None,
}
# The published sample sizes and bounds. They are DEFAULT_CONFIG's own
# values, so a run reproduces the paper unless the user changes them (the
# GUI exposes the trial count); amici2022_config() says so when they differ.
AMICI2022_PUBLISHED_SIZES = {'N_bkg_points': 20, 'N_trials': 15, 'emissivity': 1,
                             'T_min': 100, 'T_max': 2000}


def amici2022_config(config):
    """The stage-5 configuration: `config` with the structural values of
    the published method (AMICI2022_PUBLISHED_CONFIG: window, blanking
    rule, abundance bounds, solver settings) imposed on top. The sample
    sizes and temperature bounds (AMICI2022_PUBLISHED_SIZES) come from
    `config`, whose defaults are the published ones; a departure is
    reported on stdout so a run is never mistaken for the published
    configuration."""
    cfg = dict(config)
    cfg.update(AMICI2022_PUBLISHED_CONFIG)
    changed = {k: cfg.get(k) for k, v in AMICI2022_PUBLISHED_SIZES.items()
               if cfg.get(k) is not None and float(cfg.get(k)) != float(v)}
    if changed:
        print('  NOTE: the reference retrieval departs from the published configuration of '
              'Amici et al. (2022) in ' + ', '.join(f'{k} = {v}' for k, v in changed.items())
              + ' (published: ' + ', '.join(f'{k} = {AMICI2022_PUBLISHED_SIZES[k]}' for k in changed) + ').')
    return cfg


DEFAULT_CONFIG = {
    'only_SWIR': True,
    # Full SWIR (1400-2445 nm), not the legacy 1400-1900 nm sub-window.
    # The short window leaves 28 bands against 22 free parameters
    # (2*N_sources + N_bkg_points) - 7 degrees of freedom - and measured
    # -85.6 K bias with 103 K IQR at T=800/1000 K, p=0.02. The full window
    # gives 89 bands, 68 degrees of freedom, and -3.5 K bias with 9.9 K IQR
    # on the same test. Set True to reproduce the legacy configuration.
    'only_short_SWIR': False,
    'only_feasible_SWIR': False,
    'saturation_threshold': 0.8,
    'plot': False,
    'N_sources': 1,
    'N_bkg_points': 20,
    'N_trials': 15,
    'emissivity': 1,
    'verbose': 0,
    'T_min': 100,
    'T_max': 2000,
    'method': 'trf',
    # Remove opaque water-vapour channels from the fit entirely. This
    # replaces the legacy `ydata < 0.003` rule, which zeroed only the Planck
    # term (leaving the background term to fit the data there) at an
    # absolute, unit-dependent radiance cut. Over the full SWIR that rule
    # blanked 62 of 89 bands and cost -223 K of bias at 600 K.
    'exclude_atmospheric_windows': True,
    'atmospheric_exclusions': _ATMOSPHERIC_EXCLUSIONS_NM,
    # None disables the legacy blanking; set a float (0.003 was the old
    # hard-coded value) only to reproduce older results.
    'absorption_blank_threshold': None,
    # Lower bound on each unnormalised fire abundance. Sets the reported
    # fire_fraction_floor = fire_abundance_min / (N_sources + N_bkg_points).
    'fire_abundance_min': 0.05,
    # Base seed; each pixel derives its own stream from this plus its index.
    'random_seed': 0,
    # Evaluation budget per curve_fit call. scipy's own default for 'trf' is
    # 100*n_params, i.e. 2200 for the default 22-parameter model, which a
    # small minority of pixels exhaust - the exact scale degeneracy
    # (VERIFICATION.md section 2.2) leaves a flat direction the optimiser
    # can wander along, spending evaluations without reducing the residual.
    # A more generous budget converts most of those aborts into converged
    # fits; the cost is bounded because it only applies to the fits that
    # would otherwise have failed.
    'max_nfev': 20000,
    # Adaptive two-component escalation (see the module note above
    # two_component_escalation() in the source, and VERIFICATION.md section
    # 15): re-fit a pixel with a second Planck component ONLY when its
    # single-component temperature is pinned at T_max (T_at_bound), and keep
    # the escalated result only when escalation_significance() confirms it.
    # Every pixel that is not bound-pinned is completely unaffected. Set
    # False to reproduce results from before this existed.
    'escalate_bound_pinned_pixels': True,
    # Deliberately far from textbook alpha=0.05/any-improvement defaults -
    # see two_component_escalation()'s docstring for why: a single true
    # component can look "significantly" improved by a spurious second one
    # at loose gates, because the joint NLLS rarely converges to the exact
    # optimum and a second free Planck curve partly absorbs that residual.
    'escalation_p_value_max': 0.001,
    'escalation_min_rmse_reduction_frac': 0.5,
    'escalation_min_temperature_separation_K': 150.0,
    # Hand curve_fit the exact Jacobian of the mixture model instead of
    # letting it build one by finite differences. OFF by default, and it
    # must stay off for any published Amici et al. (2022) number: the
    # released code uses scipy's default, and although the analytic
    # derivative agrees with the finite difference to the step size
    # (2e-8 relative), the fit is a cold-start search over a multi-modal
    # surface whose result is averaged over 15 random trials, so a
    # different-but-equivalent descent path lands some trials in different
    # local minima. Measured over the published window on the t014 fire
    # pixels: single trials agree to a median 0.003 K, but 15-trial pixel
    # temperatures move by a median of 37 K (p90 136 K, max 174 K), while
    # the fit runs 2.9x faster. Use it for development turnaround, never
    # for a number that goes in the paper.
    'amici_analytic_jacobian': False,
}


def _load_thermal_partners(scene, acq_dt, download_dir, viirs_path=None,
                            viirs_auto_fetch=False, slstr_path=None,
                            slstr_auto_fetch=False, enmap_radiance_path=None,
                            max_time_diff_hours=12):
    """Resolve the VIIRS and Sentinel-3 SLSTR partners into the
    `extra_references` list validate_temperature() takes.

    Returns [] and warns rather than raising, in every failure mode. These
    two sensors are an ADDITION to the ECOSTRESS/Landsat pair the pipeline
    has always used, so nothing here may turn a working run into a failed
    one: a missing thermal_partners.py, an uninstalled h5py/netCDF4, absent
    credentials, an archive being down, or simply no granule in the window
    all end the same way - a warning, an empty list, and a pipeline that
    carries on exactly as it did before.

    SLSTR is a swath product and has to be resampled onto a regular grid
    before it can be sampled per pixel, so it needs the scene footprint;
    that comes from the EnMAP metadata when enmap_radiance_path is given,
    and otherwise from the scene's own lat/lon arrays."""
    if not (viirs_path or viirs_auto_fetch or slstr_path or slstr_auto_fetch):
        return []
    try:
        import thermal_partners as tpart
    except ImportError as exc:
        warnings.warn(f'VIIRS/SLSTR partners requested but thermal_partners.py could '
                      f'not be imported ({exc}); continuing without them.')
        return []

    lat, lon = scene['lat'], scene['lon']
    bbox = (float(np.nanmin(lon)), float(np.nanmin(lat)),
            float(np.nanmax(lon)), float(np.nanmax(lat)))
    polygon = None
    if enmap_radiance_path:
        try:
            ref = tpart.enmap_scene_reference(enmap_radiance_path,
                                              acquisition_datetime=acq_dt)
            bbox, polygon = ref['bbox'], ref['polygon']
        except Exception:
            pass          # the lat/lon bbox above is a fine fallback

    refs = []

    # ---- VIIRS ----
    if viirs_path is None and viirs_auto_fetch and acq_dt is not None:
        try:
            print(f'Searching NASA Earthdata for a VIIRS LST granule near '
                  f'{acq_dt.isoformat()} UTC...')
            found = tpart.search_viirs_lst(bbox, acq_dt,
                                           window_hours=max_time_diff_hours,
                                           scene_polygon=polygon)
            if found:
                viirs_path = tpart.download_viirs_lst(found[0], download_dir)
            else:
                print('No VIIRS LST granule found in the window.')
        except Exception as exc:
            warnings.warn(f'VIIRS auto-fetch failed ({exc.__class__.__name__}: {exc}); '
                           'continuing without VIIRS.')
            viirs_path = None
    if viirs_path:
        try:
            data = tpart.load_viirs_lst(viirs_path, bbox=bbox)
            refs.append({
                'label': 'VIIRS LST (%s)' % (data.get('platform') or 'VIIRS'),
                'data': data,
                # The composite's OWN observation time, decoded from its
                # View_Time layer - not the granule date, which is just the
                # day. See load_viirs_lst() for why that distinction is
                # worth this much trouble.
                'datetime': data.get('observation_utc_median'),
                'ceiling_K': data.get('ceiling_K', tpart.VIIRS_LST_CEILING_K),
                'column': 'viirs_LST_K',
            })
        except Exception as exc:
            warnings.warn(f'Could not load VIIRS LST from {viirs_path} '
                          f'({exc.__class__.__name__}: {exc}).')

    # ---- Sentinel-3 SLSTR ----
    if slstr_path is None and slstr_auto_fetch and acq_dt is not None:
        try:
            print(f'Searching Copernicus Data Space for a Sentinel-3 SLSTR LST '
                  f'product near {acq_dt.isoformat()} UTC...')
            found = tpart.search_slstr(bbox, acq_dt, window_hours=max_time_diff_hours,
                                       product_types=('SL_2_LST___',))
            if found:
                slstr_path = tpart.download_slstr(found[0], download_dir)
            else:
                print('No Sentinel-3 SLSTR LST product found in the window.')
        except Exception as exc:
            warnings.warn(f'Sentinel-3 auto-fetch failed ({exc.__class__.__name__}: {exc}); '
                           'continuing without SLSTR.')
            slstr_path = None
    if slstr_path:
        try:
            data = tpart.load_slstr_lst(slstr_path, target_bbox=bbox)
            refs.append({
                'label': 'Sentinel-3 SLSTR LST',
                'data': data,
                'datetime': data.get('observation_utc_median'),
                # Read from the file: SLSTR's int16 packing saturates at
                # ~355.5 K, BELOW ECOSTRESS's ~380 K, so using a shared
                # default here would under-report saturated comparisons.
                'ceiling_K': data.get('ceiling_K', 355.5),
                'column': 'slstr_LST_K',
            })
        except Exception as exc:
            warnings.warn(f'Could not load Sentinel-3 SLSTR from {slstr_path} '
                          f'({exc.__class__.__name__}: {exc}).')

    return refs


# The eight stages of run_pipeline(), in the order they run. Exposed so a
# front end (the GUI's workflow panel) can list them and mark them off as the
# progress hook reports each one.
PIPELINE_STAGES = (
    'Load the EnMAP scene',
    'Detect fire pixels (GLRT, this work) and build the background pool',
    'Reference data for validation (field measurements, thermal partners)',
    "Temperature retrieval, this work: profile-likelihood inversion with the scene "
    "fire-term transmittance",
    'Temperature retrieval, Amici et al. (2022): multi-trial Planck-mixture fit (reference method)',
    'Other published methods and optional corrections',
    'Validation against the reference data',
    'Outputs: temperature maps and figures',
)
PIPELINE_N_STAGES = len(PIPELINE_STAGES)
# Stage 2 names the detector that actually runs; PIPELINE_STAGES[1] above is
# the default ('glrt') title.
DETECTION_STAGE_TITLES = {
    'glrt': PIPELINE_STAGES[1],
    'robust': 'Detect fire pixels (two-part index test, robust MAD threshold) and build the '
              'background pool',
    'otsu': 'Detect fire pixels (two-part index test, Otsu threshold) and build the background pool',
}


def pipeline_stages(detection_method='glrt'):
    """PIPELINE_STAGES with the stage-2 title for `detection_method`."""
    stages = list(PIPELINE_STAGES)
    stages[1] = DETECTION_STAGE_TITLES.get(detection_method, stages[1])
    return stages
# Labels passed as the third argument of the progress hook during the two
# per-pixel loops, so a front end can say which retrieval is running.
PROGRESS_LABEL_PROFILE = "this work: profile-likelihood retrieval"
PROGRESS_LABEL_AMICI = 'Amici et al. (2022) multi-trial fit'


def _progress_reporter(progress_hook):
    """Wrap a caller's progress_hook into report(i, n, stage=None).

    Two hook signatures are supported. A two-argument hook, hook(i, n), is
    the original contract: it receives per-pixel progress only. A hook that
    accepts a third positional argument, hook(i, n, label), also receives
    the label of the loop that is running (PROGRESS_LABEL_*) and, with i and
    n counting stages, a 'Stage k/8: ...' string at every stage boundary, so
    a front end can drive a workflow display. The signature is inspected
    once here rather than probed with a TypeError at call time, so an error
    raised inside the hook is never mistaken for the two-argument form."""
    if progress_hook is None:
        return lambda i, n, stage=None: None
    import inspect
    try:
        params = list(inspect.signature(progress_hook).parameters.values())
        accepts_stage = (len(params) >= 3
                         or any(p.kind == p.VAR_POSITIONAL for p in params))
    except (TypeError, ValueError):
        accepts_stage = False

    def report(i, n, stage=None):
        if accepts_stage:
            progress_hook(i, n, stage)
        elif stage is None or not str(stage).startswith('Stage '):
            progress_hook(i, n)
    return report


def run_pipeline(enmap_radiance_path, landsat_lst_path=None, field_data_csv=None,
                  ecostress_path=None, ecostress_auto_fetch=False, ecostress_max_time_diff_hours=24,
                  landsat_auto_fetch=False, landsat_max_time_diff_hours=24,
                  viirs_path=None, viirs_auto_fetch=False,
                  slstr_path=None, slstr_auto_fetch=False,
                  partner_max_time_diff_hours=12,
                  other_detector_path=None, merge_vnir_swir=True, save_merged_cube=True,
                  results_dir='results_temperature', results_subdir_per_scene=True,
                  save_results=True,
                  extra_methods=(),
                  apply_profile_ml=True, profile_device=None, apply_profile_ml_v2=False,
                  profile_ml_variant='v2', apply_profile_ml_v1=False,
                  apply_smoke_correction=False, smoke_correction_half_window=5,
                  plot_overview=False, run_parallel=True,
                  hotspot_wl_lo=2060.0, hotspot_wl_hi=2430.0,
                  hotspot_margin_px=5, hotspot_min_cluster_size=1,
                  detection_method='glrt', detection_k=6.0,
                  require_radiance_anomaly=True, detection_radiance_k=6.0,
                  detection_p_fa=1e-5,
                  config=None, amici_analytic_jacobian=False,
                  progress_hook=None, cancel_event=None,
                  generate_figures=True, figure_dpi=300, figure_hot_threshold=600.0,
                  generate_report=True,
                  fire_partner_sources=(), fire_partner_files=(), firms_map_key=None,
                  fire_partner_window_hours=3.0, method_uncertainty=True, n_monte_carlo=100,
                  dennison_transmittance='template', coregister=False, coreg_max_days=20,
                  sampling='systematic'):
    """Run the full EnMAP fire-temperature pipeline end to end.

    Co-registration (coregistration.py): with coregister=True, stage 3 first
    co-registers the scene's geolocation to the nearest cloud-free HLS 30 m
    NIR image with AROSICS (local tie points, affine shift model) and every
    comparison with another sensor - LST sampled at the fire pixels, FRP
    detections matched to the fire, the multi-sensor maps - then uses the
    corrected positions; the LST is also sampled at the nominal positions
    (columns '*_LST_K_nominal') and an independent check (EnMAP -NDVI against
    each LST product over trial offsets) is reported for both. L1B is not
    orthorectified: on t014 the nominal positions are off by ~1 km. The
    retrieval itself does not use positions and is unchanged; the GeoTIFFs
    keep the product's own L1B affine.

    Uncertainty (uncertainty.py, VERIFICATION.md S30): this work's results
    carry FRP_MW with its 95% interval; with method_uncertainty=True the
    published methods that report no uncertainty (Dozier 1981, Dennison
    2006) get an `n_monte_carlo`-replicate Monte Carlo interval
    (T_mc_p2_5 / T_mc_p97_5 / T_mc_sd columns); stage 8 writes
    Uncertainty_<scene>_<date>.csv, the scene-level budget.
    dennison_transmittance: 'template' (default; the scene's one-way gas
    transmittance on the emitted endmembers, the analogue of their MODTRAN
    step) or 'unit' (no atmosphere on the emitted term; S31 sensitivity).

    This is the function both the CLI entry point (main(), below) and the
    GUI call - it takes every knob as a parameter instead of reading a
    hardcoded config block, precisely so it can be driven from something
    other than editing this file.

    The run is eight stages, in this order (PIPELINE_STAGES), each announced
    on stdout as "[k/8] ..." and to the progress hook:

      1  load the EnMAP scene (cube, wavelengths, geolocation, acquisition time)
      2  detect the fire pixels and build the background pool
      3  assemble the reference data: field measurements and thermal
         partners (Landsat, ECOSTRESS, VIIRS, Sentinel-3), loaded or fetched
      4  temperature retrieval, this work: L1B quality masks, scene fire-term
         transmittance, profile-likelihood inversion with a 95% interval
      5  temperature retrieval, Amici et al. (2022): the multi-trial
         Planck-mixture fit inherited from the PRISMA pipeline (reference)
      6  other published methods (Dozier 1981, Dennison et al. 2006) and the
         optional smoke-transmittance correction
      7  validation of both retrievals against the reference data
      8  outputs: temperature-map rasters and every figure

    Inputs are assembled before any fitting (stages 1-3) so a missing file or
    login fails in the first minute; the two retrievals follow, this work's
    first because it is the method this repository publishes; comparison,
    validation and outputs come last.

    Fire pixels are detected, by default (detection_method='glrt'), with
    this work's scene-wide generalised likelihood-ratio test
    (fire_detection.py): every band of the fit window, the retrieval's own
    forward model, a background subspace learned from the scene, a fit-
    adequacy gate and a false-alarm rate `detection_p_fa` per pixel
    calibrated on the scene itself. The published two-part index test is
    kept (detection_method='robust' or 'otsu') - the SWIR hotspot index
    (spectral shape, thresholded at median + detection_k*1.4826*MAD) AND an
    absolute SWIR radiance anomaly (magnitude, same form, detection_radiance_k);
    it also bootstraps the GLRT's fire-free pool.
    require_radiance_anomaly=True (the default) enforces both. Leave it on:
    the index is a ratio whose variance diverges as its denominator
    approaches the noise floor, so thresholded alone it preferentially
    selects the DARKEST pixels in a scene - measured on the two reference
    scenes, 70% and 87% of index-only detections fell below the scene's 1st
    percentile of SWIR radiance. See detect_fire_pixels() for the derivation
    and Section 13 of algorithm_verification/VERIFICATION.md for the full
    measurement. Setting it False reproduces the older index-only behaviour
    for comparison, and warns.

    progress_hook, if given, is called as progress_hook(i, n) after each
    fire pixel finishes (on top of the tqdm bar), so a GUI can drive its own
    progress display without scraping stdout. A hook that accepts a third
    positional argument, progress_hook(i, n, label), also receives the
    label of the retrieval that is running (PROGRESS_LABEL_PROFILE /
    PROGRESS_LABEL_AMICI) and a 'Stage k/8: ...' string at every stage
    boundary with i = k and n = 8 - see _progress_reporter().

    cancel_event, if given (any object with .is_set(), typically a
    threading.Event), lets a caller abort a running pipeline: it's checked
    at each major stage boundary and, during the per-pixel temperature
    retrieval loop, either every iteration (serial) or via the input
    generator so joblib simply stops dispatching new work and lets already-
    running tasks finish rather than killing worker processes (parallel -
    see the comment at the parallel branch for why). Raises
    PipelineCancelled when honored; no CSV/GeoTIFF/figures are written for
    a cancelled run.

    other_detector_path/merge_vnir_swir/save_merged_cube control combining
    a VNIR+SWIR detector pair into one cube - see load_enmap_radiance() for
    the full explanation; the short version is that pointing
    enmap_radiance_path at either the VNIR or SWIR file from a stock DLR
    EnMAP download is enough, the sibling is found and merged automatically
    by default, and the merged result is cached to disk next to the source
    files so it never needs to be re-merged on a later run of the same scene.

    generate_figures controls whether the publication-style PNGs (see
    generate_publication_figures()) are saved into results_dir - this
    happens independently of save_results, since the figures aren't the
    CSV/GeoTIFF numeric outputs that flag governs.

    enmap_radiance_path may point at an EnMAP *-METADATA.XML sidecar
    instead of a raster file directly - resolve_enmap_product_files() finds
    the VNIR/SWIR rasters next to it automatically, so there's no need to
    hunt down and pick the (much larger, oddly-named) *-SPECTRAL_IMAGE_
    VNIR/SWIR.TIF files yourself; a raster path still works exactly as
    before.

    ecostress_path, if given, points at an already-downloaded ECOSTRESS
    L2(T)_LSTE GeoTIFF (see load_ecostress_lst()) for the multi-sensor
    comparison figure plus bias/RMSE in the validated CSV.
    ecostress_auto_fetch=True instead tries to find and download a matching
    granule itself via download_ecostress_lst() - this needs `earthaccess`
    installed and a free NASA Earthdata login (see that function's
    docstring); a failure there is logged as a warning and the run
    continues without ECOSTRESS validation rather than aborting.

    landsat_lst_path/landsat_auto_fetch work the same way for a Landsat 8/9
    Collection-2 ST_B10 GeoTIFF, via download_landsat_lst() - unlike
    ECOSTRESS, this needs no account or login at all (it's served from
    Microsoft's free public Planetary Computer mirror of the USGS archive),
    so there's no equivalent credentials-missing failure mode to plan for;
    a search/network failure is still just a warning, not an abort.

    extra_methods names additional published retrievals to run in stage 6
    over the same fire pixels, from temperature_methods.METHOD_REGISTRY:
    'Dozier1981' (Dozier 1981), 'Dennison2006' (Dennison et al. 2006),
    'Amici2022' and 'ProfileML' (the two retrievals of stages 5 and 4 run
    again through the same driver, so the comparison isolates the
    estimator), or 'all'. Each writes its own
    `MethodCompare_<scene>_<Method><Year>_<date>.csv`, so every output file
    names the method that produced it, and together they produce the
    method-comparison figures. Running one of the others is a cheap
    independent check: Dennison et al. (2006) grids the temperature and
    solves the abundances linearly, so it cannot land in a local optimum,
    and its agreement with the joint fit bounds how much the nonlinear
    search is contributing. Adds runtime roughly proportional to the number
    of fire pixels, so it is off by default.

    sampling: 'systematic' (default since 15 Oct 2026) - this work's
    detector and retrieval take every subset (training pool, template
    sample, background libraries, split halves) as a fixed function of the
    scene (sampling.py), so they have no seed and a rerun on the same scene
    reproduces them bit for bit; 'random' - the earlier seeded draws
    (config['random_seed'] for the libraries, seed 0 for the detector), to
    reproduce runs before that date. The published methods keep their own
    random elements (Amici et al. random starts, Dennison et al. k-means,
    Monte Carlo uncertainty), seeded from config['random_seed'].

    profile_ml_variant selects this work's PRIMARY method, the one stages 5-8,
    the validation, the preview map and the report compare against: 'v2'
    (default since 13 Oct 2026; VERIFICATION.md S22-S23 - the reweighting
    iterated to its fixed point and the scene-level estimates made from
    adequately fitted pixels only; files ..._ProfileMLv2_...) or 'v1' (the
    chain as first published; files ..._ProfileML_...).
    apply_profile_ml_v1=True also runs the legacy v1 on the same pixels as a
    comparison when the primary is v2 (it reproduces earlier results);
    apply_profile_ml_v2=True does the converse when the primary is v1. The
    comparison variant writes its own files and figure and appears in the
    report; it does not enter stages 5-7.

    fire_partner_sources / fire_partner_files bring ACTIVE-FIRE products of
    other satellites (fire_partners.py) - the products that measure what this
    work also derives, fire radiative power (and, for GOES, a sub-pixel fire
    temperature and area): any of 'FIRMS' (VIIRS 375 m and MODIS 1 km
    detections; free MAP_KEY via firms_map_key or FIRMS_MAP_KEY), 'GOES' (ABI
    fire product, 5-10 min, public bucket, no login; the Americas only),
    'SLSTR' (Sentinel-3 SL_2_FRP, Copernicus Data Space login CDSE_USERNAME /
    CDSE_PASSWORD), plus product files already on disk (LSA SAF SEVIRI/FCI
    FRP-PIXEL, SLSTR .SEN3, GOES FDC .nc, FIRMS .csv). Polar-orbiter products
    within +/-fire_partner_window_hours, geostationary slots within +/-20
    min. Stage 7 compares their FRP with this work's (retrieved, and with
    the model-free lower bounds of the saturated pixels) and writes
    ActiveFire_<scene>.csv and FRPComparison_<scene>.csv; the report shows it.

    generate_report=True (the default, with save_results=True) typesets
    Report_<scene>.pdf at the end of stage 8 (scene_report.py): every
    method's statistics, temperature classes and model-derived FRP, maps,
    distributions, pairwise agreement, this work's uncertainty and fit
    diagnostics, the transmittance, the v1/v2 sensitivity, the reference
    data, the methods and caveats, every figure, and a reproducibility
    record (settings, versions, SHA-256 of every output). It also writes
    RunInfo_<scene>.json and ReportContext_<scene>.npz so the report can be
    rebuilt later from the folder alone (python scene_report.py <folder>).
    A report failure is a warning; the results are unaffected.

    apply_profile_ml=True (the default) runs stage 4, this work's retrieval
    (profile_pipeline.py): the L1B quality masks (enmap_quality.py), the
    scene fire-term transmittance estimated from the fire pixels themselves
    (fire_transmittance.py) and the profile-likelihood retrieval with a
    calibrated covariance, censored saturated bands and a 95% interval per
    pixel (profile_retrieval.py). It writes
    Results_<scene>_ProfileML_<date>.csv,
    Transmittance_<scene>_ProfileML_<date>.csv, T_map_<scene>_ProfileML.tif
    and, in stage 8, Fig_ProfileML_<scene>_<date>.png; stage 5 prints the
    median difference between the two retrievals and stage 7 validates both.
    Set False to skip it. profile_device says where its per-pixel
    retrievals run: None -> every CPU core (or the PROFILEML_DEVICE
    environment variable), 'joblib' (every core), 'serial' (one core, the
    reference path); the results are identical. On 16 cores the stage
    takes about two minutes for a 750-pixel fire front (it took an hour
    serially). See VERIFICATION.md Section 17.

    apply_smoke_correction=True runs temperature_methods.run_smoke_corrected_pixels()
    in stage 6 (this repository's candidate local-background /
    smoke-transmittance correction of the fire Planck term, VERIFICATION.md
    Section 16) over the same fire pixels, independently of extra_methods,
    writing `MethodCompare_<scene>_SmokeSTCR_<date>.csv`. Off by default: it
    is an unpublished candidate, not a peer-reviewed method, and Section
    16.8 found its three-gate significance test does not fire on either of
    the two scenes checked so far - it is offered as a diagnostic a user can
    opt into on a scene suspected of local smoke veiling, not as a routine
    correction. smoke_correction_half_window sets the local background
    neighbourhood (default 5, i.e. 11x11 pixels).

    Returns (results_df, scene, temp_map, figure_paths): the Amici et al.
    (2022) frame (validated when references were given), the scene, its
    temperature map and every figure written. This work's results travel on
    results_df.attrs: 'profile_ml' (the stage-4 dict with the results frame
    'df', the transmittance 'tau', the quality masks, the kernel and eps),
    'profile_ml_temp_map' and 'profile_ml_validated'. The other methods'
    results are written to disk and summarised on stdout.
    """
    # the call's own arguments, recorded in the run record of the report
    _run_args = {k: v for k, v in locals().items()
                 if isinstance(v, (str, int, float, bool, type(None))) and not k.startswith('_')}
    if cancel_event is not None and cancel_event.is_set():
        raise PipelineCancelled('Cancelled before starting.')
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    if amici_analytic_jacobian:
        # Development speed at the cost of reproducing the published
        # numbers - see DEFAULT_CONFIG['amici_analytic_jacobian'].
        cfg['amici_analytic_jacobian'] = True
        print('NOTE: the Amici et al. (2022) fits run with the analytic Jacobian '
              '(--fast-baseline). About 2.9x faster, but the retrieved temperatures '
              'are NOT those of the published configuration - do not report them.')

    if run_parallel and cfg['plot'] is True:
        # Popping matplotlib figures from inside worker processes doesn't
        # work, and even single-process it would open one figure per pixel.
        # config['plot'] is only meaningful for serial, single-pixel debugging.
        warnings.warn("config['plot']=True is ignored under run_parallel=True; "
                       "set run_parallel=False to inspect individual fits.")
        cfg['plot'] = False

    os.makedirs(results_dir, exist_ok=True)

    # Accepts a *-METADATA.XML sidecar as well as a raster path directly -
    # resolved here, once, so every downstream use of enmap_radiance_path
    # (loading, scene id, acquisition datetime) sees an actual raster file.
    enmap_radiance_path, resolved_sibling = resolve_enmap_product_files(
        enmap_radiance_path, other_detector_path=other_detector_path)
    other_detector_path = resolved_sibling

    # One output folder per EnMAP scene, inside results_dir. Every output
    # filename already carries the scene id, so a flat directory is
    # unambiguous - but it is not READABLE: a single scene writes a results
    # CSV, a validated CSV, up to three method-comparison CSVs, a GeoTIFF
    # and six or more figures, so three scenes put forty-odd files in one
    # listing with the distinguishing part of the name buried mid-string.
    # The folder is named by the acquisition (_scene_folder_id: no detector
    # suffix); the files inside keep scene_id, as before.
    scene_id = _short_scene_id(enmap_radiance_path)
    if results_subdir_per_scene:
        results_dir = os.path.join(results_dir, _scene_folder_id(enmap_radiance_path))
        os.makedirs(results_dir, exist_ok=True)
        print(f'Outputs for this scene go to {results_dir}')

    # Everything downloaded for this scene (HLS co-registration reference,
    # ECOSTRESS / Landsat / VIIRS / SLSTR thermal partners, GOES / SLSTR /
    # LSA SAF fire products) goes to ONE folder inside the scene folder, so a
    # scene's results and its inputs travel together and results_dir holds
    # nothing but scene folders. The cost: a partner granule shared by two
    # neighbouring scenes is downloaded once per scene.
    download_dir = os.path.join(results_dir, 'downloads')
    os.makedirs(download_dir, exist_ok=True)

    report = _progress_reporter(progress_hook)

    def _cancelled():
        return cancel_event is not None and cancel_event.is_set()

    # Wall-clock per stage, printed at the next stage boundary and at the end
    # of the run. A run is long enough (and the stages uneven enough) that
    # "which stage am I waiting for" is a question worth answering from the
    # log rather than by profiling the whole thing again.
    _stage_times = {}
    _stage_clock = [None, None]  # [stage number, start time]

    def _close_stage():
        k, t0 = _stage_clock
        if k is None:
            return
        dt = time.time() - t0
        _stage_times[k] = dt
        print(f'[{k}/{PIPELINE_N_STAGES}] done in {dt:.1f} s')

    def _stage(k, title):
        """Announce a stage on stdout and to the progress hook, and honour a
        pending cancellation at the boundary."""
        _close_stage()
        if _cancelled():
            raise PipelineCancelled(f'Cancelled before stage {k} ({title}).')
        print(f'\n[{k}/{PIPELINE_N_STAGES}] {title}')
        _stage_clock[0], _stage_clock[1] = k, time.time()
        report(k, PIPELINE_N_STAGES, f'Stage {k}/{PIPELINE_N_STAGES}: {title}')

    #%% ---- Stage 1: the scene ----
    _stage(1, PIPELINE_STAGES[0])
    print(f'Loading {enmap_radiance_path} ...')
    scene = load_enmap_radiance(enmap_radiance_path, other_detector_path=other_detector_path,
                                 merge_vnir_swir=merge_vnir_swir, save_merged_cube=save_merged_cube)
    cube, wavelength = scene['cube'], scene['wavelength']
    lat, lon = scene['lat'], scene['lon']
    print(f'Cube shape (rows, bands, cols): {cube.shape}, '
          f'{len(wavelength)} bands from {wavelength.min():.0f}-{wavelength.max():.0f} nm')
    # The acquisition time is a property of the scene, so it is read with the
    # scene: stage 3 needs it to find thermal partners and stage 8 stamps it
    # on the figures.
    acq_dt = _scene_acquisition_datetime(enmap_radiance_path)
    if acq_dt is not None:
        print(f'Acquisition time: {acq_dt.isoformat()} UTC')
    else:
        print('Acquisition time: not found (no METADATA.XML sidecar and no EnMAP filename '
              'timestamp); thermal-partner auto-fetch is not possible for this input.')

    #%% ---- Stage 2: fire pixels and background ----
    _stage(2, pipeline_stages(detection_method)[1])
    print('Computing SWIR hotspot index and auto-detecting fire pixels...')
    hotspot_index = compute_hotspot_index(cube, wavelength, hotspot_wl_lo, hotspot_wl_hi)

    # Two-part test: spectral SHAPE (the hotspot index) AND absolute
    # MAGNITUDE (total SWIR radiance). The index alone is a ratio whose
    # variance diverges as its denominator approaches the noise floor, so
    # on its own it preferentially selects the DARKEST pixels in the scene
    # - water, terrain shadow - rather than the hottest. See
    # detect_fire_pixels()'s docstring for the derivation and the measured
    # false-positive rates. Set require_radiance_anomaly=False only to
    # reproduce the old index-only behaviour.
    swir_radiance = radiance_threshold = None
    if require_radiance_anomaly:
        swir_radiance = swir_radiance_magnitude(cube, wavelength, hotspot_wl_lo, hotspot_wl_hi)
        radiance_threshold = robust_background_threshold(swir_radiance, k=detection_radiance_k)

    detection_info = None
    if detection_method == 'glrt':
        import fire_detection as _fdet
        band_mask = cfg_band_mask_for_methods(wavelength, cfg)
        fire_points, background_mask, detection_info = _fdet.detect_fire_pixels_glrt(
            cube, wavelength, band_mask, hotspot_index=hotspot_index,
            swir_radiance=swir_radiance if swir_radiance is not None else swir_radiance_magnitude(cube, wavelength, hotspot_wl_lo, hotspot_wl_hi),
            p_fa=detection_p_fa, margin_px=hotspot_margin_px, raster_path=enmap_radiance_path,
            bootstrap_k=detection_k, verbose=True, sampling=sampling,
            rng=np.random.default_rng(int(cfg.get('random_seed', 0))))
        threshold_used = float(detection_info['threshold'])
        if hotspot_min_cluster_size > 1 and len(fire_points):
            fm = np.zeros(hotspot_index.shape, bool); fm[fire_points[:, 0], fire_points[:, 1]] = True
            lab, n_lab = ndi.label(fm)
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            keep = sizes[lab[fire_points[:, 0], fire_points[:, 1]]] >= hotspot_min_cluster_size
            for key in ('point_class', 'point_lambda', 'point_T_best'):
                detection_info[key] = detection_info[key][keep]
            fire_points = fire_points[keep]
    else:
        fire_points, background_mask, threshold_used = detect_fire_pixels(
            hotspot_index, method=detection_method, k=detection_k,
            margin_px=hotspot_margin_px, min_cluster_size=hotspot_min_cluster_size,
            swir_radiance=swir_radiance, radiance_threshold=radiance_threshold)

    if detection_method == 'glrt':
        print(f'{len(fire_points)} fire pixels ({detection_info["n_fire"]} fire class, '
              f'{detection_info["n_candidate"]} candidates).')
    elif require_radiance_anomaly:
        index_only = int((np.nan_to_num(hotspot_index, nan=-np.inf) > threshold_used).sum())
        print(f'Hotspot-index threshold = {threshold_used:.4f} ({index_only} pixels); '
              f'SWIR radiance threshold = {radiance_threshold:.4g} '
              f'W m-2 sr-1 nm-1 at {hotspot_wl_lo:.0f}+{hotspot_wl_hi:.0f} nm.')
        print(f'{len(fire_points)} candidate fire pixels pass BOTH tests '
              f'({index_only - len(fire_points)} index-only pixels rejected as too dark to '
              'carry a fire signal).')
    else:
        print(f'Threshold = {threshold_used:.4f}; {len(fire_points)} candidate fire pixels found.')
        warnings.warn('require_radiance_anomaly=False: fire pixels were selected from the '
                      'hotspot index alone. That index is a ratio whose noise diverges in dark '
                      'pixels, so this detection typically includes a large fraction of shadow '
                      'and water pixels (measured: 70-87% on the reference scenes). Use it for '
                      'comparison only, not for a published retrieval.')

    if len(fire_points) == 0:
        raise ValueError('No fire pixels were detected automatically. Inspect '
                          'compute_hotspot_index() output, lower min_cluster_size, '
                          'or lower detection_k / detection_radiance_k (or raise detection_p_fa).')

    bkg_rows, bkg_cols = np.where(background_mask)
    bkg_pool = cube[bkg_rows, :, bkg_cols]
    print(f'{bkg_pool.shape[0]} candidate background pixels available.')
    if bkg_pool.shape[0] < cfg['N_bkg_points']:
        warnings.warn(
            f"Only {bkg_pool.shape[0]} background pixels are available, but "
            f"N_bkg_points={cfg['N_bkg_points']}: each trial will sample with "
            "replacement from a pool smaller than one draw, so the randomized "
            "background draws that the confidence interval relies on will "
            "contain duplicates. Consider lowering N_bkg_points or "
            "hotspot_margin_px, or check the scene has enough non-fire area.")

    fire_spectra = cube[fire_points[:, 0], :, fire_points[:, 1]]
    fire_spectra_2D = fire_spectra / np.nanmax(fire_spectra, axis=1, keepdims=True)
    n_fire = len(fire_points)
    band_mask = cfg_band_mask_for_methods(wavelength, cfg)
    today = date.today().strftime('%d-%m-%Y')

    #%% ---- Stage 3: reference data ----
    # Everything the run will be validated against is assembled BEFORE the
    # retrievals: a missing login, a failed download or an unreadable file
    # then shows up in the first minute, not after the per-pixel fits.
    _stage(3, PIPELINE_STAGES[2])
    # Co-registration first: everything below that compares positions with
    # another sensor reads scene['lat'] / scene['lon'], so correcting them
    # here carries through to the LST sampling, the FRP matching and the maps.
    coreg_info = None
    if coregister:
        try:
            import coregistration as _cg
            coreg_info = _cg.coregister_scene(scene, acq_dt, download_dir, max_days=coreg_max_days)
        except Exception as exc:
            coreg_info = {'status': f'not applied: {exc.__class__.__name__}: {exc}'}
        if coreg_info.get('status') == 'applied':
            scene['lat_nominal'], scene['lon_nominal'] = scene['lat'], scene['lon']
            scene['lat'], scene['lon'] = coreg_info['lat'], coreg_info['lon']
            lat, lon = scene['lat'], scene['lon']
        else:
            print(f"Co-registration {coreg_info['status']}; the comparisons use the L1B positions.")
    field_df = load_field_data(field_data_csv) if field_data_csv else None
    if field_df is not None:
        print(f'Field measurements: {len(field_df)} points from {field_data_csv}')

    if acq_dt is None and (ecostress_auto_fetch or landsat_auto_fetch or viirs_auto_fetch
                           or slstr_auto_fetch):
        print('Could not determine this scene\'s acquisition datetime (checked its '
              'METADATA.XML sidecar and EnMAP filename convention); skipping the '
              'thermal-partner auto-fetch. Pass ecostress_path=/landsat_lst_path=/... '
              'pointing at manually downloaded files instead.')

    if ecostress_path is None and ecostress_auto_fetch and acq_dt is not None:
        try:
            print(f'Searching NASA Earthdata for an ECOSTRESS LST granule near '
                  f'{acq_dt.isoformat()} UTC...')
            ecostress_path = download_ecostress_lst(
                scene, acq_dt, download_dir, max_time_diff_hours=ecostress_max_time_diff_hours)
        except Exception as exc:
            warnings.warn(f'ECOSTRESS auto-fetch failed ({exc.__class__.__name__}: {exc}); '
                           'continuing without ECOSTRESS validation.')
            ecostress_path = None

    if landsat_lst_path is None and landsat_auto_fetch and acq_dt is not None:
        try:
            print(f'Searching Planetary Computer for a Landsat 8/9 scene near '
                  f'{acq_dt.isoformat()} UTC...')
            landsat_lst_path = download_landsat_lst(
                scene, acq_dt, download_dir, max_time_diff_hours=landsat_max_time_diff_hours)
        except Exception as exc:
            warnings.warn(f'Landsat auto-fetch failed ({exc.__class__.__name__}: {exc}); '
                           'continuing without Landsat validation.')
            landsat_lst_path = None

    landsat = load_landsat_lst(landsat_lst_path) if landsat_lst_path else None
    ecostress = load_ecostress_lst(ecostress_path) if ecostress_path else None

    # VIIRS / Sentinel-3 SLSTR partners (thermal_partners.py). Kept in its
    # own helper, and imported lazily inside it, so that this module keeps
    # working unchanged when thermal_partners.py is absent or when its
    # optional dependencies (h5py, netCDF4) are not installed.
    extra_references = _load_thermal_partners(
        scene, acq_dt, download_dir,
        viirs_path=viirs_path, viirs_auto_fetch=viirs_auto_fetch,
        slstr_path=slstr_path, slstr_auto_fetch=slstr_auto_fetch,
        enmap_radiance_path=enmap_radiance_path,
        max_time_diff_hours=partner_max_time_diff_hours)

    # Independent check of the co-registration against data AROSICS never
    # saw: where does EnMAP's -NDVI best match each LST product, at the
    # nominal and at the corrected positions? (coregistration.py)
    if coreg_info is not None:
        try:
            import coregistration as _cg
            _proxy = _cg.proxy_neg_ndvi(scene)
            _checks = []
            _lst_refs = [(r.get('label', 'LST'), r.get('data')) for r in (extra_references or [])]
            if landsat is not None:
                _lst_refs.append(('Landsat', landsat))
            if ecostress is not None:
                _lst_refs.append(('ECOSTRESS', ecostress))
            for _label, _data in _lst_refs:
                if _data is None:
                    continue
                _c = {'label': _label,
                      'nominal': _cg.thermal_offset_scan(_proxy, scene.get('lat_nominal', scene['lat']),
                                                         scene.get('lon_nominal', scene['lon']), _data)}
                if 'lat_nominal' in scene:
                    _c['coregistered'] = _cg.thermal_offset_scan(_proxy, scene['lat'], scene['lon'], _data)
                if _c['nominal'] is not None:
                    _checks.append(_c)
                    _msg = (f"  {_label} LST against EnMAP -NDVI: offset EnMAP still needs "
                            f"{_c['nominal']['peak_dx_m']:+.0f} m E / {_c['nominal']['peak_dy_m']:+.0f} m N "
                            f"at the L1B positions")
                    if _c.get('coregistered'):
                        _msg += (f", {_c['coregistered']['peak_dx_m']:+.0f} / {_c['coregistered']['peak_dy_m']:+.0f}"
                                 f" m after co-registration (r {_c['nominal']['r_zero']:.3f} -> "
                                 f"{_c['coregistered']['r_zero']:.3f})")
                    print(_msg)
            coreg_info['thermal_check'] = _checks
            del _proxy
        except Exception as exc:
            warnings.warn(f'Co-registration check against the LST products failed '
                          f'({exc.__class__.__name__}: {exc}).')
        if save_results:
            try:
                import coregistration as _cg
                _cg.summary_table(coreg_info).to_csv(
                    os.path.join(results_dir, f'CoregSummary_{scene_id}_{today}.csv'), index=False)
                if coreg_info.get('tiepoints') is not None and len(coreg_info['tiepoints']):
                    coreg_info['tiepoints'].to_csv(
                        os.path.join(results_dir, f'CoregTiepoints_{scene_id}_{today}.csv'), index=False)
            except Exception as exc:
                warnings.warn(f'Could not write the co-registration tables ({exc}).')

    # Parsed from the reference filenames (the same helpers that label the
    # figure panels in stage 8) so validate_temperature() can report how far
    # apart the acquisitions actually are, and warn about day/night
    # mismatches, rather than printing a bias/RMSE that looks equally
    # authoritative whether the granule is 20 minutes or 13 days away.
    fire_partner_det, fire_partner_log = None, []
    if fire_partner_sources or fire_partner_files:
        try:
            import fire_partners as _fp
            _flat = scene['lat'][fire_points[:, 0], fire_points[:, 1]]
            _flon = scene['lon'][fire_points[:, 0], fire_points[:, 1]]
            _fbbox = (float(np.nanmin(_flon)) - 0.05, float(np.nanmin(_flat)) - 0.05,
                      float(np.nanmax(_flon)) + 0.05, float(np.nanmax(_flat)) + 0.05)
            if acq_dt is None and fire_partner_sources:
                print('Active-fire products: the acquisition time is unknown, so none can be searched; '
                      'files given explicitly are still read.')
            print('Active-fire products of other satellites (FRP), around the fire extent '
                  + ', '.join(f'{v:.3f}' for v in _fbbox) + ':')
            fire_partner_det, fire_partner_log = _fp.gather_fire_partners(
                _fbbox, acq_dt, sources=tuple(fire_partner_sources) if acq_dt is not None else (),
                files=tuple(fire_partner_files), cache_dir=download_dir, firms_map_key=firms_map_key,
                window_hours=fire_partner_window_hours)
            if save_results and fire_partner_det is not None and len(fire_partner_det):
                _p = os.path.join(results_dir, f'ActiveFire_{scene_id}_{date.today().strftime("%d-%m-%Y")}.csv')
                fire_partner_det.to_csv(_p, index=False)
                print(f'Saved {_p}')
        except Exception as exc:
            warnings.warn(f'Active-fire products failed ({exc.__class__.__name__}: {exc}); the run continues '
                          'without them.')

    ref_ecostress_dt = _ecostress_scene_datetime(ecostress_path) if ecostress_path else None
    if landsat_lst_path:
        # Fills in the overpass time (and caches it) when the file arrived
        # without one - see fetch_landsat_acquisition_time(). No-op if the
        # time is already known or there is no network.
        fetch_landsat_acquisition_time(landsat_lst_path)
    ref_landsat_dt = _landsat_scene_date(landsat_lst_path) if landsat_lst_path else None
    landsat_label = _landsat_platform_label(landsat_lst_path, default='Landsat')

    have_references = (landsat is not None or field_df is not None or ecostress is not None
                       or bool(extra_references))
    if have_references:
        found = [name for name, ok in (('field measurements', field_df is not None),
                                       (landsat_label, landsat is not None),
                                       ('ECOSTRESS', ecostress is not None))
                 if ok] + [r.get('label', 'LST') for r in (extra_references or [])]
        print('Reference data ready: ' + ', '.join(found))
    else:
        print('No reference data requested or found; the retrievals run without external '
              'validation (stage 7 will be skipped).')

    #%% ---- Stage 4: this work's retrieval ----
    # profile_pipeline.py: the L1B quality masks, the scene fire-term
    # transmittance estimated from the fire pixels themselves, and the
    # profile-likelihood retrieval with a calibrated covariance, censored
    # saturated bands and a 95% interval per pixel. It writes its own
    # Results_/Transmittance_/T_map_ files; its figure is drawn in stage 8
    # once the reference retrieval of stage 5 exists to compare against.
    _stage(4, PIPELINE_STAGES[3])
    profile_ml = None
    _profile_method_kwargs = None
    _variant_seconds = {}
    if profile_ml_variant not in ('v1', 'v2'):
        raise ValueError(f"profile_ml_variant must be 'v1' or 'v2', got {profile_ml_variant!r}")
    _alt_variant = None
    if profile_ml_variant == 'v2' and apply_profile_ml_v1:
        _alt_variant = 'v1'
    elif profile_ml_variant == 'v1' and apply_profile_ml_v2:
        _alt_variant = 'v2'
    _tag_of = {'v1': 'ProfileML', 'v2': 'ProfileMLv2'}
    _t_v = time.time()
    if apply_profile_ml:
        try:
            import profile_pipeline as _pp
            print(f"  This work, primary method: {profile_ml_variant}"
                  + (f" (the legacy {_alt_variant} also runs, for comparison)" if _alt_variant else ''))
            profile_ml = _pp.run_profile_ml_stage(
                cube, wavelength, fire_points, background_mask, bkg_pool,
                band_mask, cfg, scene, enmap_radiance_path,
                results_dir, scene_id, today, native_results=None,
                save_results=save_results, generate_figures=False, figure_dpi=figure_dpi,
                progress=lambda key, i, n: report(i, n, PROGRESS_LABEL_PROFILE),
                device=profile_device, variant=profile_ml_variant, sampling=sampling)
            if profile_ml is not None and profile_ml.get('tau_applied'):
                from profile_retrieval import column_median as _pr_cm
                _tau_fit = profile_ml['tau'][band_mask]
                _profile_method_kwargs = {'ProfileML': {
                    'tau': _tau_fit, 'snr': _pp.DEFAULT_SNR,
                    'reference_radiance': _pr_cm(bkg_pool, band_mask),
                    'rel_model_error': float(profile_ml['eps']) if profile_ml.get('eps') is not None else 0.05,
                    'corr_length_nm': 0.0}}
        except PipelineCancelled:
            raise
        except Exception as exc:
            profile_ml = None
            warnings.warn(f"This work's retrieval failed ({exc.__class__.__name__}: {exc}); "
                          'the remaining stages run on the reference method alone.')
    else:
        print("Skipped (apply_profile_ml=False); only the reference method of stage 5 is run.")
    _variant_seconds[_tag_of[profile_ml_variant]] = time.time() - _t_v
    profile_ml_alt = None
    _t_v = time.time()
    if apply_profile_ml and _alt_variant is not None:
        try:
            import profile_pipeline as _pp
            profile_ml_alt = _pp.run_profile_ml_stage(
                cube, wavelength, fire_points, background_mask, bkg_pool,
                band_mask, cfg, scene, enmap_radiance_path,
                results_dir, scene_id, today, native_results=None,
                save_results=save_results, generate_figures=False, figure_dpi=figure_dpi,
                progress=lambda key, i, n: report(i, n, PROGRESS_LABEL_PROFILE),
                device=profile_device, variant=_alt_variant, sampling=sampling)
        except PipelineCancelled:
            raise
        except Exception as exc:
            profile_ml_alt = None
            warnings.warn(f"This work ({_alt_variant}, comparison) failed ({exc.__class__.__name__}: {exc}); "
                          'the primary results are unaffected.')
        _variant_seconds[_tag_of[_alt_variant]] = time.time() - _t_v

    #%% ---- Stage 5: the reference retrieval, Amici et al. (2022) ----
    # The simultaneous multi-trial Planck-mixture fit inherited from the
    # PRISMA pipeline (main_temp_estimation). It is run on every scene: it
    # is the published method the new retrieval is compared against, and
    # the frame it produces carries the trial bookkeeping, the fire
    # significance test and the two-component escalation that the
    # publication figures and the validation of stage 7 are built on.
    _stage(5, PIPELINE_STAGES[4])
    # The published configuration, imposed on the scene configuration: the
    # reference retrieval is Amici et al. (2022) as published, not a variant
    # (see AMICI2022_PUBLISHED_CONFIG).
    amici_cfg = amici2022_config(cfg)
    _amici_mask = _swir_band_window(wavelength, amici_cfg, spectrum_norm=None, warn=False)
    print(f"Amici et al. (2022) configuration: {int(_amici_mask.sum())} channels in 1400-1900 nm, "
          f"m = {amici_cfg['N_bkg_points']} background spectra, N = {amici_cfg['N_trials']} trials, "
          f"T in [{amici_cfg['T_min']}, {amici_cfg['T_max']}] K, fire abundance >= "
          f"{amici_cfg['fire_abundance_min']}, Planck term zeroed below "
          f"{amici_cfg['absorption_blank_threshold']} W m-2 sr-1 nm-1, trf with tolerances 1e-8.")
    # tqdm's bar is disabled whenever a progress_hook is given (the GUI's
    # signal that it wants programmatic progress instead) - tqdm normally
    # overwrites its line with '\r', which a Tk Text widget renders as log
    # clutter rather than an in-place update.
    tqdm_disabled = progress_hook is not None

    if run_parallel:
        from joblib import Parallel, delayed
        print('Starting parallel temperature estimation...')
        t0 = time.time()
        pbar = tqdm(total=n_fire, disable=tqdm_disabled)

        def _on_batch_done(n_done, total):
            pbar.n = n_done
            pbar.refresh()
            report(n_done, total, PROGRESS_LABEL_AMICI)

        def _pixel_indices():
            # A plain cancel_event.set() from another thread can't safely
            # interrupt joblib's dispatch loop or kill its worker processes
            # mid-task - tried exactly that (shutting down the loky
            # executor from outside while Parallel() is running) and it
            # hangs instead of cancelling cleanly. Instead, this generator
            # just stops YIELDING new work once cancelled; joblib's own
            # pre-dispatch (2*n_jobs by default) means only a small,
            # bounded batch of already-in-flight tasks finishes naturally,
            # then Parallel() returns on its own with whatever results it
            # has - no killing, no hangs.
            for i in range(n_fire):
                if _cancelled():
                    return
                yield i

        with _joblib_progress(n_fire, _on_batch_done):
            all_results = Parallel(n_jobs=-1)(
                delayed(main_temp_estimation)(
                    i, wavelength, fire_spectra[i, :], fire_spectra_2D[i, :], bkg_pool, amici_cfg)
                for i in _pixel_indices()
            )
        pbar.close()
        print(f'Done in {time.time() - t0:.1f} s')
    else:
        print('Starting serial temperature estimation...')
        all_results = []
        for i in tqdm(range(n_fire), disable=tqdm_disabled):
            if _cancelled():
                break
            all_results.append(main_temp_estimation(
                i, wavelength, fire_spectra[i, :], fire_spectra_2D[i, :], bkg_pool, amici_cfg))
            report(i + 1, n_fire, PROGRESS_LABEL_AMICI)

    if _cancelled():
        raise PipelineCancelled(
            f'Cancelled after {len(all_results)}/{n_fire} fire pixels; no results saved.')

    results_pd = pd.DataFrame.from_dict(all_results)
    results_pd['ind_x'] = fire_points[:, 0]
    results_pd['ind_y'] = fire_points[:, 1]
    if detection_info is not None:
        results_pd['detection_class'] = np.where(detection_info['point_class'] == 1, 'fire', 'candidate')
        results_pd['detection_lambda'] = detection_info['point_lambda']
        results_pd['detection_T_grid_K'] = detection_info['point_T_best']
    results_pd = attach_pixel_geolocation(results_pd, scene)

    # main_temp_estimation() returns Kelvin, matching config['T_min']/['T_max']
    # and the Planck fit those came from - relabelled here with an explicit
    # _K suffix (rather than left as bare 'mean_temp' etc.) so the unit is
    # unambiguous in the saved CSV.
    rename_map = {
        'mean_temp': 'mean_temp_K',
        'median_temp': 'median_temp_K',
        'std_temp': 'std_temp_K',
        'CI_temp_low': 'CI_temp_low_K',
        'CI_temp_high': 'CI_temp_high_K',
        'temp_p2_5': 'temp_p2_5_K',
        'temp_p97_5': 'temp_p97_5_K',
    }
    for col in list(results_pd.columns):
        if col.startswith('mean_temp_src') or col.startswith('std_temp_src'):
            rename_map[col] = f'{col}_K'
    results_pd = results_pd.rename(columns=rename_map)

    # The method tag goes into every output filename so a results file
    # always says which published method produced it, and in which year -
    # otherwise files from different methods over the same scene are
    # indistinguishable once they leave this directory.
    # temperature_methods.METHOD_REGISTRY holds the other tags.
    method_tag = 'Amici2022'
    if save_results:
        results_csv = os.path.join(results_dir, f'Results_{scene_id}_{method_tag}_{today}.csv')
        results_pd.to_csv(results_csv, index=False)
        print(f'Saved {results_csv}')

    if profile_ml is not None and profile_ml.get('df') is not None:
        _pdf = profile_ml['df']
        _ok = (_pdf['fit_quality'].values == 'ok') & np.isfinite(_pdf['T_K'].values)
        _Tn = results_pd['median_temp_K'].values
        if len(_Tn) == len(_pdf) and _ok.sum() >= 2:
            _d = _pdf['T_K'].values[_ok] - _Tn[_ok]
            print(f"This work vs Amici et al. (2022) on the {int(_ok.sum())} adequately fitted "
                  f"pixels: median T {np.nanmedian(_pdf['T_K'].values[_ok]):.0f} K vs "
                  f"{np.nanmedian(_Tn[_ok]):.0f} K; median difference {np.nanmedian(_d):+.0f} K "
                  f"(quartiles {np.nanpercentile(_d, 25):+.0f}/{np.nanpercentile(_d, 75):+.0f} K)")

    #%% ---- Stage 6: other published methods, optional corrections ----
    # Each method writes its own file tagged with its name and year, so the
    # outputs sit side by side and can be compared directly. Running a
    # second, independently formulated method is a cheap check on the
    # first: agreement between Dennison et al. (2006) (gridded linear
    # search) and Amici et al. (2022) (joint nonlinear fit) is evidence
    # that both found the true optimum. The comparison figures are drawn in
    # stage 8 with everything else.
    _stage(6, PIPELINE_STAGES[5])
    method_results = {}
    bkg_spectrum = None
    _fig_method_kwargs = _profile_method_kwargs
    if extra_methods:
        import temperature_methods as _tm
        try:
            bkg_spectrum, bkg_info = _tm.background_from_histogram(
                cube, wavelength, hotspot_index=hotspot_index,
                wl_lo=hotspot_wl_lo, wl_hi=hotspot_wl_hi)
            print(f"Background for Dozier (1981) from the Otsu split of the hotspot-index "
                  f"histogram: threshold {bkg_info['otsu_threshold']:.4f}, "
                  f"{bkg_info['n_background_px']} px "
                  f"({100 * bkg_info['fraction_of_scene']:.1f}% of scene)")
        except Exception as exc:
            warnings.warn(f'Histogram background estimate failed ({exc}); '
                          'Dozier (1981) will fall back to the background-pool median.')
        # 'all' runs every registered method. Amici et al. (2022) is NOT
        # fitted a second time: the comparison uses the published-
        # configuration results of stage 5 (temperature_methods.
        # amici2022_comparison_frame). A second fit through the driver used
        # to cost ~90 s on 757 pixels, drew different random backgrounds,
        # and - cut to the scene window - lost the 1449 nm channel the paper
        # fits, so two "Amici et al. (2022)" results disagreed (median
        # |dT| 52 K on t014). ProfileML runs through the driver as a control
        # on the estimator (no transmittance chain, no quality masks).
        requested = (tuple(_tm.METHOD_REGISTRY)
                     if isinstance(extra_methods, str) and extra_methods.lower() == 'all'
                     else tuple(extra_methods))
        _precomputed = ({'Amici2022': _tm.amici2022_comparison_frame(results_pd)}
                        if 'Amici2022' in requested else None)
        print('Running methods: ' + ', '.join(_tm.method_display_name(k) for k in requested))
        # Dennison et al. (2006) model "only unsaturated channels": hand
        # their function the per-pixel saturated-channel masks from the L1B
        # quality layers when stage 4 loaded them - censored (DN at the
        # ceiling) or flagged by the processor (its saturation / defect
        # masks). Dozier (1981) gets the same mask (a clipped channel has
        # no solution) and, per pixel, the median of its neighbouring
        # background pixels as the background term.
        _per_pixel = {}
        if profile_ml is not None and profile_ml.get('quality') is not None:
            _q = profile_ml['quality']
            if _q.get('censored') is not None and len(_q['censored']) == len(fire_points):
                _sat = np.asarray(_q['censored'], bool)
                if _q.get('valid') is not None:
                    _sat = _sat | ~np.asarray(_q['valid'], bool)
                _per_pixel = {'Dennison2006': {'saturated': _sat}, 'Dozier1981': {'saturated': _sat}}
        _loc_pools = None
        if 'Dozier1981' in requested:
            _excl = ~np.asarray(background_mask, bool)
            _loc = []
            _loc_pools = []
            for _r, _c in np.asarray(fire_points):
                _lp = _tm.local_background_pool(cube, wavelength, int(_r), int(_c), _excl,
                                                half_window=5, min_pixels=10)
                _loc_pools.append(_lp if _lp is not None and len(_lp) >= 10 else None)
                _loc.append(np.median(_lp, axis=0) if _lp is not None and len(_lp) >= 10 else
                            (bkg_spectrum if bkg_spectrum is not None
                             else np.full(len(wavelength), np.nan)))
            _loc = np.asarray(_loc, float)
            # pixels with no local background use the scene background (the
            # driver's default) - pass only finite rows
            if np.all(np.isfinite(_loc)):
                _per_pixel.setdefault('Dozier1981', {})['background'] = _loc
        _method_kwargs = dict(_profile_method_kwargs or {})
        # Dennison et al. (2006): image-based path radiance, EAR-selected
        # class endmembers and one-way transmittance of the emitted
        # endmembers (temperature_methods.dennison_scene_inputs).
        dennison_inputs = None
        if 'Dennison2006' in requested:
            try:
                import enmap_quality as _eq
                _sz = _vz = None
                try:
                    _qf = _eq.find_quality_files(enmap_radiance_path)
                    if _qf.get('metadata'):
                        _sz, _vz = _eq.sun_view_zenith(_qf['metadata'])
                except OSError:
                    pass
                dennison_inputs = _tm.dennison_scene_inputs(bkg_pool, wavelength, _sz, _vz,
                                                            rng=np.random.default_rng(int(cfg.get('random_seed', 0))))
                _method_kwargs['Dennison2006'] = {
                    'subtract_spectrum': dennison_inputs['path_radiance'],
                    'fixed_endmembers': dennison_inputs['endmembers']}
                if dennison_transmittance == 'template':
                    _method_kwargs['Dennison2006']['transmittance'] = dennison_inputs['transmittance']
                elif dennison_transmittance != 'unit':
                    raise ValueError(f"dennison_transmittance must be 'template' or 'unit', "
                                     f"got {dennison_transmittance!r}")
                print(f"Dennison et al. (2006) inputs: {len(dennison_inputs['endmembers'])} EAR endmembers "
                      f"(class sizes {dennison_inputs['class_sizes'].tolist()}), path radiance from the darkest "
                      f"0.1% of the background, one-way transmittance share {dennison_inputs['one_way_share']:.2f}"
                      + (f" (sun zenith {_sz:.1f}, view zenith {_vz:.1f} deg)" if _sz is not None and _vz is not None
                         else ' (geometry not found: 0.5)'))
            except Exception as exc:
                warnings.warn(f'Dennison et al. (2006) scene inputs failed ({exc.__class__.__name__}: {exc}); '
                              'random background draws, no path radiance, unit transmittance.')
        _per_pixel = _per_pixel or None
        _fig_method_kwargs = _method_kwargs or None
        method_results = _tm.run_methods_on_pixels(
            cube, wavelength, fire_points, bkg_pool, methods=requested,
            band_mask=band_mask, background_spectrum=bkg_spectrum,
            emissivity=cfg['emissivity'], n_endmembers=cfg['N_bkg_points'],
            method_kwargs=_method_kwargs or None, per_pixel_kwargs=_per_pixel,
            precomputed=_precomputed)
        # Monte Carlo uncertainty of the methods that publish none
        # (uncertainty.method_monte_carlo); point estimates unchanged
        _mc_keys = [k for k in ('Dozier1981', 'Dennison2006') if k in method_results]
        if method_uncertainty and _mc_keys and len(fire_points):
            try:
                import uncertainty as _unc
                from profile_retrieval import estimate_band_noise as _ebn
                _rs = np.random.default_rng(7)
                _bp = np.atleast_2d(bkg_pool)
                _ps = np.asarray(_bp[np.sort(_rs.choice(_bp.shape[0], size=min(5000, _bp.shape[0]),
                                                        replace=False))], dtype=float)
                _pos = np.all(_ps > 0, axis=0) & np.all(np.isfinite(_ps), axis=0)
                _sig = np.zeros(len(wavelength))
                _sig[_pos] = _ebn(_ps[:, _pos])
                _ref = np.maximum(np.median(_ps, axis=0), 1e-12)
                for _k in _mc_keys:
                    _t0 = time.time()
                    _mc = _unc.method_monte_carlo(
                        _k, cube, wavelength, fire_points, _sig, _ref, n_mc=int(n_monte_carlo),
                        background=bkg_spectrum if bkg_spectrum is not None else np.median(_ps, axis=0),
                        local_backgrounds=_loc_pools if _k == 'Dozier1981' else None,
                        method_kwargs=_method_kwargs, per_pixel_kwargs=_per_pixel, band_mask=band_mask,
                        n_jobs=-1 if run_parallel else 1)
                    for _c in _mc.columns:
                        method_results[_k][_c] = _mc[_c].values
                    _w = (_mc['T_mc_p97_5'] - _mc['T_mc_p2_5']).values
                    print(f"  {_tm.method_display_name(_k)}: Monte Carlo uncertainty ({int(n_monte_carlo)} "
                          f"replicates of noise{' and neighbour background' if _k == 'Dozier1981' else ''}) - "
                          f"median 95% interval width {np.nanmedian(_w):.0f} K over "
                          f"{int(np.isfinite(_w).sum())} pixels ({time.time() - _t0:.0f} s)")
            except Exception as exc:
                warnings.warn(f'Monte Carlo uncertainty failed ({exc.__class__.__name__}: {exc})')
        for key, df in method_results.items():
            n_ok = int(df['converged'].sum())
            med = float(np.nanmedian(df['T_K'])) if n_ok else float('nan')
            print(f"  {_tm.method_display_name(key):>40}: {n_ok}/{len(df)} converged, "
                  f"median T = {med:.1f} K")
            # A temperature sitting on the edge of the method's own search
            # interval is a lower limit, not a retrieval: the data pulled the
            # solution past the end of the interval and the optimiser
            # returned the endpoint. Say so here rather than letting it pass
            # into a median or a bias unnoticed.
            n_bound = int(df.get('T_at_bound', pd.Series(dtype=bool)).sum())
            if n_bound:
                print(f"  {'':>40}  !! {n_bound} of {n_ok} sit at this method's "
                      "search bound - lower limits, not measurements; exclude them "
                      "from aggregates (column T_at_bound)")
            if save_results and not (_precomputed and key in _precomputed):
                # (stage 5 already wrote Results_<scene>_Amici2022_<date>.csv)
                # MethodCompare_, not Results_: the primary outputs of stages
                # 4 and 5 keep their full schemas under Results_; these
                # frames share the ten-column comparison schema and belong
                # with the Fig_Method*.png figures.
                path = os.path.join(results_dir,
                                    f'MethodCompare_{scene_id}_{key}_{today}.csv')
                df.to_csv(path, index=False)
                print(f'  Saved {path}')
    else:
        print('No additional published methods requested (extra_methods / --methods).')

    # Optional: local-background / smoke-transmittance correction. See
    # temperature_methods.py's module header above local_background_pool()
    # and VERIFICATION.md Section 16. Independent of extra_methods; off by
    # default, so every existing call and every previously published result
    # is reproduced unchanged.
    if apply_smoke_correction:
        import temperature_methods as _tm
        try:
            bkg_clear_smoke, _bkg_info_smoke = _tm.background_from_histogram(
                cube, wavelength, hotspot_index=hotspot_index,
                wl_lo=hotspot_wl_lo, wl_hi=hotspot_wl_hi)
        except Exception as exc:
            bkg_clear_smoke = None
            warnings.warn(f'Histogram background estimate failed ({exc}); the smoke '
                          'correction cannot run without a scene clear-sky reference.')
        if bkg_clear_smoke is not None:
            n_rows_smoke, _n_bands_smoke, n_cols_smoke = cube.shape
            hotspot_mask_smoke = np.zeros((n_rows_smoke, n_cols_smoke), dtype=bool)
            hotspot_mask_smoke[fire_points[:, 0], fire_points[:, 1]] = True
            print(f'Running the smoke-transmittance correction (candidate method of this '
                  f'repository, unpublished - see VERIFICATION.md Section 16) over '
                  f'{len(fire_points)} fire pixels...')
            try:
                smoke_df = _tm.run_smoke_corrected_pixels(
                    cube, wavelength, fire_points, hotspot_mask_smoke, bkg_clear_smoke,
                    band_mask, emissivity=cfg['emissivity'], n_endmembers=cfg['N_bkg_points'],
                    half_window=smoke_correction_half_window)
            except Exception as exc:
                smoke_df = None
                warnings.warn(f'Smoke correction failed ({exc.__class__.__name__}: {exc}); '
                              'other outputs are unaffected.')
            if smoke_df is not None:
                n_applied = int((~smoke_df['fallback']).sum())
                print(f"  Smoke correction: gate fired on {n_applied}/{len(smoke_df)} pixels "
                      f"(the rest fell back to the plain single-Planck fit - see "
                      "'fallback_reason' in the output CSV)")
                if n_applied:
                    corrected_T = smoke_df.loc[~smoke_df['fallback'], 'T_K'].values
                    print(f"     mean corrected T on those pixels: {np.nanmean(corrected_T):.1f} K")
                method_results['SmokeSTCR'] = smoke_df
                if save_results:
                    path = os.path.join(results_dir, f'MethodCompare_{scene_id}_SmokeSTCR_{today}.csv')
                    smoke_df.to_csv(path, index=False)
                    print(f'  Saved {path}')

    #%% ---- Stage 7: validation ----
    _stage(7, PIPELINE_STAGES[6])
    profile_validated = None
    if have_references:
        _val_kwargs = dict(landsat=landsat, field_df=field_df, ecostress=ecostress,
                           enmap_dt=acq_dt, landsat_dt=ref_landsat_dt,
                           ecostress_dt=ref_ecostress_dt, landsat_label=landsat_label,
                           extra_references=extra_references)
        print('Amici et al. (2022) retrieval against the references:')
        results_pd = validate_temperature(results_pd, **_val_kwargs)
        if save_results:
            results_pd.to_csv(os.path.join(
                results_dir, f'Results_{scene_id}_{method_tag}_validated_{today}.csv'), index=False)
        if profile_ml is not None and profile_ml.get('df') is not None:
            # The same references against this work's temperatures.
            # validate_temperature() reads the retrieved value from
            # 'mean_temp_K', so the profile frame is given that column.
            print("This work's retrieval against the references:")
            pv = profile_ml['df'].copy()
            pv['mean_temp_K'] = pv['T_K'].values
            pv = attach_pixel_geolocation(pv, scene)
            profile_validated = validate_temperature(pv, **_val_kwargs)
            if save_results:
                profile_validated.to_csv(os.path.join(
                    results_dir, f"Results_{scene_id}_{profile_ml.get('tag', 'ProfileML')}_validated_{today}.csv"),
                    index=False)
    else:
        print('Skipped: no reference data (see stage 3).')
    frp_comparison = None
    if (fire_partner_det is not None and len(fire_partner_det) and profile_ml is not None
            and profile_ml.get('df') is not None):
        try:
            import fire_partners as _fp
            _d = profile_ml['df']
            _en = _fp.enmap_frp_table(_d, scene['lat'][_d['ind_x'].values, _d['ind_y'].values],
                                      scene['lon'][_d['ind_x'].values, _d['ind_y'].values],
                                      emissivity=float(cfg.get('emissivity', 1.0)))
            _summ, _pairs = _fp.compare_frp(_en, fire_partner_det, acq_dt if acq_dt is not None else
                                            fire_partner_det['datetime_utc'].min())
            frp_comparison = {'summary': _summ, 'pairs': _pairs}
            if len(_summ):
                print("Fire radiative power, this work against the other satellites' products "
                      '(EnMAP: retrieved / with the saturated-pixel lower bounds):')
                for _, r in _summ.iterrows():
                    print(f"  {r['source']} {r['platform']} {r['datetime_utc']} (dt {r['dt_minutes']:+.0f} min): "
                          f"{int(r['n_detections_on_enmap_fire'])} detections on the EnMAP fire, "
                          f"{r['partner_frp_MW']:.0f} MW vs EnMAP {r['enmap_frp_retrieved_MW']:.0f} / "
                          f"{r['enmap_frp_with_bounds_MW']:.0f} MW")
                if save_results:
                    _summ.to_csv(os.path.join(results_dir, f'FRPComparison_{scene_id}_{today}.csv'), index=False)
                    _pairs.to_csv(os.path.join(results_dir, f'FRPPairs_{scene_id}_{today}.csv'), index=False)
        except Exception as exc:
            warnings.warn(f'FRP comparison failed ({exc.__class__.__name__}: {exc})')

    #%% ---- Stage 8: outputs ----
    _stage(8, PIPELINE_STAGES[7])
    n_rows, n_cols = cube.shape[0], cube.shape[2]
    temp_map = np.full((n_rows, n_cols), np.nan, dtype='float32')
    temp_map[results_pd['ind_x'].values, results_pd['ind_y'].values] = results_pd['mean_temp_K'].values

    if save_results:
        geoTIFF_creation(n_rows, n_cols, 1, temp_map[np.newaxis, :, :],
                          scene['geotransform'], scene['projection_wkt'],
                          'float32', os.path.join(results_dir, f'T_map_{scene_id}.tif'),
                          unit_type='K')

    # This work's temperature map, for callers that display a preview (the
    # GUI); its GeoTIFF was written by stage 4.
    profile_temp_map = None
    if profile_ml is not None and profile_ml.get('df') is not None:
        _pdf = profile_ml['df']
        _good = (_pdf['fit_quality'].values == 'ok') & np.isfinite(_pdf['T_K'].values)
        profile_temp_map = np.full((n_rows, n_cols), np.nan, dtype='float32')
        profile_temp_map[_pdf['ind_x'].values[_good], _pdf['ind_y'].values[_good]] = \
            _pdf['T_K'].values[_good]

    if plot_overview:
        fig, ax = plt.subplots(figsize=(10, 10))
        pcm = ax.pcolormesh(lon, lat, temp_map, cmap='inferno', shading='auto')
        fig.colorbar(pcm, ax=ax, label='Temperature (K)')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        plt.show()

    figure_paths = []
    if generate_figures:
        figure_paths = generate_publication_figures(
            results_pd, scene, temp_map, hotspot_index, threshold_used,
            fire_points, background_mask, wavelength, results_dir, scene_id,
            hotspot_wl_lo=hotspot_wl_lo, hotspot_wl_hi=hotspot_wl_hi,
            hot_threshold=figure_hot_threshold, acquisition_dt=acq_dt, dpi=figure_dpi,
            detection_info=detection_info)

        # Every thermal partner that was actually found, not just the
        # ECOSTRESS/Landsat pair.
        partner_sources = [{'label': r.get('label', 'LST'), 'data': r.get('data'),
                            'datetime': r.get('datetime')}
                           for r in (extra_references or [])]
        if ecostress is not None or landsat is not None or partner_sources:
            multi_sensor_path = os.path.join(results_dir, f'Fig4_multi_sensor_comparison_{scene_id}.png')
            plot_multi_sensor_comparison_figure(
                results_pd, scene, ecostress=ecostress, landsat=landsat,
                enmap_dt=acq_dt, ecostress_dt=ref_ecostress_dt, landsat_dt=ref_landsat_dt,
                landsat_label=_landsat_platform_label(landsat_lst_path),
                extra_sources=partner_sources,
                output_path=multi_sensor_path, dpi=figure_dpi)
            figure_paths.append(multi_sensor_path)
            print(f'Saved {multi_sensor_path}')

        # Fig5 (retrieved sub-pixel T vs. each thermal reference's
        # coarse-footprint LST) is not generated automatically: a 60-1000 m
        # footprint mean and a sub-pixel flame temperature are different
        # physical quantities, and the fitted "relationship" describes the
        # resolution mismatch rather than agreement. The helper is kept for
        # the comparable subset it separates out and can be called directly
        # as plot_reference_agreement_figure.

        if profile_ml is not None and profile_ml.get('df') is not None:
            try:
                import profile_pipeline as _pp
                pml_figs = _pp.profile_ml_figure(profile_ml, scene, results_pd, results_dir,
                                                 scene_id, today, dpi=figure_dpi)
                for p in pml_figs:
                    print(f'Saved {p}')
                figure_paths.extend(pml_figs)
            except Exception as exc:
                warnings.warn(f"This work's figure failed ({exc.__class__.__name__}: {exc}); "
                              'its CSVs and GeoTIFF were written in stage 4.')
        if profile_ml_alt is not None and profile_ml_alt.get('df') is not None:
            try:
                import profile_pipeline as _pp
                v2_figs = _pp.profile_ml_figure(profile_ml_alt, scene, results_pd, results_dir,
                                                scene_id, today, dpi=figure_dpi)
                for p in v2_figs:
                    print(f'Saved {p}')
                figure_paths.extend(v2_figs)
            except Exception as exc:
                warnings.warn(f"This work's comparison-variant figure failed ({exc.__class__.__name__}: {exc}).")

        if len(method_results) > 0:
            import temperature_methods as _tm
            try:
                method_figs = _tm.plot_method_comparison(
                    method_results, cube, wavelength, band_mask, bkg_pool,
                    bkg_spectrum, scene, results_dir, scene_id, today,
                    reference='Amici2022', emissivity=cfg['emissivity'],
                    n_endmembers=cfg['N_bkg_points'], dpi=figure_dpi,
                    method_kwargs=_fig_method_kwargs)
                for p in method_figs:
                    print(f'Saved {p}')
                figure_paths.extend(method_figs)
            except Exception as exc:
                warnings.warn(f'Method-comparison figures failed '
                              f'({exc.__class__.__name__}: {exc}); the results CSVs were still '
                              'written.')

    _close_stage()
    if _stage_times:
        _tot = sum(_stage_times.values())
        print('\nStage timings (s): ' + ', '.join(
            f'{k}={_stage_times[k]:.1f}' for k in sorted(_stage_times)) +
            f' | total {_tot:.1f} s')

    # The PDF report of the scene (scene_report.py), from the files written
    # above plus a run record and the context images it draws its maps on.
    report_path = None
    if generate_report and save_results:
        try:
            import scene_report as _sr
            methods_run = ['Amici2022']
            _pml = {}
            for _res in (profile_ml, profile_ml_alt):
                _tag = (_res or {}).get('tag')
                if _res is not None and _res.get('df') is not None:
                    methods_run.append(_tag)
                    _k = _res.get('kernel') or {}
                    _pml[_tag] = {'eps': _res.get('eps'), 'kernel_length_nm': _k.get('corr_length_nm'),
                                  'kernel_nugget': _k.get('nugget'), 'n_tau_pixels': _res.get('n_tau_pixels'),
                                  'scene_estimates': _res.get('scene_estimates', 'all'),
                                  'tau_applied': _res.get('tau_applied'),
                                  'seconds': _variant_seconds.get(_tag)}
            methods_run += [k for k in method_results if k not in methods_run]
            _sw = {'python': sys.version.split()[0], 'numpy': np.__version__, 'pandas': pd.__version__}
            for _mod in ('scipy', 'matplotlib', 'osgeo.gdal', 'numba', 'joblib'):
                try:
                    _m = __import__(_mod, fromlist=['_'])
                    _sw[_mod] = getattr(_m, '__version__', None) or getattr(_m, 'VersionInfo', lambda: '?')()
                except Exception:
                    pass
            try:
                import fcls_jit as _fj
                _sw['compiled FCLS continuation (fcls_jit)'] = 'active' if getattr(_fj, 'available', False) else 'not active (numba missing)'
            except Exception:
                _sw['compiled FCLS continuation (fcls_jit)'] = 'not available'
            _det = {}
            if detection_info is not None:
                _det = {'n_fire': detection_info.get('n_fire'), 'n_candidate': detection_info.get('n_candidate'),
                        'p_fa': detection_p_fa, 'method': detection_method}
            _settings = dict(_run_args)
            _settings.update({f'config.{k}': v for k, v in cfg.items()
                              if isinstance(v, (str, int, float, bool, type(None)))})
            _settings['input_raster'] = enmap_radiance_path
            _sr.write_run_info(results_dir, scene_id, {
                'scene_id': scene_id, 'date_tag': today, 'methods_run': methods_run,
                'primary': _tag_of[profile_ml_variant],
                'acquisition_utc': acq_dt.isoformat() if acq_dt is not None else None,
                'detection': _det, 'profile_ml': _pml, 'stage_seconds': _stage_times,
                'settings': _settings, 'software': _sw, 'emissivity': cfg.get('emissivity', 1.0),
                'fire_partner_log': [list(x) for x in fire_partner_log],
                'coregistration': (None if coreg_info is None else
                                   {k: coreg_info.get(k) for k in ('status', 'reference', 'model', 'crs_wkt',
                                                                   'n_grid_points', 'arosics_params',
                                                                   'thermal_check')})})
            _sr.write_report_context(scene, fire_points, results_dir, scene_id)
            report_path = _sr.build_scene_report(results_dir, scene_id=scene_id)
        except Exception as exc:
            warnings.warn(f'The PDF report failed ({exc.__class__.__name__}: {exc}); every result file '
                          'was written. Rebuild it with: python scene_report.py <results folder>')
    print('Done.')
    # Everything a caller might want beyond the four return values travels
    # on the frame's attrs (in memory only; to_csv() ignores it): this
    # work's results and temperature map, and the validated copy.
    results_pd.attrs['profile_ml'] = profile_ml
    results_pd.attrs['profile_ml_alt'] = profile_ml_alt
    results_pd.attrs['profile_ml_variant'] = profile_ml_variant
    results_pd.attrs['detection'] = (None if detection_info is None else
                                     {k: v for k, v in detection_info.items() if k != 'detector'})
    results_pd.attrs['profile_ml_temp_map'] = profile_temp_map
    results_pd.attrs['profile_ml_validated'] = profile_validated
    results_pd.attrs['stages'] = pipeline_stages(detection_method)
    results_pd.attrs['report_path'] = report_path
    results_pd.attrs['frp_comparison'] = frp_comparison
    results_pd.attrs['fire_partner_log'] = fire_partner_log
    results_pd.attrs['coregistration'] = (None if coreg_info is None else
                                          {k: v for k, v in coreg_info.items() if k not in ('lat', 'lon')})
    results_pd.attrs['results_dir'] = results_dir
    # stage 6: the other methods' per-pixel frames (only those that ran), for the GUI comparison
    results_pd.attrs['method_results'] = method_results
    return results_pd, scene, temp_map, figure_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('enmap_radiance_path', nargs='?',
                         default='EnMAP_data/ENMAP01_L1C_radiance.hdr',
                         help='Path to the EnMAP L1B/L1C radiance product: its *-METADATA.XML '
                              'sidecar (preferred - the VNIR and SWIR rasters are found next to '
                              'it automatically), or a raster file directly (.hdr, or either the '
                              'VNIR or SWIR file from a stock DLR EnMAP download - the sibling '
                              'detector file is found and merged automatically; see '
                              '--other-detector / --no-vnir-swir-merge).')
    parser.add_argument('--other-detector', dest='other_detector_path', default=None,
                         help='Explicit path to the second detector file (VNIR or SWIR) to merge '
                              'with enmap_radiance_path, overriding auto-detection.')
    parser.add_argument('--no-vnir-swir-merge', dest='merge_vnir_swir', action='store_false',
                         help='Process enmap_radiance_path alone, even if a matching VNIR/SWIR '
                              'sibling file sits next to it.')
    parser.add_argument('--no-save-merged-cube', dest='save_merged_cube', action='store_false',
                         help="Don't cache a VNIR+SWIR merge to disk (ENVI .dat/.hdr next to the "
                              'source files) for reuse on later runs.')
    parser.add_argument('--landsat', dest='landsat_lst_path', default=None,
                         help='Optional Landsat 8/9 Collection-2 L2 ST_B10 GeoTIFF for validation.')
    parser.add_argument('--field-data', dest='field_data_csv', default=None,
                         help='Optional CSV of field temperature measurements for validation.')
    parser.add_argument('--ecostress', dest='ecostress_path', default=None,
                         help='Optional ECOSTRESS L2(T)_LSTE GeoTIFF (from AppEEARS/LP DAAC) for '
                              'an EnMAP-vs-ECOSTRESS comparison figure and bias/RMSE.')
    parser.add_argument('--ecostress-auto-fetch', dest='ecostress_auto_fetch', action='store_true',
                         help='Search NASA Earthdata and download the closest-in-time ECOSTRESS '
                              'granule automatically, instead of passing --ecostress. Needs '
                              '`pip install earthaccess` and a free NASA Earthdata login.')
    parser.add_argument('--landsat-auto-fetch', dest='landsat_auto_fetch', action='store_true',
                         help='Search Microsoft Planetary Computer and download the closest-in-'
                              'time Landsat 8/9 scene automatically, instead of passing --landsat. '
                              'No account or login needed (a free public mirror of the USGS '
                              'archive).')
    parser.add_argument('--viirs', dest='viirs_path', default=None,
                         help='A VIIRS daily gridded LST granule (VNP21A1D/N, VJ121A1D/N, '
                              'VJ221A1D/N .h5) to validate against. Its true observation time '
                              'is decoded from the granule View_Time layer, not from its date.')
    parser.add_argument('--viirs-auto-fetch', dest='viirs_auto_fetch', action='store_true',
                         help='Search NASA Earthdata and download the closest VIIRS LST granule '
                              'automatically. Needs a free NASA Earthdata login. Unlike '
                              'ECOSTRESS, VIIRS covers every latitude, so this works above 52 N.')
    parser.add_argument('--slstr', dest='slstr_path', default=None,
                         help='A Sentinel-3 SLSTR SL_2_LST .SEN3 directory to validate against.')
    parser.add_argument('--slstr-auto-fetch', dest='slstr_auto_fetch', action='store_true',
                         help='Search Copernicus Data Space and download the closest Sentinel-3 '
                              'SLSTR LST product automatically. Needs CDSE_USERNAME/CDSE_PASSWORD '
                              '(free registration at https://dataspace.copernicus.eu).')
    parser.add_argument('--partner-window-hours', dest='partner_max_time_diff_hours',
                         type=float, default=12,
                         help='Search window for --viirs-auto-fetch/--slstr-auto-fetch '
                              '(default 12 h).')
    parser.add_argument('--detection', dest='detection_method', default='glrt',
                         choices=['glrt', 'robust', 'otsu'],
                         help="Fire-pixel detector: 'glrt' (default; this work's scene-wide "
                              "likelihood-ratio test with a calibrated false-alarm rate, "
                              "fire_detection.py) or the published two-part index test with a "
                              "'robust' (MAD) or 'otsu' threshold.")
    parser.add_argument('--p-fa', dest='detection_p_fa', type=float, default=1e-5,
                         help='Per-pixel false-alarm rate of the GLRT detector (default 1e-5, '
                              'about ten false alarms per EnMAP tile).')
    parser.add_argument('--no-radiance-gate', dest='require_radiance_anomaly',
                         action='store_false',
                         help='Detect fire pixels from the SWIR hotspot index alone, without also '
                              'requiring an absolute SWIR radiance anomaly. NOT recommended: the '
                              'index is a ratio whose noise diverges in dark pixels, so on its own '
                              'it selects mostly shadow/water (measured: 70-87%% of detections on '
                              'the reference scenes). Provided only to reproduce the older '
                              'behaviour for comparison.')
    parser.add_argument('--results-dir', default='results_temperature')
    parser.add_argument('--no-scene-subdir', dest='results_subdir_per_scene',
                         action='store_false',
                         help='Write this run\'s outputs straight into --results-dir '
                              'instead of into a per-scene subfolder named after the '
                              'scene. Downloaded partner granules are cached in '
                              '--results-dir either way.')
    parser.add_argument('--no-save', dest='save_results', action='store_false')
    parser.add_argument('--plot', dest='plot_overview', action='store_true')
    parser.add_argument('--serial', dest='run_parallel', action='store_false')
    parser.add_argument('--fast-baseline', dest='amici_analytic_jacobian',
                        action='store_true',
                        help='Fit the Amici et al. (2022) reference method with the exact '
                             "analytic Jacobian instead of scipy's finite difference: about "
                             '2.9x faster, and the rest of the run is unaffected. The fit is a '
                             'cold-start search over a multi-modal surface averaged over 15 '
                             'random trials, so an equivalent descent path lands some trials in '
                             'different local minima: pixel temperatures move by a median of '
                             '37 K. For development turnaround only - never for a published '
                             'number.')
    parser.add_argument('--no-figures', dest='generate_figures', action='store_false',
                         help='Skip generating the publication-style PNG figures.')
    parser.add_argument('--methods', default='',
                         help="Additional published retrievals to run over the same fire "
                              "pixels in stage 6, comma-separated, or 'all'. Available: "
                              "Dozier1981 (Dozier 1981), Dennison2006 (Dennison et al. 2006), "
                              "Amici2022 (Amici et al. 2022, the reference retrieval of stage 5 "
                              "run again through the comparison driver) and ProfileML (this "
                              "work's estimator on the same inputs as the others). Each writes "
                              "its own MethodCompare_<scene>_<Method><Year>_<date>.csv, and "
                              "together they produce the method-comparison figures (maps, "
                              "agreement, spectral fit).")
    parser.add_argument('--profile-device', dest='profile_device', default=None,
                         choices=('joblib', 'serial'),
                         help="Where stage 4's per-pixel retrievals run: 'joblib' (every CPU "
                              "core, the default) or 'serial' (one core). Same results on both; "
                              "the PROFILEML_DEVICE environment variable sets the default.")
    parser.add_argument('--fire-partners', dest='fire_partner_sources', default='',
                        help="Comma-separated active-fire products to fetch for the FRP comparison: "
                             "FIRMS (MAP_KEY), GOES (Americas, no login), SLSTR (CDSE login), LSASAF (SEVIRI, "
                             "LSA SAF login). Keys/logins: the ones remembered by the GUI (credentials.py), or "
                             "FIRMS_MAP_KEY, CDSE_USERNAME/CDSE_PASSWORD, LSASAF_USERNAME/LSASAF_PASSWORD.")
    parser.add_argument('--fire-partner-file', dest='fire_partner_files', action='append', default=[],
                        help='An active-fire product on disk (LSA SAF FRP-PIXEL .h5/.nc, SLSTR SL_2_FRP .SEN3, '
                             'GOES ABI-L2-FDC .nc, FIRMS .csv); repeatable.')
    parser.add_argument('--firms-map-key', dest='firms_map_key', default=None)
    parser.add_argument('--coregister', dest='coregister', action='store_true',
                        help='Co-register the scene to the nearest cloud-free HLS 30 m NIR image with AROSICS '
                             'before comparing it with other satellites (coregistration.py; needs an Earthdata '
                             'login and arosics, installed in the emit environment).')
    parser.add_argument('--coreg-max-days', dest='coreg_max_days', type=int, default=20,
                        help='Search window for the HLS reference, days either side (default 20).')
    parser.add_argument('--no-report', dest='generate_report', action='store_false',
                        help="Don't typeset the scene's PDF report (Report_<scene>.pdf, scene_report.py).")
    parser.add_argument('--sampling', dest='sampling', choices=('systematic', 'random'), default='systematic',
                        help="How this work's detector and retrieval take subsets: systematic (default; no "
                             'seed, bit-reproducible) or random (the seeded draws used before 15 Oct 2026).')
    parser.add_argument('--seed', dest='random_seed', type=int, default=None,
                        help="Base seed of the random elements that remain (published methods' random starts "
                             'and k-means, Monte Carlo uncertainty, and --sampling random). Default 0.')
    parser.add_argument('--profile-ml-variant', dest='profile_ml_variant', choices=('v1', 'v2'), default='v2',
                        help="This work's primary method: v2 (default; VERIFICATION.md S22-S23) or the "
                             'legacy v1 as first published.')
    parser.add_argument('--profile-ml-v1', dest='apply_profile_ml_v1', action='store_true',
                        help='Also run the legacy v1 for comparison when the primary is v2.')
    parser.add_argument('--profile-ml-v2', dest='apply_profile_ml_v2', action='store_true',
                        help="With --profile-ml-variant v1: also run v2 for comparison. (v2 is the "
                             "default primary method; this flag is kept for older scripts.)")
    parser.add_argument('--no-profile-ml', dest='apply_profile_ml', action='store_false',
                         help="Skip stage 4, this work's retrieval (profile-likelihood "
                              "inversion with the scene fire-term transmittance, L1B quality "
                              "masks and 95%% intervals; VERIFICATION.md Section 17), which "
                              "otherwise runs before the Amici et al. (2022) reference "
                              "retrieval and writes its own Results_/Transmittance_/T_map_/"
                              "Fig_ProfileML files.")
    parser.add_argument('--no-method-uncertainty', dest='method_uncertainty', action='store_false',
                        help='Skip the Monte Carlo uncertainty of Dozier (1981) and Dennison et al. (2006) '
                             '(uncertainty.py); their temperatures are the same either way.')
    parser.add_argument('--n-monte-carlo', dest='n_monte_carlo', type=int, default=100,
                        help='Monte Carlo replicates per pixel for that uncertainty (default 100).')
    parser.add_argument('--dennison-transmittance', dest='dennison_transmittance', choices=('template', 'unit'),
                        default='template',
                        help="Atmosphere on the Dennison et al. (2006) emitted endmembers: 'template' (the "
                             "scene's one-way gas transmittance, analogue of their MODTRAN step; default) or "
                             "'unit' (none). VERIFICATION.md S31.")
    parser.add_argument('--smoke-correction', dest='apply_smoke_correction',
                         action='store_true',
                         help="Also run this repository's candidate smoke-transmittance "
                              "correction (unpublished - local-background / smoke-transmittance "
                              "correction of the fire Planck term; VERIFICATION.md Section 16), "
                              "writing its own MethodCompare CSV. "
                              "Off by default: it is this repository's own candidate, gated by "
                              "three significance tests, and found not to fire on either of the "
                              "two reference scenes checked so far (Section 16.8) - offered as a "
                              "diagnostic for a scene suspected of local smoke veiling.")
    args = parser.parse_args()

    methods = args.methods.strip()
    extra_methods = ('all' if methods.lower() == 'all'
                     else tuple(m.strip() for m in methods.split(',') if m.strip()))

    run_pipeline(args.enmap_radiance_path, landsat_lst_path=args.landsat_lst_path,
                 require_radiance_anomaly=args.require_radiance_anomaly,
                 detection_method=args.detection_method, detection_p_fa=args.detection_p_fa,
                 field_data_csv=args.field_data_csv, ecostress_path=args.ecostress_path,
                 ecostress_auto_fetch=args.ecostress_auto_fetch,
                 landsat_auto_fetch=args.landsat_auto_fetch,
                 viirs_path=args.viirs_path, viirs_auto_fetch=args.viirs_auto_fetch,
                 slstr_path=args.slstr_path, slstr_auto_fetch=args.slstr_auto_fetch,
                 partner_max_time_diff_hours=args.partner_max_time_diff_hours,
                 other_detector_path=args.other_detector_path, merge_vnir_swir=args.merge_vnir_swir,
                 save_merged_cube=args.save_merged_cube,
                 results_dir=args.results_dir,
                 results_subdir_per_scene=args.results_subdir_per_scene,
                 save_results=args.save_results, plot_overview=args.plot_overview,
                 run_parallel=args.run_parallel, generate_figures=args.generate_figures,
                 extra_methods=extra_methods,
                 apply_profile_ml=args.apply_profile_ml, profile_device=args.profile_device,
                 apply_profile_ml_v2=args.apply_profile_ml_v2, generate_report=args.generate_report,
                 profile_ml_variant=args.profile_ml_variant, apply_profile_ml_v1=args.apply_profile_ml_v1,
                 fire_partner_sources=tuple(x.strip().upper() for x in args.fire_partner_sources.split(',') if x.strip()),
                 fire_partner_files=tuple(args.fire_partner_files), firms_map_key=args.firms_map_key,
                 apply_smoke_correction=args.apply_smoke_correction,
                 amici_analytic_jacobian=args.amici_analytic_jacobian,
                 method_uncertainty=args.method_uncertainty, n_monte_carlo=args.n_monte_carlo,
                 dennison_transmittance=args.dennison_transmittance,
                 coregister=args.coregister, coreg_max_days=args.coreg_max_days,
                 sampling=args.sampling,
                 config=None if args.random_seed is None else {'random_seed': args.random_seed})


if __name__ == '__main__':
    main()

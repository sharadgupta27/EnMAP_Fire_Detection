#!/usr/bin/env python
# coding: utf-8
"""Thermal sharpening: coarse land-surface temperature onto the EnMAP grid.

WHAT THIS IS FOR
----------------
The thermal partners available to an EnMAP scene are all coarser than
EnMAP's 30 m pixel - ECOSTRESS 70 m, Landsat TIRS 100 m, VIIRS and SLSTR
1 km. thermal_partners.py finds and loads them; this module puts one of
them onto the EnMAP grid so that a per-EnMAP-pixel background temperature
exists at all.

That background is not a curiosity, it is an input. The sub-pixel fire
retrieval in Fire_analysis_with_EnMAP.py and temperature_methods.py solves

    L_obs(lambda) = f * eps * B(lambda, T_fire) + (1 - f) * L_bkg(lambda)

and its background term is currently drawn from the scene's own
non-fire spectra. A physically measured background temperature at 30 m,
from a sensor that actually observes the thermal infrared, is a stronger
constraint and an independent check on the one the unmixing assumes.

THE METHOD, AND WHY THIS ONE
----------------------------
DisTrad (Kustas et al. 2003, RSE 85:429-440) and TsHARP (Agam et al. 2007,
RSE 107:545-558). Both rest on the same empirical observation: at a fixed
time and over a fixed region, land surface temperature is strongly and
monotonically related to vegetation cover - bare soil runs hot, dense
canopy runs cool, and the relation is tight enough to carry most of the
sub-pixel variance.

The procedure implemented in sharpen_thermal():

  1. Aggregate the fine-resolution predictor (from EnMAP) to the coarse
     thermal grid, by averaging every fine pixel falling inside each
     coarse cell. Averaging the PREDICTOR up rather than interpolating the
     TEMPERATURE down is the whole point - it is what keeps the regression
     honest, because both sides of the fit then describe the same ground.
  2. Fit T_coarse = f(predictor_coarse) across coarse cells.
  3. Evaluate f at fine resolution.
  4. Add the residual correction (below).

RESIDUAL CORRECTION AND EXACT CONSERVATION
------------------------------------------
A regression alone would throw away everything the thermal sensor measured
that vegetation cover does not explain. The residual step puts it back:

    p_i   = f(x_i)                        fine-scale prediction
    P_c   = mean over cell c of p_i       its own coarse-cell mean
    r_c   = T_c - P_c                     what the model missed
    s_i   = p_i + r_{c(i)}                sharpened result

Note that r_c is defined against the cell mean of the FINE predictions,
P_c, and not against f(mean predictor), which is the form usually written
down. The two agree only when f is linear. Using the cell mean of the fine
predictions makes the identity

    mean over cell c of s_i  ==  T_c

hold exactly for ANY f, linear or not. So the sharpened field never
invents energy: aggregated back to the thermal sensor's own grid it
reproduces that sensor's measurement to machine precision. That is checked
numerically in the self-test, and it is the property that makes the output
safe to compare against the original.

WHY FIRE PIXELS ARE EXCLUDED FROM THE FIT
-----------------------------------------
The temperature/vegetation relation the method depends on is a relation
about ambient surfaces heated by the sun. A burning pixel is not on that
curve - it is hot because it is combusting, not because it is bare - and a
handful of such cells will drag the regression slope badly, since they sit
at extreme temperature and arbitrary vegetation cover.

sharpen_thermal() therefore fits only on cells that pass `fit_mask`, and
`exclude_hot_K` drops any cell above a temperature threshold by default.
The resulting model is then applied EVERYWHERE, fire cells included. This
is deliberate and is the scientifically useful behaviour: over a fire cell
the sharpened field is the model's estimate of what the BACKGROUND would
be there, which is exactly the quantity the sub-pixel unmixing needs, and
the residual term still carries the cell's measured excess.

WHEN THIS METHOD DOES NOT WORK, AND HOW YOU ARE TOLD
----------------------------------------------------
Sharpening is not free. It helps only where the predictor genuinely
explains sub-pixel temperature, and that is a property of the SCENE, not
of the code.

Measured on the scene shipped with this repository - EnMAP DT0000099384,
2024-10-19, over the Mojave - sharpening ECOSTRESS, SLSTR and VIIRS onto
the 30 m grid and scoring each against Landsat 9's independent 30 m ST
gave pattern-RMSE changes of -4.3%, +1.6% and -33.3%. That is: on this
scene the method ranges from useless to actively harmful. The reason is
visible in the predictor - scene-median NDVI is -0.118 and median
fractional vegetation cover 0.10, because this is near-bare desert. The
temperature/vegetation relation the method depends on is not present, so
there is nothing for it to exploit.

Three findings are built into the code rather than left to the reader:

  1. In-sample R2 must not be used to decide. Of those three cases the
     VIIRS fit had the HIGHEST in-sample R2 (0.346 against 0.315 and
     0.067) and gave the WORST result by a wide margin. sharpen_thermal()
     therefore reports `cv_r2` from spatially-blocked cross-validation and
     gates on that; `min_cv_r2`/`on_low_skill` control what happens when it
     fails. Blocked rather than random folds because both fields are
     strongly autocorrelated - see _blocked_cv_r2().

  2. cv_r2 is NECESSARY BUT NOT SUFFICIENT, and this module says so rather
     than pretending otherwise. On the same three cases it scored -0.028,
     0.293 and 0.314: it correctly rejected SLSTR, but passed VIIRS at
     0.293 even though VIIRS went on to be 33% worse than not sharpening.
     The reason is structural, not a tuning problem. cv_r2 measures whether
     the predictor explains variation BETWEEN coarse cells; sharpening
     depends on the same relation holding WITHIN them, and no statistic
     computed at the coarse scale can settle that. (A scale-invariance test
     - refitting on 2x and 4x aggregations and comparing slopes - was tried
     and does not separate these cases either: VIIRS had the most stable
     slope of the three, at 4.4% spread.)

     So the gate is a floor, not a certificate. To actually establish that
     sharpening helped on a given scene, use validate_sharpening() against
     an independent fine-resolution thermal product - for EnMAP, Landsat
     8/9 ST at 30 m. That is the only test that answers the question, and
     it is shipped as a function for exactly that reason.

  3. Failure is graceful. With a predictor carrying no information the
     regression flattens, the residual term then supplies the whole answer,
     and the output returns to the nearest-neighbour coarse field rather
     than to noise. Confirmed numerically: three zero-R2 predictors (VNIR
     albedo, SWIR brightness, a SWIR band ratio) each reproduced the
     unsharpened correlation of 0.8164 to four decimal places. A useless
     predictor therefore costs nothing; only a spuriously correlated one
     does damage.

Where the relation IS present the method works as designed - against a
synthetic field with a known linear temperature/vegetation relation it
recovers the coefficients to 0.04% and cuts RMSE by 77.5%. Good candidate
scenes are vegetated and heterogeneous; arid scenes are not.

ONE CONFOUND, STATED PLAINLY
----------------------------
The -33.3% measured for VIIRS is not all sharpening error. VIIRS observed
this scene at ~21:00 UTC and the Landsat reference at 18:22 UTC, 2.6 h
apart, so part of that gap is genuine diurnal change in the surface rather
than a defect in the redistribution. This does not rescue the result - the
comparison still shows no demonstrated benefit - but any figure quoted from
it should carry the caveat, and validate_sharpening() reports the mean
offset separately so the size of the time effect stays visible.

WHAT THIS METHOD CANNOT DO
--------------------------
It cannot recover the fire itself. Sharpening redistributes a coarse
measurement using a fine-resolution correlate; it adds no thermal
information the coarse sensor did not have. A 1 km SLSTR cell containing a
30 m flaming front reports that front smeared over a million square metres
and saturated by its own encoding, and no amount of sharpening will
un-smear it. The sub-pixel fire temperature comes from the EnMAP SWIR
retrieval - that is the entire reason the retrieval exists. Use this module
for the background, and read the ceiling caveats in thermal_partners.py
before comparing anything against a flaming front.

Self-test:
    python algorithm_verification/test_thermal_sharpening.py
"""

import numpy as np

__all__ = [
    'aggregate_to_coarse_cells',
    'fractional_vegetation_cover',
    'ndvi_from_enmap',
    'sharpen_thermal',
    'validate_sharpening',
]


# --------------------------------------------------------------------------
# predictors
# --------------------------------------------------------------------------

def ndvi_from_enmap(cube, wavelength, red_nm=670.0, nir_nm=860.0,
                    is_reflectance=False, verbose=True):
    """NDVI from an EnMAP cube, using the bands nearest `red_nm`/`nir_nm`.

    `cube` is (rows, bands, cols) - the band-interleaved layout
    load_enmap_radiance() returns and that search_band() in the main module
    expects - and `wavelength` its band centres in nm. Band selection goes
    through that same search_band(), so this function and the rest of the
    pipeline can never disagree about which band is which.

    THE RADIANCE CAVEAT, WHICH MATTERS
    ----------------------------------
    EnMAP L1B is at-sensor RADIANCE, not reflectance, and an NDVI formed
    from radiance is not the NDVI of the literature. Extraterrestrial solar
    irradiance at 860 nm is roughly two thirds of its value at 670 nm, so a
    radiance ratio carries that solar-spectrum slope on top of the surface
    signal and comes out systematically lower than the reflectance NDVI of
    the same ground.

    For sharpening this is tolerable, because DisTrad/TsHARP need a
    predictor that varies monotonically with vegetation cover across the
    scene, not one on an absolute scale - the regression re-fits the
    scaling every time. It is NOT tolerable if you intend to quote the
    NDVI, compare it against another scene, or feed it to a threshold taken
    from the literature.

    So: pass L2A reflectance when you have it and set is_reflectance=True.
    With L1B radiance the function still works and warns once. Note also
    that fractional_vegetation_cover() normalises by scene percentiles,
    which absorbs most of this offset - which is why it is the recommended
    predictor here."""
    from Fire_analysis_with_EnMAP import search_band

    cube = np.asarray(cube)
    wavelength = np.asarray(wavelength, dtype=float)
    if cube.ndim != 3:
        raise ValueError('cube must be (rows, bands, cols), got %r' % (cube.shape,))
    if cube.shape[1] != wavelength.size:
        raise ValueError('cube has %d bands but %d wavelengths given'
                         % (cube.shape[1], wavelength.size))

    red_band, i_red = search_band(cube, wavelength, red_nm)
    nir_band, i_nir = search_band(cube, wavelength, nir_nm)
    if i_red == i_nir:
        raise ValueError('red and NIR resolve to the same band (%d) - check '
                         'the wavelength axis' % i_red)

    red = red_band.astype('float64')
    nir = nir_band.astype('float64')
    denom = nir + red
    with np.errstate(invalid='ignore', divide='ignore'):
        ndvi = np.where(np.abs(denom) > 0, (nir - red) / denom, np.nan)
    ndvi = np.where(np.isfinite(ndvi), np.clip(ndvi, -1.0, 1.0), np.nan)

    if verbose:
        print('[NDVI] red %.1f nm (band %d), NIR %.1f nm (band %d)'
              % (wavelength[i_red], i_red, wavelength[i_nir], i_nir))
        finite = np.isfinite(ndvi)
        if finite.any():
            print('[NDVI] range %.3f to %.3f (median %.3f)'
                  % (np.nanmin(ndvi), np.nanmax(ndvi), np.nanmedian(ndvi)))
        if not is_reflectance:
            print('[NDVI] NOTE: computed from at-sensor radiance, so this is a '
                  'relative index, not a reflectance NDVI - fine as a '
                  'sharpening predictor, not quotable as NDVI. See the '
                  'docstring.')
    return ndvi


def fractional_vegetation_cover(ndvi, ndvi_soil=None, ndvi_veg=None,
                                percentiles=(2.0, 98.0), verbose=True):
    """Fractional vegetation cover from NDVI, after Carlson & Ripley (1997):

        fc = ((NDVI - NDVI_soil) / (NDVI_veg - NDVI_soil)) ** 2

    This is the predictor TsHARP uses, and it is preferred over raw NDVI
    here for two reasons. It is more nearly linear against temperature than
    NDVI is (which is the empirical finding Agam et al. 2007 reported), and
    because its endpoints are taken from scene percentiles it absorbs the
    absolute-scale offset that an L1B radiance NDVI carries - see
    ndvi_from_enmap().

    The endpoints default to the 2nd and 98th percentiles of the finite
    NDVI rather than its min and max, so a handful of bad pixels, a cloud
    edge or a specular water return cannot set the scale for the whole
    scene. Pass ndvi_soil/ndvi_veg to fix them explicitly."""
    ndvi = np.asarray(ndvi, dtype='float64')
    finite = np.isfinite(ndvi)
    if not finite.any():
        raise ValueError('NDVI array has no finite values')
    if ndvi_soil is None:
        ndvi_soil = float(np.nanpercentile(ndvi, percentiles[0]))
    if ndvi_veg is None:
        ndvi_veg = float(np.nanpercentile(ndvi, percentiles[1]))
    if not np.isfinite(ndvi_veg - ndvi_soil) or abs(ndvi_veg - ndvi_soil) < 1e-9:
        raise ValueError('degenerate NDVI endpoints (soil=%r, veg=%r) - the '
                         'scene has no usable NDVI contrast to sharpen with'
                         % (ndvi_soil, ndvi_veg))
    fc = ((ndvi - ndvi_soil) / (ndvi_veg - ndvi_soil)) ** 2
    fc = np.where(finite, np.clip(fc, 0.0, 1.0), np.nan)
    if verbose:
        print('[fc] NDVI_soil=%.3f NDVI_veg=%.3f -> fc median %.3f'
              % (ndvi_soil, ndvi_veg, np.nanmedian(fc)))
    return fc


# --------------------------------------------------------------------------
# geometry: which coarse cell does each fine pixel fall in
# --------------------------------------------------------------------------

def _inverse_geotransform(gt):
    """Invert a GDAL geotransform. Raises on a degenerate one rather than
    returning garbage that would silently mis-place every pixel."""
    a, b, c, d, e, f = (float(v) for v in gt)
    det = b * f - c * e
    if abs(det) < 1e-30:
        raise ValueError('degenerate geotransform %r' % (gt,))
    return (a, d, f / det, -c / det, -e / det, b / det)


def _coarse_cell_index(lon, lat, coarse_gt, coarse_wkt, coarse_shape):
    """Map fine-pixel lon/lat to (row, col) in the coarse raster.

    Returns row/col arrays plus a validity mask. Out-of-raster pixels come
    back invalid rather than clipped to the edge - clipping would silently
    attribute ground outside the thermal scene to its border cells."""
    from pyproj import CRS, Transformer

    lon = np.asarray(lon, dtype='float64')
    lat = np.asarray(lat, dtype='float64')
    target = CRS.from_wkt(coarse_wkt) if isinstance(coarse_wkt, str) \
        else CRS.from_user_input(coarse_wkt)
    tr = Transformer.from_crs(CRS.from_epsg(4326), target, always_xy=True)
    x, y = tr.transform(lon, lat)

    inv = _inverse_geotransform(coarse_gt)
    ox, oy, ixx, ixy, iyx, iyy = inv
    dx, dy = x - ox, y - oy
    col = ixx * dx + ixy * dy
    row = iyx * dx + iyy * dy

    ny, nx = coarse_shape
    ci = np.floor(col).astype('int64')
    ri = np.floor(row).astype('int64')
    valid = np.isfinite(x) & np.isfinite(y) & (ci >= 0) & (ci < nx) & (ri >= 0) & (ri < ny)
    ci = np.where(valid, ci, 0)
    ri = np.where(valid, ri, 0)
    return ri, ci, valid


def aggregate_to_coarse_cells(values, cell_id, n_cells, min_count=1):
    """Mean of `values` within each coarse cell.

    `cell_id` is a flat cell index per fine pixel, `n_cells` the total.
    NaNs are ignored, and cells with fewer than `min_count` contributing
    fine pixels come back NaN - a coarse cell covered by three valid fine
    pixels out of several hundred is not a fair average of that cell and
    would otherwise enter the regression with full weight."""
    values = np.asarray(values, dtype='float64').ravel()
    cell_id = np.asarray(cell_id).ravel()
    good = np.isfinite(values) & (cell_id >= 0)
    total = np.bincount(cell_id[good], weights=values[good], minlength=n_cells)
    count = np.bincount(cell_id[good], minlength=n_cells).astype('float64')
    with np.errstate(invalid='ignore', divide='ignore'):
        mean = np.where(count >= max(1, min_count), total / count, np.nan)
    return mean, count


# --------------------------------------------------------------------------
# the regression
# --------------------------------------------------------------------------

def _design_matrix(predictors, model):
    """Columns of the least-squares design, including the intercept.

    predictors is (n_samples, n_predictors)."""
    p = np.atleast_2d(np.asarray(predictors, dtype='float64'))
    if p.shape[0] == 1 and p.shape[1] != 1:
        p = p.T
    cols = [np.ones(p.shape[0])]
    cols.extend(p[:, k] for k in range(p.shape[1]))
    if model == 'quadratic':
        # Quadratic in the FIRST predictor only. DisTrad fits T against a
        # second-order polynomial in NDVI; adding squares of every
        # predictor instead would multiply the free parameters without any
        # of that empirical support behind it.
        cols.append(p[:, 0] ** 2)
    elif model != 'linear':
        raise ValueError("model must be 'linear' or 'quadratic', got %r" % model)
    return np.column_stack(cols)


def _fit_ols(design, y, robust_iterations=0, verbose=True):
    """Least squares, optionally re-weighted to resist outliers.

    The robust option is Tukey bisquare IRLS. It is off by default: with
    fire cells already removed by `exclude_hot_K` the fit is usually clean,
    and a silent robust re-weighting can hide a genuinely bad predictor by
    quietly discarding the cells that disagree with it."""
    coeff, *_ = np.linalg.lstsq(design, y, rcond=None)
    weights = np.ones_like(y)
    for _ in range(int(robust_iterations)):
        resid = y - design @ coeff
        mad = np.median(np.abs(resid - np.median(resid)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale <= 0:
            break
        u = resid / (4.685 * scale)
        weights = np.where(np.abs(u) < 1.0, (1.0 - u ** 2) ** 2, 0.0)
        sw = np.sqrt(weights)
        coeff, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    fitted = design @ coeff
    resid = y - fitted
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    if verbose:
        print('[sharpen] regression on %d coarse cells: R2=%.3f, RMSE=%.2f K'
              % (len(y), r2, rmse))
    return coeff, r2, rmse, weights


def _blocked_cv_r2(design, y, rows, cols, n_folds=5, block=8):
    """Cross-validated R2 using SPATIALLY BLOCKED folds.

    This exists because in-sample R2 is not merely optimistic for this
    problem, it is actively misleading, and that was established here on
    real data rather than assumed. Sharpening ECOSTRESS, SLSTR and VIIRS
    onto one EnMAP scene gave in-sample R2 of 0.315, 0.067 and 0.346; the
    VIIRS fit had the HIGHEST R2 of the three and produced by far the WORST
    sharpened field, 33% worse than not sharpening at all. A criterion
    based on in-sample R2 would have recommended exactly the wrong choice.

    The folds are contiguous blocks of coarse cells rather than random
    cells. Land surface temperature and vegetation cover are both strongly
    spatially autocorrelated, so a randomly held-out cell almost always has
    a near neighbour in the training set carrying nearly the same
    information - random k-fold therefore measures interpolation between
    neighbours, not the model's ability to generalise, and reports skill
    that does not exist. Blocking by `block` x `block` cell tiles forces
    each fold to predict ground it has not seen.

    Returns NaN when there are too few distinct blocks to split, which is
    the honest answer rather than a fabricated score."""
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    fold = ((rows // block) + (cols // block)) % n_folds
    if np.unique(fold).size < 2:
        return np.nan
    pred = np.full(y.shape, np.nan)
    for k in np.unique(fold):
        train, test = fold != k, fold == k
        if train.sum() < design.shape[1] + 2 or test.sum() == 0:
            continue
        coeff, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
        pred[test] = design[test] @ coeff
    ok = np.isfinite(pred)
    if ok.sum() < 3:
        return np.nan
    ss_res = float(np.sum((y[ok] - pred[ok]) ** 2))
    ss_tot = float(np.sum((y[ok] - np.mean(y[ok])) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def sharpen_thermal(coarse, predictor, lon, lat, model='linear',
                    exclude_hot_K=330.0, fit_mask=None, min_fine_per_cell=4,
                    min_cells=25, robust_iterations=0, residual_correction=True,
                    min_cv_r2=0.0, on_low_skill='warn', cv_folds=5, cv_block=8,
                    verbose=True):
    """Sharpen a coarse LST field onto the fine grid described by
    `lon`/`lat`, using DisTrad/TsHARP with residual correction.

    Parameters
    ----------
    coarse : dict
        {'lst_K', 'geotransform', 'projection_wkt'} - anything
        thermal_partners.py or Fire_analysis_with_EnMAP.py loads.
    predictor : 2-D array, or sequence of 2-D arrays
        Fine-resolution predictor(s), same shape as `lon`/`lat`. Typically
        fractional_vegetation_cover(ndvi_from_enmap(...)). Several may be
        given for a multivariate fit - EnMAP offers far more than NDVI, and
        a second predictor such as a SWIR-based soil or moisture index
        often lifts R2 materially over greenness alone.
    lon, lat : 2-D arrays
        Geolocation of every fine pixel, in degrees. For EnMAP L1B these are
        scene['lon'] and scene['lat'] - the product is not map-projected, so
        a geotransform would not describe it.
    model : 'linear' or 'quadratic'
    exclude_hot_K : float or None
        Coarse cells at or above this temperature are kept out of the fit.
        The default of 330 K (~57 C) is above any ambient land surface the
        partners here retrieve reliably and below the temperature at which
        a coarse cell is only that hot because something in it is burning.
        Set None to disable, and see the module docstring for why this
        matters.
    fit_mask : 2-D bool array, optional
        Extra fine-resolution mask of pixels allowed to inform the fit -
        e.g. ~fire_mask from detect_fire_pixels(), or a cloud mask.
    min_fine_per_cell : int
        Coarse cells backed by fewer fine pixels than this are not used.
    min_cells : int
        Refuse to fit at all below this many usable cells.
    residual_correction : bool
        Leave True. See the module docstring - it is what makes the
        sharpened field aggregate back to the measurement exactly.
    min_cv_r2 : float
        Skill floor, judged by spatially-blocked cross-validation
        (_blocked_cv_r2), not by the in-sample R2. The default of 0.0 means
        "the predictor must beat simply using the cell mean". Below this
        the sharpened field is not trustworthy - see on_low_skill.
    on_low_skill : 'warn' | 'coarse' | 'raise'
        What to do when cv_r2 < min_cv_r2. 'warn' returns the sharpened
        field with skill_ok=False set in the result and a printed warning.
        'coarse' returns the coarse field resampled to the fine grid
        instead, which is the safe choice for an automated pipeline. 'raise'
        refuses outright.

    Returns
    -------
    dict with
        'lst_K'          sharpened temperature on the fine grid
        'prediction_K'   the regression part alone, before the residual
        'residual_K'     the residual added, on the fine grid
        'coarse_lst_K'   the coarse temperature resampled to the fine grid,
                         for a like-for-like before/after comparison
        'r2', 'rmse_K', 'coefficients', 'model', 'n_cells_fit'
        'cv_r2'          spatially-blocked cross-validated R2 - the number
                         to judge the result by, not 'r2'
        'skill_ok'       whether cv_r2 cleared min_cv_r2
        'fell_back_to_coarse'
        'cell_id'        flat coarse-cell index per fine pixel
        'conservation_max_error_K'
                         largest departure from exact conservation over any
                         fitted cell; ~1e-13 when residual_correction is on
    """
    lon = np.asarray(lon, dtype='float64')
    lat = np.asarray(lat, dtype='float64')
    if lon.shape != lat.shape:
        raise ValueError('lon and lat must match: %r vs %r' % (lon.shape, lat.shape))
    # Validated here rather than inside the low-skill branch below: a typo
    # would otherwise pass silently on every scene where the fit HAS skill and
    # only surface, much later, on one where it does not.
    if on_low_skill not in ('warn', 'coarse', 'raise'):
        raise ValueError("on_low_skill must be 'warn', 'coarse' or 'raise', "
                         'got %r' % on_low_skill)

    preds = [np.asarray(predictor, dtype='float64')] if np.ndim(predictor) == 2 \
        else [np.asarray(p, dtype='float64') for p in predictor]
    for p in preds:
        if p.shape != lon.shape:
            raise ValueError('predictor shape %r does not match lon/lat %r'
                             % (p.shape, lon.shape))

    coarse_lst = np.asarray(coarse['lst_K'], dtype='float64')
    ny, nx = coarse_lst.shape
    n_cells = ny * nx

    ri, ci, in_raster = _coarse_cell_index(lon, lat, coarse['geotransform'],
                                           coarse['projection_wkt'], (ny, nx))
    cell_id = np.where(in_raster, ri * nx + ci, -1)

    # coarse temperature seen by each fine pixel (nearest cell)
    flat_T = coarse_lst.ravel()
    coarse_at_fine = np.where(in_raster, flat_T[np.clip(cell_id, 0, n_cells - 1)], np.nan)

    finite_pred = np.ones(lon.shape, dtype=bool)
    for p in preds:
        finite_pred &= np.isfinite(p)
    usable = in_raster & finite_pred
    if fit_mask is not None:
        fit_mask = np.asarray(fit_mask, dtype=bool)
        if fit_mask.shape != lon.shape:
            raise ValueError('fit_mask shape %r does not match lon/lat %r'
                             % (fit_mask.shape, lon.shape))

    # ---- aggregate predictors to the coarse grid ----
    fit_cell = np.where(usable, cell_id, -1)
    if fit_mask is not None:
        fit_cell = np.where(fit_mask, fit_cell, -1)
    coarse_preds, counts = [], None
    for p in preds:
        mean_p, counts = aggregate_to_coarse_cells(
            np.where(usable, p, np.nan), fit_cell, n_cells,
            min_count=min_fine_per_cell)
        coarse_preds.append(mean_p)

    good_cell = np.isfinite(flat_T) & (counts >= min_fine_per_cell)
    for mp in coarse_preds:
        good_cell &= np.isfinite(mp)
    if exclude_hot_K is not None:
        n_hot = int(np.sum(good_cell & (flat_T >= exclude_hot_K)))
        good_cell &= (flat_T < exclude_hot_K)
        if verbose and n_hot:
            print('[sharpen] %d coarse cell(s) at/above %.0f K excluded from the '
                  'fit (kept in the output)' % (n_hot, exclude_hot_K))

    n_fit = int(good_cell.sum())
    if n_fit < min_cells:
        raise ValueError(
            'only %d usable coarse cells (need %d). The thermal granule and the '
            'EnMAP scene may barely overlap, the predictor may be empty, or '
            'min_fine_per_cell=%d may be too strict for this resolution ratio.'
            % (n_fit, min_cells, min_fine_per_cell))

    X = _design_matrix(np.column_stack([mp[good_cell] for mp in coarse_preds]), model)
    y = flat_T[good_cell]
    coeff, r2, rmse, _ = _fit_ols(X, y, robust_iterations, verbose=verbose)

    cell_rows, cell_cols = np.divmod(np.flatnonzero(good_cell), nx)
    cv_r2 = _blocked_cv_r2(X, y, cell_rows, cell_cols,
                           n_folds=cv_folds, block=cv_block)
    skill_ok = bool(np.isfinite(cv_r2) and cv_r2 >= min_cv_r2)
    if verbose:
        print('[sharpen] blocked cross-validated R2 = %s (floor %.2f) -> %s'
              % ('%.3f' % cv_r2 if np.isfinite(cv_r2) else 'n/a', min_cv_r2,
                 'usable' if skill_ok else 'NOT usable'))
    if not skill_ok:
        message = (
            'Thermal sharpening has no demonstrable skill on this scene: '
            'blocked cross-validated R2 = %s against a floor of %.2f (in-sample '
            'R2 = %.3f). The predictor does not generalise, so the sharpened '
            'field would redistribute the coarse measurement using a relation '
            'that is not there. This is the expected outcome over arid or '
            'sparsely vegetated ground, where the temperature/vegetation '
            'relation DisTrad and TsHARP rely on largely collapses.'
            % ('%.3f' % cv_r2 if np.isfinite(cv_r2) else 'undefined',
               min_cv_r2, r2))
        if on_low_skill == 'raise':
            raise ValueError(message)
        import warnings as _w
        _w.warn(message)
        if verbose:
            print('[sharpen] %s' % message)

    # ---- evaluate at fine resolution ----
    fine_stack = np.column_stack([np.where(usable, p, np.nan).ravel() for p in preds])
    ok_fine = np.all(np.isfinite(fine_stack), axis=1)
    prediction = np.full(fine_stack.shape[0], np.nan)
    if ok_fine.any():
        Xf = _design_matrix(fine_stack[ok_fine], model)
        prediction[ok_fine] = Xf @ coeff

    residual_fine = np.zeros_like(prediction)
    conservation = np.nan
    if residual_correction:
        # r_c = T_c - (cell mean of the FINE predictions). Defined this way
        # so that conservation is exact for any model - see the module
        # docstring.
        pred_cell_mean, pred_count = aggregate_to_coarse_cells(
            prediction, np.where(ok_fine, cell_id.ravel(), -1), n_cells, min_count=1)
        r_cell = np.where(np.isfinite(pred_cell_mean) & np.isfinite(flat_T),
                          flat_T - pred_cell_mean, np.nan)
        idx = np.clip(cell_id.ravel(), 0, n_cells - 1)
        residual_fine = np.where(cell_id.ravel() >= 0, r_cell[idx], np.nan)
        residual_fine = np.where(np.isfinite(residual_fine), residual_fine, 0.0)

    sharp = prediction + residual_fine

    if residual_correction:
        # verify the identity rather than assert it in a comment
        back, back_n = aggregate_to_coarse_cells(
            sharp, np.where(np.isfinite(sharp), cell_id.ravel(), -1), n_cells, min_count=1)
        check = np.isfinite(back) & np.isfinite(flat_T) & (back_n >= 1) & good_cell
        conservation = float(np.max(np.abs(back[check] - flat_T[check]))) if check.any() else np.nan

    shape = lon.shape
    if not skill_ok and on_low_skill == 'coarse':
        # Fall back to the measurement itself. Note this is not a silent
        # substitution: skill_ok=False and fell_back_to_coarse=True are
        # both in the returned dict, and a warning has already been raised.
        sharp = coarse_at_fine.copy()
        prediction = np.full_like(sharp, np.nan)
        residual_fine = np.zeros_like(sharp)

    out = {
        'lst_K': sharp.reshape(shape),
        'prediction_K': prediction.reshape(shape),
        'residual_K': residual_fine.reshape(shape),
        'coarse_lst_K': coarse_at_fine.reshape(shape),
        'cell_id': cell_id.reshape(shape),
        'r2': r2, 'rmse_K': rmse, 'coefficients': coeff, 'model': model,
        'n_cells_fit': n_fit,
        'conservation_max_error_K': conservation,
        'cv_r2': cv_r2, 'skill_ok': skill_ok,
        'fell_back_to_coarse': bool(not skill_ok and on_low_skill == 'coarse'),
    }
    if verbose:
        finite = np.isfinite(out['lst_K'])
        if finite.any():
            print('[sharpen] output %d valid fine pixels, %.1f-%.1f K'
                  % (int(finite.sum()), float(np.nanmin(out['lst_K'])),
                     float(np.nanmax(out['lst_K']))))
        if residual_correction and np.isfinite(conservation):
            print('[sharpen] conservation check: max |aggregated - measured| '
                  '= %.3e K' % conservation)
    return out


def validate_sharpening(result, reference, lon, lat, reference_label='reference',
                        remove_mean=True, verbose=True):
    """Score a sharpen_thermal() result against an INDEPENDENT fine-
    resolution thermal reference, and say whether sharpening actually
    helped on this scene.

    This function exists because the diagnostics available at fit time -
    in-sample R2, and even the blocked cross-validated cv_r2 that
    sharpen_thermal() gates on - are necessary but NOT sufficient. Measured
    on the Mojave scene in this repository, sharpening VIIRS cleared the
    cv_r2 gate at 0.293 and still produced a field 33% worse than not
    sharpening. cv_r2 asks whether the predictor explains variation BETWEEN
    coarse cells; sharpening lives or dies on whether the same relation
    holds WITHIN them, which no coarse-scale statistic can settle. Only an
    independent fine-resolution measurement can, and that is what this
    checks.

    Parameters
    ----------
    result : dict from sharpen_thermal()
    reference : dict with 'lst_K'/'geotransform'/'projection_wkt' - an
        independent thermal product at or near the fine resolution.
        Landsat 8/9 Collection-2 ST at 30 m is the natural choice for
        EnMAP. It must NOT be the product that was sharpened.
    lon, lat : the same fine-grid geolocation passed to sharpen_thermal()

    remove_mean : bool
        Compare spatial PATTERN rather than absolute level, by removing the
        mean of each field first. Leave this True whenever the reference
        was acquired at a different time from the sharpened product, which
        is almost always. A one-hour offset around local noon moves land
        surface temperature by several kelvin uniformly across a scene;
        that offset is real, it is not a sharpening error, and leaving it
        in buries the sub-pixel signal this is meant to measure. The mean
        difference is still reported, just not scored.

    Returns a dict of correlation and pattern-RMSE for the sharpened field
    and for the unsharpened coarse field, plus 'improvement_pct' and a
    boolean 'sharpening_helped'. A negative improvement means the coarse
    field was closer to the reference and the sharpening should be
    discarded for this scene - which is a legitimate and not-uncommon
    outcome over arid ground; see the module docstring."""
    from Fire_analysis_with_EnMAP import _sample_raster_at_lonlat

    lon = np.asarray(lon, dtype='float64')
    lat = np.asarray(lat, dtype='float64')
    ref = _sample_raster_at_lonlat(reference['lst_K'], reference['geotransform'],
                                   reference['projection_wkt'],
                                   lon.ravel(), lat.ravel()).reshape(lon.shape)

    sharp = np.asarray(result['lst_K'], dtype='float64')
    coarse = np.asarray(result['coarse_lst_K'], dtype='float64')
    ok = np.isfinite(sharp) & np.isfinite(coarse) & np.isfinite(ref)
    n = int(ok.sum())
    if n < 100:
        raise ValueError('only %d pixels where the sharpened field, the coarse '
                         'field and the reference are all valid - too few to '
                         'score. Do they overlap?' % n)

    s, c, r = sharp[ok], coarse[ok], ref[ok]
    mean_offset = float(s.mean() - r.mean())
    if remove_mean:
        s, c, r = s - s.mean(), c - c.mean(), r - r.mean()

    def _score(a):
        corr = float(np.corrcoef(a, r)[0, 1])
        rmse = float(np.sqrt(np.mean((a - r) ** 2)))
        return corr, rmse

    corr_s, rmse_s = _score(s)
    corr_c, rmse_c = _score(c)
    improvement = 100.0 * (1.0 - rmse_s / rmse_c) if rmse_c > 0 else np.nan
    helped = bool(np.isfinite(improvement) and improvement > 0)

    out = {
        'n_pixels': n,
        'reference_label': reference_label,
        'mean_offset_K': mean_offset,
        'corr_sharpened': corr_s, 'rmse_sharpened_K': rmse_s,
        'corr_coarse': corr_c, 'rmse_coarse_K': rmse_c,
        'improvement_pct': improvement,
        'sharpening_helped': helped,
    }
    if verbose:
        print('[validate] against %s over %d pixels%s'
              % (reference_label, n, ' (mean removed)' if remove_mean else ''))
        print('[validate]   mean offset sharpened - reference: %+.2f K' % mean_offset)
        print('[validate]   coarse    : r=%.4f  RMSE=%.3f K' % (corr_c, rmse_c))
        print('[validate]   sharpened : r=%.4f  RMSE=%.3f K' % (corr_s, rmse_s))
        print('[validate]   sharpening %s (%+.1f%%)'
              % ('HELPED' if helped else 'did NOT help', improvement))
        if not helped:
            print('[validate]   -> use the coarse field for this scene. A weak '
                  'temperature/vegetation relation (arid or homogeneous ground) '
                  'is the usual cause; check result["cv_r2"] and the predictor.')
    return out

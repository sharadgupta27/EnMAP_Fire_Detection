#!/usr/bin/env python
# coding: utf-8
"""
This work's retrieval stage of run_pipeline(): the profile-likelihood chain run
over a scene's detected fire pixels, end to end.

Order of operations (each step is what the corresponding module documents):

  1. enmap_quality.load_pixel_quality()   per-pixel band validity and
                                          saturation (censored) masks from
                                          the L1B pixel mask and DN rasters;
  2. background libraries                 five independent draws per pixel,
                                          each half from the pixel's own
                                          11x11 neighbourhood (the design of
                                          the original PRISMA retrieval,
                                          Amici et al. 2022, select_bkg_point)
                                          and half from the scene-wide
                                          background pool, so a burnt
                                          surround and the scene's variety are
                                          both represented;
  3. fire_transmittance.estimate_fire_term_transmittance()
                                          the scene's fire-term transmittance
                                          tau(lambda) from all fire pixels
                                          (draw 1 of the libraries);
  4. profile_retrieval.run_profile_on_pixels()
                                          the profile-likelihood retrieval
                                          with tau, the masks, the calibrated
                                          model-error fraction and the
                                          residual-derived kernel, best-of-5
                                          libraries for the point estimate
                                          and the envelope interval;
  5. outputs                              Results_<scene>_ProfileML_<date>.csv,
                                          Transmittance_<scene>_ProfileML_<date>.csv,
                                          T_map_<scene>_ProfileML.tif and
                                          Fig_ProfileML_<scene>_<date>.png.

Everything here is additive: the native Amici2022 outputs are untouched,
and a failure in this stage is reported as a warning rather than aborting
the run. The stage needs at least three fire pixels (the transmittance estimate's floor); below
that it runs the retrieval without a transmittance and says so in the log and in the
`tau_applied` column.
"""

import os
import warnings

import numpy as np

import profile_retrieval as pr
import fire_transmittance as ft
import enmap_quality as eq

__all__ = ['run_profile_ml_stage', 'build_libraries']

# EnMAP SWIR SNR at reference radiance (>= 150:1 at 2200 nm per the mission
# specification); the instrument term is minor next to the calibrated
# model-error term on a fire pixel, so this only needs to be the right order.
DEFAULT_SNR = 150.0


DEFAULT_N_LOCAL = 40      # neighbourhood spectra per library (11x11 window has up to 120)
DEFAULT_N_SCENE = 20      # scene-wide spectra per library


def build_libraries(cube, wavelength, fire_points, background_mask, bkg_pool, n_draws=5,
                    n_endmembers=None, half_window=5, rng=None, n_local_max=DEFAULT_N_LOCAL,
                    n_scene=DEFAULT_N_SCENE, selection='systematic'):
    """Per-pixel background libraries: `n_draws` arrays of full-band spectra,
    up to `n_local_max` from the pixel's own 11x11 neighbourhood
    (background-mask pixels only) plus `n_scene` scene-wide ones; scene-wide
    only (n_scene + n_local_max of them) when the neighbourhood has fewer
    than 10 usable pixels (scene edge, dense cluster).

    The library is deliberately larger than the band count allows for an
    ordinary least-squares fit: the retrieval is non-negative and closed, so
    it selects a sparse active set and is charged only for that (active-set
    degrees of freedom, profile_retrieval). Measured on synthetic fires
    embedded in real neighbours of the scene's fire pixels, the mean
    absolute temperature error falls from 40 K with 20 random scene spectra
    to 12 K with 40 local + 20 scene spectra, at 94% single-library
    coverage; larger libraries gain nothing further and cost five times the
    runtime. `n_endmembers`, if given, is the legacy total (split half/half).

    `selection` says how the subsets are taken. 'systematic' (default) has
    no random element (sampling.py): draw k takes the neighbours at evenly
    spaced positions of their raster order, offset by the k-th Weyl phase,
    and the scene part is a systematic sample of the (raster-ordered) pool
    with a phase keyed on the pixel's raster index and k, so a pixel's
    libraries are a fixed function of the scene and of the pixel's
    position. 'random' is the earlier seeded draw from `rng`, kept to
    reproduce earlier runs.
    Returns (libraries, n_local_per_pixel)."""
    import temperature_methods as tm
    import sampling as smp
    if selection not in ('systematic', 'random'):
        raise ValueError(f"selection must be 'systematic' or 'random', got {selection!r}")
    rng = rng if rng is not None else np.random.default_rng(0)
    n_cols = np.asarray(cube).shape[2] if np.ndim(cube) == 3 else 1
    if n_endmembers is not None:
        n_local_max = int(n_endmembers) // 2; n_scene = int(n_endmembers) - n_local_max
    exclude = ~np.asarray(background_mask, dtype=bool)
    # the pool stays in its own dtype; only the drawn rows are converted
    # (a float64 copy of a 10^6-spectrum pool is ~1.8 GB)
    bkg_pool = np.atleast_2d(bkg_pool)
    libs, n_local = [], []
    for r_, c_ in np.asarray(fire_points):
        local = tm.local_background_pool(cube, wavelength, int(r_), int(c_), exclude,
                                         half_window=half_window, min_pixels=10)
        if local is not None:
            local = local[np.all(local > 0, axis=1)]
        n_loc = 0 if local is None or len(local) < 10 else len(local)
        n_local.append(n_loc)
        draws = []
        key = int(r_) * n_cols + int(c_)
        for k in range(max(int(n_draws), 1)):
            if selection == 'systematic':
                n_pool = bkg_pool.shape[0]
                ph_s = smp.weyl_phase(key * max(int(n_draws), 1) + k + 1)
                if n_loc:
                    loc_part = local[smp.systematic_indices(n_loc, min(n_local_max, n_loc), smp.weyl_phase(k))]
                    scene_part = np.asarray(bkg_pool[smp.systematic_indices(n_pool, min(n_scene, n_pool), ph_s)],
                                            dtype=float)
                    draws.append(np.vstack([loc_part, scene_part]))
                else:
                    draws.append(np.asarray(bkg_pool[smp.systematic_indices(n_pool, min(n_scene + n_local_max, n_pool),
                                                                            ph_s)], dtype=float))
                continue
            if n_loc:
                loc_part = local[rng.choice(n_loc, size=min(n_local_max, n_loc), replace=False)]
                scene_part = np.asarray(bkg_pool[rng.choice(bkg_pool.shape[0], size=min(n_scene, bkg_pool.shape[0]),
                                                            replace=False)], dtype=float)
                draws.append(np.vstack([loc_part, scene_part]))
            else:
                draws.append(np.asarray(bkg_pool[rng.choice(bkg_pool.shape[0], size=min(n_scene + n_local_max,
                                                                                          bkg_pool.shape[0]),
                                                             replace=False)], dtype=float))
        libs.append(draws)
    return libs, np.asarray(n_local)


# Cap of the trimmed scene-level objective, in units of the scene's median
# reduced chi-square: the 'poor fit' threshold of run_profile_on_pixels()
# (reduced chi2 > 5 with the error model calibrated to a median of 1).
ADEQUACY_CAP = 5.0


def run_profile_ml_stage(cube, wavelength, fire_points, background_mask, bkg_pool, band_mask, cfg,
                         scene, enmap_radiance_path, results_dir, scene_id, date_tag,
                         native_results=None, save_results=True, generate_figures=True,
                         figure_dpi=300, snr=DEFAULT_SNR, n_draws=5, n_rounds=8, progress=None,
                         verbose=True, n_tau_draws=2, use_template_prior=False, device=None,
                         variant='v1', kernel_method='autocorrelation', scene_estimates=None,
                         sampling='systematic', quality=None):
    """Run this work's chain over `fire_points`. Returns a dict with keys
    df (the ProfileML results frame), tau (full-band array, NaN outside the
    fit window), tau_applied (bool), quality (enmap_quality dict), kernel,
    eps, figure_paths, paths (files written).

    `device` says where the per-pixel retrievals run: None -> all CPU cores
    (or the PROFILEML_DEVICE environment variable), 'joblib' (all cores),
    'serial' (one core). Every step of the chain - the transmittance rounds,
    the split-half validation, the two calibrations, the diagonal pass, the
    library and transmittance draws - goes through
    profile_batch.profile_retrieval_many(), so the whole stage scales with
    the cores available; the result is the same on either device.

    `variant` selects the method. 'v1' (default) is the published chain,
    unchanged. 'v2' (ProfileML-v2, VERIFICATION.md S22-S23) differs in two
    places. (i) Every retrieval of the chain iterates the reweighting to its
    FIXED POINT (profile_retrieval(n_weight_passes='converge')) instead of
    stopping after two passes, so the estimate is the solution of the
    quasi-likelihood score equations and does not depend on the starting
    proxy or an arbitrary pass count; a pixel that does not reach it is
    flagged (irls_converged=False; on t014 38 of 39 such pixels are 'poor'
    fits). Measured against truth on synthetic fires carrying the REAL
    t014 residual structure, v2 and v1 have the same accuracy and coverage
    (MAE 30.8/26.1 K vs 30.5/26.1 K, 95% coverage 0.94/0.95 both); on the
    real tile (i) alone moved the scene median by +46 K, through the
    transmittance estimate (S22). (ii) The scene-level estimates use only
    adequately fitted pixels (scene_estimates='adequate', below): the ~110
    bright, poorly fitted t014 pixels had dominated the transmittance band
    solve, whose weights grow with the square of the fire signal. With
    (ii), v1-with-(ii) and v2 agree to 2 K on t014 (was 46 K), the result
    moves by -2/+9 K for caps 3/10, and accuracy and coverage on the
    synthetic-truth scenes are unchanged (S23).

    `kernel_method`: 'autocorrelation' (default, both variants) or 'reml'
    (estimate_model_error_kernel_reml(); experimental - better coverage
    on smooth synthetic errors, but 4-5 K WORSE MAE and 30-40% wider
    intervals under the real error structure, S22).

    `scene_estimates`: None -> 'all' for v1, 'adequate' for v2. 'all' - every converged pixel enters the
    scene-level estimates (transmittance rounds, split-half draws, residual
    kernel); 'adequate' - they use a capped (trimmed) objective, so only
    pixels the mixture model describes to within ADEQUACY_CAP times the
    scene's median misfit enter them
    (ft.estimate_fire_term_transmittance(adequacy_cap=),
    pr.estimate_model_error_kernel(max_reduced_chi2=)). Every pixel is still
    retrieved and reported.

    `sampling`: 'systematic' (default) - every subset of the stage (the
    background libraries, the halves of the split-half transmittance draws,
    the noise sample of the saturation bounds) is a fixed function of the
    data (sampling.py), so the stage has no seed and repeated runs on the
    same scene give the same numbers bit for bit; 'random' - the earlier
    draws from cfg['random_seed'], kept to reproduce runs before
    15 Oct 2026 and for the seed-sensitivity comparison (VERIFICATION.md S34).

    `quality`: a per-pixel band-quality dict in the form of
    enmap_quality.load_pixel_quality() (valid, censored, censor_level, and
    optionally dn / flag / available) to use instead of reading the EnMAP
    product files - for another sensor, e.g. EMIT's interpolated
    (saturated) channels from its L2A band mask.

    v2 writes its own files (..._ProfileMLv2_...)."""
    import time as _time
    from profile_batch import resolve_device
    device = resolve_device(device)
    if variant not in ('v1', 'v2'):
        raise ValueError(f"variant must be 'v1' or 'v2', got {variant!r}")
    if kernel_method not in ('autocorrelation', 'reml'):
        raise ValueError(f"kernel_method must be 'autocorrelation' or 'reml', got {kernel_method!r}")
    if scene_estimates is None:
        scene_estimates = 'adequate' if variant == 'v2' else 'all'
    if sampling not in ('systematic', 'random'):
        raise ValueError(f"sampling must be 'systematic' or 'random', got {sampling!r}")
    if scene_estimates not in ('all', 'adequate'):
        raise ValueError(f"scene_estimates must be 'all' or 'adequate', got {scene_estimates!r}")
    gate = scene_estimates == 'adequate'
    v2 = variant == 'v2'
    tag = 'ProfileMLv2' if v2 else 'ProfileML'
    who = 'This work (v2)' if v2 else 'This work'
    irls = dict(n_weight_passes='converge') if v2 else {}
    if v2 and device not in ('joblib', 'serial'):
        warnings.warn(f'{who}: the converged reweighting runs on the CPU engines; device {device!r} -> joblib')
        device = 'joblib'

    def print(*a, **k):                               # the log names the variant
        import builtins
        builtins.print(*(x.replace('This work:', who + ':', 1) if isinstance(x, str) else x for x in a), **k)
    _t_stage = _time.time()
    # Wall clock of each step of the chain, printed at the end of the stage.
    # The steps are very uneven (the transmittance rounds and the main pass
    # dominate), so a run that feels slow is worth being able to attribute
    # from the log rather than by profiling it again.
    _steps = {}
    _mark = [_time.time()]

    def _step(name):
        now = _time.time()
        _steps[name] = _steps.get(name, 0.0) + (now - _mark[0])
        _mark[0] = now
    cube = np.asarray(cube)
    wavelength = np.asarray(wavelength, dtype=float)
    fire_points = np.asarray(fire_points)
    band_mask = np.asarray(band_mask, dtype=bool)
    n_pix = len(fire_points)
    out = {'df': None, 'tau': None, 'tau_applied': False, 'quality': None, 'kernel': None,
           'template_scales': None, 'tau_halves': [], 'tau_template': None,
           'eps': None, 'figure_paths': [], 'paths': [], 'wavelength': wavelength,
           'band_mask': band_mask, 'variant': variant, 'tag': tag, 'kernel_reml': None,
           'scene_estimates': scene_estimates, 'n_tau_pixels': None}
    if n_pix == 0:
        return out
    if verbose:
        print(f'  This work: per-pixel retrievals on {device!r} '
              f'({os.cpu_count()} CPU cores available)')
    w = wavelength[band_mask]
    pool_m = np.atleast_2d(bkg_pool)          # own dtype; subsets are converted where used
    ref = pr.column_median(pool_m, band_mask)
    fire_full = cube[fire_points[:, 0], :, fire_points[:, 1]].astype(float)
    fire = fire_full[:, band_mask]
    emissivity = float(cfg.get('emissivity', 1.0))
    T_bounds = (float(cfg.get('T_min', 300.0)), float(cfg.get('T_max', 2000.0)))
    if T_bounds[0] < 300.0:
        # The native retrieval's default T_min is 100 K, far below anything
        # the SWIR can constrain; the profile grid starts at 300 K.
        T_bounds = (300.0, T_bounds[1])

    # ---- 1. quality masks --------------------------------------------------
    q = quality if quality is not None else eq.load_pixel_quality(
        enmap_radiance_path, fire_points, n_bands_expected=len(wavelength), verbose=verbose)
    out['quality'] = q
    _step('quality masks')
    pk = [dict(valid_mask=q['valid'][i][band_mask], censored_mask=q['censored'][i][band_mask],
               censor_level=q['censor_level'][i][band_mask]) for i in range(n_pix)]

    # ---- 2. libraries -------------------------------------------------------
    rng = np.random.default_rng(int(cfg.get('random_seed', 0)))
    libs_full, n_local = build_libraries(cube, wavelength, fire_points, background_mask, pool_m,
                                         n_draws=n_draws, rng=rng, selection=sampling)
    if verbose:
        print(f'  This work: libraries built ({int((n_local > 0).sum())}/{n_pix} pixels with a usable '
              f'11x11 neighbourhood; the rest scene-wide only)')
    libs_first = [L[0][:, band_mask] for L in libs_full]
    _step('libraries')

    # ---- 3. scene transmittance ----------------------------------------------
    base = dict(snr=snr, reference_radiance=ref, rel_model_error=0.06, corr_length_nm=0.0,
                emissivity=emissivity, T_bounds=T_bounds, **irls)
    tau_m = None
    tpl = None
    if use_template_prior:
        try:
            tpl = ft.background_absorption_template(pool_m[:, band_mask], w)
        except Exception as exc:
            warnings.warn(f'This work: background absorption template failed ({exc}); free ridge toward 1.')
    # The template is a diagnostic and the detector's prior, NOT the ridge
    # target of the estimate: measured on the Portugal tile, a ridge toward
    # the scaled template re-estimated each round overshoots the absorber
    # scales (0.8-1.2 against 0.5-0.7 from the full sample) and agrees WORSE
    # with the full-sample estimate on subsets of 4-32 pixels (RMS
    # difference of ln tau in the absorber regions 0.15-0.4 against
    # 0.01-0.09 for the ridge toward unity). The free estimate with the
    # ridge toward unity is kept; the absorber scales are reported.
    est_kw = dict(prior_template=tpl) if (tpl is not None and use_template_prior) else {}
    if gate:
        est_kw['adequacy_cap'] = ADEQUACY_CAP
    if n_pix >= 3:
        try:
            est = ft.estimate_fire_term_transmittance(fire, w, libs_first, n_rounds=n_rounds,
                                                      per_pixel_kwargs=pk, verbose=verbose, device=device,
                                                      **est_kw, **base)
            tau_m = est['tau_fine']
            h = est['history']
            out['n_tau_pixels'] = est['n_pixels_used']
            out['template_scales'] = est.get('template_scales')
            if tpl is not None:
                out['tau_template'] = ft.template_scale_factors(tau_m, tpl, weights=est['band_information'])['tau_template']
            if verbose:
                print(f"  This work: scene fire-term transmittance from {est['n_pixels_used']} pixels - "
                      f"median relative misfit {100 * h[0]['median_rmse_rel']:.1f}% -> "
                      f"{100 * min(x['median_rmse_rel'] for x in h):.1f}%; bands with tau < 0.9: "
                      f"{np.round(w[tau_m < 0.9]).astype(int).tolist()}")
                if out['template_scales']:
                    print('  This work: absorber scales relative to the background two-way column '
                          '(0.5 = one-way path, more = plume excess): '
                          + ', '.join(f'{k} {v:.2f}' for k, v in out['template_scales'].items() if np.isfinite(v)))
            # half-sample transmittance draws: the interval envelope below
            # also spans the uncertainty of tau itself
            if n_tau_draws > 0 and n_pix >= 6:
                try:
                    sh = ft.split_half_validation(fire, w, libs_first, rng=np.random.default_rng(1), n_rounds=max(n_rounds // 2, 2),
                                                  split='interleaved' if sampling == 'systematic' else 'random',
                                                  per_pixel_kwargs=pk, estimator_kwargs=est_kw, device=device,
                                                  round0_retrievals=est.get('retrievals_round0'), **base)
                    out['tau_halves'] = [sh['tau_A'], sh['tau_B']][:n_tau_draws]
                    if verbose:
                        print(f"  This work: split-half transmittance - held-out misfit "
                              f"{100*sh['A']['median_rmse_rel_before']:.1f}% -> {100*sh['A']['median_rmse_rel_after']:.1f}% and "
                              f"{100*sh['B']['median_rmse_rel_before']:.1f}% -> {100*sh['B']['median_rmse_rel_after']:.1f}%; "
                              f"halves agree to {sh['tau_agreement_median_abs_diff']:.3f}")
                except Exception as exc:
                    warnings.warn(f'This work: split-half transmittance failed ({exc.__class__.__name__}: {exc})')
        except Exception as exc:
            warnings.warn(f'This work: transmittance estimate failed ({exc.__class__.__name__}: {exc}); '
                          'running the retrieval without it.')
    elif verbose:
        print(f'  This work: only {n_pix} fire pixels (< 3), no scene transmittance estimated')
    _step('transmittance + split-half')
    out['tau_applied'] = tau_m is not None
    tau_full = np.full(len(wavelength), np.nan)
    if tau_m is not None:
        tau_full[band_mask] = tau_m
    out['tau'] = tau_full

    # ---- 4. kernel from a diagonal pass, then the retrieval ----------------
    kernel_kw = {}
    try:
        eps_d, _ = pr.calibrate_model_error(fire, w, libs_first, tau=tau_m, per_pixel_kwargs=pk,
                                            snr=snr, reference_radiance=ref, corr_length_nm=0.0,
                                            emissivity=emissivity, T_bounds=T_bounds, device=device, **irls)
        diag_fits = pr._many(list(fire), w, libs_first, per_pixel=pk, device=device, tau=tau_m,
                             rel_model_error=eps_d, snr=snr, reference_radiance=ref, corr_length_nm=0.0,
                             emissivity=emissivity, T_bounds=T_bounds, **irls)
        if gate:
            red_d = np.array([r['reduced_chi2'] for r in diag_fits if r.get('converged')], float)
            red_d = red_d[np.isfinite(red_d)]
            cap_d = ADEQUACY_CAP * float(np.median(red_d)) if len(red_d) else None
            ker = pr.estimate_model_error_kernel(diag_fits, fire, w, max_reduced_chi2=cap_d)
        else:
            ker = pr.estimate_model_error_kernel(diag_fits, fire, w)
        if kernel_method == 'reml':
            _t_k = _time.time()
            kre = pr.estimate_model_error_kernel_reml(diag_fits, list(fire), w, libs_first, tau=tau_m,
                                                      snr=snr, reference_radiance=ref, emissivity=emissivity)
            out['kernel_reml'] = {k_: v_ for k_, v_ in kre.items() if not k_.startswith(('table', 'grid'))}
            if verbose:
                print(f"  This work: REML model-error kernel from {kre['n_pixels']} pixels - correlation length "
                      f"{kre['corr_length_nm']:.0f} nm, nugget {kre['nugget']:.3f} "
                      f"(residual-autocorrelation estimate {ker['corr_length_nm']:.0f} nm, {ker['nugget']:.2f}; "
                      f"{_time.time() - _t_k:.0f} s)"
                      + (f"; WARNING: optimum at the grid edge in {kre.get('edge_parameters')}"
                         if kre.get('at_grid_edge') else ''))
            ker = dict(ker, corr_length_nm=kre['corr_length_nm'], nugget=kre['nugget'],
                       n_pixels=kre['n_pixels'], method='REML')
        out['kernel'] = ker
        if ker['n_pixels'] >= 3 and ker['nugget'] < 0.9 and 5.0 <= ker['corr_length_nm'] <= 500.0:
            # the REML nugget is an estimate and may be below the 0.05 floor
            kernel_kw = dict(corr_length_nm=ker['corr_length_nm'],
                             nugget=ker['nugget'] if kernel_method == 'reml' else max(ker['nugget'], 0.05))
            if verbose:
                print(f"  This work: model-error kernel from residuals - correlation length "
                      f"{ker['corr_length_nm']:.0f} nm, nugget {ker['nugget']:.2f}")
        else:
            kernel_kw = dict(corr_length_nm=0.0)
            if verbose:
                print('  This work: residuals show no usable band correlation; diagonal covariance')
    except Exception as exc:
        warnings.warn(f'This work: kernel estimation failed ({exc.__class__.__name__}: {exc}); '
                      'diagonal covariance')
        kernel_kw = dict(corr_length_nm=0.0)

    _step('model-error kernel')
    df = pr.run_profile_on_pixels(
        fire_full, wavelength, pool_m, band_mask=band_mask, n_draws=n_draws, snr=snr,
        emissivity=emissivity, T_bounds=T_bounds, fire_points=fire_points,
        valid_masks=q['valid'], censored_masks=q['censored'], censor_levels=q['censor_level'],
        libraries=libs_full, tau=tau_m, progress=progress, device=device,
        method_label=tag, **irls, **kernel_kw)
    _step('main retrieval pass')
    df['tau_applied'] = bool(tau_m is not None)
    df['n_local_endmembers'] = n_local
    # Model-free lower bounds from the saturated bands (saturation_bounds.py):
    # FRP >= FRP_min and T_hot >= T_hot_min for ANY sub-pixel temperature
    # distribution. They are what the brightest pixels - those with no
    # retrieval - still say; the report adds them to the scene FRP.
    try:
        import saturation_bounds as sbd
        if q.get('available', True) and np.asarray(q['censored']).any():
            # the joint (LP) bound also uses the linear, unflagged bands, less
            # 3 noise sigma (per-band HySime noise of a background sample)
            if sampling == 'systematic':
                from sampling import systematic_indices as _sysidx
                _ps = np.asarray(pool_m[_sysidx(pool_m.shape[0], min(5000, pool_m.shape[0]))], dtype=float)
            else:
                _rs = np.random.default_rng(7)
                _ps = np.asarray(pool_m[np.sort(_rs.choice(pool_m.shape[0], size=min(5000, pool_m.shape[0]),
                                                           replace=False))], dtype=float)
            _pos = np.all(_ps > 0, axis=0) & np.all(np.isfinite(_ps), axis=0)
            _sig = np.zeros(len(wavelength))
            _sig[_pos] = pr.estimate_band_noise(_ps[:, _pos])
            _ref = np.maximum(np.median(_ps, axis=0), 1e-12)
            _noise = _sig[None, :] * np.sqrt(np.maximum(fire_full / _ref[None, :], 1.0))
            _usable = np.asarray(q['valid'], bool) & _pos[None, :]
            if q.get('dn') is not None:
                _usable &= np.asarray(q['dn']) < sbd.LINEAR_DN_FRACTION * 65535
            bd = sbd.bounds_for_pixels(wavelength, np.asarray(q['censored'], bool),
                                       np.asarray(q['censor_level'], float), libs_full, pool_m,
                                       radiance=fire_full, usable=_usable, noise_sigma=_noise)
            df['FRP_min_MW_saturated'] = bd['FRP_min_MW'].values
            df['FRP_min_joint_MW_saturated'] = bd['FRP_min_joint_MW'].values
            df['T_hot_min_K_saturated'] = bd['T_hot_min_K'].values
        else:
            df['FRP_min_MW_saturated'] = np.nan
            df['FRP_min_joint_MW_saturated'] = np.nan
            df['T_hot_min_K_saturated'] = np.nan
    except Exception as exc:
        warnings.warn(f'This work: saturation bounds failed ({exc.__class__.__name__}: {exc})')
    eps_main = df.attrs.get('eps_model_error')
    # fire radiative power with its interval along the temperature profile
    # (uncertainty.py); widened below with the transmittance draws like T
    import uncertainty as unc
    unc.add_frp_columns(df, emissivity)
    df['n_tau_draws'] = 0
    for tau_alt in out['tau_halves']:
        try:
            df_alt = pr.run_profile_on_pixels(
                fire_full, wavelength, pool_m, band_mask=band_mask, n_draws=1, snr=snr,
                emissivity=emissivity, T_bounds=T_bounds, fire_points=fire_points,
                valid_masks=q['valid'], censored_masks=q['censored'], censor_levels=q['censor_level'],
                libraries=[[L[0]] for L in libs_full], tau=tau_alt, rel_model_error=eps_main,
                progress=None, device=device, method_label=tag, **irls, **kernel_kw)
            _merge_envelope(df, df_alt)
            unc.widen_frp(df, df_alt, emissivity)
            df['n_tau_draws'] += 1
        except Exception as exc:
            warnings.warn(f'This work: transmittance draw failed ({exc.__class__.__name__}: {exc})')
    _step('transmittance draws')
    out['df'] = df
    out['eps'] = eps_main
    ok = df['fit_quality'].values == 'ok'
    if verbose:
        print(f'  This work: retrieval chain done in {_time.time() - _t_stage:.0f} s for {n_pix} pixels')
        print('  This work: step timings (s): '
              + ', '.join(f'{k} {v:.0f}' for k, v in _steps.items() if v >= 0.5))
        fin = ~(df['T_lower_unbounded'].values | df['T_upper_unbounded'].values)
        print(f"  This work: {int(df['converged'].sum())}/{n_pix} converged, {int(ok.sum())} with an "
              f"adequate fit (reduced chi2 <= 5), {int((df['n_bands_censored'] > 0).sum())} with censored "
              f"(saturated) bands; median T over adequate fits {np.nanmedian(df['T_K'].values[ok]):.0f} K, "
              f"median 95% interval width {np.nanmedian((df['T_ci_high'] - df['T_ci_low']).values[ok & fin]):.0f} K "
              f"({int((ok & ~fin).sum())} intervals reach the search bound)")
        Tn = _native_median(native_results)
        if Tn is not None and len(Tn) == n_pix:
            if True:
                d = df['T_K'].values[ok] - Tn[ok]
                print(f"  This work vs Amici et al. (2022) on the same pixels: median difference "
                      f"{np.nanmedian(d):+.0f} K (quartiles {np.nanpercentile(d, 25):+.0f}/{np.nanpercentile(d, 75):+.0f})")

    # ---- 5. outputs ---------------------------------------------------------
    if save_results:
        os.makedirs(results_dir, exist_ok=True)
        p = os.path.join(results_dir, f'Results_{scene_id}_{tag}_{date_tag}.csv')
        df.to_csv(p, index=False); out['paths'].append(p)
        print(f'  Saved {p}')
        import pandas as pd
        tdf = pd.DataFrame({'wavelength_nm': wavelength, 'in_fit_window': band_mask, 'tau': tau_full})
        if tpl is not None:
            g_full = np.full(len(wavelength), np.nan); g_full[band_mask] = tpl['template']
            tdf['background_absorption_template'] = g_full
        if out.get('tau_template') is not None:
            t_full = np.full(len(wavelength), np.nan); t_full[band_mask] = out['tau_template']
            tdf['tau_template'] = t_full
        for k_, th in enumerate(out['tau_halves']):
            h_full = np.full(len(wavelength), np.nan); h_full[band_mask] = th
            tdf[f'tau_half_{"AB"[k_]}'] = h_full
        if out.get('template_scales'):
            for k_, v_ in out['template_scales'].items():
                tdf.attrs[k_] = v_
        p = os.path.join(results_dir, f'Transmittance_{scene_id}_{tag}_{date_tag}.csv')
        tdf.to_csv(p, index=False); out['paths'].append(p)
        print(f'  Saved {p}')
        # scene-level uncertainty budget (uncertainty.scene_budget)
        try:
            _q = df['fit_quality'].values
            _lbc = ('FRP_min_joint_MW_saturated' if 'FRP_min_joint_MW_saturated' in df
                    else 'FRP_min_MW_saturated')
            _lb = (np.where(_q != 'ok', df[_lbc].values.astype(float), np.nan) if _lbc in df else None)
            ub = pd.DataFrame(unc.scene_budget(df, emissivity, saturated_bounds=_lb))
            p = os.path.join(results_dir, f'Uncertainty_{scene_id}_{tag}_{date_tag}.csv')
            ub.to_csv(p, index=False); out['paths'].append(p)
            out['uncertainty'] = ub
            print(f'  Saved {p}')
            for _, r_ in ub.iterrows():
                if r_['unit'] == 'MW' or r_['quantity'].startswith('median temperature'):
                    print(f"    {r_['quantity']}: {r_['value']:.1f} [{r_['low']:.1f}, {r_['high']:.1f}] {r_['unit']}")
        except Exception as exc:
            warnings.warn(f'This work: uncertainty budget failed ({exc.__class__.__name__}: {exc})')
        try:
            from Fire_analysis_with_EnMAP import geoTIFF_creation
            n_rows, n_cols = cube.shape[0], cube.shape[2]
            tmap = np.full((n_rows, n_cols), np.nan, dtype='float32')
            good = ok & np.isfinite(df['T_K'].values)
            tmap[df['ind_x'].values[good], df['ind_y'].values[good]] = df['T_K'].values[good]
            p = os.path.join(results_dir, f'T_map_{scene_id}_{tag}.tif')
            geoTIFF_creation(n_rows, n_cols, 1, tmap[np.newaxis, :, :], scene['geotransform'],
                             scene['projection_wkt'], 'float32', p, unit_type='K')
            out['paths'].append(p)
            print(f'  Saved {p}')
        except Exception as exc:
            warnings.warn(f'This work: temperature map raster not written ({exc.__class__.__name__}: {exc})')

    if generate_figures:
        try:
            out['figure_paths'] = _figure(df, tau_full, wavelength, band_mask, scene, native_results,
                                          results_dir, scene_id, date_tag, figure_dpi, tag=tag)
            for p in out['figure_paths']:
                print(f'  Saved {p}')
        except Exception as exc:
            warnings.warn(f'This work: figure failed ({exc.__class__.__name__}: {exc}); the CSVs were written.')
    return out


def _merge_envelope(df, df_alt):
    """Widen df's per-pixel interval to cover df_alt's (a retrieval with an
    alternative transmittance draw); the point estimate stays the
    best-chi2 one of the main run."""
    ok = df_alt['converged'].values & np.isfinite(df_alt['T_K'].values)
    lo = np.where(ok, np.fmin(df['T_ci_low'].values, df_alt['T_ci_low'].values), df['T_ci_low'].values)
    hi = np.where(ok, np.fmax(df['T_ci_high'].values, df_alt['T_ci_high'].values), df['T_ci_high'].values)
    df['T_ci_low'] = lo; df['T_ci_high'] = hi
    df['T_lower_unbounded'] = df['T_lower_unbounded'].values | (ok & df_alt['T_lower_unbounded'].values)
    df['T_upper_unbounded'] = df['T_upper_unbounded'].values | (ok & df_alt['T_upper_unbounded'].values)
    spread = np.abs(df_alt['T_K'].values - df['T_K'].values)
    df['T_draw_spread'] = np.where(ok, np.fmax(df['T_draw_spread'].values, spread), df['T_draw_spread'].values)


def _native_median(native_results):
    """The reference retrieval's median temperature per pixel, whichever of
    the two column names it carries (run_pipeline() renames 'median_temp'
    to 'median_temp_K' before saving)."""
    if native_results is None:
        return None
    for col in ('median_temp_K', 'median_temp'):
        if col in native_results.columns:
            return np.asarray(native_results[col].values, dtype=float)
    return None


def profile_ml_figure(profile_ml, scene, native_results, results_dir, scene_id, date_tag, dpi=300):
    """Draw the stage figure for a run_profile_ml_stage() result.

    Separate from the stage so run_pipeline() can draw it in its outputs
    stage, after the Amici et al. (2022) reference retrieval exists to
    compare against (panel b). Returns the list of paths written."""
    if profile_ml is None or profile_ml.get('df') is None:
        return []
    return _figure(profile_ml['df'], profile_ml['tau'], profile_ml['wavelength'],
                   profile_ml['band_mask'], scene, native_results, results_dir, scene_id,
                   date_tag, dpi, tag=profile_ml.get('tag', 'ProfileML'))


def _figure(df, tau_full, wavelength, band_mask, scene, native_results, results_dir, scene_id,
            date_tag, dpi, tag='ProfileML'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from Fire_analysis_with_EnMAP import (_build_rgb_composite, _scatter_over_composite,
                                          _add_detection_inset, _plot_agreement_axes)
    ok = df['fit_quality'].values == 'ok'
    good = ok & np.isfinite(df['T_K'].values)
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) transmittance
    ax = axes[0]
    w = wavelength[band_mask]
    tau = tau_full[band_mask]
    if np.any(np.isfinite(tau)):
        steps = np.diff(w)
        gaps = np.where(steps > 2.5 * np.median(steps))[0]
        wp, tp = list(w), list(tau)
        for k, gi in enumerate(gaps):
            wp.insert(gi + 1 + k, 0.5 * (w[gi] + w[gi + 1])); tp.insert(gi + 1 + k, np.nan)
        ax.plot(wp, tp, '-', color='#B3552C', lw=1.4)
        ax.plot(w, tau, '.', color='#B3552C', ms=3)
        ax.axhline(1.0, color='grey', lw=0.8, ls=':')
        ax.set_ylim(min(0.3, np.nanmin(tau) - 0.05), 1.05)
        ax.set_title('Scene fire-term transmittance (this work)\nridge toward 1, bounded at 1; dips = gas absorption',
                     fontsize=10)
    else:
        ax.text(0.5, 0.5, 'not estimated\n(< 3 fire pixels)', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Scene fire-term transmittance (this work)', fontsize=10)
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel(r'$\tau(\lambda)$')

    # (b) chain vs native
    ax = axes[1]
    Tn = _native_median(native_results)
    if Tn is not None and len(Tn) == len(df):
        m_ = good & np.isfinite(Tn)
        lo = df['T_ci_low'].values; hi = df['T_ci_high'].values
        ax.errorbar(Tn[m_], df['T_K'].values[m_], yerr=[df['T_K'].values[m_] - lo[m_], hi[m_] - df['T_K'].values[m_]],
                    fmt='none', ecolor='#B3552C', elinewidth=0.7, alpha=0.6, zorder=1)
        if m_.sum() >= 2:
            allv = np.concatenate([Tn[m_], df['T_K'].values[m_]])
            pad = 0.05 * (allv.max() - allv.min() + 1)
            _plot_agreement_axes(ax, Tn[m_], df['T_K'].values[m_], limits=(allv.min() - pad, allv.max() + pad),
                                 xlabel='Amici et al. (2022), median T (K)',
                                 ylabel='this work, T (K)')
            ax.legend(fontsize=7, loc='lower right')
        n_poor = int((~ok).sum())
        ax.set_title(f'Retrieved temperature, {int(m_.sum())} adequate fits'
                     f'{" (" + str(n_poor) + " poor fits omitted)" if n_poor else ""}\n'
                     'bars: 95% profile-likelihood envelope', fontsize=10)
    else:
        ax.hist(df['T_K'].values[good], bins=20, color='#B3552C')
        ax.set_xlabel('this work, T (K)'); ax.set_ylabel('pixels')
        ax.set_title('Retrieved temperature (adequate fits)', fontsize=10)

    # (c) map
    ax = axes[2]
    lat, lon = scene['lat'], scene['lon']
    rgb = _build_rgb_composite(scene)
    vals = df['T_K'].values[good]
    if vals.size:
        vlo, vhi = np.percentile(vals, 2), np.percentile(vals, 98)
        px, py = df['ind_x'].values[good], df['ind_y'].values[good]
        sc_ = _scatter_over_composite(ax, lon, lat, rgb, px, py, vals, 'inferno', vlo, vhi)
        aspect = 1.0 / max(np.cos(np.radians(np.nanmean(lat))), 1e-3)
        if sc_ is not None:
            fig.colorbar(sc_, ax=ax, label='this work, T (K)', shrink=0.85)
            _add_detection_inset(ax, lon, lat, rgb, px, py, vals, 'inferno', vlo, vhi, aspect, scene=scene)
        ax.set_aspect(aspect)
    ax.set_title('Temperature map, this work', fontsize=10)
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude'); ax.tick_params(axis='x', rotation=30)
    fig.suptitle(f'Profile-likelihood retrieval with scene transmittance - {scene_id}', fontsize=11)
    fig.tight_layout()
    p = os.path.join(results_dir, f'Fig_{tag}_{scene_id}_{date_tag}.png')
    fig.savefig(p, dpi=dpi, bbox_inches='tight'); plt.close(fig)
    return [p]

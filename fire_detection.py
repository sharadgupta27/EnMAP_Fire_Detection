#!/usr/bin/env python
# coding: utf-8
"""
Scene-wide fire detection by a generalised likelihood-ratio test (this work).

The published VSWIR fire detectors threshold a two-band index (the SWIR
"hotspot index", HFDI-type ratios) and, in this repository's host pipeline, an
absolute SWIR radiance anomaly on top. An index uses two of the ~76 usable
SWIR bands and has no false-alarm calibration beyond a robust threshold. The
detector here uses every band of the fit window and the same forward model as
the temperature retrieval:

    y = f * tau * B(T) + background,       background in a K-dimensional
                                            subspace learned from the scene

and asks, per pixel, how much the fire term earns:

    Lambda = max_T  [ (P_perp b_T)^T P_perp y_w ]_+^2 / || P_perp b_T ||^2

where y_w is the whitened spectrum, P_perp projects out the background
subspace (mean plus the leading K principal directions of the whitened
fire-free background), b_T the whitened Planck spectrum at T (times the scene
fire-term transmittance when known), and [.]_+ keeps the physical sign f >= 0.
This is the matched-subspace detector (Scharf & Friedlander 1994); under the
null with Gaussian noise Lambda is the maximum over T of a squared
half-normal, so it is a chi-square-like statistic, but the real null is set
by background-library inadequacy rather than noise, and it is therefore
CALIBRATED EMPIRICALLY on the scene's own fire-free background:

  * fit adequacy: a saturated roof or a specular target projects on the
    Planck direction too, but the residual AFTER the fire term stays large
    because a plateau at the detector ceiling is not a Planck curve. The
    per-pixel relative misfit after the fire term (with the retrieval's
    multiplicative error model) must be below the 99.9th percentile of the
    fire-free background's own misfit, or the pixel is not a fire candidate;
  * false-alarm rate: the (1 - p_fa) quantile of Lambda under the null is
    extrapolated from the fire-free background sample by a generalised
    Pareto fit to its upper tail (peaks over threshold; Pickands 1975), so a
    scene-appropriate p_fa (1e-5 per pixel, about ten false alarms per
    EnMAP tile) can be set from a sample of twenty thousand pixels.

On the three EnMAP tiles of this study the detector recovers injected fires
at about half the fire fraction the index needs at equal false-alarm rate,
and at a quarter of what the two-part index+magnitude test needs; on a
fire-free tile it flags the expected handful of pixels.
"""
import numpy as np
from scipy import stats
from scipy import ndimage as ndi

try:
    from profile_retrieval import _planck_nm
except ImportError:                      # pragma: no cover - standalone use
    def _planck_nm(wavelength_nm, T, emissivity=1.0):
        c1, c2 = 1.191042e-16, 1.438776e-2
        lam = np.asarray(wavelength_nm, float) * 1e-9
        return emissivity * c1 / lam ** 5 / np.expm1(c2 / (lam * T)) * 1e-9

DEFAULT_T_GRID = (500.0, 600.0, 700.0, 800.0, 900.0, 1000.0, 1200.0, 1500.0)


def _whitened_mean_cov(X, scale, chunk=65536):
    """Mean and covariance of X * scale over all rows, in two chunked passes
    (float64 accumulators; X stays in its own dtype)."""
    n, p = X.shape
    s = np.zeros(p)
    for i in range(0, n, chunk):
        s += (np.asarray(X[i:i + chunk], float) * scale).sum(axis=0)
    mu = s / n
    C = np.zeros((p, p))
    for i in range(0, n, chunk):
        Z = np.asarray(X[i:i + chunk], float) * scale - mu
        C += Z.T @ Z
    return mu, C / max(n - 1, 1)


class FireGLRT:
    """The detector for one scene: learn the background subspace and the null
    from a fire-free pool, then score any set of spectra or a whole cube."""

    def __init__(self, wavelength, pool, tau=None, K=12, snr=150.0, rel_model_error=0.05,
                 T_grid=DEFAULT_T_GRID, adequacy_quantile=0.999, n_fit=8000, rng=None,
                 ceiling=None, ceiling_fraction=0.75, tau_prior_scale=0.5, pool_rc=None,
                 sampling='random', pool_trim=None):
        self.w = np.asarray(wavelength, float)
        pool = np.asarray(pool)
        keep = np.all(np.isfinite(pool), axis=1) & np.all(pool > 0, axis=1)
        # n_fit=None: the WHOLE pool trains the subspace and sets the adequacy
        # gate and the pool null (no subset is chosen at all); the pool stays
        # in its own dtype and the statistics are accumulated in chunks, so a
        # whole-scene pool (~10^6 spectra) fits in memory
        whole = n_fit is None
        self.whole_pool = whole
        pool = pool[keep] if whole else np.asarray(pool[keep], dtype=float)
        self.pool_rc = None if pool_rc is None else np.asarray(pool_rc)[keep]
        # sampling: 'random' (subsets drawn from rng) or 'systematic' (evenly
        # spaced in the pool's order, sampling.py; no random element)
        if sampling not in ('random', 'systematic'):
            raise ValueError(f"sampling must be 'random' or 'systematic', got {sampling!r}")
        self.sampling = sampling
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.snr = float(snr); self.eps = float(rel_model_error)
        self.ref = np.median(pool, axis=0).astype(float)
        self.sig_ref = self.ref / self.snr
        # scene-level whitening for the projection (a fixed metric keeps the
        # statistic a single matrix product per temperature)
        self.Wt = 1.0 / np.sqrt(self.sig_ref ** 2 + (self.eps * self.ref) ** 2)
        # bands at the detector ceiling (saturated) are not measurements and
        # are left out of the fit-adequacy statistic
        self.ceiling = None if ceiling is None else np.asarray(ceiling, float)
        self.ceiling_fraction = float(ceiling_fraction)
        if tau is None and tau_prior_scale is not None:
            # No scene transmittance yet (detection runs before the retrieval):
            # take the physical prior - the one-way path through the column the
            # background sees, exp(-0.5 g) with g the background's two-way
            # absorption template - so the fire term already carries the gas
            # dips that would otherwise be left in the residual of every
            # strong fire pixel.
            try:
                from fire_transmittance import background_absorption_template, template_scale_factors
                if whole:
                    from sampling import systematic_indices
                    _tp = pool[systematic_indices(len(pool), min(3000, len(pool)))]
                    tpl = background_absorption_template(_tp, self.w, rng=self.rng, sampling='systematic')
                else:
                    tpl = background_absorption_template(pool, self.w, rng=self.rng, sampling=sampling)
                region = np.any(list(tpl['region_masks'].values()), axis=0)
                tau = np.where(region, np.exp(-float(tau_prior_scale) * tpl['template']), 1.0)
                self.tau_source = f'background template, scale {tau_prior_scale}'
            except Exception:
                tau = None; self.tau_source = 'none'
        else:
            self.tau_source = 'given' if tau is not None else 'none'
        self.tau = np.ones(len(self.w)) if tau is None else np.where(np.isfinite(tau), tau, 1.0)
        self.K = int(K)
        self.T_grid = np.asarray(T_grid, float)
        # The subspace is the leading eigenvectors of the covariance of the
        # WHOLE pool (deterministic; a covariance over n pixels costs the
        # same as one matrix product). Fitting it on a random subsample was
        # tried first and is unstable on a scene with rare bright classes:
        # whether those classes are spanned depends on the draw, and the
        # detection count varied by a factor of ten between seeds.
        if whole:
            self.mu, C = _whitened_mean_cov(pool, self.Wt)
            # pool_trim (whole pool only): a robust subspace - the pool pixels
            # whose whitened residual outside the subspace lies above the
            # (1 - pool_trim) quantile (rare bright classes, smoke, missed weak
            # fires) are left out and the mean and covariance refitted, twice.
            # A rule on the data, not a draw: the same pixels go every time.
            self.pool_trim = pool_trim
            if pool_trim:
                for _ in range(2):
                    ev, evec = np.linalg.eigh(C)
                    V0 = evec[:, np.argsort(ev)[::-1][:int(K)]]
                    r2 = np.concatenate([np.sum(((np.asarray(pool[i:i + 65536], float) * self.Wt - self.mu)
                                                 - ((np.asarray(pool[i:i + 65536], float) * self.Wt - self.mu) @ V0)
                                                 @ V0.T) ** 2, axis=1)
                                         for i in range(0, len(pool), 65536)])
                    keep_t = r2 <= np.quantile(r2, 1.0 - float(pool_trim))
                    self.mu, C = _whitened_mean_cov(pool[keep_t], self.Wt)
                self.n_trimmed = int((~keep_t).sum())
        else:
            Xw = pool * self.Wt
            self.mu = Xw.mean(axis=0)
            C = np.cov(Xw - self.mu, rowvar=False)
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]; evals, evecs = np.maximum(evals[order], 0), evecs[:, order]
        self.V = evecs[:, :self.K]
        self.explained = float(evals[:self.K].sum() / max(evals.sum(), 1e-300))
        if whole:
            fit_idx = np.zeros(0, dtype=np.int64)
        elif sampling == 'systematic':
            from sampling import systematic_indices
            fit_idx = systematic_indices(len(pool), min(n_fit, len(pool)))
        else:
            fit_idx = self.rng.choice(len(pool), size=min(n_fit, len(pool)), replace=False)
        # whitened, projected Planck directions
        self.pb = []
        self.planck_retained = []      # ||P_perp b_T|| / ||b_T||: 1 = the subspace spans no fire signal
        for T in self.T_grid:
            b = _planck_nm(self.w, T) * self.tau * self.Wt
            pb = b - self.V @ (self.V.T @ b)
            self.planck_retained.append(float(np.linalg.norm(pb) / np.linalg.norm(b)))
            self.pb.append(pb / np.linalg.norm(pb))
        self.pb = np.array(self.pb)                       # (nT, n_bands)
        # null and adequacy from the held-out part of the pool
        if whole:
            held = np.arange(len(pool))
            parts = [self.score(pool[i:i + 65536]) + (self._last_chi_before,) for i in range(0, len(pool), 65536)]
            lam0 = np.concatenate([x[0] for x in parts]); chi0 = np.concatenate([x[1] for x in parts])
            self._last_chi_before = np.concatenate([x[3] for x in parts])
        else:
            held = np.setdiff1d(np.arange(len(pool)), fit_idx)
            held = held if len(held) >= 2000 else np.arange(len(pool))
            lam0, chi0, _ = self.score(pool[held])
        self.adequacy_quantile = float(adequacy_quantile)
        self.chi_gate = float(np.nanquantile(chi0, self.adequacy_quantile))
        ok0 = chi0 <= self.chi_gate
        self.null_chi = chi0[ok0]; self.null_chi_before = self._last_chi_before[ok0]
        self.held_rc = None if self.pool_rc is None else self.pool_rc[held][ok0]
        self.null_sample_all = lam0[ok0]
        self.null_sample = self.null_sample_all
        self.n_null = int(ok0.sum())
        self._fit_tail()

    # ------------------------------------------------------------ statistic
    def score(self, spectra):
        """Lambda, the relative misfit after the fire term, and the best grid
        T for (n, n_bands) spectra (NaN rows give NaN)."""
        Y = np.asarray(spectra, float)
        good = np.all(np.isfinite(Y), axis=1) & np.all(Y > 0, axis=1)
        lam = np.full(len(Y), np.nan); chi = np.full(len(Y), np.nan); Tb = np.full(len(Y), np.nan)
        chi_b = np.full(len(Y), np.nan)
        self._last_chi_before = chi_b
        if not good.any():
            return lam, chi, Tb
        Yg = Y[good]
        Yw = Yg * self.Wt - self.mu
        Z = Yw - (Yw @ self.V) @ self.V.T
        S = Z @ self.pb.T                                  # (n, nT) projections
        S = np.maximum(S, 0.0)
        k = np.argmax(S, axis=1)
        s_best = S[np.arange(len(S)), k]
        lam_g = s_best ** 2
        # residual after the fire term, back in radiance units, judged with
        # the per-pixel multiplicative error model of the retrieval
        R = Z - s_best[:, None] * self.pb[k]
        r = R / self.Wt
        # the retrieval's two-regime instrument noise (read-noise floor at the
        # reference level, shot-noise growth above it) plus the multiplicative
        # model error; a fire pixel is judged with its own noise, not the
        # background's
        var = self.sig_ref ** 2 * np.maximum(np.abs(Yg) / self.ref, 1.0) + (self.eps * np.abs(Yg)) ** 2
        use = np.ones_like(Yg, dtype=bool)
        if self.ceiling is not None:
            use &= Yg < self.ceiling_fraction * self.ceiling[None, :]
        n_use = np.maximum(use.sum(axis=1) - self.K - 2, 1)
        chi_g = np.sum(np.where(use, r ** 2 / var, 0.0), axis=1) / n_use
        r0 = Z / self.Wt
        chib_g = np.sum(np.where(use, r0 ** 2 / var, 0.0), axis=1) / n_use
        # Pixels with bands at the ceiling: a plateau at the ceiling is not a
        # measurement, so the statistic is recomputed on the unsaturated
        # bands alone (per pixel; there are few). A fire hot enough to
        # saturate the 2.1-2.4 um bands still rises steeply across
        # 1.5-1.8 um, where a sunlit roof falls with the solar spectrum.
        sat = ~use.all(axis=1)
        if sat.any():
            for i in np.where(sat)[0]:
                u = use[i]
                if u.sum() < self.K + 6:
                    # too few unsaturated bands to fit anything: not a
                    # measurement, but almost certainly a fire - a candidate
                    lam_g[i] = np.inf; chi_g[i] = np.inf; chib_g[i] = np.inf; continue
                Vr = self.V[u]
                yw = Yw[i][u]
                coef, *_ = np.linalg.lstsq(Vr, yw, rcond=None)
                z = yw - Vr @ coef
                best = 0.0; kb = 0; sb = 0.0; pbb = None
                for j, T in enumerate(self.T_grid):
                    b = (_planck_nm(self.w, T) * self.tau * self.Wt)[u]
                    cb, *_ = np.linalg.lstsq(Vr, b, rcond=None)
                    pb = b - Vr @ cb
                    nrm = np.linalg.norm(pb)
                    if nrm <= 0:
                        continue
                    pb = pb / nrm
                    sv = max(float(z @ pb), 0.0)
                    if sv ** 2 > best:
                        best = sv ** 2; kb = j; sb = sv; pbb = pb
                lam_g[i] = best; k[i] = kb
                rr = ((z - sb * pbb) if pbb is not None else z) / self.Wt[u]
                chi_g[i] = float(np.sum(rr ** 2 / var[i][u]) / max(u.sum() - self.K - 2, 1))
                chib_g[i] = float(np.sum((z / self.Wt[u]) ** 2 / var[i][u]) / max(u.sum() - self.K - 2, 1))
        lam[good] = lam_g; chi[good] = chi_g; Tb[good] = self.T_grid[k]
        chi_b[good] = chib_g
        return lam, chi, Tb

    # ------------------------------------------------------------ null tail
    def _fit_tail(self, tail_fraction=0.05, sample=None):
        x = self.null_sample if sample is None else sample
        x = np.sort(x[np.isfinite(x)])
        if sample is not None:
            self.null_sample = x
        if len(x) < 200:
            self.gpd = None; return
        u = x[int((1 - tail_fraction) * len(x))]
        exc = x[x > u] - u
        try:
            c, loc, scale = stats.genpareto.fit(exc, floc=0.0)
        except Exception:
            c, scale = 0.0, float(np.mean(exc)) if len(exc) else 1.0
        self.gpd = dict(u=float(u), c=float(c), scale=float(scale), n_exc=int(len(exc)), n=int(len(x)),
                        p_u=float(len(exc) / len(x)))

    def threshold(self, p_fa=1e-5, min_exceedances=5):
        """The (1 - p_fa) quantile of Lambda under the null: the empirical
        order statistic whenever at least `min_exceedances` null values lie
        above it (on a full tile at p_fa = 1e-5 that is the tenth largest of
        a million), else the generalised-Pareto extrapolation. The tail
        shape (self.gpd['c']) is the diagnostic: above 0.5 the tail is heavy
        and the extrapolated quantile is uncertain by a factor of several."""
        x = self.null_sample
        self.threshold_source = 'empirical'
        if p_fa * len(x) >= min_exceedances:
            return float(np.quantile(x, 1 - p_fa))
        self.threshold_source = 'generalised Pareto extrapolation'
        if self.gpd is None:
            return float(np.max(x))
        g = self.gpd
        if p_fa >= g['p_u']:
            return float(np.quantile(x, 1 - p_fa))
        q = p_fa / g['p_u']
        c, s = g['c'], g['scale']
        if abs(c) < 1e-6:
            return float(g['u'] + s * (-np.log(q)))
        return float(g['u'] + s / c * (q ** (-c) - 1.0))

    # ------------------------------------------------------------ whole scene
    def detect_cube(self, window_cube, p_fa=1e-5, chunk_rows=64, cluster_min=1,
                    exclusion_radius=20, first_pass_p_fa=1e-4, ratio_gate=0.6, scene_null=True):
        """Score a (rows, cols, n_bands) cube (the fit-window bands, any
        dtype). Returns a dict with lam, chi, T_best maps, the boolean
        detection map, the threshold and the fire points (row, col).

        Two classes come back: `detections` (above the threshold AND either
        adequately fitted or with the fire term explaining at least
        1 - ratio_gate of the excess: fires) and `candidates` (above the
        threshold, but the residual after the fire term stays large: a
        saturated or specular target, or a fire whose bands sit on a
        detector plateau; the full retrieval with censoring decides). The
        null is calibrated on the scene itself with candidate clusters and
        their surroundings removed (see the comment in the body)."""
        n_r, n_c, nb = window_cube.shape
        lam = np.full((n_r, n_c), np.nan, np.float32); chi = np.full((n_r, n_c), np.nan, np.float32)
        Tb = np.full((n_r, n_c), np.nan, np.float32); chib = np.full((n_r, n_c), np.nan, np.float32)
        for r0 in range(0, n_r, chunk_rows):
            blk = np.asarray(window_cube[r0:r0 + chunk_rows]).reshape(-1, nb).astype(float)
            l_, c_, t_ = self.score(blk)
            lam[r0:r0 + chunk_rows] = l_.reshape(-1, n_c); chi[r0:r0 + chunk_rows] = c_.reshape(-1, n_c)
            Tb[r0:r0 + chunk_rows] = t_.reshape(-1, n_c); chib[r0:r0 + chunk_rows] = self._last_chi_before.reshape(-1, n_c)
        with np.errstate(invalid='ignore', divide='ignore'):
            ratio = np.where(chib > 0, chi / chib, np.nan).astype(np.float32)
        # ---- null calibration. The pool sample (a few thousand pixels)
        # gives a first threshold; the scene itself, with the neighbourhoods
        # of candidate CLUSTERS removed, then supplies a null sample of the
        # order of a million pixels, so the (1 - p_fa) quantile is set by its
        # own upper tail (generalised Pareto fit to the top 1 percent, some
        # ten thousand exceedances) rather than extrapolated a hundredfold.
        # Isolated exceedances stay in the null: they are the false alarms
        # the rate allows, and removing them would strip the tail.
        self.null_sample = self.null_sample_all
        self._fit_tail()
        # first pass on the pool null at a rate its size supports (an
        # order statistic, never an extrapolation)
        thr1 = self.threshold(max(first_pass_p_fa, 20.0 / max(len(self.null_sample), 1)))
        adequate = chi <= self.chi_gate
        cand = (lam > thr1) & adequate
        lab, n_lab = ndi.label(cand)
        if n_lab:
            sizes = np.bincount(lab.ravel()); sizes[0] = 0
            cand = sizes[lab] >= 3
        near = ndi.binary_dilation(cand, iterations=int(exclusion_radius)) if cand.any() else np.zeros_like(cand)
        n_excluded = int(near.sum())
        null_source = 'pool'
        if scene_null:
            samp = lam[np.isfinite(lam) & adequate & ~near]
            if len(samp) >= 50000 and near.mean() <= 0.5:
                self._fit_tail(tail_fraction=0.01, sample=samp)
                null_source = f'scene ({len(samp)} pixels, {n_excluded} near candidate clusters excluded)'
        thr = self.threshold(p_fa)
        thr_source = self.threshold_source
        tail_shape = float(self.gpd['c']) if self.gpd else float('nan')
        above = lam > thr
        with np.errstate(invalid='ignore'):
            rt = ratio if ratio is not None else np.ones_like(lam)
        fire_like = above & (adequate | (rt <= ratio_gate))
        candidate = above & ~fire_like
        det = fire_like
        if cluster_min > 1:
            lab, n = ndi.label(det)
            sizes = np.bincount(lab.ravel())
            det = det & (sizes[lab] >= cluster_min)
        rows, cols = np.where(det)
        crow, ccol = np.where(candidate)
        return dict(lam=lam, chi=chi, chi_before=chib, ratio=ratio, T_best=Tb, detections=det, threshold=float(thr), p_fa=float(p_fa),
                    candidates=candidate, candidate_points=np.column_stack([crow, ccol]), above=above,
                    null_source=null_source, ratio_gate=float(ratio_gate), threshold_source=thr_source,
                    tail_shape=tail_shape, heavy_tail=bool(tail_shape > 0.5),
                    chi_gate=self.chi_gate, fire_points=np.column_stack([rows, cols]),
                    n_inadequate=int(candidate.sum()),
                    n_null_excluded=n_excluded, n_null_used=int(len(self.null_sample)),
                    explained_variance=self.explained, gpd=self.gpd, n_null=self.n_null)


def inject_and_score(detector, pool_full, band_mask, wavelength_full, T, f, n=400, snr=150.0, rng=None,
                     tau_full=None):
    """Detection rate of a synthetic fire (T, f) injected into fire-free pool
    spectra, for the GLRT at p_fa and for the two-band index; returns the
    raw statistics so any threshold can be applied."""
    rng = rng if rng is not None else np.random.default_rng(0)
    idx = rng.choice(len(pool_full), size=min(n, len(pool_full)), replace=False)
    B = _planck_nm(np.asarray(wavelength_full, float), T)
    if tau_full is not None:
        B = B * np.where(np.isfinite(tau_full), tau_full, 1.0)
    Y = (1 - f) * pool_full[idx] + f * B[None, :]
    Y = Y + rng.normal(0, np.abs(Y) / snr)
    lam, chi, _ = detector.score(Y[:, band_mask])
    return Y, lam, chi


def hotspot_index(spectra_full, wavelength_full, wl_lo=2060.0, wl_hi=2430.0):
    wl = np.asarray(wavelength_full, float)
    i_lo = int(np.argmin(np.abs(wl - wl_lo))); i_hi = int(np.argmin(np.abs(wl - wl_hi)))
    lo, hi = spectra_full[:, i_lo], spectra_full[:, i_hi]
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(hi + lo > 0, (hi - lo) / (hi + lo), np.nan)


def detect_fire_pixels_glrt(cube, wavelength, band_mask, hotspot_index=None, swir_radiance=None,
                            p_fa=1e-5, margin_px=5, K=12, n_pool=60000, raster_path=None,
                            include_candidates=True, bootstrap_k=6.0, rng=None, verbose=True,
                            sampling='systematic', ceiling=None, pool_trim=None, pool_passes=1,
                            clean_p_fa=1e-3):
    """Scene-wide fire detection for run_pipeline(): the GLRT of FireGLRT on
    every pixel of the fit window, calibrated on the scene itself.

    The fire-free pool that trains the background subspace is bootstrapped
    from the two-part index test (hotspot index AND SWIR radiance anomaly,
    robust thresholds at `bootstrap_k`): everything outside a margin around
    its detections. The pool only needs to be free of the strong fires, so
    the loose bootstrap is enough; the detector's own null is then set on
    the scene with candidate clusters and their surroundings removed.

    Returns (fire_points, background_mask, info): fire_points are the
    'fire' class plus, when include_candidates, the 'candidate' class
    (above threshold but inadequately fitted: saturated or specular, or a
    fire on a detector plateau; the retrieval with censoring decides);
    background_mask excludes both classes and a margin of `margin_px`. info
    carries the detector, the maps (lam, chi, ratio, T_best), the per-point
    class and statistic, the threshold and the null description.

    `sampling`: 'systematic' (default) takes the `n_pool` training pixels
    evenly spaced through the bootstrap background in raster order, and
    every later subset the same way (sampling.py), so detection has no
    seed; 'random' is the earlier draw from `rng` (default seed 0), kept to
    reproduce runs before 15 Oct 2026 (VERIFICATION.md S34).

    `ceiling` (full band axis): per-band radiance ceiling to use instead of
    the EnMAP metadata's (another sensor); bands above
    ceiling_fraction * ceiling are left out of the adequacy statistic.
    `pool_passes` > 1: a self-consistent fire-free pool. The bootstrap
    background of a burning scene still holds smouldering and weakly burning
    pixels; in the covariance they make the background subspace span part of
    the Planck directions, which then projects real fire signal away (lower
    sensitivity, and a detection count that depends on how many such pixels a
    training subset happens to contain). After each pass, every pixel above
    the scene null's (1 - clean_p_fa) quantile or classed fire / candidate,
    with a margin of `margin_px`, leaves the pool, and the detector is
    retrained; the passes stop when the pool no longer changes (<0.1 %).
    `det.planck_retained` reports how much of a Planck spectrum survives the
    projection, per grid temperature.
    """
    from Fire_analysis_with_EnMAP import (compute_hotspot_index, swir_radiance_magnitude,
                                          robust_background_threshold, detect_fire_pixels)
    rng = rng if rng is not None else np.random.default_rng(0)
    cube = np.asarray(cube)
    wavelength = np.asarray(wavelength, float)
    m = np.asarray(band_mask, bool)
    w = wavelength[m]
    n_r, n_b, n_c = cube.shape
    if hotspot_index is None:
        hotspot_index = compute_hotspot_index(cube, wavelength)
    if swir_radiance is None:
        swir_radiance = swir_radiance_magnitude(cube, wavelength)
    _, boot_bkg, _ = detect_fire_pixels(hotspot_index, method='robust', k=bootstrap_k,
                                        margin_px=margin_px, swir_radiance=swir_radiance,
                                        radiance_threshold=robust_background_threshold(swir_radiance, k=bootstrap_k))
    pool_bkg = boot_bkg
    for _pass in range(max(int(pool_passes), 1)):
        rows, cols = np.where(pool_bkg)
        if n_pool is None:
            sel = np.arange(len(rows))                    # the whole bootstrap background
        elif sampling == 'systematic':
            from sampling import systematic_indices
            sel = systematic_indices(len(rows), min(n_pool, len(rows)))
        elif sampling == 'random':
            sel = rng.choice(len(rows), size=min(n_pool, len(rows)), replace=False)
        else:
            raise ValueError(f"sampling must be 'systematic' or 'random', got {sampling!r}")
        if n_pool is None:
            # whole background: kept float32 and gathered in chunks (a float64
            # copy of every band of a million pixels would be ~1.8 GB)
            pool = np.empty((len(sel), int(m.sum())), np.float32)
            for i in range(0, len(sel), 65536):
                s_ = sel[i:i + 65536]
                pool[i:i + len(s_)] = np.asarray(cube[rows[s_], :, cols[s_]], np.float32)[:, m]
        else:
            pool = cube[rows[sel], :, cols[sel]].astype(float)[:, m]
        pool_rc = np.column_stack([rows[sel], cols[sel]])
        # full band axis on the first pass; already the fit window afterwards
        ceiling = None if ceiling is None else (np.asarray(ceiling, float)[m] if np.size(ceiling) == len(m)
                                                else np.asarray(ceiling, float))
        if ceiling is None and raster_path is not None:
            try:
                import enmap_quality as eq
                files = eq.find_quality_files(raster_path)
                if files.get('metadata'):
                    ceiling = np.asarray(eq.band_ceilings_from_metadata(files['metadata'])[0], float)[m]
            except Exception as exc:                          # pragma: no cover
                if verbose:
                    print(f'  detector: band ceilings unavailable ({exc.__class__.__name__}: {exc})')
        det = FireGLRT(w, pool, K=K, ceiling=ceiling, pool_rc=pool_rc, rng=rng, sampling=sampling,
                       n_fit=None if n_pool is None else 8000, pool_trim=pool_trim if n_pool is None else None)
        window = np.ascontiguousarray(np.transpose(cube[:, m, :], (0, 2, 1)))
        out = det.detect_cube(window, p_fa=p_fa)
        del window
        if _pass + 1 >= int(pool_passes):
            break
        # leave out everything the detector itself finds fire-like, then retrain
        thr_c = det.threshold(clean_p_fa)
        hot = (np.nan_to_num(out['lam'], nan=0.0) > thr_c) | out['detections'] | out['candidates']
        excl = ndi.binary_dilation(hot, iterations=int(margin_px)) if hot.any() else hot
        new_bkg = boot_bkg & ~excl
        changed = float((new_bkg != pool_bkg).sum()) / max(int(pool_bkg.sum()), 1)
        if verbose:
            print(f'  detector pass {_pass + 1}: {int(hot.sum())} pixels above the {clean_p_fa:g} null quantile or '
                  f'classed fire/candidate; pool {int(pool_bkg.sum())} -> {int(new_bkg.sum())} px '
                  f'(Planck at 1100 K retained {det.planck_retained[int(np.argmin(abs(det.T_grid - 1100)))]:.3f})')
        pool_bkg = new_bkg
        if changed < 1e-3:
            break
    fire_mask = out['detections'].copy()
    cls = np.where(out['detections'], 1, 0)
    if include_candidates:
        fire_mask |= out['candidates']
        cls = np.where(out['candidates'], 2, cls)
    fire_points = np.argwhere(fire_mask)
    buffered = ndi.binary_dilation(fire_mask, iterations=int(margin_px)) if fire_mask.any() else fire_mask
    background_mask = (~buffered) & np.isfinite(hotspot_index)
    info = dict(detector=det, lam=out['lam'], chi=out['chi'], ratio=out['ratio'], T_best=out['T_best'],
                threshold=out['threshold'], p_fa=p_fa, null_source=out['null_source'],
                threshold_source=out['threshold_source'], tail_shape=out['tail_shape'], heavy_tail=out['heavy_tail'],
                chi_gate=det.chi_gate, explained_variance=det.explained, K=K,
                n_fire=int(out['detections'].sum()), n_candidate=int(out['candidates'].sum()),
                point_class=cls[fire_points[:, 0], fire_points[:, 1]] if len(fire_points) else np.zeros(0, int),
                point_lambda=out['lam'][fire_points[:, 0], fire_points[:, 1]] if len(fire_points) else np.zeros(0),
                point_T_best=out['T_best'][fire_points[:, 0], fire_points[:, 1]] if len(fire_points) else np.zeros(0),
                tau_source=det.tau_source, n_pool=int(len(pool)), n_pool_passes=_pass + 1,
                planck_retained=dict(zip(det.T_grid.tolist(), det.planck_retained)))
    if verbose:
        print(f'  Fire detection (GLRT, this work): background subspace K={K} explains '
              f'{100 * det.explained:.2f}% of the fire-free pool ({len(pool)} px); fire-term transmittance '
              f'prior: {det.tau_source}; null from the {out["null_source"]}; threshold {out["threshold"]:.1f} '
              f'at p_fa {p_fa:g} per pixel ({p_fa * n_r * n_c:.0f} false alarms expected per tile; '
              f'{out["threshold_source"]}, null tail shape {out["tail_shape"]:.2f}'
              + (', HEAVY-TAILED: the rate is uncertain by a factor of several on this scene' if out['heavy_tail'] else '')
              + f'); {info["n_fire"]} fire pixels + {info["n_candidate"]} candidates (above threshold, inadequate fit).')
    return fire_points, background_mask, info

#!/usr/bin/env python
# coding: utf-8
"""
Step 2 of the two-step retrieval: two-sided 95% bounds on the fire radiative
power (FRP) of a whole fire CLUSTER, including its saturated and badly
fitted pixels.

Step 1 (ProfileML v2, profile_retrieval.py) retrieves one temperature per
pixel and fails where most bands are saturated or flagged (VERIFICATION.md
Section 24.7). Section 26 showed that the temperature of a single saturated
pixel is not identified without a shape assumption. What IS constrained is
the power of the cluster: the PSF only moves radiance between neighbouring
pixels, so a sum over a cluster does not depend on where each fire sits.

MODEL (linear in every unknown, so all problems are convex):

    y_p(l) = sum_j a_pj L_pj(l) + tau(l) e sum_t G_pt B(l, T_t)              (1)
    G_pt   = sum_{d in 3x3} H_{p-d, d, t}       radiance RECEIVED by pixel p
    hmin_d E_qt <= H_qdt <= hmax_d E_qt,  (1 - m_far) E_qt <= sum_d H_qdt <= E_qt
    sum_t E_qt <= 1 (area fractions),  a >= 0, sum_j a_pj <= 1

  E_qt : non-negative area fraction of pixel q radiating at grid temperature
         T_t (non-parametric temperature distribution);
  H_qdt: part of it recorded by the neighbour q + d;
  hmin_d, hmax_d: the smallest / largest share of offset d over EVERY point
         position inside the pixel and every PSF width in `sigma_scales` x
         the EnMAP MTF requirement (psf.psf_point_weights);
         m_far: the largest share falling outside the 3 x 3 neighbourhood.

The share constraints are a RELAXATION of 'a fire somewhere in pixel q,
blurred by the PSF': every fire position (and mixture of positions) satisfies
them, so no sub-pixel position grid is needed. (Section 26 used a 3 x 3
position grid; its bounds could miss the truth when the fire lay between grid
points.) They also force at least 1 - max hmax_self (~50%) of a pixel's
radiance into its neighbours, whose unsaturated bands then limit what a
saturated core can hide: this is the source of the UPPER bound.

The target is FRP_c = sigma A sum_{q in footprint} sum_t E_qt T_t^4 over the
cluster footprint (the detected pixels and a 1-pixel ring). Data rows: every
pixel within 1 of the footprint. Free fire: every pixel within 2 of the
footprint. Fire outside the footprint is a nuisance that may explain the
ring, so the bounds are valid but can be conservative. As in psf_convex,
bounds are profile-likelihood (strict) bounds: the extremes of FRP_c over
every solution with J <= J* + chi2_1(0.95) max(1, J*/n). Also bounded: the
flaming power P900_c (fire >= 900 K).

The lower bound of 0 or an upper bound at the area limit means not identified.

DIFFERENCES FROM Keely et al. (2026, arXiv:2608.07580): they fit each pixel
independently with a two-Gaussian p(T), mask saturated bands, optimise a
non-convex loss with Adam, and report per-pixel E[T], sigma[T]. Here: a
cluster-level radiance budget through the PSF, a non-parametric p(T),
saturated bands as one-sided constraints, a convex problem solved to its
global optimum, and bounds on cluster FRP instead of a per-pixel temperature
(which Section 26 found not identified in saturated cores).

STATUS: experimental, VERIFICATION.md Section 27. Runs in the uv environment
of psf_convex (cvxpy + Clarabel).
"""

import time

import numpy as np

from psf_convex import planck_nm, temperature_grid, SIGMA_SB, PIXEL_AREA_M2, _CHI2_1_95

__all__ = ['offset_shares', 'cluster_footprints', 'cluster_bounds']

_OFFSETS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]


def offset_shares(sigma_along=None, sigma_across=None, sigma_scales=(1.0,), n_u=41):
    """(hmin, hmax, m_far): min / max share of each 3 x 3 offset, and the max
    share outside the 3 x 3, over point positions in [-0.5, 0.5]^2 and PSF
    widths sigma x sigma_scales. The PSF is separable, so per axis."""
    from psf import aperture_weights as _aperture_weights, default_sigmas
    sa, sc = default_sigmas(sigma_along, sigma_across)
    u = np.linspace(-0.5, 0.5, n_u)
    lo, hi, far = np.full((3, 3), np.inf), np.zeros((3, 3)), 0.0
    for s in sigma_scales:
        wa = np.array([_aperture_weights(sa * s, x, 4) for x in u])        # (n_u, 9)
        wc = np.array([_aperture_weights(sc * s, x, 4) for x in u])
        wa /= wa.sum(1, keepdims=True); wc /= wc.sum(1, keepdims=True)
        ia, ic = wa[:, 3:6], wc[:, 3:6]                                    # offsets -1, 0, 1
        lo = np.minimum(lo, np.outer(ia.min(0), ic.min(0)))
        hi = np.maximum(hi, np.outer(ia.max(0), ic.max(0)))
        inner = ia.sum(1)[:, None] * ic.sum(1)[None, :]                    # all position pairs
        far = max(far, float(1.0 - inner.min()))
    return lo, hi, far


def axis_hulls(sigma_along=None, sigma_across=None, sigma_scales=(1.0,), n_u=21, tol=1e-3, max_facets=None):
    """Joint constraints on the shares of one axis. Along an axis, a point fire
    at u sends (w_-1, w_0, w_1)(u) to the three pixels, times the share
    gamma in [gamma_min, 1] that the other axis keeps inside its 3 pixels. Any
    fire (a mixture of points) has its marginal share vector in the convex
    hull of these points: returned as facets (N, b), N v <= b, per axis
    ('along', 'across'), with b loosened by `tol` (the hull of a sample of
    the curve is slightly inside the hull of the curve). Unlike per-offset
    min/max shares this excludes a fire that is both centred (large own
    share) and at an edge (large neighbour share)."""
    from scipy.spatial import ConvexHull
    from psf import aperture_weights as _aperture_weights, default_sigmas
    sa, sc = default_sigmas(sigma_along, sigma_across)
    u = np.linspace(-0.5, 0.5, n_u)
    inner = {}
    for name, sg in (('along', sa), ('across', sc)):
        pts = []
        for s in sigma_scales:
            w = np.array([_aperture_weights(sg * s, x, 4) for x in u])
            w /= w.sum(1, keepdims=True)
            pts.append(w[:, 3:6])
        inner[name] = np.vstack(pts)
    out = {}
    for name, other in (('along', 'across'), ('across', 'along')):
        g_min = float(inner[other].sum(1).min())
        P = np.vstack([inner[name], g_min * inner[name], np.zeros((1, 3))])
        h = ConvexHull(P)
        N, b = h.equations[:, :3], -h.equations[:, 3]
        if max_facets is not None and len(b) > max_facets:
            # any subset of the facets is still an outer approximation (valid,
            # looser); keep well-spread normals (farthest-point sampling)
            keep = [int(np.argmax(N[:, 1]))]                 # the own-share facet first
            while len(keep) < max_facets:
                d = np.min(1.0 - N @ N[keep].T, axis=1)
                keep.append(int(np.argmax(d)))
            N, b = N[keep], b[keep]
        out[name] = (N, b + tol * np.linalg.norm(N, axis=1))
    return out


def grid_frp_factors(wavelength_nm, T_grid, delta=0.005, T_eval=None, solver='CLARABEL'):
    """(kappa_lo, kappa_hi) of a temperature grid. For a fire at any T in
    T_eval, the non-negative grid mixtures whose spectrum is within relative
    `delta` of B(T) have FRP in [kappa(T) lo, kappa(T) hi] x FRP(T); kappa_lo
    is the largest lower and kappa_hi the smallest upper ratio. Mixtures of
    grid Planck curves OVERstate the FRP of an in-between temperature (T^4 is
    more convex than the SWIR radiance), so a lower bound computed on the grid
    must be divided by kappa_lo (>= 1) to stay valid; a sum of fires is
    represented by the sum of their mixtures, so the factor also holds for any
    temperature distribution. Physics only, no data."""
    import cvxpy as cp
    Tg = np.asarray(T_grid, float)
    T_eval = np.linspace(Tg[0], Tg[-1], 4 * len(Tg) + 1) if T_eval is None else np.asarray(T_eval, float)
    B = np.stack([planck_nm(wavelength_nm, t) for t in Tg]).T
    x = cp.Variable(len(Tg), nonneg=True)
    k_lo, k_hi = 1.0, 1.0
    for t in T_eval:
        bt = planck_nm(wavelength_nm, t)
        Br = B / bt[:, None]                      # relative to B(T): rows O(1)
        cs = Br.max(axis=0)                       # column scale, x = y / cs
        cons = [cp.abs((Br / cs) @ x - 1.0) <= delta]
        frp_ratio = ((Tg / t) ** 4 / cs) @ x
        p_lo, p_hi = cp.Problem(cp.Minimize(frp_ratio), cons), cp.Problem(cp.Maximize(frp_ratio), cons)
        p_lo.solve(solver=solver)
        p_hi.solve(solver=solver)
        if p_lo.status != 'optimal' or p_hi.status != 'optimal':
            raise RuntimeError(f'grid_frp_factors: {p_lo.status}/{p_hi.status} at {t:.0f} K')
        k_lo, k_hi = max(k_lo, p_lo.value), min(k_hi, p_hi.value)
    return float(k_lo), float(k_hi)


def cluster_footprints(detected, ring=1, tile=None):
    """Footprints: 8-connected components of the detected mask grown by `ring`,
    each cut into tile x tile blocks when `tile` is given (sums of the block
    bounds bound the cluster; smaller blocks are cheaper but looser, since
    fire across a block edge is a free nuisance)."""
    from scipy.ndimage import binary_dilation, label
    grown = binary_dilation(np.asarray(detected, bool), np.ones((3, 3), bool), iterations=ring)
    lab, n = label(grown, np.ones((3, 3), int))
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if tile is None:
            out.append(m)
            continue
        ii, jj = np.nonzero(m)
        for i0 in range(ii.min(), ii.max() + 1, tile):
            for j0 in range(jj.min(), jj.max() + 1, tile):
                b = np.zeros_like(m)
                b[i0:i0 + tile, j0:j0 + tile] = True
                if (b & m).any():
                    out.append(b & m)
    return out


def _dilate(m, n):
    from scipy.ndimage import binary_dilation
    return binary_dilation(m, np.ones((3, 3), bool), iterations=n) if n else m.copy()


def _solve(problem, solver):
    """Solve; on a solver failure or a non-optimal status retry with more
    iterations, then with stronger regularisation. The final status is kept
    (callers flag anything but 'optimal')."""
    import cvxpy as cp
    tries = [{}, dict(max_iter=1000), dict(max_iter=1000, static_regularization_constant=1e-7,
                                           equilibrate_max_iter=50)]
    status = 'solver_error'
    for opts in tries:
        try:
            problem.solve(solver=solver, **opts)
        except cp.error.SolverError:
            continue
        status = problem.status          # only a status from a solve that returned
        if status == 'optimal':
            break
    return status


def cluster_bounds(Y, valid, censored, level, libraries, sigma, wavelength, tau, footprint,
                   T_grid=None, emissivity=1.0, censor_weight=100.0, rel_fire_error=0.02,
                   T_split=900.0, sigma_scales=(1.0,), frp_factors=None, use_hulls=True, max_facets=12, area_slack=0.02, solver='CLARABEL',
                   verbose=False):
    """95% bounds of the FRP (W) and flaming power P900 (W) of the fire inside
    `footprint` (bool R x C). libraries: (R, C) object array of background
    spectra (n_j, K), None = no background model (pixel not used as data)."""
    import cvxpy as cp
    from scipy.sparse import coo_matrix
    t0 = time.time()
    Y = np.asarray(Y, float)
    R, C, K = Y.shape
    T = temperature_grid(450.0, 2000.0, 24) if T_grid is None else np.asarray(T_grid, float)
    if frp_factors is None:                      # grid representation bias (grid_frp_factors)
        frp_factors = grid_frp_factors(wavelength, T)
    k_lo, k_hi = frp_factors
    nT = len(T)
    fp = np.asarray(footprint, bool)
    has_lib = np.array([[libraries[i][j] is not None for j in range(C)] for i in range(R)])
    data_m = _dilate(fp, 1) & has_lib
    fire_m = _dilate(fp, 2)
    data_px = list(zip(*np.nonzero(data_m)))
    fire_px = list(zip(*np.nonzero(fire_m)))
    nF, nP = len(fire_px), len(data_px)
    pidx = {p: n for n, p in enumerate(data_px)}
    hmin, hmax, m_far = offset_shares(sigma_scales=sigma_scales)
    B = np.stack([emissivity * np.asarray(tau, float) * planck_nm(wavelength, t) for t in T])   # (nT, K)
    s = np.sqrt(np.mean(B ** 2, axis=1))
    Bn = B / s[:, None]                         # columns in units of s_t x area fraction
    # G = M @ vec(H): H indexed (f, d, t) -> G (p, t)
    rr, cc = [], []
    for f_n, (i, j) in enumerate(fire_px):
        for d_n, (di, dj) in enumerate(_OFFSETS):
            p = (i + di, j + dj)
            if p in pidx:
                base_h = (f_n * 9 + d_n) * nT
                rr.append(pidx[p] * nT + np.arange(nT)); cc.append(base_h + np.arange(nT))
    rr = np.concatenate(rr) if rr else np.zeros(0, int)
    cc = np.concatenate(cc) if cc else np.zeros(0, int)
    M = coo_matrix((np.ones(len(rr)), (rr, cc)), shape=(nP * nT, nF * 9 * nT)).tocsr()
    # data rows over [G (nP nT), a (sum n_j)]
    nlib = [libraries[p[0]][p[1]].shape[0] for p in data_px]
    a0 = nP * nT + np.concatenate([[0], np.cumsum(nlib)[:-1]]).astype(int)
    ncol = nP * nT + int(sum(nlib))
    acc = {'v': ([], [], [], []), 'c': ([], [], [], [])}
    nrow = {'v': 0, 'c': 0}
    vcen = np.asarray(censored, bool) & ~np.asarray(valid, bool)
    lvl = np.nan_to_num(np.asarray(level, float))
    for n, p in enumerate(data_px):
        L = np.asarray(libraries[p[0]][p[1]], float)
        v, c = np.asarray(valid[p], bool), vcen[p]
        sig2 = np.asarray(sigma[p], float) ** 2 + (rel_fire_error * np.maximum(Y[p] - L.min(0), 0.0)) ** 2
        w = 1.0 / sig2
        w_med = float(np.median(w[v])) if v.any() else float(np.median(w))
        for kind, msk in (('v', v), ('c', c)):
            kk = np.flatnonzero(msk)
            if not len(kk):
                continue
            sw = np.sqrt(w[kk]) if kind == 'v' else np.full(len(kk), np.sqrt(censor_weight * w_med))
            rows = nrow[kind] + np.arange(len(kk))
            R_, C_, V_, b_ = acc[kind]
            for jl in range(L.shape[0]):
                R_.append(rows); C_.append(np.full(len(kk), a0[n] + jl)); V_.append(L[jl, kk] * sw)
            R_.append(np.tile(rows, nT)); C_.append(np.repeat(n * nT + np.arange(nT), len(kk)))
            V_.append((Bn[:, kk] * sw[None, :]).ravel())
            b_.append((Y[p][kk] if kind == 'v' else lvl[p][kk]) * sw)
            nrow[kind] += len(kk)
    cat = lambda x: np.concatenate(x) if x else np.zeros(0)
    mats = {}
    for kind in ('v', 'c'):
        R_, C_, V_, b_ = acc[kind]
        mats[kind] = (coo_matrix((cat(V_), (cat(R_).astype(int), cat(C_).astype(int))),
                                 shape=(nrow[kind], ncol)).tocsr(), cat(b_))
    out = dict(n_footprint=int(fp.sum()), n_data_px=nP, n_fire_px=nF, n_rows=nrow['v'],
               n_cens_rows=nrow['c'], m_far=m_far, use_hulls=use_hulls)
    if nrow['v'] == 0:
        out['status'] = 'no data'
        return out

    E = cp.Variable((nF, nT), nonneg=True)
    H = cp.Variable((nF * 9, nT), nonneg=True)
    z = cp.Variable(ncol, nonneg=True)                        # [G, a]
    Hr = [H[d::9, :] for d in range(9)]                       # (nF, nT) per offset
    cons = [z[:nP * nT] == M @ cp.vec(H.T, order='F')]        # vec over (f, d, t) with t fastest
    hsum = sum(Hr)
    cons += [hsum <= E, hsum >= (1.0 - m_far) * E]
    for d in range(9):
        di, dj = _OFFSETS[d]
        cons += [Hr[d] <= hmax[di + 1, dj + 1] * E]
        if hmin[di + 1, dj + 1] > 1e-6:
            cons += [Hr[d] >= hmin[di + 1, dj + 1] * E]
    hulls = axis_hulls(sigma_scales=sigma_scales, max_facets=max_facets) if use_hulls else {}
    Hd = {(di, dj): Hr[n] for n, (di, dj) in enumerate(_OFFSETS)}
    marg = {'along': [sum(Hd[(di, dj)] for dj in (-1, 0, 1)) for di in (-1, 0, 1)],
            'across': [sum(Hd[(di, dj)] for di in (-1, 0, 1)) for dj in (-1, 0, 1)]}
    for name, (N, b) in hulls.items():
        A = marg[name]
        cons += [N[f, 0] * A[0] + N[f, 1] * A[1] + N[f, 2] * A[2] <= b[f] * E for f in range(len(b))]
    out['hull_facets'] = {k: len(v[1]) for k, v in hulls.items()}
    cons += [E @ (1.0 / s) <= 1.0]                            # area fraction of each pixel
    # burning area replaces background, as in step 1 (f + sum q <= 1 with shade);
    # area_slack absorbs library brightness errors
    fidx = {q: n for n, q in enumerate(fire_px)}
    cons += [cp.sum(z[a0[n]:a0[n] + nlib[n]]) + (E[fidx[p], :] @ (1.0 / s) if p in fidx else 0)
             <= 1 + area_slack for n, p in enumerate(data_px)]
    J = cp.sum_squares(mats['v'][0] @ z - mats['v'][1])
    if nrow['c']:
        J = J + cp.sum_squares(cp.pos(mats['c'][1] - mats['c'][0] @ z))
    pr0 = cp.Problem(cp.Minimize(J), cons)
    fit_status = _solve(pr0, solver)
    out['fit_status'] = fit_status
    if fit_status not in ('optimal', 'optimal_inaccurate'):
        out['status'] = f'fit {fit_status}'
        return out
    J_star = float(pr0.value)
    scale = max(1.0, J_star / nrow['v'])
    Delta = _CHI2_1_95 * scale
    in_fp = np.array([fp[q] for q in fire_px], float)
    cmat = cp.Parameter((nF, nT))
    prb = cp.Problem(cp.Maximize(cp.sum(cp.multiply(cmat, E))), cons + [J <= J_star + Delta])
    statuses, sols = [], []

    def extreme(weight_T, sense):
        c = np.outer(in_fp, SIGMA_SB * PIXEL_AREA_M2 * T ** 4 * weight_T / s)
        cn = float(np.max(np.abs(c))) or 1.0
        cmat.value = sense * c / cn
        st = _solve(prb, solver)
        statuses.append(st)
        sols.append(None if st not in ('optimal', 'optimal_inaccurate') else
                    (sense, np.array(E.value), np.array(z.value)))
        if st not in ('optimal', 'optimal_inaccurate'):
            return np.nan
        return sense * float(prb.value) * cn

    one = np.ones(nT)
    hot = (T >= T_split).astype(float)
    # raw grid bounds, then corrected for the grid's FRP representation bias
    out['FRP_lo_grid'] = max(extreme(one, -1), 0.0)
    out['FRP_hi_grid'] = extreme(one, +1)
    out['P900_lo_grid'] = max(extreme(hot, -1), 0.0)
    out['P900_hi_grid'] = extreme(hot, +1)
    out['frp_factors'] = (k_lo, k_hi)
    out['FRP_lo'] = out['FRP_lo_grid'] / k_lo
    out['FRP_hi'] = out['FRP_hi_grid'] / k_hi
    # the 900 K split is itself on the grid: a fire just below 900 K may be
    # represented partly above it, so P900 is reported uncorrected and
    # flagged as grid-level in the docs
    out['P900_lo'], out['P900_hi'] = out['P900_lo_grid'], out['P900_hi_grid']
    # the largest FRP the area constraint alone allows (bound 'open' if reached)
    out['FRP_area_limit'] = float(fp.sum() * SIGMA_SB * PIXEL_AREA_M2 * T[-1] ** 4)
    out.update(status='ok' if all(st == 'optimal' for st in statuses) else 'inaccurate',
               bound_statuses=statuses, solutions=sols, fire_px=fire_px, T_grid=T, s=s, J_star=J_star, scale=scale, Delta=Delta, seconds=time.time() - t0)
    if verbose:
        print(f"  cluster {int(fp.sum())} px: FRP [{out['FRP_lo'] / 1e6:.2f}, {out['FRP_hi'] / 1e6:.2f}] MW "
              f"({out['seconds']:.0f} s)", flush=True)
    return out

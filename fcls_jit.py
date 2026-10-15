#!/usr/bin/env python
# coding: utf-8
"""
Certified active-set continuation for the profile grid of
profile_retrieval._FCLSProblem, compiled with numba (optional).

WHY
---
The profile chi2_p(T) is evaluated on a 200-point temperature grid, twice
per retrieval, about 26 retrievals per fire pixel in the stage-4 chain:
~10,000 NNLS solves per pixel, each solved from scratch by Lawson-Hanson
over a ~62-column library (85-110 us compiled). On the fire pixels of
DT0000149658 the solution uses a median of 3 columns and the active set is
the same at neighbouring grid temperatures 93% of the time.

WHAT
----
The Lawson-Hanson solution at one grid point hands its active set P to the
following points. At each of them the problem restricted to P is solved
directly and accepted only if it is CERTIFIED optimal for the full problem
(KKT: every coefficient on P positive, no excluded column with a positive
dual - the problem is convex, so this is the global optimum). The first
point that fails goes back to Lawson-Hanson, whose P is carried on. Nothing
is approximated; a certified point is the same optimum as Lawson-Hanson's
to rounding (max relative chi2 difference 2e-13 over 58,400 real grid
problems in the NumPy prototype this ports).

Numerics (each of these produced wrong certificates when done naively):
* The closure is a ROW weighted omega = 1e4 x the row scale (up to ~4e8).
  It is never formed here: eps*omega^2 of rounding noise (tens) would swamp
  the duals being tested. The closure multiplier mu is taken from
  stationarity on the passive columns instead.
* The weighted-row problem is not exactly closed (sum(a) - 1 reaches 3e-4 on
  censored pixels), so the problem with the closure level c as a free
  right-hand side is solved for c = 1 and its derivative, everything is
  affine in c, and the penalty's stationarity mu(c) = omega^2 (1 - c) gives
  c in closed form: the SAME estimator as the weighted row, not the exactly
  closed one.
* The 1e3-weighted censor rows carry the same kind of noise, so problems
  with censored bands are not handled here (the caller solves them in full).
* Only the fire column depends on T: the library part of P is factorised
  once per active set, every grid point is a rank-1 extension of it, and
  the 60-column dual comes from H = FW @ Bw computed once per sweep.

* Active sets without the fire column are left to Lawson-Hanson: the
  profile is flat in T there and Lawson-Hanson's exact ties must be kept.
* Grid values within TIE_RTOL of a neighbour or of the minimum are re-solved
  by Lawson-Hanson after the sweep (see GridContinuation.sweep).

REPRODUCIBILITY (measured on DT0000149658 tile 014, 757 fire pixels)
--------------------------------------------------------------------
The stage-4 outputs are bit-identical to the plain loop: every temperature,
fraction, chi2, likelihood-ratio statistic, local-minimum count, tau and
eps_m. Only the interval ends move, by <= 6e-12 K, because the interval-edge
root finding uses _FCLSProblem.chi2_at() (certified values). Brent's search
deliberately does NOT: its T_hat sets the next IRLS pass's covariance, and a
1e-13 change there shifted every value of that pass and the plateau
diagnostics read from it.

Speed: the uncensored grid sweep is 5.5-6.6x faster than one Lawson-Hanson
per point, a retrieval ~1.8x (kernel covariance) - but stage 4 as a whole
only 144 s -> 114 s, because bit-identity keeps Lawson-Hanson on the
plateaus (~35% of grid points), in Brent and on censored pixels, and the
16 workers share 8 physical cores. A mode that used every certified value
as is (rounding-level differences, T within ~1e-6 K) was measured no
faster at stage level and was dropped.

Without numba, `available` is False and the caller keeps its plain loop.
Environment: numba is not in `emit` (which must not be modified); it lives
in .venv-numba, a uv venv on emit's own interpreter with
--system-site-packages (emit's numpy/scipy/GDAL, numba 0.68 only). Run with
    conda run -n emit .venv-numba\\Scripts\\python.exe <script>
(conda activation is still needed for the DLL paths).
"""

import math
import os

import numpy as np

__all__ = ['available', 'GridContinuation']

DUAL_RTOL = 1e-8   # accept dual_c <= DUAL_RTOL ||D_c|| ||r||: entering c could
                   # lower ||r||^2 by at most (dual_c/||D_c||)^2 <= 1e-16 ||r||^2
TIE_RTOL = 1e-11   # grid values closer than this are compared as Lawson-Hanson values
                   # (>= 50x the largest certified error measured, 1.8e-13)

try:
    if os.environ.get('PROFILEML_JIT', '1') == '0':
        raise ImportError('disabled by PROFILEML_JIT=0')
    from numba import njit
    available = True
except ImportError:                                   # pragma: no cover - depends on the env
    available = False

    def njit(*a, **k):
        return (lambda f: f) if not (a and callable(a[0])) else a[0]


@njit(cache=True, fastmath=False)
def _sweep(k0, FW, H, fwn2, scale, d_e, QL, RL, b_p, d_p, Qb, Qd, QB, bB, dB, Bd,
           lib_cols, lib_e, has_fire, others_cols, out_cols, cn_lib_sorted, cn_lib,
           chi2_out):
    """Certify grid points k0, k0+1, ... with one active set; stops at the
    first point that is not certified. Returns that index (len(FW) if all)."""
    nT, R = FW.shape
    m = QL.shape[1]
    nlib_all = cn_lib.shape[0]
    n_ab = lib_cols.shape[0] + (1 if has_fire else 0)
    f = np.empty(R)
    f_p = np.empty(R)
    r1 = np.empty(R)
    rdp = np.empty(R)
    Qf = np.empty(m)
    v1 = np.empty(m)
    vd = np.empty(m)
    L1 = np.empty(m)
    Ld = np.empty(m)
    g1 = np.empty(nlib_all + 1)
    gd = np.empty(nlib_all + 1)
    cn = np.empty(nlib_all + 1)
    for k in range(k0, nT):
        xf1 = 0.0
        xfd = 0.0
        if has_fire:
            ff_raw = 0.0
            for r in range(R):
                f[r] = FW[k, r] - d_e[r]
                ff_raw += f[r] * f[r]
            for j in range(m):
                s = 0.0
                for r in range(R):
                    s += QL[r, j] * f[r]
                Qf[j] = s
            ff = 0.0
            fb = 0.0
            fd = 0.0
            for r in range(R):
                s = f[r]
                for j in range(m):
                    s -= QL[r, j] * Qf[j]
                f_p[r] = s
                ff += s * s
                fb += s * b_p[r]
                fd += s * d_p[r]
            if not (ff > 1e-20 * max(ff_raw, 1e-300)):
                return k
            xf1 = fb / ff
            xfd = fd / ff
            for r in range(R):
                r1[r] = b_p[r] - xf1 * f_p[r]
                rdp[r] = d_p[r] - xfd * f_p[r]
        else:
            for j in range(m):
                Qf[j] = 0.0
            for r in range(R):
                r1[r] = b_p[r]
                rdp[r] = d_p[r]
        # library coefficients: RL z = Qb - xf Qf (back substitution)
        for j in range(m):
            v1[j] = Qb[j] - xf1 * Qf[j]
            vd[j] = Qd[j] - xfd * Qf[j]
        for j in range(m - 1, -1, -1):
            s1 = v1[j]
            sd = vd[j]
            for t in range(j + 1, m):
                s1 -= RL[j, t] * L1[t]
                sd -= RL[j, t] * Ld[t]
            L1[j] = s1 / RL[j, j]
            Ld[j] = sd / RL[j, j]
        # gradients D^T r: fire column directly, library from H
        g1f = 0.0
        gdf = 0.0
        kap = 0.0
        for r in range(R):
            g1f += FW[k, r] * r1[r]
            gdf += FW[k, r] * rdp[r]
            kap += rdp[r] * rdp[r]
        g1[0] = g1f
        gd[0] = gdf
        for c in range(nlib_all):
            fpb = H[k, c] - Bd[c]
            for j in range(m):
                fpb -= Qf[j] * QB[j, c]
            if has_fire:
                g1[c + 1] = bB[c] - xf1 * fpb
                gd[c + 1] = dB[c] - xfd * fpb
            else:
                g1[c + 1] = bB[c]
                gd[c + 1] = dB[c]
        # closure multiplier from stationarity on the passive abundance columns
        s = g1[0] if has_fire else 0.0
        for i in range(lib_cols.shape[0]):
            s += g1[lib_cols[i]]
        mu1 = -s / n_ab
        dc = -mu1 / (kap + scale[k] * scale[k])            # closure level c - 1
        mu = mu1 + dc * kap
        # coefficients on P must be positive
        a_f = xf1 - dc * xfd
        if has_fire and not (a_f > 0.0):
            return k
        a_sum = a_f if has_fire else 0.0
        bad = False
        for j in range(m):
            a_j = L1[j] - dc * Ld[j]
            if not (a_j > 0.0):
                bad = True
                break
            a_sum += a_j
        if bad:
            return k
        a_e = 1.0 + dc - a_sum
        if not (a_e > 0.0):
            return k
        # residual norm (= chi2 over the data rows)
        rn2 = 0.0
        for r in range(R):
            t = r1[r] - dc * rdp[r]
            rn2 += t * t
        rn = math.sqrt(rn2)
        # column norms, floored at their median (the shade has no data rows)
        cnf = math.sqrt(fwn2[k])
        nL = cn_lib_sorted.shape[0]
        ntot = nL + 1
        # median of [cnf] + cn_lib_sorted
        def_lo = (ntot - 1) // 2
        def_hi = ntot // 2
        med_lo = 0.0
        med_hi = 0.0
        ins = 0
        while ins < nL and cn_lib_sorted[ins] < cnf:
            ins += 1
        for q in range(2):
            idx = def_lo if q == 0 else def_hi
            if idx < ins:
                val = cn_lib_sorted[idx]
            elif idx == ins:
                val = cnf
            else:
                val = cn_lib_sorted[idx - 1]
            if q == 0:
                med_lo = val
            else:
                med_hi = val
        med = 0.5 * (med_lo + med_hi)
        # dual feasibility on every column outside P
        for i in range(out_cols.shape[0]):
            c = out_cols[i]
            w = g1[c] - dc * gd[c] + mu
            cnc = cnf if c == 0 else cn_lib[c - 1]
            if cnc < med:
                cnc = med
            if w > DUAL_RTOL * cnc * rn:
                return k
        chi2_out[k] = rn2
    return nT


@njit(cache=True, fastmath=False)
def _segment(k0, FW, H, fwn2, scale, Bw, yw, cn_lib, cn_lib_sorted, act, p, chi2_out):
    """Factorise the library part of the active set `act` (column indices,
    ascending, abundances only) and certify grid points from k0 on with it.
    Returns the first uncertified index, or -1 if the set cannot be used."""
    R, nL = Bw.shape
    has_fire = act.shape[0] > 0 and act[0] == 0
    n_lib = act.shape[0] - (1 if has_fire else 0)
    if n_lib < 1 or not has_fire:
        # Without the fire column the profile is flat in T, and Lawson-
        # Hanson returns EXACTLY tied chi2 values along it; a certified
        # value differs from those by ~1e-13, and that jitter creates
        # spurious local minima in the grid diagnostics and moves the
        # argmin along the plateau. Such sets are left to Lawson-Hanson.
        return -1
    lib = np.empty(n_lib, np.int64)
    for i in range(n_lib):
        lib[i] = act[i + (1 if has_fire else 0)]
    # eliminated column: smallest norm (the shade, when present)
    e = 0
    best = np.inf
    for i in range(n_lib):
        nrm = cn_lib[lib[i] - 1]
        if nrm < best:
            best = nrm
            e = i
    d_e = np.empty(R)
    for r in range(R):
        d_e[r] = Bw[r, lib[e] - 1]
    m = n_lib - 1
    QL = np.zeros((R, m))
    RL = np.zeros((m, m))
    j = 0
    for i in range(n_lib):
        if i == e:
            continue
        v = np.empty(R)
        nv0 = 0.0
        for r in range(R):
            v[r] = Bw[r, lib[i] - 1] - d_e[r]
            nv0 += v[r] * v[r]
        for _ in range(2):                                 # CGS2
            for t in range(j):
                c = 0.0
                for r in range(R):
                    c += QL[r, t] * v[r]
                for r in range(R):
                    v[r] -= c * QL[r, t]
                RL[t, j] += c
        nv = 0.0
        for r in range(R):
            nv += v[r] * v[r]
        nv = math.sqrt(nv)
        if not (nv > 1e-10 * math.sqrt(nv0)) or nv == 0.0:
            return -1
        RL[j, j] = nv
        for r in range(R):
            QL[r, j] = v[r] / nv
        j += 1
    b_p = np.empty(R)
    d_p = np.empty(R)
    for r in range(R):
        b_p[r] = yw[r] - d_e[r]
        d_p[r] = d_e[r]
    Qb = np.empty(m)
    Qd = np.empty(m)
    for t in range(m):
        cb = 0.0
        cd = 0.0
        for r in range(R):
            cb += QL[r, t] * b_p[r]
            cd += QL[r, t] * d_p[r]
        Qb[t] = cb
        Qd[t] = cd
    for t in range(m):
        for r in range(R):
            b_p[r] -= QL[r, t] * Qb[t]
            d_p[r] -= QL[r, t] * Qd[t]
    QB = np.zeros((m, nL))
    bB = np.zeros(nL)
    dB = np.zeros(nL)
    Bd = np.zeros(nL)
    for r in range(R):
        br = b_p[r]
        dr = d_p[r]
        er = d_e[r]
        for c in range(nL):
            x = Bw[r, c]
            bB[c] += br * x
            dB[c] += dr * x
            Bd[c] += er * x
            for t in range(m):
                QB[t, c] += QL[r, t] * x
    inP = np.zeros(p, np.bool_)
    for i in range(act.shape[0]):
        inP[act[i]] = True
    n_out = p - act.shape[0]
    out = np.empty(n_out, np.int64)
    i = 0
    for c in range(p):
        if not inP[c]:
            out[i] = c
            i += 1
    return _sweep(k0, FW, H, fwn2, scale, d_e, QL, RL, b_p, d_p, Qb, Qd, QB, bB, dB, Bd,
                  lib, lib[e], has_fire, lib, out, cn_lib_sorted, cn_lib, chi2_out)


class GridContinuation:
    """Certified grid sweep for one uncensored _FCLSProblem. `solve_lh(k)`
    is the caller's Lawson-Hanson at grid point k, returning (chi2, x) with x
    the full NNLS vector (abundances first)."""

    def __init__(self, Bw, yw, p):
        self.Bw = np.ascontiguousarray(Bw)              # (R, p-1) whitened library + shade
        self.yw = np.ascontiguousarray(yw)
        self.p = p
        self.cn_lib = np.sqrt(np.einsum('ij,ij->j', self.Bw, self.Bw))
        self.cn_lib_sorted = np.sort(self.cn_lib)
        self.n_lh = 0
        self.n_cert = 0
        self.last_act = None             # active set of the latest Lawson-Hanson solve
        self._one = np.empty(1)

    def sweep(self, FW, scale, solve_lh):
        """chi2 at every grid point (rows of the whitened fire columns FW)."""
        FW = np.ascontiguousarray(FW)
        nT = FW.shape[0]
        H = np.ascontiguousarray(FW @ self.Bw)
        fwn2 = np.einsum('ij,ij->i', FW, FW)
        scale = np.ascontiguousarray(scale, dtype=float)
        chi2 = np.full(nT, np.nan)
        from_lh = np.zeros(nT, bool)
        k = 0
        while k < nT:
            c2, x = solve_lh(k)
            self.n_lh += 1
            chi2[k] = c2
            from_lh[k] = True
            k += 1
            if k >= nT:
                break
            act = np.flatnonzero(x[:self.p] > 0).astype(np.int64)
            self.last_act = act
            k_stop = _segment(k, FW, H, fwn2, scale, self.Bw, self.yw, self.cn_lib,
                              self.cn_lib_sorted, act, self.p, chi2)
            if k_stop < 0:
                continue
            self.n_cert += k_stop - k
            k = k_stop
        # Ties at rounding level. The grid is read through comparisons
        # (argmin, local minima, the endpoint tests); a certified value can
        # differ from Lawson-Hanson's by ~1e-13 relative, which decides a
        # comparison only where two values are that close - on plateaus
        # where the fire term contributes nothing measurable, which
        # Lawson-Hanson resolves into exact or monotone last digits and a
        # certified value into jitter (spurious local minima). Every
        # certified point within TIE_RTOL of a neighbour or of the grid
        # minimum is therefore re-solved by Lawson-Hanson: every close
        # comparison is then between Lawson-Hanson values, the others are
        # decided by a margin >= 50x the certified error, and the grid reads
        # exactly as the plain loop's does.
        tol = TIE_RTOL * np.abs(chi2)
        near = np.zeros(nT, bool)
        close = np.abs(np.diff(chi2)) <= np.maximum(tol[:-1], tol[1:])
        near[:-1] |= close
        near[1:] |= close
        near |= chi2 - np.min(chi2) <= tol
        for k in np.flatnonzero(near & ~from_lh):
            chi2[k] = solve_lh(k)[0]
            self.n_lh += 1
            self.n_cert -= 1
        return chi2

    def point(self, fw, scale, solve_lh):
        """chi2 at one whitened fire column (the Brent and interval-edge
        evaluations between grid points): certified with the latest active
        set if possible, else Lawson-Hanson (`solve_lh()` -> (chi2, x)),
        whose active set is kept for the next call."""
        if self.last_act is not None:
            FW = np.ascontiguousarray(fw[None, :])
            self._one[0] = np.nan
            k = _segment(0, FW, FW @ self.Bw, np.array([float(fw @ fw)]), np.array([scale]),
                         self.Bw, self.yw, self.cn_lib, self.cn_lib_sorted, self.last_act, self.p,
                         self._one)
            if k == 1:
                self.n_cert += 1
                return float(self._one[0])
        c2, x = solve_lh()
        self.n_lh += 1
        self.last_act = np.flatnonzero(x[:self.p] > 0).astype(np.int64)
        return c2

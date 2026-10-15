#!/usr/bin/env python
# coding: utf-8
"""
Profile-likelihood sub-pixel fire temperature retrieval (this work).

A separable maximum-likelihood formulation of the VSWIR sub-pixel fire
mixture model, with a globally optimal one-dimensional temperature search,
an explicit noise + model-error covariance, profile-likelihood confidence
intervals, a likelihood-ratio fire test, and a temperature-dependent
detection limit. Written against the same interface as the published
methods in temperature_methods.py so it can be run side by side with them.

--------------------------------------------------------------------------
1. THE GAP THIS FILLS
--------------------------------------------------------------------------

Every published VSWIR sub-pixel fire retrieval fits the additive mixture

    L(lambda) = f * eps * B(lambda, T) + sum_j q_j L_j(lambda)             (1)

(Veraverbeke et al. 2018, Eq. 2) by some form of least squares:

  * Dozier (1981)                two bands, closed form, no uncertainty;
  * Dennison et al. (2006),
    Dennison & Matheson (2011)   a discrete library of Planck endmembers,
                                 lowest-RMSE temperature wins, no uncertainty;
  * Waigl et al. (2019)          two fire + two background endmembers,
                                 Hyperion, least squares;
  * Amici et al. (2022)          joint nonlinear least squares with random
                                 restarts over background draws; the quoted
                                 "95% CI" is a t-interval on the MEAN of the
                                 trial distribution (VERIFICATION.md S4), a
                                 standard error that shrinks as 1/sqrt(N_trials)
                                 while the true scatter does not;
  * arXiv:2608.07580 (2026)      full-physics AVIRIS-3 inversion with a
                                 Gaussian temperature distribution, Adam
                                 optimiser; the reported sigma[T] is the
                                 PHYSICAL spread of the fitted distribution,
                                 not an estimation uncertainty, and no
                                 coverage test is presented.

Common to all of them: (a) a local optimiser (or a coarse grid) with no
guarantee of the global optimum; (b) ordinary (unweighted) least squares,
which weights every band by its radiance and so lets the brightest bands -
exactly where sensor nonlinearity and unmodelled transmittance bite hardest -
dominate the answer; (c) no confidence interval for the temperature whose
coverage has been checked; (d) no principled answer to "what is the
smallest fire fraction this pixel/scene could have detected at temperature
T"; (e) a retrieved temperature that silently sits at the search bound for
pixels hotter than the model can represent (VERIFICATION.md S14).

--------------------------------------------------------------------------
2. METHOD
--------------------------------------------------------------------------

2.1 Separable structure. Model (1) is NONLINEAR in T only and LINEAR in the
abundances a = (f, q_1..q_J[, s]) where s is an optional shade (zero
radiance) endmember. Write it as

    m(T, a) = A(T) a,     A(T) = [ eps*tau*B(.,T) | L_1 | ... | L_J | 0 ]

with the physical constraints a >= 0 and sum(a) = 1 (with shade present the
closure means f + sum q_j <= 1, i.e. areal fractions cannot exceed the
pixel). For Gaussian errors with per-band standard deviation sigma_i the
negative log-likelihood is, up to a constant,

    chi2(T, a) = sum_i [ (y_i - (A(T) a)_i) / sigma_i ]^2                 (2)

2.2 Variable projection (Golub & Pereyra 1973). For FIXED T, (2) is a
convex quadratic in a over a compact convex set (the simplex), so its
minimum value is unique and is found exactly by non-negative least squares
with the closure imposed as a heavily weighted extra row (the FCLS
construction of Heinz & Chang 2001, the same solve the rest of this
repository uses). Define the PROFILE

    chi2_p(T) = min_{a in simplex} chi2(T, a)                                (3)

Then min_{T,a} chi2(T,a) = min_T chi2_p(T): the joint minimum is the
minimum of a ONE-dimensional continuous function on a closed interval
(continuity by Berge's maximum theorem, the feasible set being compact and
independent of T). That function is evaluated on a dense grid, uniform in
1/T (the Wien-regime slope is linear in 1/T, so this spaces the grid by the
spectral shape change rather than by Kelvin), and the best bracket is
refined by bounded Brent minimisation. The result is the global optimum to
within the grid's ability to resolve a local minimum narrower than one grid
cell - which is checked: every local minimum on the grid is counted and the
depth of the second-best one is reported, so a genuinely multi-modal
profile is flagged rather than silently resolved by whichever restart won.

Consequences: no random restarts, no initial guess, deterministic output,
no convergence failures, no abundance-scale gauge (the abundances are
solved in the closed simplex, so the null direction of VERIFICATION.md S2.2
does not exist here), and no artificial floor on the fire fraction (S2.3):
f >= 0 is the only bound.

2.3 Error model. sigma_i is not assumed constant. The covariance is

    Sigma = diag(sigma_n^2) + eps_m^2 * D K D,   D = diag(|m|)               (4)

the sum of the instrument noise (a two-regime model: the reference noise
L_ref/SNR below the per-band reference radiance L_ref - read-noise
dominated, so a dark band never gets an infinite weight - growing as the
square root of the radiance above it - shot-noise dominated) and a
MULTIPLICATIVE model-error term proportional to the modelled radiance m,
with a correlation kernel K between bands (squared-exponential plus a
nugget). The second term is what the residuals of every VSWIR fire
retrieval actually look like (VERIFICATION.md S11.3-11.4: an 8-15%
relative misfit that scales with the signal; after the scene transmittance
correction, 2-3%), and is far above the instrument noise on a fire pixel.
Weighting by (4) turns (2) into the maximum-likelihood estimator under
that error model. Because m depends on the fit, the weights are refined in
a second pass (two passes are ample). Three quantities of (4) are
CALIBRATED per scene rather than assumed: eps_m so that the median reduced
chi-square over the fire pixels is 1 (calibrate_model_error(), a
variance-component estimate), and the kernel's correlation length and
nugget from the pooled autocorrelation of the diagonally-weighted
residuals (estimate_model_error_kernel()); on the Portugal scene these are
eps_m = 0.03, length 27 nm, nugget 0.4.

Which weighting to use is not a matter of taste; it interacts with the
completeness of the forward model. With the unattenuated Planck term, the
relative weighting is WORSE than ordinary least squares (synthetic truth
with a gas transmittance imposed: median bias -18 K against -4 K), because
it gives full weight to the dark long-wave bands where the unmodelled
absorption sits, which OLS effectively ignores. Once the transmittance is
in the model (fire_transmittance.py) the relative/kernel weighting is the
better estimator (bias -0.4 K, and a valid likelihood for the interval),
and under a 5% smooth multiplicative model error its bias is 2-10 times
smaller than that of OLS (verify17_profile.py, section C).

2.4 Profile-likelihood confidence interval. For the parameter of interest T
with the abundances profiled out, the likelihood-ratio statistic
chi2_p(T) - chi2_p(T_hat) is asymptotically chi2_1 when Sigma is known
(Wilks). With the scale of Sigma estimated as s^2 = chi2_min / dof the
interval is

    { T : chi2_p(T) - chi2_min <= s^2 * F_{1,dof}(1 - alpha) }              (5)

(Bates & Watts 1988, Ch. 6, the "profile t"/likelihood-ratio interval for
nonlinear regression). It is exact for linear models and far better than a
Wald interval for a strongly asymmetric parameter such as T, whose profile
is steep on the cold side and flat on the hot side (the SWIR spectral shape
changes as 1/T^2). Residual correlation is handled in the likelihood by
the kernel of (4) (with a diagonal covariance, the AR(1) effective-sample-
size correction is applied instead and the lag-1 autocorrelation is
reported either way). When the hot side of (5) reaches T_max the upper
limit is reported as UNBOUNDED, which is the correct statement for the
bound-pinned pixels of VERIFICATION.md S14: "at least T_lo, and the data
cannot say how much hotter".

The background library is itself uncertain (a random 20-spectrum draw
cannot represent every background exactly), and an interval conditional
on one library under-covers: on synthetic truth the single-draw kernel
interval covered 88% at nominal 95%. run_profile_on_pixels() therefore
fits several independent library draws, takes the lowest-chi2 draw as the
point estimate (maximum likelihood over the discrete set of libraries as
well as over T; this halves the mean absolute error of a single draw,
27 K against 45 K) and reports the ENVELOPE of the per-draw intervals: 97%
coverage at nominal 95%, median width 50 K, on the same synthetic set.

2.4b Invalid and saturated bands. A non-positive at-sensor radiance is
never fitted. Bands the EnMAP L1B pixel mask flags are excluded, and bands
at the detector ceiling (DN >= 63000 of 65535) are CENSORED: the fit is
constrained to model >= recorded radiance instead of fitting the clipped
value (weighted_fcls(), `censored`; enmap_quality.py builds the masks).
On the Portugal scene 11 of 55 fire pixels have saturated SWIR bands and
35 have processor-flagged ones; fitting them as data moved the affected
pixels' temperatures by a median 45 K. Pixels the model cannot describe
at all (reduced chi-square above 5 with eps_m calibrated; on this scene
unflagged soft-saturation plateaus that no Planck mixture can fit) are
marked fit_quality='poor' and should be excluded from aggregates.

2.5 Fire test and detection limit. The likelihood-ratio statistic

    Lambda = chi2_p(background only) - chi2_min                              (6)

measures how much the fire term earns. Because T is searched, Lambda's
null distribution is not chi2_1 (a look-elsewhere effect), so it is
calibrated EMPIRICALLY from the scene's own non-fire pixels
(`empirical_null_threshold()`): the (1 - alpha_FA) quantile of Lambda over
background pixels is the detection threshold at false-alarm rate alpha_FA.
For a background b, endmember library L and noise sigma, adding fire at
fraction f changes the weighted spectrum by f * (eps B(T) - b)/sigma, of
which only the part orthogonal to span(L/sigma) is detectable, so

    f_min(T) = sqrt(Lambda_thr) / || P_perp (eps B(T) - b) / sigma ||        (7)

is the minimum detectable fire fraction at temperature T
(`minimum_detectable_fraction()`). The unconstrained projector P_perp
absorbs at least as much as the constrained NNLS can, so (7) is a
conservative (upper) bound on the true limit. This replaces the tuning-
artefact floor of S2.3 with a radiometric quantity.

2.6 Two components. `profile_two_component()` profiles over (T_hot, T_cool)
on a 2-D grid uniform in 1/T with T_hot > T_cool, refines with Nelder-Mead,
and returns the nested F test against the single-component fit with the
same weights. It is the deterministic counterpart of the gated escalation
in Fire_analysis_with_EnMAP.two_component_escalation(); the gating policy
(only bound-pinned pixels, strict effect-size and separation gates) is
unchanged and lives with the caller.

--------------------------------------------------------------------------
3. WHAT IS NOT CLAIMED
--------------------------------------------------------------------------

* The grey-body emissivity is still degenerate with f (VERIFICATION.md S9);
  the retrieved fraction is f_eff = f * eps.
* The fire term is still unattenuated unless a fixed transmittance `tau` is
  supplied (the smoke-transmittance correction of temperature_methods.py supplies a local one).
* Coverage of (5) is verified on synthetic spectra with real EnMAP
  backgrounds (algorithm_verification/verify17_profile.py); on real pixels
  it inherits whatever the error model (4) fails to describe. That is why
  eps_m and the kernel are calibrated on the scene rather than assumed.
* The 40 K difference between this chain and the published unattenuated
  fits on the Portugal scene is physically expected in sign and size but
  has not been checked against an independent temperature measurement.

Units: radiance in W m^-2 sr^-1 nm^-1 and wavelength in nm throughout,
matching Fire_analysis_with_EnMAP.Planck_function().

References
  Bates, D.M., Watts, D.G. (1988). Nonlinear Regression Analysis and Its
    Applications. Wiley. Ch. 6.
  Golub, G.H., Pereyra, V. (1973). SIAM J. Numer. Anal. 10, 413-432.
  Heinz, D.C., Chang, C.-I. (2001). IEEE TGRS 39, 529-545.
  Bioucas-Dias, J.M., Nascimento, J.M.P. (2008). IEEE TGRS 46, 2435-2445
    (noise estimation by inter-band regression).
  Veraverbeke, S. et al. (2018). RSE 216, 105-121.
  Amici, S. et al. (2022). JGR Biogeosciences, 10.1029/2022JG007055.
  Dennison, P.E. et al. (2006). RSE 100, 212-222.
  Dennison, P.E., Matheson, D.S. (2011). RSE 115, 876-886.
"""

import math
import warnings

import numpy as np
from scipy import optimize, stats

from Fire_analysis_with_EnMAP import Planck_function

__all__ = [
    'profile_retrieval', 'profile_two_component', 'weighted_fcls', 'noise_sigma',
    'estimate_band_noise', 'calibrate_model_error', 'estimate_model_error_kernel',
    'estimate_model_error_kernel_reml', 'empirical_null_threshold',
    'minimum_detectable_fraction', 'run_profile_on_pixels', 'METHOD_NAME',
]

METHOD_NAME = 'profile-likelihood ML retrieval (this work)'
METHOD_NAME_V2 = 'profile-likelihood ML retrieval v2: reweighting iterated to its fixed point (this work)'
MIN_FIT_BANDS = 8      # data rows needed for a retrieval (T, f and a few abundances)


def _bind_nnls():
    """scipy.optimize.nnls, with its per-call input re-validation removed.

    The public function runs np.asarray_chkfinite over the whole design and
    the right-hand side on every call - a scan of some 4,800 elements that
    costs about a tenth of the solve at the sizes here, repeated ~400 times
    per pixel for the profile grid. _FCLSProblem owns its design: it builds
    it once from data it has already checked and rewrites only the fire
    column, which it checks itself, so the scan re-establishes nothing.

    The compiled routine is private, so it is bound only if it is present
    AND reproduces scipy.optimize.nnls exactly on a set of test problems of
    the shape this module actually solves; otherwise the public function is
    used and nothing is lost but the speed. Returns (callable, bypassed)."""
    try:
        from scipy.optimize._slsqplib import nnls as _compiled
    except Exception:
        return optimize.nnls, False

    def fast_nnls(A, b):
        x, rnorm, info = _compiled(A, b, 3 * A.shape[1])
        if info == 3:
            raise RuntimeError('NNLS: maximum number of iterations reached')
        return x, rnorm

    try:
        rng = np.random.default_rng(0)
        for (m_, n_) in ((80, 62), (40, 12), (120, 3), (9, 9)):
            A = np.ascontiguousarray(rng.normal(size=(m_, n_)))
            A[:, 0] *= 1e4                      # the closure row's scaling
            b = np.ascontiguousarray(rng.normal(size=m_))
            xa, ra = optimize.nnls(A, b)
            xb, rb = fast_nnls(A, b)
            if not (np.array_equal(xa, xb) and ra == rb):
                return optimize.nnls, False
    except Exception:
        return optimize.nnls, False
    return fast_nnls, True


_nnls, _NNLS_BYPASSES_VALIDATION = _bind_nnls()

# The profile grid of an uncensored problem runs through the compiled,
# certified active-set continuation of fcls_jit.py when numba is importable
# (the .venv-numba environment); otherwise, or with PROFILEML_JIT=0, through
# the plain Lawson-Hanson loop. Both give the same optimum to rounding.
try:
    import fcls_jit as _fcls_jit
    _JIT_GRID = bool(_fcls_jit.available)
except Exception:                                 # pragma: no cover
    _JIT_GRID = False


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _planck_nm(wavelength_nm, T, emissivity=1.0):
    """Planck radiance, wavelength in nm, result in W m^-2 sr^-1 nm^-1."""
    return Planck_function(np.asarray(wavelength_nm, dtype=float) * 1e-3, float(T),
                           float(emissivity))


def weighted_fcls(A, y, sigma, closure_weight=1e4, censored=None, censor_level=None,
                  censor_weight=1e3):
    """Weighted fully-constrained least squares: minimise
    (y - A a)^T Sigma^-1 (y - A a) subject to a >= 0 and sum(a) = 1,
    optionally with CENSORED bands whose true radiance is only known to be
    at or above `censor_level` (a saturated detector).

    `sigma` is either a 1-D array of per-band standard deviations (diagonal
    Sigma) or the LOWER Cholesky factor L of a full covariance Sigma = L L^T
    (generalised least squares); in both cases the data rows are whitened
    first and the closure is then imposed as one extra row weighted
    `closure_weight` times the mean row norm of the whitened design (Heinz
    & Chang 2001), which keeps the problem a plain NNLS - guaranteed to
    converge, no initialisation - while enforcing the sum to a relative
    precision that is reported so it can be checked rather than assumed.

    Censoring. A saturated band records the detector ceiling, not the
    radiance, so its measurement is the one-sided statement m_k >= c_k.
    Treating it as data biases the fit toward the ceiling; dropping it
    throws away exactly the constraint that pins the temperature of the
    hottest pixels from below. Here it is kept as an inequality: the band's
    data row is removed and replaced by the row m_k - s_k = c_k with a
    NON-NEGATIVE slack s_k (one extra NNLS column, -e_k, excluded from the
    closure), weighted `censor_weight` times the mean whitened row norm.
    The band then costs nothing while m_k >= c_k and (m_k - c_k)^2 scaled
    by that weight below it - the sharp-boundary limit of the Tobit
    (censored-Gaussian) likelihood. Censored rows are constraints, so like
    the closure row they are excluded from the returned chi2. Under a full
    covariance the censored rows are removed BEFORE whitening: the
    covariance is restricted to the uncensored rows and re-factorised, so a
    censored row is never mixed into its neighbours (marginalising a
    Gaussian over a subset of coordinates is exactly the sub-covariance).

    Returns (a, chi2, whitened_residual, closure_error) with `a` holding
    only the abundances (slacks stripped) and chi2/residual over the
    UNCENSORED data rows only.
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n, p = A.shape
    keep = np.ones(n, dtype=bool)
    idx = np.zeros(0, dtype=int)
    if censored is not None and np.any(censored):
        censored = np.asarray(censored, dtype=bool)
        keep &= ~censored
        idx = np.where(censored)[0]
    n_slack = len(idx)
    # whiten the DATA rows only (the censored rows are constraints)
    if sigma.ndim == 1:
        Aw = A[keep] / sigma[keep, None]
        yw = y[keep] / sigma[keep]
    else:
        from scipy.linalg import solve_triangular, cholesky
        if n_slack:
            S = sigma @ sigma.T
            L = cholesky(S[np.ix_(keep, keep)], lower=True, check_finite=False)
        else:
            L = sigma
        Aw = solve_triangular(L, A[keep], lower=True, check_finite=False)
        yw = solve_triangular(L, y[keep], lower=True, check_finite=False)
    row_scale = np.linalg.norm(Aw) / max(n, 1) + 1e-300
    scale = closure_weight * row_scale
    extra_rows, extra_rhs = [], []
    if n_slack:
        cw = censor_weight * row_scale
        for j, k in enumerate(idx):
            row = np.zeros(p + n_slack)
            row[:p] = A[k] * cw
            row[p + j] = -cw
            extra_rows.append(row)
            extra_rhs.append(float(censor_level[k]) * cw)
    Aw_k = np.hstack([Aw, np.zeros((Aw.shape[0], n_slack))]) if n_slack else Aw
    closure = np.concatenate([np.full(p, scale), np.zeros(n_slack)])
    M = np.vstack([Aw_k] + extra_rows + [closure])
    rhs = np.concatenate([yw, extra_rhs, [scale]])
    sol, _ = optimize.nnls(M, rhs)
    a = sol[:p]
    r = Aw @ a - yw
    return a, float(r @ r), r, float(abs(a.sum() - 1.0))


class _FCLSProblem:
    """weighted_fcls() prepared once for a fixed covariance, library and
    censoring, to be solved many times with only the fire column changing.

    The profile chi2_p(T) evaluates weighted_fcls() a few hundred times per
    pixel with the SAME whitened library, data and constraint rows, and a
    different first column (the Planck term) each time. Whitening the 60
    library columns through the Cholesky factor for every T, and rebuilding
    the stacked NNLS matrix from scratch, was 85% of the retrieval's run
    time (the compiled NNLS itself 12%). Here the T-independent parts are
    whitened and laid out once; solve() whitens the new column, updates the
    entries that depend on it - the closure and censor weights scale with
    the Frobenius norm of the whitened design, which includes that column -
    and calls the same NNLS on the same matrix weighted_fcls() would build,
    so the result is the same to rounding.

    `A_fixed` holds the T-independent columns (library, shade) in their
    weighted_fcls column order minus the first."""

    def __init__(self, A_fixed, y, sigma, censored=None, censor_level=None,
                 closure_weight=1e4, censor_weight=1e3):
        A_fixed = np.asarray(A_fixed, dtype=float)
        y = np.asarray(y, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        n, p1 = A_fixed.shape
        self.n, self.p = n, p1 + 1
        keep = np.ones(n, dtype=bool)
        idx = np.zeros(0, dtype=int)
        if censored is not None and np.any(censored):
            censored = np.asarray(censored, dtype=bool)
            keep &= ~censored
            idx = np.where(censored)[0]
        self.keep, self.idx = keep, idx
        self.n_slack = n_slack = len(idx)
        self.A_fixed_raw = A_fixed
        if sigma.ndim == 1:
            self.L = None
            self.sig_keep = sigma[keep]
            Bw = A_fixed[keep] / self.sig_keep[:, None]
            yw = y[keep] / self.sig_keep
        else:
            from scipy.linalg import solve_triangular, cholesky
            if n_slack:
                S = sigma @ sigma.T
                self.L = cholesky(S[np.ix_(keep, keep)], lower=True, check_finite=False)
            else:
                self.L = sigma
            Bw = solve_triangular(self.L, A_fixed[keep], lower=True, check_finite=False)
            yw = solve_triangular(self.L, y[keep], lower=True, check_finite=False)
        self.yw = yw
        self.Bw_norm2 = float(np.sum(Bw * Bw))
        # The one finiteness check of the fixed part, standing in for the
        # per-call scan scipy.optimize.nnls would do (see _bind_nnls).
        if not (math.isfinite(self.Bw_norm2) and np.all(np.isfinite(yw))):
            raise ValueError('non-finite whitened design or data in the FCLS problem')
        n_keep = int(keep.sum())
        self.n_keep = n_keep
        p = self.p
        # rows: whitened data | censor constraints | closure; cols: fire | fixed | slacks
        M = np.zeros((n_keep + n_slack + 1, p + n_slack))
        M[:n_keep, 1:p] = Bw
        self.M = M
        self.rhs = np.zeros(n_keep + n_slack + 1)
        self.rhs[:n_keep] = yw
        self.closure_weight, self.censor_weight = closure_weight, censor_weight
        self.censor_level = None if not n_slack else np.asarray(censor_level, dtype=float)[idx]
        # the T-independent part of the constraint rows, unweighted
        self.cons_fixed = A_fixed[idx] if n_slack else None
        self._slack_cols = p + np.arange(n_slack)

    def whiten_column(self, col):
        col = np.asarray(col, dtype=float)
        if self.L is None:
            return col[self.keep] / self.sig_keep
        from scipy.linalg import solve_triangular
        return solve_triangular(self.L, col[self.keep], lower=True, check_finite=False)

    def solve(self, fire_col):
        """Same return as weighted_fcls(A, y, sigma, ...) with
        A = [fire_col | A_fixed]: (a, chi2, whitened residual, closure error)."""
        fire_col = np.asarray(fire_col, dtype=float)
        fw = self.whiten_column(fire_col)
        M, rhs, p, nk, ns = self.M, self.rhs, self.p, self.n_keep, self.n_slack
        # The design's fixed part was checked when it was whitened in
        # __init__; the fire column is the only thing that changes, so it is
        # the only thing that has to be checked here. `fw @ fw` below is the
        # check: it is needed anyway for the row scale, and is non-finite
        # exactly when fw is.
        row_scale = math.sqrt(float(fw @ fw) + self.Bw_norm2) / max(self.n, 1) + 1e-300
        if not math.isfinite(row_scale):
            raise ValueError('non-finite fire column in the FCLS design')
        scale = self.closure_weight * row_scale
        M[:nk, 0] = fw
        if ns:
            cw = self.censor_weight * row_scale
            rows = slice(nk, nk + ns)
            M[rows, 0] = fire_col[self.idx] * cw
            M[rows, 1:p] = self.cons_fixed * cw
            M[rows, self._slack_cols] = 0.0
            M[nk + np.arange(ns), self._slack_cols] = -cw
            rhs[rows] = self.censor_level * cw
        M[-1, :p] = scale
        M[-1, p:] = 0.0
        rhs[-1] = scale
        sol, _ = _nnls(M, rhs)
        self._last_x = sol
        a = sol[:p]
        r = M[:nk, :p] @ a - self.yw
        return a, float(r @ r), r, float(abs(a.sum() - 1.0))

    def chi2_at(self, fire_col):
        """solve(fire_col)[1] - the chi2 alone, for the interval-edge root
        finding, which needs nothing else and feeds nothing downstream. On an
        uncensored problem with the compiled continuation available, the
        latest active set is tried first and accepted only with its KKT
        certificate (fcls_jit.py), so the value is the same optimum to
        rounding; otherwise this is solve()."""
        if self.n_slack or not _JIT_GRID or getattr(self, '_grid_cont', None) is None:
            return self.solve(fire_col)[1]
        fw = self.whiten_column(fire_col)
        row_scale = math.sqrt(float(fw @ fw) + self.Bw_norm2) / max(self.n, 1) + 1e-300
        if not math.isfinite(row_scale):
            raise ValueError('non-finite fire column in the FCLS design')

        def solve_lh():
            a, c2, _, _ = self.solve(fire_col)
            return c2, self._last_x

        return self._grid_cont.point(np.ascontiguousarray(fw), self.closure_weight * row_scale, solve_lh)

    def chi2_grid(self, fire_cols):
        """chi2 of solve() at every row of `fire_cols` (n_T, n): the profile
        grid in one call. Same NNLS on the same matrix as solve(), entry for
        entry; what is gone is the per-call overhead, which was more than
        half of each grid point (measured on the main-pass kernel covariance:
        25 us whitening + 99 us of indexing and argument validation around a
        109 us compiled NNLS). The whitening is one triangular solve for all
        columns instead of one per column - equal to solve()'s to rounding,
        not necessarily bit for bit, since the multi-right-hand-side LAPACK
        call may sum in a different order - and only the entries of M that
        depend on the column are rewritten."""
        fire_cols = np.asarray(fire_cols, dtype=float)
        if self.L is None:
            FW = fire_cols[:, self.keep] / self.sig_keep
        else:
            from scipy.linalg import solve_triangular
            FW = solve_triangular(self.L, fire_cols[:, self.keep].T, lower=True,
                                  check_finite=False).T
        FW = np.ascontiguousarray(FW)
        M, rhs, p, nk, ns = self.M, self.rhs, self.p, self.n_keep, self.n_slack
        M_data = M[:nk, :p]
        closure_row = M[-1]
        closure_row[p:] = 0.0
        if ns:
            rows = slice(nk, nk + ns)
            diag_slack = (nk + np.arange(ns), self._slack_cols)
            M[rows, self._slack_cols] = 0.0
        n_scale = max(self.n, 1)
        out = np.empty(len(FW))
        if not ns and _JIT_GRID:
            # compiled certified continuation (fcls_jit.py): Lawson-Hanson
            # only where the active set changes, exactly as below there
            from fcls_jit import GridContinuation
            row_scales = np.empty(len(FW))
            for k in range(len(FW)):
                fw = FW[k]
                row_scales[k] = math.sqrt(float(fw @ fw) + self.Bw_norm2) / n_scale + 1e-300
            if not np.all(np.isfinite(row_scales)):
                raise ValueError('non-finite fire column in the FCLS design')
            scales = self.closure_weight * row_scales

            def solve_lh(k):
                M[:nk, 0] = FW[k]
                closure_row[:p] = scales[k]
                rhs[-1] = scales[k]
                sol, _ = _nnls(M, rhs)
                r = M_data @ sol[:p] - self.yw
                return float(r @ r), sol

            if getattr(self, '_grid_cont', None) is None:
                self._grid_cont = GridContinuation(M[:nk, 1:p], self.yw, p)
            return self._grid_cont.sweep(FW, scales, solve_lh)
        for k in range(len(FW)):
            fw = FW[k]
            row_scale = math.sqrt(float(fw @ fw) + self.Bw_norm2) / n_scale + 1e-300
            if not math.isfinite(row_scale):
                raise ValueError('non-finite fire column in the FCLS design')
            scale = self.closure_weight * row_scale
            M[:nk, 0] = fw
            if ns:
                cw = self.censor_weight * row_scale
                M[rows, 0] = fire_cols[k, self.idx] * cw
                M[rows, 1:p] = self.cons_fixed * cw
                M[diag_slack] = -cw
                rhs[rows] = self.censor_level * cw
            closure_row[:p] = scale
            rhs[-1] = scale
            sol, _ = _nnls(M, rhs)
            r = M_data @ sol[:p] - self.yw
            out[k] = float(r @ r)
        return out

def column_median(X, cols=None, nan=False):
    """Per-column median of a (n, n_bands) array, as float64, one column at a
    time: the same values as np.median(np.asarray(X[:, cols], float), axis=0)
    without the float64 copy of the whole array (a 10^6-spectrum background
    pool is ~0.6-1.8 GB in float64). nan=True uses nanmedian."""
    X = np.atleast_2d(X)
    if cols is None:
        idx = np.arange(X.shape[1])
    else:
        c = np.asarray(cols)
        idx = np.flatnonzero(c) if c.dtype == bool else c
    f = np.nanmedian if nan else np.median
    return np.array([f(np.asarray(X[:, j], dtype=float)) for j in idx])


def _planck_nm_grid(wavelength_nm, T_values, emissivity=1.0):
    """Planck radiance for many temperatures at once, (n_T, n_bands), the
    same expression as Planck_function() (wavelength in nm)."""
    x = np.asarray(wavelength_nm, dtype=float)[None, :] * 1e-3
    T = np.asarray(T_values, dtype=float)[:, None]
    c1 = 1e-3 * 1.191042 * 1e8
    c2 = 1.4387752 * 1e4
    return float(emissivity) * (c1 / (x ** 5)) * (1 / (np.exp(c2 / (T * x)) - 1))


def noise_sigma(y, model=None, snr=None, sigma_instrument=None, reference_radiance=None,
                rel_model_error=0.05, wavelength=None, corr_length_nm=0.0, nugget=0.2):
    """The error model (4), optionally with a correlated model-error term.

        Sigma = diag(sigma_n^2) + eps_m^2 * D K D,   D = diag(|m|)           (4')

    sigma_n,i (instrument noise) is a two-regime model: at or below the
    per-band reference radiance L_ref,i it equals the reference noise
    sigma_ref,i (read-noise dominated - the noise does NOT fall to zero
    for a dark band, which is what a naive |y|/SNR would claim and what
    turns a zero-radiance band into an infinite weight); above L_ref it
    grows as sqrt(|m|/L_ref) (shot-noise dominated):

        sigma_n,i = sigma_ref,i * sqrt(max(|m_i| / L_ref,i, 1))

    sigma_ref,i is `sigma_instrument` if given (e.g. from
    estimate_band_noise()), else L_ref,i / snr. L_ref,i is
    `reference_radiance` if given (per band, e.g. the scene's median
    background radiance), else the median of |y| over bands. If neither
    snr nor sigma_instrument is given the instrument term is zero and
    rel_model_error must be > 0.

    m_i is `model` if given (second IRLS pass) else the observation itself.

    K is the model-error correlation between bands: the identity when
    corr_length_nm == 0 (returns a 1-D sigma, diagonal weighting), else a
    squared-exponential kernel in wavelength with correlation length
    corr_length_nm and a diagonal `nugget` share, (1-nugget)*exp(-d^2/2l^2)
    + nugget*delta - in which case the LOWER Cholesky factor of Sigma is
    returned (for weighted_fcls' generalised-least-squares path). A smooth
    multiplicative discrepancy is what an unmodelled transmittance, a
    grey-body emissivity shape or a background-library mismatch actually
    produce, and treating it as band-independent noise both under-states
    the temperature uncertainty (a smooth tilt across the window is partly
    degenerate with T) and leaves a lag-1 residual autocorrelation of
    0.7-0.9 on real EnMAP fire pixels.
    """
    y = np.asarray(y, dtype=float)
    m = np.abs(y if model is None else np.asarray(model, dtype=float))
    n = len(y)
    if reference_radiance is None:
        L_ref = np.full(n, max(float(np.nanmedian(np.abs(y))), 1e-300))
    else:
        L_ref = np.maximum(np.asarray(reference_radiance, dtype=float), 1e-300)
    if sigma_instrument is not None:
        s_ref = np.asarray(sigma_instrument, dtype=float)
    elif snr is not None and snr > 0:
        s_ref = L_ref / float(snr)
    else:
        s_ref = np.zeros(n)
    if rel_model_error <= 0 and not np.any(s_ref > 0):
        raise ValueError('noise_sigma: need an instrument noise (snr or sigma_instrument) '
                         'or rel_model_error > 0; otherwise every weight is infinite.')
    sn2 = (s_ref * np.sqrt(np.maximum(m / L_ref, 1.0))) ** 2
    floor = (1e-6 * max(float(np.nanmax(np.abs(y))), 1e-300)) ** 2
    diag = np.maximum(sn2 + (float(rel_model_error) * m) ** 2, floor)
    if corr_length_nm is None or corr_length_nm <= 0 or wavelength is None:
        return np.sqrt(diag)
    lam = np.asarray(wavelength, dtype=float)
    d2 = (lam[:, None] - lam[None, :]) ** 2
    K = (1.0 - nugget) * np.exp(-0.5 * d2 / float(corr_length_nm) ** 2) + nugget * np.eye(n)
    em = float(rel_model_error) * m
    Sigma = np.diag(np.maximum(sn2, floor)) + (em[:, None] * em[None, :]) * K
    # nugget*eps^2 m^2 already sits on the diagonal through K, so Sigma is PD
    return np.linalg.cholesky(Sigma)


def _design(wavelength, T, endmembers, emissivity, tau, include_shade):
    fire = emissivity * tau * _planck_nm(wavelength, T)
    cols = [fire] + [endmembers[j] for j in range(endmembers.shape[0])]
    if include_shade:
        cols.append(np.zeros_like(wavelength))
    return np.column_stack(cols)


def _lag1_autocorrelation(r):
    r = np.asarray(r, dtype=float)
    r = r - r.mean()
    d = float(r @ r)
    if len(r) < 3 or d <= 0:
        return 0.0
    return float((r[:-1] @ r[1:]) / d)


def _effective_n(n, rho):
    """AR(1) effective sample size. Negative rho is not rewarded (clipped to
    0): anti-correlated residuals would inflate n_eff above n, which is
    not a claim this module wants to make about a fit residual."""
    rho = min(max(float(rho), 0.0), 0.95)
    return float(n) * (1.0 - rho) / (1.0 + rho)


def _bound_flag(T, lo, hi, rel_tol=1e-6, abs_tol=1e-6):
    if not np.isfinite(T):
        return False
    span = max(abs(hi - lo), 1.0)
    return bool(T <= lo + max(abs_tol, rel_tol * span) or
                T >= hi - max(abs_tol, rel_tol * span))


def _t_grid(T_bounds, n_grid):
    """Grid uniform in 1/T from T_min to T_max, ascending in T."""
    u = np.linspace(1.0 / float(T_bounds[1]), 1.0 / float(T_bounds[0]), int(n_grid))
    return np.sort(1.0 / u)


def _nan_result():
    """The result dict of a retrieval that could not be made (converged=False)."""
    return {'T_K': np.nan, 'fire_fraction': np.nan, 'T_ci_low': np.nan, 'T_ci_high': np.nan,
            'T_lower_unbounded': False, 'T_upper_unbounded': False,
            'f_at_ci_low': np.nan, 'f_at_ci_high': np.nan, 'chi2_min': np.nan,
            'dof': 0, 'dof_eff': 0.0, 'reduced_chi2': np.nan, 'rho_lag1': np.nan,
            'n_eff': 0.0, 'n_bands': 0, 'n_bands_invalid': 0, 'n_bands_censored': 0,
            'censor_violation': np.nan, 'n_params': 0, 'n_params_nominal': 0, 'n_active': 0, 'fire_lrt': np.nan,
            'chi2_background_only': np.nan, 'rmse': np.nan, 'model': None,
            'abundances': None, 'sigma': None, 'n_local_minima': 0,
            'second_minimum_delta_chi2': np.nan, 'ci_possibly_disconnected': False,
            'closure_error': np.nan, 'delta_chi2_threshold': np.nan,
            'T_at_bound': False, 'converged': False, 'method': METHOD_NAME,
            'irls_passes': 0, 'irls_converged': None}


def _pixel_masks(radiance, wavelength, B, tau_arr, valid_mask, censored_mask, censor_level):
    """Which bands of one pixel enter the likelihood (`good`), which are
    censored constraints (`cens`, or None) with their ceilings (`lvl_full`),
    and how many finite radiances were excluded (`n_invalid`). Shared by
    profile_retrieval() and the batched engine (profile_batch.py)."""
    # A non-positive at-sensor radiance is not a measurement (EnMAP L1B
    # writes 0 into bands its per-band pixel mask flags as defective or
    # saturated, and every earlier method fitted those zeros as data); such
    # bands are excluded here and counted in `n_bands_invalid`.
    good = (np.isfinite(radiance) & (radiance > 0) & np.isfinite(wavelength)
            & np.all(np.isfinite(B), axis=0) & np.isfinite(tau_arr))
    if valid_mask is not None:
        good &= np.asarray(valid_mask, dtype=bool)
    n_invalid = int(np.isfinite(radiance).sum() - good.sum())
    # censored bands must be finite and carry a ceiling; they are fitted as
    # inequalities, so they are NOT part of `good` (the likelihood rows)
    cens = None
    lvl_full = None
    if censored_mask is not None and np.any(censored_mask):
        cens_full = (np.asarray(censored_mask, dtype=bool) & np.isfinite(wavelength)
                     & np.all(np.isfinite(B), axis=0) & np.isfinite(tau_arr))
        lvl_full = np.asarray(censor_level, dtype=float) if censor_level is not None else radiance
        cens_full &= np.isfinite(lvl_full) & (lvl_full > 0)
        # NOT intersected with valid_mask: a censored band is by definition
        # not a valid measurement (enmap_quality marks it valid=False), it
        # is a constraint, and it must survive here. An earlier version
        # applied valid_mask to it too, which silently switched censoring
        # off on every real pixel while leaving the unit tests (no
        # valid_mask) green.
        good &= ~cens_full
        if cens_full.any():
            cens = cens_full
    return good, cens, lvl_full, n_invalid


def _dof_and_threshold(chi2_min, a_hat, r_hat, n, residual_correlation, scale_variance, alpha):
    """Active-set degrees of freedom and the profile-likelihood threshold q
    of (5) for one fitted pixel. Returns (n_active, n_params_eff, dof, rho,
    n_eff, dof_eff, q)."""
    # Effective parameter count: T plus the abundances that are strictly
    # positive at the solution (the active face of the simplex), minus the
    # closure. For a constrained least-squares estimator the degrees of
    # freedom are the expected number of active coordinates (Meyer &
    # Woodroofe 2000; the lasso analogue is Zou, Hastie & Tibshirani 2007),
    # so a 60-spectrum library that uses six spectra costs six parameters,
    # not sixty. Floor of 2 (T and f).
    n_active = int(np.sum(a_hat > 1e-10))
    n_params_eff = max(n_active, 2)
    dof = max(n - n_params_eff, 1)
    rho = _lag1_autocorrelation(r_hat)
    n_eff = _effective_n(n, rho) if residual_correlation == 'ar1' else float(n)
    dof_eff = max(n_eff - n_params_eff, 1.0)
    s2 = chi2_min / dof_eff
    if scale_variance:
        q = s2 * float(stats.f.ppf(1.0 - alpha, 1, dof_eff))
    else:
        q = float(stats.chi2.ppf(1.0 - alpha, 1))
    return n_active, n_params_eff, dof, rho, n_eff, dof_eff, q


def _grid_diagnostics(chi2_grid, chi2_min, q):
    """Multi-modality diagnostics on the profile grid: (n_local_minima,
    second_minimum_delta_chi2, ci_possibly_disconnected, k_best)."""
    interior = (chi2_grid[1:-1] < chi2_grid[:-2]) & (chi2_grid[1:-1] <= chi2_grid[2:])
    local_idx = np.where(interior)[0] + 1
    # endpoints count as local minima when the profile rises away from them
    if chi2_grid[0] < chi2_grid[1]:
        local_idx = np.concatenate([[0], local_idx])
    if chi2_grid[-1] < chi2_grid[-2]:
        local_idx = np.concatenate([local_idx, [len(chi2_grid) - 1]])
    k_best = int(np.argmin(chi2_grid))
    others = [i for i in local_idx if abs(int(i) - k_best) > 1]
    second_delta = float(min(chi2_grid[i] for i in others) - chi2_min) if others else np.nan
    ci_disconnected = bool(np.isfinite(second_delta) and second_delta <= q)
    return int(len(local_idx)), second_delta, ci_disconnected, k_best


def _walk_edge(chi2_grid, T_grid, k_best, T_hat, chi2_min, q, direction, T_lo_b, T_hi_b):
    """Walk the profile grid from the best point outward until the excess
    chi2 is positive. Returns (bracket (a, b) or None, unbounded_flag,
    fallback_T): a bracket to refine the interval edge in, or the search
    bound when the excess never turns positive."""
    idx = np.arange(k_best, -1, -1) if direction < 0 else np.arange(k_best, len(T_grid))
    prev_T = T_hat
    for i in idx:
        T_i = float(T_grid[i])
        if (direction < 0 and T_i >= T_hat) or (direction > 0 and T_i <= T_hat):
            continue
        if chi2_grid[i] - chi2_min - q > 0:
            a_, b_ = (T_i, prev_T) if direction < 0 else (prev_T, T_i)
            return (a_, b_), False, T_i
        prev_T = T_i
    return None, True, (T_lo_b if direction < 0 else T_hi_b)


# --------------------------------------------------------------------------
# single component
# --------------------------------------------------------------------------

def _build_cov(y, m, w, snr, sigma_instrument, reference_radiance, eps, corr_length_nm, nugget):
    """The covariance (4') for one pixel, as a 1-D sigma or a Cholesky factor."""
    return noise_sigma(y, m, snr=snr, sigma_instrument=sigma_instrument,
                       reference_radiance=reference_radiance, rel_model_error=eps,
                       wavelength=w, corr_length_nm=corr_length_nm, nugget=nugget)


def _diag_sigma(cov):
    cov = np.asarray(cov, dtype=float)
    return cov if cov.ndim == 1 else np.sqrt(np.sum(cov ** 2, axis=1))


def profile_retrieval(radiance, wavelength, background_endmembers, emissivity=1.0,
                      sigma=None, snr=None, sigma_instrument=None, reference_radiance=None,
                      rel_model_error=0.05, corr_length_nm=150.0, nugget=0.2,
                      T_bounds=(300.0, 2000.0), n_grid=200, include_shade=True, tau=None,
                      alpha=0.05, residual_correlation='auto', scale_variance=True,
                      n_weight_passes=2, closure_weight=1e4, return_profile=False,
                      valid_mask=None, censored_mask=None, censor_level=None, rng=None,
                      irls_rtol=1e-4, irls_max_passes=12):
    """Single-component profile-likelihood retrieval of (T, f) for one pixel.

    Parameters
    ----------
    radiance, wavelength : 1-D arrays over the fit window (already masked).
    background_endmembers : (J, n_bands) library of background spectra.
    emissivity : grey-body emissivity assumed for the fire term (degenerate
        with f - see module docstring, section 3).
    sigma : optional FIXED per-band standard deviation (1-D) or lower
        Cholesky factor of a full covariance (2-D). If given, the error
        model (4') is not used and no reweighting pass is made.
    snr, sigma_instrument, reference_radiance : the instrument-noise term
        of (4') - see noise_sigma().
    rel_model_error : eps_m of (4'), the multiplicative model-error
        fraction; calibrate per scene with calibrate_model_error().
    corr_length_nm, nugget : the model-error correlation kernel of (4');
        corr_length_nm=0 gives band-independent (diagonal) weighting.
    T_bounds : search interval (K). The CI can report "unbounded" at either
        end; T_at_bound flags a point estimate sitting on the interval edge.
    n_grid : number of profile evaluations, uniform in 1/T.
    include_shade : add a zero-radiance shade endmember (MESMA convention),
        so closure becomes f + sum q_j <= 1.
    tau : optional fixed transmittance spectrum multiplying the fire term
        (the (S2) formulation of the smoke-transmittance correction in temperature_methods.py).
    alpha : 1 - confidence level for the interval (0.05 -> 95%).
    residual_correlation : 'ar1' (measure the lag-1 autocorrelation of the
        whitened residual and shrink the degrees of freedom by the AR(1)
        effective sample size), 'none', or 'auto' (= 'ar1' only when the
        covariance is diagonal; with a correlated model-error covariance the
        correlation is already in the likelihood and must not be counted
        twice). The lag-1 autocorrelation is reported in every case.
    scale_variance : True -> interval (5) with s^2 = chi2_min/dof (sigma
        known up to scale); False -> chi2_1 quantile (sigma taken as
        absolute).
    n_weight_passes : IRLS passes (the covariance depends on the modelled
        radiance; pass 1 uses the observation as the proxy, pass 2 the
        pass-1 model). The covariance is FIXED within a pass, so every
        profile is a proper generalised-least-squares profile.
        'converge' iterates to the FIXED POINT instead (ProfileML-v2): the
        estimate whose covariance is the one its own fitted radiance
        implies - the solution of the quasi-likelihood score equations,
        which does not depend on where the iteration started or on an
        arbitrary pass count. Converged when the fitted radiance changes by
        less than `irls_rtol` (relative, every band) between passes; at most
        `irls_max_passes`. On the fire pixels of DT0000149658 two passes are
        within a median 0.1-2 K of the fixed point (p90 8-9 K) and 4-6
        passes reach it; a pixel whose iteration cycles between two
        distant minima is returned with irls_converged=False.
    return_profile : include the (T_grid, chi2_grid) arrays in the output.
    valid_mask : optional per-band bool, False = do not fit this band (e.g.
        the EnMAP L1B per-band pixel mask). Non-positive radiances are
        always excluded.
    censored_mask, censor_level : optional per-band bool and radiance
        ceiling for SATURATED bands: the fit is constrained to model(band)
        >= censor_level instead of fitting the recorded value (see
        weighted_fcls), under a diagonal or a full covariance alike.
    rng : accepted and ignored, for signature compatibility with the
        registry driver (this method is deterministic).

    Many pixels at once: profile_batch.profile_retrieval_many() runs this
    same retrieval for a list of pixels over every CPU core and returns the
    same result dicts.

    Returns
    -------
    dict with T_K, fire_fraction, T_ci_low, T_ci_high, T_lower_unbounded,
    T_upper_unbounded, f_at_ci_low, f_at_ci_high, chi2_min, dof, dof_eff,
    reduced_chi2, rho_lag1, n_eff, n_bands, n_params, fire_lrt,
    chi2_background_only, rmse (unweighted, for comparison with the other
    methods), model, abundances, sigma (diagonal std, full-length),
    n_local_minima, second_minimum_delta_chi2, ci_possibly_disconnected,
    closure_error, delta_chi2_threshold, T_at_bound, converged, method.
    """
    wavelength = np.asarray(wavelength, dtype=float)
    radiance = np.asarray(radiance, dtype=float)
    B = np.atleast_2d(np.asarray(background_endmembers, dtype=float))
    tau_arr = np.ones_like(wavelength) if tau is None else np.asarray(tau, dtype=float)
    T_lo_b, T_hi_b = float(T_bounds[0]), float(T_bounds[1])
    if not (T_hi_b > T_lo_b > 0):
        raise ValueError(f'T_bounds must satisfy 0 < T_min < T_max, got {T_bounds}')

    nan_out = _nan_result()

    good, cens, lvl_full, n_invalid = _pixel_masks(radiance, wavelength, B, tau_arr,
                                                   valid_mask, censored_mask, censor_level)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=float)
        if sigma.ndim == 1:
            good &= np.isfinite(sigma) & (sigma > 0)
    n_params = 1 + B.shape[0] + (1 if include_shade else 0)   # nominal: T + abundances - closure + fire abundance
    # The library may be larger than the band count: the non-negative,
    # closed fit is still well posed (the NNLS active set is at most the
    # number of rows) and its effective parameter count is the size of that
    # active set (Meyer & Woodroofe 2000), not the number of columns - see
    # n_params_eff below. Only a floor on the number of data rows is needed.
    if good.sum() < MIN_FIT_BANDS:
        nan_out['reason'] = f'only {int(good.sum())} usable bands (minimum {MIN_FIT_BANDS})'
        return nan_out
    fit = good | cens if cens is not None else good      # rows entering the NNLS
    w = wavelength[fit]
    y = radiance[fit]
    Bg = B[:, fit]
    tg = tau_arr[fit]
    cens_f = cens[fit] if cens is not None else None
    lvl_f = lvl_full[fit] if cens is not None else None
    sig_inst = None if sigma_instrument is None else np.asarray(sigma_instrument, float)[fit]
    ref_rad = None if reference_radiance is None else np.asarray(reference_radiance, float)[fit]

    converge = isinstance(n_weight_passes, str)
    if converge and n_weight_passes != 'converge':
        raise ValueError(f"n_weight_passes must be an int or 'converge', got {n_weight_passes!r}")
    if sigma is not None:
        if sigma.ndim == 1:
            cov = sigma[fit]
        else:
            if not np.all(fit):
                raise ValueError('a full covariance factor cannot be masked band-wise; '
                                 'pass finite inputs on every band or a 1-D sigma')
            cov = sigma
        passes = 1
    else:
        cov = _build_cov(y, None, w, snr, sig_inst, ref_rad, rel_model_error, corr_length_nm, nugget)
        passes = max(int(irls_max_passes), 2) if converge else max(int(n_weight_passes), 1)
    correlated = np.asarray(cov).ndim == 2
    if residual_correlation == 'auto':
        residual_correlation = 'none' if correlated else 'ar1'

    T_grid = _t_grid((T_lo_b, T_hi_b), n_grid)
    # The design is [fire(T) | library | shade]; only the first column
    # depends on T, so the rest is whitened once per covariance pass
    # (_FCLSProblem) and the Planck column is evaluated on the whole grid
    # at once. Same matrices as _design() + weighted_fcls(), to rounding.
    A_fixed = _design(w, T_grid[0], Bg, emissivity, tg, include_shade)[:, 1:]
    fire_scale = emissivity * tg
    fire_grid = fire_scale[None, :] * _planck_nm_grid(w, T_grid, 1.0)

    def fire_col(T):
        return fire_scale * _planck_nm(w, T)

    def chi2_profile(T, prob):
        a, c2, r, ce = prob.solve(fire_col(T))
        return c2, a, r, ce

    T_hat = a_hat = r_hat = None
    chi2_grid = None
    chi2_min = np.inf
    closure_err = np.nan
    m_prev = None
    irls_converged = None if not (converge and sigma is None) else False
    n_passes_done = 0
    for _pass in range(passes):
        n_passes_done += 1
        prob = _FCLSProblem(A_fixed, y, cov, censored=cens_f, censor_level=lvl_f,
                            closure_weight=closure_weight)
        chi2_grid = prob.chi2_grid(fire_grid)
        k = int(np.argmin(chi2_grid))
        lo = T_grid[max(k - 1, 0)]
        hi = T_grid[min(k + 1, len(T_grid) - 1)]
        if hi > lo:
            # Brent sees exactly solve()'s values: its T_hat sets the next
            # pass's covariance, and a ~1e-13 difference there would move
            # every grid value of that pass (and the diagnostics read from
            # its near-ties). Only the interval-edge search uses chi2_at().
            res = optimize.minimize_scalar(lambda T: chi2_profile(T, prob)[0],
                                           bounds=(lo, hi), method='bounded',
                                           options={'xatol': 1e-3})
            T_hat = float(res.x) if res.fun <= chi2_grid[k] else float(T_grid[k])
        else:
            T_hat = float(T_grid[k])
        chi2_min, a_hat, r_hat, closure_err = chi2_profile(T_hat, prob)
        if chi2_min > chi2_grid[k]:                      # never worse than the grid
            T_hat = float(T_grid[k])
            chi2_min, a_hat, r_hat, closure_err = chi2_profile(T_hat, prob)
        A_hat = np.column_stack([fire_col(T_hat), A_fixed])
        if irls_converged is not None:
            m_now = A_hat @ a_hat
            if m_prev is not None and np.all(np.abs(m_now - m_prev) <= irls_rtol * np.abs(m_now)):
                # the covariance this pass used is, to irls_rtol, the one
                # its own fit implies: the fixed point
                irls_converged = True
                break
            m_prev = m_now
        if _pass < passes - 1:
            cov = _build_cov(y, A_hat @ a_hat, w, snr, sig_inst, ref_rad, rel_model_error,
                             corr_length_nm, nugget)

    model = A_hat @ a_hat
    n = int(len(y) - (cens_f.sum() if cens_f is not None else 0))
    (n_active, n_params_eff, dof, rho, n_eff, dof_eff,
     q) = _dof_and_threshold(chi2_min, a_hat, r_hat, n, residual_correlation, scale_variance, alpha)

    # ---- multi-modality diagnostics on the grid --------------------------
    n_local_minima, second_delta, ci_disconnected, k_best = _grid_diagnostics(chi2_grid, chi2_min, q)

    # ---- profile-likelihood interval (5) ---------------------------------
    def excess(T):
        return prob.chi2_at(fire_col(T)) - chi2_min - q

    def _find_edge(direction):
        bracket, unbounded, fallback = _walk_edge(chi2_grid, T_grid, k_best, T_hat, chi2_min, q,
                                                  direction, T_lo_b, T_hi_b)
        if bracket is None:
            return fallback, unbounded
        try:
            return float(optimize.brentq(excess, bracket[0], bracket[1], xtol=1e-2, maxiter=100)), False
        except ValueError:
            return fallback, False

    T_ci_low, low_unb = _find_edge(-1)
    T_ci_high, high_unb = _find_edge(+1)
    _, a_lo, _, _ = chi2_profile(T_ci_low, prob)
    _, a_hi, _, _ = chi2_profile(T_ci_high, prob)

    # ---- fire likelihood-ratio statistic (6) -----------------------------
    A0 = A_hat[:, 1:]
    if A0.shape[1] == 0:
        chi2_bkg = float(((y / _diag_sigma(cov)) ** 2).sum())
    else:
        _, chi2_bkg, _, _ = weighted_fcls(A0, y, cov, closure_weight, censored=cens_f,
                                          censor_level=lvl_f)
    fire_lrt = float(max(chi2_bkg - chi2_min, 0.0))

    out = {'T_K': float(T_hat), 'fire_fraction': float(a_hat[0]),
           'T_ci_low': float(T_ci_low), 'T_ci_high': float(T_ci_high),
           'T_lower_unbounded': bool(low_unb), 'T_upper_unbounded': bool(high_unb),
           'f_at_ci_low': float(a_lo[0]), 'f_at_ci_high': float(a_hi[0]),
           'chi2_min': float(chi2_min), 'dof': int(dof), 'dof_eff': float(dof_eff),
           'reduced_chi2': float(chi2_min / dof), 'rho_lag1': float(rho), 'n_eff': float(n_eff),
           'n_bands': int(n), 'n_bands_invalid': n_invalid, 'n_params': int(n_params_eff),
           'n_params_nominal': int(n_params), 'n_active': int(n_active),
           'fire_lrt': fire_lrt,
           'chi2_background_only': float(chi2_bkg),
           'rmse': float(np.sqrt(np.mean((model[~cens_f] - y[~cens_f]) ** 2))) if cens_f is not None
                   else float(np.sqrt(np.mean((model - y) ** 2))),
           'model': _unmask(model, fit, len(radiance)),
           'abundances': a_hat, 'sigma': _unmask(_diag_sigma(cov), fit, len(radiance)),
           '_cov': cov, '_good': fit, '_censored': cens_f, '_censor_level': lvl_f,
           'n_bands_censored': int(cens_f.sum()) if cens_f is not None else 0,
           'censor_violation': (float(np.max(np.maximum(lvl_f[cens_f] - model[cens_f], 0.0)
                                             / lvl_f[cens_f])) if cens_f is not None else 0.0),
           'n_local_minima': int(n_local_minima),
           'second_minimum_delta_chi2': second_delta,
           'ci_possibly_disconnected': ci_disconnected,
           'closure_error': float(closure_err),
           'T_at_bound': _bound_flag(T_hat, T_lo_b, T_hi_b),
           'converged': True, 'method': METHOD_NAME, 'delta_chi2_threshold': float(q),
           'irls_passes': int(n_passes_done), 'irls_converged': irls_converged}
    if return_profile:
        out['T_grid'] = T_grid
        out['chi2_grid'] = chi2_grid
    return out


def _unmask(values, good, n_full):
    out = np.full(n_full, np.nan)
    out[good] = values
    return out


def _many(radiances, wavelength, libraries, per_pixel=None, device=None, **kwargs):
    """profile_retrieval() over a list of pixels through
    profile_batch.profile_retrieval_many(): all CPU cores (the default) or
    serial - see profile_batch.resolve_device()."""
    from profile_batch import profile_retrieval_many
    return profile_retrieval_many(radiances, wavelength, libraries, per_pixel=per_pixel,
                                  device=device, **kwargs)


# --------------------------------------------------------------------------
# two components
# --------------------------------------------------------------------------

def profile_two_component(radiance, wavelength, background_endmembers, emissivity=1.0,
                          T_bounds=(300.0, 2000.0), n_grid=40, include_shade=True, tau=None,
                          closure_weight=1e4, single=None, **kwargs):
    """Two-component (T_hot > T_cool) profile fit with a nested F test
    against the single-component fit made with the SAME covariance.

    The 2-D profile chi2_p(T_hot, T_cool) is evaluated on an n_grid x n_grid
    grid uniform in 1/T restricted to T_hot > T_cool, the best cell is
    refined by Nelder-Mead in (1/T_hot, 1/T_cool) with bound violations
    rejected, and the result is compared with `single` (a
    profile_retrieval() result; computed here with **kwargs if not given)
    via

        F = [(chi2_1 - chi2_2) / 2] / [chi2_2 / dof_2]

    Returns dict with T_hot_K, T_cool_K, f_hot, f_cool, chi2_min,
    F_statistic, p_value, rmse_reduction_frac, separation_K, either_at_bound,
    converged, plus `single` for reference. The caller decides whether the
    second component is kept (see two_component_escalation() in the main
    module for the gating policy this repository uses)."""
    wavelength = np.asarray(wavelength, dtype=float)
    radiance = np.asarray(radiance, dtype=float)
    B = np.atleast_2d(np.asarray(background_endmembers, dtype=float))
    if single is None:
        single = profile_retrieval(radiance, wavelength, B, emissivity=emissivity,
                                   T_bounds=T_bounds, include_shade=include_shade, tau=tau,
                                   closure_weight=closure_weight, **kwargs)
    if not single['converged']:
        return {'converged': False, 'single': single}
    good = single['_good']
    cov = single['_cov']
    w, y, Bg = wavelength[good], radiance[good], B[:, good]
    tg = np.ones_like(w) if tau is None else np.asarray(tau, float)[good]
    T_lo_b, T_hi_b = float(T_bounds[0]), float(T_bounds[1])
    u_lo, u_hi = 1.0 / T_hi_b, 1.0 / T_lo_b

    def chi2_of(u1, u2):
        if not (u_lo <= u1 < u2 <= u_hi):
            return np.inf, None, None, None
        T1, T2 = 1.0 / u1, 1.0 / u2
        fire1 = emissivity * tg * _planck_nm(w, T1)
        fire2 = emissivity * tg * _planck_nm(w, T2)
        cols = [fire1, fire2] + [Bg[j] for j in range(Bg.shape[0])]
        if include_shade:
            cols.append(np.zeros_like(w))
        A = np.column_stack(cols)
        a, c2, r, _ = weighted_fcls(A, y, cov, closure_weight, censored=single.get('_censored'),
                                    censor_level=single.get('_censor_level'))
        return c2, a, A @ a, r

    ug = np.linspace(u_lo, u_hi, int(n_grid))
    best = (np.inf, None)
    for i in range(len(ug)):
        for j in range(i + 1, len(ug)):
            c2, _, _, _ = chi2_of(ug[i], ug[j])
            if c2 < best[0]:
                best = (c2, (ug[i], ug[j]))
    if best[1] is None:
        return {'converged': False, 'single': single}
    x0 = np.array(best[1])
    res = optimize.minimize(lambda x: chi2_of(x[0], x[1])[0], x0, method='Nelder-Mead',
                            options={'xatol': 1e-7, 'fatol': 1e-6, 'maxiter': 2000})
    x = res.x if np.isfinite(res.fun) and res.fun <= best[0] else x0
    chi2_2, a2, model2, r2 = chi2_of(x[0], x[1])
    T_hot, T_cool = 1.0 / x[0], 1.0 / x[1]
    n = single['n_bands']
    p2 = 2 + B.shape[0] + (1 if include_shade else 0)
    dof2 = max(n - p2, 1)
    chi2_1 = single['chi2_min']
    improvement = max(chi2_1 - chi2_2, 0.0)
    F = (improvement / 2.0) / (chi2_2 / dof2) if chi2_2 > 0 else (np.inf if improvement > 0 else 0.0)
    p = float(1.0 - stats.f.cdf(F, 2, dof2)) if np.isfinite(F) else 0.0
    unc = ~single['_censored'] if single.get('_censored') is not None else np.ones(len(y), bool)
    rmse2 = float(np.sqrt(np.mean((model2[unc] - y[unc]) ** 2)))
    edge = 1e-3 * (T_hi_b - T_lo_b)
    either_at_bound = bool(T_hot >= T_hi_b - edge or T_cool <= T_lo_b + edge or T_hot <= T_lo_b + edge)
    return {'T_hot_K': float(T_hot), 'T_cool_K': float(T_cool),
            'f_hot': float(a2[0]), 'f_cool': float(a2[1]), 'chi2_min': float(chi2_2),
            'F_statistic': float(F), 'p_value': p,
            'rmse_reduction_frac': float(1.0 - rmse2 / single['rmse']) if single['rmse'] > 0 else np.nan,
            'separation_K': float(T_hot - T_cool), 'either_at_bound': either_at_bound,
            'rmse': rmse2, 'model': _unmask(model2, good, len(radiance)),
            'rho_lag1': float(_lag1_autocorrelation(r2)),
            'dof': int(dof2), 'converged': True, 'single': single,
            'method': 'two-component profile-likelihood (this work)'}


# --------------------------------------------------------------------------
# noise estimation and calibration
# --------------------------------------------------------------------------

def estimate_band_noise(spectra, ridge=1e-6):
    """Per-band noise standard deviation from a set of background spectra by
    inter-band multiple regression (the estimator of Bioucas-Dias &
    Nascimento 2008, HySime): each band is regressed on all the others and
    the residual standard deviation is taken as that band's noise.

    `spectra` is (n_pixels, n_bands) with n_pixels >> n_bands. Returns
    sigma_n (n_bands,) in the units of `spectra`. The estimate includes any
    variance the other bands cannot predict, so it is an upper bound on the
    pure instrument noise - the safe direction for a weight."""
    X = np.asarray(spectra, dtype=float)
    X = X[np.all(np.isfinite(X), axis=1)]
    n, p = X.shape
    if n < 3 * p:
        raise ValueError(f'estimate_band_noise: need many more pixels ({n}) than bands ({p}).')
    Xc = X - X.mean(axis=0)
    R = Xc.T @ Xc / n
    R += ridge * np.trace(R) / p * np.eye(p)
    Rinv = np.linalg.inv(R)
    # residual variance of band i regressed on the others: 1 / (R^-1)_ii
    # (the standard partial-variance identity), scaled back by n/(n-p)
    var = 1.0 / np.diag(Rinv) * n / max(n - p, 1)
    return np.sqrt(np.maximum(var, 0.0))


def calibrate_model_error(fire_spectra, wavelength, endmember_libs, snr=None,
                          sigma_instrument=None, reference_radiance=None, corr_length_nm=150.0,
                          nugget=0.2, eps_grid=None, target=1.0, per_pixel_kwargs=None, device=None,
                          **kwargs):
    """Choose the multiplicative model-error fraction eps_m of (4') so that
    the MEDIAN reduced chi-square over the given fire pixels equals
    `target` (1.0): a variance-component calibration of the error model on
    the scene's own fire pixels.

    fire_spectra : (n_pix, n_bands) already restricted to the fit window.
    endmember_libs : list of (J, n_bands) libraries, one per pixel (or a
        single library used for all).
    Returns (eps_m, table) where table lists (eps, median reduced chi2).
    The temperature is re-fitted only once, at a mid-grid eps; for the
    other grid values chi2 is re-evaluated at that pixel's fitted T with
    the new covariance (the fitted T moves very little with eps, and this
    keeps the calibration to one retrieval per pixel)."""
    fire_spectra = np.atleast_2d(np.asarray(fire_spectra, dtype=float))
    wavelength = np.asarray(wavelength, dtype=float)
    if eps_grid is None:
        eps_grid = np.array([0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5])
    eps_grid = np.asarray(eps_grid, dtype=float)
    if not isinstance(endmember_libs, (list, tuple)):
        endmember_libs = [endmember_libs] * fire_spectra.shape[0]
    eps0 = float(np.median(eps_grid))
    common = dict(snr=snr, sigma_instrument=sigma_instrument, reference_radiance=reference_radiance,
                  corr_length_nm=corr_length_nm, nugget=nugget)
    if per_pixel_kwargs is None:
        per_pixel_kwargs = [{}] * fire_spectra.shape[0]
    fits = []
    res = _many(list(fire_spectra), wavelength, list(endmember_libs), per_pixel=list(per_pixel_kwargs),
                device=device, rel_model_error=eps0, **common, **kwargs)
    for y, lib, r in zip(fire_spectra, endmember_libs, res):
        if r['converged']:
            fits.append((y, lib, r))
    if not fits:
        return eps0, []
    include_shade = kwargs.get('include_shade', True)
    emissivity = kwargs.get('emissivity', 1.0)
    tau = kwargs.get('tau', None)
    table = []
    for eps in eps_grid:
        red = []
        for y, lib, r in fits:
            good = r['_good']
            w = wavelength[good]
            tg = np.ones_like(w) if tau is None else np.asarray(tau, float)[good]
            A = _design(w, r['T_K'], np.atleast_2d(lib)[:, good], emissivity, tg, include_shade)
            m = r['model'][good]
            cov = _build_cov(y[good], m, w, snr,
                             None if sigma_instrument is None else np.asarray(sigma_instrument, float)[good],
                             None if reference_radiance is None else np.asarray(reference_radiance, float)[good],
                             eps, corr_length_nm, nugget)
            _, c2, _, _ = weighted_fcls(A, y[good], cov, censored=r.get('_censored'),
                                        censor_level=r.get('_censor_level'))
            red.append(c2 / max(r['dof'], 1))
        table.append((float(eps), float(np.median(red))))
    eps_v = np.array([t[0] for t in table])
    red_v = np.array([t[1] for t in table])
    # reduced chi2 decreases monotonically with eps; interpolate in log-log
    if red_v.min() > target:
        return float(eps_v.max()), table
    if red_v.max() < target:
        return float(eps_v.min()), table
    order = np.argsort(red_v)
    eps_hat = float(np.exp(np.interp(np.log(target), np.log(red_v[order]), np.log(eps_v[order]))))
    return eps_hat, table


def estimate_model_error_kernel(results, spectra, wavelength, max_lag_nm=400.0, lag_step_nm=10.0,
                                max_reduced_chi2=None):
    """Correlation length and nugget of the model-error kernel of (4'),
    estimated from the pooled autocorrelation of the whitened residuals of
    a set of DIAGONALLY weighted fits (profile_retrieval results with
    corr_length_nm=0).

    The residual autocorrelation r(d) at band separation d (nm) is pooled
    over all pixels and fitted with the same squared-exponential-plus-
    nugget form the covariance uses,

        r(d) = (1 - nugget) * exp(-d^2 / (2 l^2)),   d > 0

    so that the kernel put into the likelihood is the one the residuals
    actually show, not an assumed one. Negative long-lag correlations (a
    residual that swings between the SWIR windows) are not representable
    by this kernel and are left out of the fit; they are reported as
    `min_acf` so a reader can see them.

    max_reduced_chi2 : None (every converged pixel) or a cap; pixels whose
    reduced chi-square exceeds it are left out, so the kernel describes the
    residuals of fits the model is adequate for rather than the structured
    misfit of the ones it is not.

    Returns dict corr_length_nm, nugget, lags_nm, acf, n_pixels."""
    from scipy.optimize import curve_fit
    wavelength = np.asarray(wavelength, dtype=float)
    acc = {}
    n_used = 0
    for r, y in zip(results, spectra):
        if not r.get('converged'):
            continue
        if max_reduced_chi2 is not None and not (r.get('reduced_chi2', np.inf) <= max_reduced_chi2):
            continue
        g = r['_good']
        wg = wavelength[g]
        rr = (r['model'][g] - np.asarray(y, float)[g]) / r['sigma'][g]
        if r.get('_censored') is not None:
            keep = ~r['_censored']
            wg, rr = wg[keep], rr[keep]
        if len(rr) < 8:
            continue
        rr = rr - rr.mean()
        v = float(rr @ rr) / len(rr)
        if v <= 0:
            continue
        n_used += 1
        # all pairs (a, b >= a), stopping at the first separation beyond
        # max_lag_nm for each a, binned to lag_step_nm; vectorised form of
        # the pair loop this replaced (same pairs, same rounding, same
        # order of the values within a bin)
        n_ = len(rr)
        dl_mat = wg[None, :] - wg[:, None]
        upper = np.triu(np.ones((n_, n_), dtype=bool))
        cut = np.logical_or.accumulate(upper & (dl_mat > max_lag_nm), axis=1)
        keep = upper & ~cut
        ia, ib = np.nonzero(keep)                       # row-major: a ascending, then b
        dl = dl_mat[ia, ib]
        keys = (np.rint(dl / lag_step_nm) * lag_step_nm).astype(int)
        vals = rr[ia] * rr[ib] / v
        for key in np.unique(keys):
            acc.setdefault(int(key), []).append(vals[keys == key])
    if n_used == 0 or not acc:
        return {'corr_length_nm': 0.0, 'nugget': 1.0, 'lags_nm': np.array([]), 'acf': np.array([]),
                'n_pixels': 0, 'min_acf': np.nan}
    lags = np.array(sorted(acc), dtype=float)
    acf = np.array([np.mean(np.concatenate(acc[int(l)])) for l in lags])
    pos = (lags > 0) & (acf > 0)

    def ker(dl, l, nug):
        return (1.0 - nug) * np.exp(-0.5 * dl ** 2 / l ** 2)

    try:
        popt, _ = curve_fit(ker, lags[pos], acf[pos], p0=[50.0, 0.3], bounds=([5.0, 0.0], [2000.0, 1.0]))
        l_hat, nug_hat = float(popt[0]), float(popt[1])
    except Exception:
        l_hat, nug_hat = 0.0, 1.0
    return {'corr_length_nm': l_hat, 'nugget': nug_hat, 'lags_nm': lags, 'acf': acf,
            'n_pixels': n_used, 'min_acf': float(acf.min())}


REML_L_GRID = np.array([10., 15., 20., 30., 40., 55., 75., 100., 135., 180.])
REML_NUG_GRID = np.array([0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.45, 0.65, 0.85])
REML_EPS_GRID = np.exp(np.linspace(np.log(0.004), np.log(0.3), 10))


def _reml_pixel_table(y, w, lib, tau, T, model, snr, ref, emissivity, include_shade, Ls, nugs, epss):
    """-2 x restricted log-likelihood of one pixel over the (l, nugget, eps)
    grid, its fire temperature held at the fitted value; NaN where the
    active design is singular."""
    from scipy.linalg import solve_triangular
    A = _design(w, T, lib, emissivity, tau, include_shade)
    out = np.full((len(Ls), len(nugs), len(epss)), np.nan)
    for a_, l in enumerate(Ls):
        for b_, nug in enumerate(nugs):
            for c_, eps in enumerate(epss):
                Lc = noise_sigma(y, model, snr=snr, reference_radiance=ref, rel_model_error=eps,
                                 wavelength=w, corr_length_nm=l, nugget=nug)
                a, chi2, _, _ = weighted_fcls(A, y, Lc)
                # The restricted likelihood integrates out the mean, which here
                # lives on the active face of the simplex: its design is the
                # active columns with the closure eliminated, Z_j = A_j - A_e
                # (A_e the active column of smallest norm - the all-zero shade
                # when it is active, which makes the plain active design
                # singular).
                act = np.flatnonzero(a > 1e-10)
                ld = 0.0
                if len(act) > 1:
                    e = act[int(np.argmin(np.sum(A[:, act] ** 2, axis=0)))]
                    Z = A[:, [j for j in act if j != e]] - A[:, [e]]
                    Zw = solve_triangular(Lc, Z, lower=True, check_finite=False)
                    sgn, ld = np.linalg.slogdet(Zw.T @ Zw)
                    if not sgn > 0:
                        continue
                out[a_, b_, c_] = chi2 + 2.0 * float(np.sum(np.log(np.diag(Lc)))) + ld
    return out


def estimate_model_error_kernel_reml(results, spectra, wavelength, endmember_libs, tau=None, snr=None,
                                     reference_radiance=None, emissivity=1.0, include_shade=True,
                                     max_reduced_chi2=5.0, n_jobs=-1):
    """Correlation length and nugget of the model-error kernel of (4') by
    RESTRICTED MAXIMUM LIKELIHOOD, from a set of fitted pixels
    (profile_retrieval results, e.g. the diagonal pass). EXPERIMENTAL - not
    used by default (profile_pipeline kernel_method='reml').

    WHY IT WAS TRIED. estimate_model_error_kernel() fits the kernel to the
    autocorrelation of FITTED residuals, which decorrelate faster than the
    errors they estimate. On synthetic fires in real t014 backgrounds with a
    2% SMOOTH model error that estimator gave 32 nm and 95% coverage
    0.86-0.88, and this one 0.95-0.98 at the same point accuracy.

    WHY IT IS NOT THE DEFAULT. With the model error resampled from the
    RESIDUALS OF REAL t014 FITS (the real error structure), the
    autocorrelation estimator is already calibrated (coverage 0.94-0.95)
    and this one gives the same coverage with 30-40% wider intervals and a
    4-5 K worse mean absolute error (VERIFICATION.md S22).

    WHAT. For each pixel the Gaussian restricted log-likelihood (Patterson &
    Thompson 1971; Harville 1977)

        -2 l_R = r' S^-1 r + log|S| + log|Z' S^-1 Z|

    is evaluated with S = S(eps, l, nugget) of (4') around the pixel's
    fitted radiance, r the residual of the closed FCLS fit under S at the
    pixel's fitted temperature, and Z the active design with the closure
    eliminated; the sum over pixels is minimised on a fixed grid, then on a
    fixed refinement around the grid optimum. Every pixel the forward model
    describes is used - converged, reduced chi-square <= `max_reduced_chi2`
    in `results` (the rule that marks a fit 'poor'; pass a pass whose eps
    was calibrated to a median reduced chi-square of 1). No random
    subsample: on the real tile a 200-pixel subsample moved the scene
    estimates and the temperatures by tens of kelvin. The 'poor' pixels are
    left out because a covariance model is a description of the errors of
    fits the model CAN make: on DT0000149658 t014 their structured residuals
    (median reduced chi-square 27; soft-saturation plateaus) dominated the
    likelihood, pushed the optimum to the grid edge and the median interval
    width to 350 K. Censored rows are constraints, not measurements, and are
    left out. The result is deterministic.

    The scale eps is estimated jointly and reported, but the caller keeps
    the established calibration of eps (median reduced chi-square = 1,
    calibrate_model_error()) under the REML kernel: that is the combination
    validated against synthetic truth.

    Returns dict corr_length_nm, nugget, eps_reml, n_pixels, and the coarse
    and fine tables with their grids."""
    from joblib import Parallel, delayed
    wavelength = np.asarray(wavelength, dtype=float)
    if not isinstance(endmember_libs, (list, tuple)):
        endmember_libs = [endmember_libs] * len(results)
    jobs = []
    for i, r in enumerate(results):
        if not r.get('converged') or not (r.get('reduced_chi2', np.inf) <= max_reduced_chi2):
            continue
        rows = np.flatnonzero(r['_good'])
        if r.get('_censored') is not None:
            rows = rows[~np.asarray(r['_censored'], bool)]
        if len(rows) < MIN_FIT_BANDS:
            continue
        g = np.zeros(len(wavelength), bool)
        g[rows] = True
        jobs.append((np.asarray(spectra[i], float)[g], wavelength[g], np.atleast_2d(endmember_libs[i])[:, g],
                     np.ones(g.sum()) if tau is None else np.asarray(tau, float)[g], float(r['T_K']),
                     np.asarray(r['model'], float)[g], snr,
                     None if reference_radiance is None else np.asarray(reference_radiance, float)[g],
                     emissivity, include_shade))
    if not jobs:
        return {'corr_length_nm': 0.0, 'nugget': 1.0, 'eps_reml': np.nan, 'n_pixels': 0}

    def _total(Ls, nugs, epss):
        tabs = Parallel(n_jobs=n_jobs)(delayed(_reml_pixel_table)(*j, Ls, nugs, epss) for j in jobs)
        return np.sum(tabs, axis=0)          # NaN where any pixel's design was singular

    coarse = _total(REML_L_GRID, REML_NUG_GRID, REML_EPS_GRID)
    a_, b_, c_ = np.unravel_index(np.nanargmin(coarse), coarse.shape)

    def _between(grid, k, n, log):
        lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        return np.exp(np.linspace(np.log(lo), np.log(hi), n)) if log else np.linspace(lo, hi, n)

    Lf = _between(REML_L_GRID, a_, 5, True)
    Nf = _between(REML_NUG_GRID, b_, 5, False)
    Ef = _between(REML_EPS_GRID, c_, 9, True)
    fine = _total(Lf, Nf, Ef)
    a2, b2, c2 = np.unravel_index(np.nanargmin(fine), fine.shape)
    return {'corr_length_nm': float(Lf[a2]), 'nugget': float(Nf[b2]), 'eps_reml': float(Ef[c2]),
            'n_pixels': len(jobs), 'table_coarse': coarse, 'table_fine': fine,
            'grid_fine': (Lf, Nf, Ef),
            'at_grid_edge': bool(a_ in (0, len(REML_L_GRID) - 1) or b_ == len(REML_NUG_GRID) - 1
                                 or c_ in (0, len(REML_EPS_GRID) - 1)),
            'edge_parameters': [n_ for n_, hit in (('corr_length', a_ in (0, len(REML_L_GRID) - 1)),
                                                   ('nugget', b_ == len(REML_NUG_GRID) - 1),
                                                   ('eps', c_ in (0, len(REML_EPS_GRID) - 1))) if hit]}


def empirical_null_threshold(background_spectra, wavelength, endmember_libs, alpha_fa=0.01,
                             device=None, **kwargs):
    """Detection threshold for the fire likelihood-ratio statistic (6),
    calibrated on non-fire pixels of the same scene: the (1 - alpha_fa)
    quantile of Lambda over `background_spectra`.

    Because T is searched over an interval, Lambda under H0 is the maximum
    of a family of chi2_1 statistics and its distribution depends on the
    scene, so it is measured, not assumed. Returns (threshold, lambdas)."""
    background_spectra = np.atleast_2d(np.asarray(background_spectra, dtype=float))
    if not isinstance(endmember_libs, (list, tuple)):
        endmember_libs = [endmember_libs] * background_spectra.shape[0]
    res = _many(list(background_spectra), wavelength, list(endmember_libs), device=device, **kwargs)
    lam = [r['fire_lrt'] for r in res if r['converged']]
    lam = np.asarray(lam, dtype=float)
    if lam.size == 0:
        return np.nan, lam
    return float(np.quantile(lam, 1.0 - alpha_fa)), lam


def minimum_detectable_fraction(T_values, background_spectrum, wavelength, endmembers, sigma,
                                lambda_threshold, emissivity=1.0, tau=None):
    """Minimum detectable fire fraction f_min(T), equation (7), for a pixel
    whose fire-free spectrum is `background_spectrum`, fitted against
    `endmembers` with per-band noise `sigma` (1-D diagonal std, or a lower
    Cholesky factor of a full covariance), at detection threshold
    `lambda_threshold` on the statistic (6).

    Adding fire at fraction f turns b into b + f (eps B(T) - b). The part of
    that change the endmember library can absorb is invisible; the
    detectable part is its projection onto the orthogonal complement of
    span(whitened endmembers). The unconstrained projector absorbs at least
    as much as the non-negative closed fit can, so this is a CONSERVATIVE
    (upper) bound on the true detection limit. Returns f_min array."""
    from scipy.linalg import solve_triangular
    wavelength = np.asarray(wavelength, dtype=float)
    b = np.asarray(background_spectrum, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    E = np.atleast_2d(np.asarray(endmembers, dtype=float))
    tau_arr = np.ones_like(wavelength) if tau is None else np.asarray(tau, float)
    if sigma.ndim == 1:
        good = np.isfinite(b) & np.isfinite(sigma) & (sigma > 0) & np.all(np.isfinite(E), axis=0)
        sg = sigma[good]
        whiten = lambda v: v / sg
        Ew = (E[:, good] / sg).T
    else:
        good = np.ones(len(b), dtype=bool)
        whiten = lambda v: solve_triangular(sigma, v, lower=True, check_finite=False)
        Ew = solve_triangular(sigma, E.T, lower=True, check_finite=False)
    Q, _ = np.linalg.qr(Ew)
    out = []
    for T in np.atleast_1d(T_values):
        s = whiten(emissivity * tau_arr[good] * _planck_nm(wavelength[good], T) - b[good])
        s_perp = s - Q @ (Q.T @ s)
        nrm = float(np.linalg.norm(s_perp))
        out.append(math.sqrt(max(lambda_threshold, 0.0)) / nrm if nrm > 0 else np.inf)
    return np.asarray(out)


# --------------------------------------------------------------------------
# per-scene driver
# --------------------------------------------------------------------------

def run_profile_on_pixels(fire_spectra, wavelength, pool, band_mask=None, n_endmembers=20,
                          n_draws=5, snr=None, rel_model_error=None, sigma_instrument=None,
                          reference_radiance=None, corr_length_nm=150.0, nugget=0.2,
                          calibrate=True, emissivity=1.0, T_bounds=(300.0, 2000.0),
                          alpha=0.05, fire_points=None, valid_masks=None, censored_masks=None,
                          censor_levels=None, poor_fit_reduced_chi2=5.0, libraries=None, rng=None,
                          progress=None, device=None, method_label='ProfileML',
                          citation='this work (profile_retrieval.py)', **kwargs):
    """The profile-likelihood retrieval over every fire pixel of a scene.

    fire_spectra : (n_pix, n_bands_full) spectra; `band_mask` selects the
        fit window (the same mask cfg_band_mask_for_methods() gives the
        other methods).
    pool : (n_pool, n_bands_full) background candidates; `n_endmembers` are
        drawn per pixel (and per draw), `n_draws` independent libraries are
        tried; the point estimate is the draw with the lowest chi2_min
        (maximum likelihood over the discrete set of libraries as well as
        over T) and the interval is the envelope over the draws (see the
        loop). `libraries`, if given, overrides the random draw: a list of
        n_pix lists of (J, n_bands_full) arrays (e.g. local-neighbourhood
        libraries from the caller).
    rel_model_error : eps_m of (4); None -> calibrated on these pixels by
        calibrate_model_error() (printed), unless calibrate=False, in which
        case 0.05.
    valid_masks, censored_masks : optional (n_pix, n_bands_full) bool arrays
        (per-pixel band validity from the L1B pixel mask; saturated bands),
        censor_levels the matching (n_pix, n_bands_full) radiance ceilings.
    device : where the per-pixel retrievals run - None (all CPU cores, or
        the PROFILEML_DEVICE environment variable), 'joblib' (all cores) or
        'serial'; see profile_batch.resolve_device().
        The estimator is the same on every device.
    poor_fit_reduced_chi2 : with eps_m calibrated so the MEDIAN reduced
        chi-square is 1, a pixel above this value is one the forward model
        cannot describe (on the Portugal scene: unflagged soft-saturation
        plateaus between 2100 and 2300 nm that no Planck mixture can fit);
        its row is marked fit_quality='poor' (and a pixel with no retrieval
        at all 'failed') and should be excluded from
        aggregates, as T_at_bound rows already are.
    Returns a pandas.DataFrame with the run_methods_on_pixels() schema plus
    the profile-likelihood diagnostics (T_ci_low, T_ci_high, T_upper_unbounded, ...).
    """
    import pandas as pd

    fire_spectra = np.atleast_2d(np.asarray(fire_spectra, dtype=float))
    wavelength = np.asarray(wavelength, dtype=float)
    m = np.ones(len(wavelength), dtype=bool) if band_mask is None else np.asarray(band_mask, bool)
    w = wavelength[m]
    pool = np.atleast_2d(pool)          # own dtype: rows are converted where drawn
    rng = rng if rng is not None else np.random.default_rng(0)
    n_pix = fire_spectra.shape[0]
    sig_inst = None if sigma_instrument is None else np.asarray(sigma_instrument, float)[m]
    # Reference radiance for the two-regime noise model: the scene's median
    # background radiance per band, unless the caller supplies one.
    ref_rad = (column_median(pool, m) if reference_radiance is None
               else np.asarray(reference_radiance, float)[m])
    common = dict(snr=snr, sigma_instrument=sig_inst, reference_radiance=ref_rad,
                  corr_length_nm=corr_length_nm, nugget=nugget)

    if libraries is not None:
        libs = [[np.atleast_2d(np.asarray(L, float))[:, m] for L in libraries[i]] for i in range(n_pix)]
    else:
        libs = [[np.asarray(pool[rng.integers(pool.shape[0], size=min(n_endmembers, pool.shape[0]))][:, m],
                            dtype=float)
                 for _ in range(max(int(n_draws), 1))] for _ in range(n_pix)]

    def _pix_masks(i):
        d = {}
        if valid_masks is not None:
            d['valid_mask'] = np.asarray(valid_masks[i], bool)[m]
        if censored_masks is not None:
            d['censored_mask'] = np.asarray(censored_masks[i], bool)[m]
            d['censor_level'] = (np.asarray(censor_levels[i], float)[m] if censor_levels is not None
                                 else fire_spectra[i, m])
        return d

    eps = rel_model_error
    calib_table = []
    if eps is None:
        if calibrate and n_pix > 0:
            eps, calib_table = calibrate_model_error(
                fire_spectra[:, m], w, [l[0] for l in libs], emissivity=emissivity,
                T_bounds=T_bounds, per_pixel_kwargs=[_pix_masks(i) for i in range(n_pix)],
                device=device, **common, **kwargs)
            print(f'  This work: calibrated multiplicative model error eps_m = {eps:.4f} '
                  f'(median reduced chi2 -> 1 over {n_pix} fire pixels)')
        else:
            eps = 0.05

    # Every (pixel, library draw) retrieval is independent: they are run as
    # one batch per block of pixels (all cores, see `device`),
    # the blocks only so progress can be reported.
    block = max(1, min(n_pix, 64))
    fits = [None] * n_pix
    for b0 in range(0, n_pix, block):
        b1 = min(n_pix, b0 + block)
        rad, lb, pk = [], [], []
        for i in range(b0, b1):
            for lib in libs[i]:
                rad.append(fire_spectra[i, m]); lb.append(lib); pk.append(_pix_masks(i))
        res = _many(rad, w, lb, per_pixel=pk, device=device, emissivity=emissivity,
                    rel_model_error=eps, T_bounds=T_bounds, alpha=alpha, **common, **kwargs)
        k = 0
        for i in range(b0, b1):
            fits[i] = res[k:k + len(libs[i])]; k += len(libs[i])
        if progress is not None:
            progress('ProfileML', b1, n_pix)
    rows = []
    for i in range(n_pix):
        y = fire_spectra[i, m]
        best = None
        draws = []
        for r in fits[i]:
            if r['converged']:
                draws.append(r)
                if best is None or r['chi2_min'] < best['chi2_min']:
                    best = r
        if best is None:
            best = r
        # The interval reported is the ENVELOPE of the profile intervals over
        # the library draws: the per-draw interval is conditional on that
        # library, and the library is itself uncertain (a random 20-spectrum
        # draw cannot represent every background exactly). On synthetic
        # truth the single-draw kernel interval covered 70% and the 5-draw
        # envelope 85% at nominal 95% (verify17_profile.py).
        if len(draws) > 1:
            best = dict(best)
            best['T_ci_low'] = float(min(d_['T_ci_low'] for d_ in draws))
            best['T_ci_high'] = float(max(d_['T_ci_high'] for d_ in draws))
            best['T_lower_unbounded'] = bool(any(d_['T_lower_unbounded'] for d_ in draws))
            best['T_upper_unbounded'] = bool(any(d_['T_upper_unbounded'] for d_ in draws))
            Ts = np.array([d_['T_K'] for d_ in draws])
            best['T_draw_spread'] = float(Ts.max() - Ts.min())
        else:
            best['T_draw_spread'] = 0.0
        rx, cx = (int(fire_points[i][0]), int(fire_points[i][1])) if fire_points is not None else (-1, -1)
        rows.append({'Pixel': i, 'ind_x': rx, 'ind_y': cx, 'T_K': best['T_K'],
                     'fire_fraction': best['fire_fraction'], 'rmse': best['rmse'],
                     'converged': bool(best['converged']), 'T_at_bound': bool(best['T_at_bound']),
                     'method': method_label, 'citation': citation,
                     'T_ci_low': best['T_ci_low'], 'T_ci_high': best['T_ci_high'],
                     'T_lower_unbounded': best['T_lower_unbounded'],
                     'T_upper_unbounded': best['T_upper_unbounded'],
                     'f_at_ci_low': best['f_at_ci_low'], 'f_at_ci_high': best['f_at_ci_high'],
                     'T_draw_spread': best['T_draw_spread'], 'n_draws': len(draws),
                     'chi2_min': best['chi2_min'], 'reduced_chi2': best['reduced_chi2'],
                     'dof': best['dof'], 'dof_eff': best['dof_eff'], 'rho_lag1': best['rho_lag1'],
                     'fire_lrt': best['fire_lrt'], 'n_local_minima': best['n_local_minima'],
                     'ci_possibly_disconnected': best['ci_possibly_disconnected'],
                     'closure_error': best['closure_error'], 'eps_model_error': eps,
                     # 'failed': no retrieval (reduced chi2 NaN - which the bare
                     # comparison with the threshold used to label 'ok')
                     'fit_quality': ('failed' if not (best['converged'] and np.isfinite(best['reduced_chi2']))
                                     else 'poor' if best['reduced_chi2'] > poor_fit_reduced_chi2 else 'ok'),
                     'n_bands_invalid': best['n_bands_invalid'],
                     'n_bands_censored': best.get('n_bands_censored', 0),
                     'censor_violation': best.get('censor_violation', np.nan),
                     'rmse_rel': best['rmse'] / max(float(np.mean(np.abs(y))), 1e-300),
                     'irls_passes': best.get('irls_passes', 0),
                     'irls_converged': best.get('irls_converged')})
    cols = ['Pixel', 'ind_x', 'ind_y', 'T_K', 'fire_fraction', 'rmse', 'converged', 'T_at_bound',
            'method', 'citation', 'T_ci_low', 'T_ci_high', 'T_lower_unbounded', 'T_upper_unbounded',
            'f_at_ci_low', 'f_at_ci_high', 'T_draw_spread', 'n_draws', 'chi2_min', 'reduced_chi2',
            'dof', 'dof_eff', 'rho_lag1',
            'fire_lrt', 'n_local_minima', 'ci_possibly_disconnected', 'closure_error',
            'eps_model_error', 'fit_quality', 'n_bands_invalid', 'n_bands_censored', 'censor_violation',
            'rmse_rel', 'irls_passes', 'irls_converged']
    df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame(columns=cols)
    df.attrs['calibration_table'] = calib_table
    df.attrs['eps_model_error'] = eps
    return df

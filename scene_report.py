#!/usr/bin/env python
# coding: utf-8
"""
Publication-quality PDF report for one processed EnMAP scene.

build_scene_report(results_dir) reads everything the pipeline wrote for the
scene - the per-pixel results of every retrieval that ran (this work v1 and
v2, Amici et al. 2022, Dozier 1981, Dennison et al. 2006, the smoke
candidate), the scene transmittance, the validated frames, the run record
(RunInfo_<scene>.json) and the context composite (ReportContext_<scene>.npz)
- and typesets one PDF:

    summary, per-method statistics, temperature classes, model-derived
    fire radiative power; context and temperature maps of every method;
    distributions; pairwise agreement (scatter, Bland-Altman, table);
    this work's uncertainty and fit diagnostics; the scene transmittance;
    the v1/v2 sensitivity when both ran; the reference data; methods and
    interpretation notes; every pipeline figure; a reproducibility record
    (settings, software versions, SHA-256 of every output file).

It works on the folder alone, so a report can be rebuilt for any earlier
run (`python scene_report.py <results folder>`); run_pipeline() calls it at
the end of stage 8 (generate_report=True) and the GUI exposes both.

Statistics policy - every number in the report follows these rules, which
the report itself states:
  * a pixel enters a method's statistics only if that method returned a
    usable temperature: this work - fit_quality 'ok' (converged, reduced
    chi-square <= 5); Amici et al. (2022) - finite and not at its search
    bound; Dozier / Dennison - converged and not at the search bound;
  * pairwise agreement uses the pixels BOTH methods retrieved;
  * intervals are the methods' own: this work's 95% profile-likelihood
    interval (its lower/upper end at the search bound is reported as
    unbounded, not as a width); Amici et al.'s 2.5-97.5 percentile of the
    trial spread, which is a spread over background draws and NOT a
    confidence interval;
  * fire radiative power is DERIVED from each method's own (T, f) as
    eps sigma T^4 f A_pixel; it is a model quantity, not an independent
    measurement, and a lower bound for the scene because pixels with no
    retrieval (typically the brightest, saturated ones) contribute nothing.

Requires reportlab (present in the 'emit' environment).
"""

import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import re
import sys
import warnings
from xml.sax.saxutils import escape as _esc

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SIGMA_SB = 5.670374419e-8          # W m-2 K-4
PIXEL_AREA_M2 = 30.0 * 30.0        # EnMAP ground sampling distance 30 m
T_CLASSES = (300, 600, 800, 1000, 1200, 2001)

# Okabe-Ito colours, the semantic assignment of paper/figstyle.py
COLORS = {'ProfileML': '#D55E00', 'ProfileMLv2': '#E69F00', 'Amici2022': '#0072B2',
          'Dennison2006': '#009E73', 'Dozier1981': '#CC79A7', 'SmokeSTCR': '#56B4E9',
          'Amici2022_driver': '#7F7F7F', 'ProfileML_driver': '#BBBBBB'}
LABELS = {'ProfileML': 'This work (v1)', 'ProfileMLv2': 'This work (v2)',
          'Amici2022': 'Amici et al. (2022)', 'Dennison2006': 'Dennison et al. (2006)',
          'Dozier1981': 'Dozier (1981)', 'SmokeSTCR': 'Smoke correction (candidate)',
          'Amici2022_driver': 'Amici et al. (2022), common driver',
          'ProfileML_driver': 'This work, common driver (no tau, no masks)'}
CITATIONS = {
    'ProfileML': 'This work: profile-likelihood inversion of the Planck-plus-background mixture with '
                 'the scene fire-term transmittance, L1B quality masks (saturated bands as one-sided '
                 'constraints), calibrated correlated error model and 95% profile-likelihood interval '
                 '(VERIFICATION.md Sections 17-22).',
    'ProfileMLv2': 'This work, variant 2: as v1 with (i) the error-model reweighting iterated to its fixed '
                   'point (quasi-likelihood score equations) and (ii) the scene-level estimates (transmittance, '
                   'error kernel) made from adequately fitted pixels only, through a capped (trimmed) objective; '
                   'VERIFICATION.md Sections 22-23.',
    'Amici2022': 'Amici, S. et al. (2022), J. Geophys. Res. Biogeosciences 127, e2022JG007055: multi-'
                 'trial Planck-mixture fit, 1400-1900 nm, 20 background spectra, 15 trials, as published and '
                 'checked against the authors\' released code. Input here is L1B top-of-atmosphere radiance '
                 '(theirs: PRISMA L2B bottom-of-atmosphere radiance).',
    'Dennison2006': 'Dennison, P. E. et al. (2006), Remote Sens. Environ. 100, 212-222: MESMA with a '
                    '500-1500 K Planck library, one reflected endmember and shade, their channels, encoded-'
                    'radiance RMSE and feasibility rule, unsaturated and unflagged channels only. Their three '
                    'scene inputs are reproduced from the image: path radiance from the darkest background '
                    'pixels, six EAR endmembers from six spectral classes, and the one-way gas transmittance on '
                    'the emitted endmembers in place of MODTRAN (VERIFICATION.md S31; that transmittance moves '
                    'the result by up to ~170 K on this kind of scene).',
    'Dozier1981': 'Dozier, J. (1981), Remote Sens. Environ. 11, 221-229: the two-channel inversion, here '
                  'on SWIR channels (1600/2200 nm) with the measured radiance of the pixel\'s neighbours in '
                  'place of the background Planck term - an adaptation of the thermal-infrared original; '
                  'pixels with a saturated or flagged channel are not solved.',
    'SmokeSTCR': 'Candidate smoke-transmittance correction of this repository (unpublished, '
                 'VERIFICATION.md Section 16); shown only where its significance gate fired.',
}
PRIMARY_ORDER = ('ProfileML', 'ProfileMLv2', 'Amici2022', 'Dennison2006', 'Dozier1981')


# ----------------------------------------------------------------- file discovery
def _latest(pattern):
    paths = [p for p in glob.glob(pattern) if os.path.isfile(p)]
    return max(paths, key=os.path.getmtime) if paths else None


def _scene_id_from_folder(results_dir):
    for p in sorted(glob.glob(os.path.join(results_dir, 'Results_*_Amici2022_*.csv'))
                    + glob.glob(os.path.join(results_dir, 'Results_*_ProfileML_*.csv'))):
        m = re.match(r'Results_(.+?)_(?:Amici2022|ProfileML|ProfileMLv2)(?:_validated)?_\d\d-\d\d-\d{4}\.csv$',
                     os.path.basename(p))
        if m:
            return m.group(1)
    return os.path.basename(os.path.normpath(results_dir))


def _read(path):
    try:
        return pd.read_csv(path) if path else None
    except Exception as exc:                                  # a corrupt file must not stop the report
        warnings.warn(f'scene_report: could not read {path}: {exc}')
        return None


def collect_scene_outputs(results_dir, scene_id=None):
    """Find and load every output of one scene. Returns a dict with the
    method frames (normalised, see _normalise), the raw frames, the
    transmittance tables, the figure paths and the run record."""
    sid = scene_id or _scene_id_from_folder(results_dir)
    J = lambda *a: os.path.join(results_dir, *a)
    run_info = {}
    p = J(f'RunInfo_{sid}.json')
    if os.path.exists(p):
        try:
            with open(p, encoding='utf-8') as fh:
                run_info = json.load(fh)
        except Exception as exc:
            warnings.warn(f'scene_report: could not read {p}: {exc}')
    # The run record names the date tag and the methods of the run it
    # describes, so files left in the folder by an earlier run are not mixed
    # in; without a record, the newest file of each kind is used.
    d = run_info.get('date_tag') or '[0-9]*'
    ran = run_info.get('methods_run')
    raw = {}
    pv = _latest(J(f'Results_{sid}_ProfileML_validated_{d}.csv'))
    raw['ProfileML'] = _read(pv or _latest(J(f'Results_{sid}_ProfileML_{d}.csv')))
    pv2 = _latest(J(f'Results_{sid}_ProfileMLv2_validated_{d}.csv'))
    raw['ProfileMLv2'] = _read(pv2 or _latest(J(f'Results_{sid}_ProfileMLv2_{d}.csv')))
    amici_val = _latest(J(f'Results_{sid}_Amici2022_validated_{d}.csv'))
    raw['Amici2022'] = _read(amici_val or _latest(J(f'Results_{sid}_Amici2022_{d}.csv')))
    for key in ('Dozier1981', 'Dennison2006', 'SmokeSTCR'):
        raw[key] = _read(_latest(J(f'MethodCompare_{sid}_{key}_{d}.csv')))
    raw['Amici2022_driver'] = _read(_latest(J(f'MethodCompare_{sid}_Amici2022_{d}.csv')))
    raw['ProfileML_driver'] = _read(_latest(J(f'MethodCompare_{sid}_ProfileML_{d}.csv')))
    if ran is not None:
        for k in ('ProfileML', 'ProfileMLv2', 'Dozier1981', 'Dennison2006', 'SmokeSTCR'):
            if k not in ran:
                raw[k] = None
        if 'Dozier1981' not in ran and 'Dennison2006' not in ran:
            raw['Amici2022_driver'] = raw['ProfileML_driver'] = None
    raw = {k: v for k, v in raw.items() if v is not None and len(v)}
    tau = {}
    for tag in ('ProfileML', 'ProfileMLv2'):
        t = _read(_latest(J(f'Transmittance_{sid}_{tag}_{d}.csv'))) if tag in raw else None
        if t is not None:
            tau[tag] = t
    context = None
    p = J(f'ReportContext_{sid}.npz')
    if os.path.exists(p):
        try:
            with np.load(p) as z:
                context = {k: z[k] for k in z.files}
        except Exception as exc:
            warnings.warn(f'scene_report: could not read {p}: {exc}')
    figs = sorted(p for p in glob.glob(J('Fig*.png')) if os.path.isfile(p))
    methods = {k: _normalise(k, v) for k, v in raw.items()}
    frp_cmp = {'summary': _read(_latest(J(f'FRPComparison_{sid}_{d}.csv'))),
               'pairs': _read(_latest(J(f'FRPPairs_{sid}_{d}.csv'))),
               'detections': _read(_latest(J(f'ActiveFire_{sid}_*.csv')))}
    unc_tab = _read(_latest(J(f'Uncertainty_{sid}_ProfileMLv2_{d}.csv'))) if 'ProfileMLv2' in raw else None
    coreg = {'summary': _read(_latest(J(f'CoregSummary_{sid}_{d}.csv'))),
             'tiepoints': _read(_latest(J(f'CoregTiepoints_{sid}_{d}.csv'))),
             'info': run_info.get('coregistration')}
    return {'scene_id': sid, 'results_dir': results_dir, 'raw': raw, 'methods': methods,
            'tau': tau, 'run_info': run_info, 'context': context, 'figures': figs, 'frp': frp_cmp,
            'uncertainty': unc_tab, 'coreg': coreg}


# ----------------------------------------------------------------- normalisation
def _col(df, name, default=np.nan):
    return df[name].values if name in df else np.full(len(df), default)


def _bool(df, name, default=False):
    if name not in df:
        return np.full(len(df), default, bool)
    v = df[name]
    if v.dtype == bool:
        return v.values
    return v.astype(str).str.lower().isin(('true', '1', '1.0')).values


def _normalise(key, df):
    """One frame per method with common columns: ind_x, ind_y, T, lo, hi,
    lo_unb, hi_unb, f, valid, interval_kind, plus lat/lon when present."""
    out = pd.DataFrame({'ind_x': _col(df, 'ind_x'), 'ind_y': _col(df, 'ind_y')})
    if key in ('ProfileML', 'ProfileMLv2'):
        T = _col(df, 'T_K').astype(float)
        out['T'] = T
        out['lo'], out['hi'] = _col(df, 'T_ci_low'), _col(df, 'T_ci_high')
        out['lo_unb'], out['hi_unb'] = _bool(df, 'T_lower_unbounded'), _bool(df, 'T_upper_unbounded')
        out['f'] = _col(df, 'fire_fraction')
        q = df['fit_quality'].astype(str).values if 'fit_quality' in df else np.full(len(df), 'ok')
        out['quality'] = q
        # an interval open at BOTH ends (it spans the whole search range) means the data do not
        # constrain the temperature at all: counted as 'unconstrained', not as a retrieval
        out['unconstrained'] = (q == 'ok') & np.isfinite(T) & out['lo_unb'].values & out['hi_unb'].values
        out['valid'] = (q == 'ok') & np.isfinite(T) & ~_bool(df, 'T_at_bound') & ~out['unconstrained'].values
        out['interval_kind'] = '95% profile-likelihood interval'
    elif key == 'Amici2022':
        T = _col(df, 'mean_temp_K').astype(float)
        out['T'] = T
        out['lo'], out['hi'] = _col(df, 'temp_p2_5_K'), _col(df, 'temp_p97_5_K')
        out['lo_unb'] = np.zeros(len(df), bool); out['hi_unb'] = np.zeros(len(df), bool)
        out['f'] = _col(df, 'mean_P_coeff')
        out['valid'] = np.isfinite(T) & ~_bool(df, 'T_at_bound')
        out['interval_kind'] = '2.5-97.5 percentile of the trial spread (not a confidence interval)'
    else:
        T = _col(df, 'T_K').astype(float)
        if key == 'SmokeSTCR' and 'fallback' in df:
            T = np.where(_bool(df, 'fallback'), np.nan, T)
        out['T'] = T
        out['lo'] = np.nan; out['hi'] = np.nan
        out['lo_unb'] = np.zeros(len(df), bool); out['hi_unb'] = np.zeros(len(df), bool)
        out['f'] = _col(df, 'fire_fraction')
        conv = _bool(df, 'converged', True) if 'converged' in df else np.ones(len(df), bool)
        out['valid'] = conv & np.isfinite(T) & ~_bool(df, 'T_at_bound')
        out['interval_kind'] = 'none'
    for c in ('lat_wgs84', 'lon_wgs84'):
        if c in df:
            out[c] = df[c].values
    return out


# ----------------------------------------------------------------- statistics
def method_statistics(m, emissivity=1.0):
    v = m['valid'].values
    T = m['T'].values[v]
    row = {'n_detected': len(m), 'n_retrieved': int(v.sum()),
           'n_unconstrained': int(m['unconstrained'].sum()) if 'unconstrained' in m else 0}
    if len(T) == 0:
        return row
    p = np.percentile(T, [0, 5, 25, 50, 75, 95, 100])
    row.update(min=p[0], p5=p[1], p25=p[2], median=p[3], p75=p[4], p95=p[5], max=p[6],
               mean=float(np.mean(T)), sd=float(np.std(T, ddof=1)) if len(T) > 1 else np.nan)
    f = m['f'].values[v].astype(float)
    good_f = np.isfinite(f) & (f > 0)
    if good_f.sum():
        row['T_fweighted'] = float(np.sum(f[good_f] * T[good_f]) / np.sum(f[good_f]))
        frp = emissivity * SIGMA_SB * T[good_f] ** 4 * f[good_f] * PIXEL_AREA_M2
        row['FRP_MW'] = float(np.sum(frp) / 1e6)
        row['fire_area_m2'] = float(np.sum(f[good_f]) * PIXEL_AREA_M2)
    lo, hi = m['lo'].values[v].astype(float), m['hi'].values[v].astype(float)
    fin = np.isfinite(lo) & np.isfinite(hi) & ~m['lo_unb'].values[v] & ~m['hi_unb'].values[v]
    if fin.any():
        row['width_median'] = float(np.median(hi[fin] - lo[fin]))
        row['n_bounded'] = int(fin.sum())
    return row


def agreement(a, b):
    """Pairwise agreement of method b against a on pixels both retrieved."""
    key = ['ind_x', 'ind_y']
    j = a[key + ['T', 'valid']].merge(b[key + ['T', 'valid']], on=key, suffixes=('_a', '_b'))
    s = j['valid_a'].values & j['valid_b'].values
    if s.sum() < 3:
        return {'n': int(s.sum())}
    x, y = j['T_a'].values[s], j['T_b'].values[s]
    d = y - x
    from scipy.stats import spearmanr
    rho = float(spearmanr(x, y).correlation)
    return {'n': int(s.sum()), 'median_diff': float(np.median(d)), 'mean_diff': float(np.mean(d)),
            'q25': float(np.percentile(d, 25)), 'q75': float(np.percentile(d, 75)),
            'mad_abs': float(np.median(np.abs(d))), 'rmsd': float(np.sqrt(np.mean(d ** 2))),
            'loa_lo': float(np.mean(d) - 1.96 * np.std(d, ddof=1)),
            'loa_hi': float(np.mean(d) + 1.96 * np.std(d, ddof=1)), 'spearman': rho, 'x': x, 'y': y}


# ----------------------------------------------------------------- figures
def _style():
    matplotlib.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5, 'xtick.labelsize': 7,
        'ytick.labelsize': 7, 'legend.fontsize': 6.8, 'legend.frameon': False,
        'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.0,
        'pdf.fonttype': 42, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
        'mathtext.default': 'regular'})


def _panel(ax, letter):
    ax.text(-0.02, 1.02, letter, transform=ax.transAxes, fontsize=9, fontweight='bold',
            ha='right', va='bottom')


def _save(fig, fig_dir, name):
    os.makedirs(fig_dir, exist_ok=True)
    png = os.path.join(fig_dir, name + '.png')
    fig.savefig(png, dpi=300)
    # no creation date in the vector copy, so a rerun gives the same file
    fig.savefig(os.path.join(fig_dir, name + '.pdf'), metadata={'CreationDate': None})
    plt.close(fig)
    return png


def _window(methods, n_rows=None, n_cols=None, pad=8, min_side=40):
    """Map window on the dense core of the fire: the detections are often a
    compact front plus a few scattered pixels kilometres away, and a window
    spanning all of them makes the front unreadable. The core is the
    densest 30-px neighbourhood; the window covers every detection within
    150 px of it (the outliers are counted and named in the captions)."""
    m0 = next(iter(methods.values()))
    xs = m0['ind_x'].values.astype(float); ys = m0['ind_y'].values.astype(float)
    ok = np.isfinite(xs) & np.isfinite(ys)
    xs, ys = xs[ok], ys[ok]
    d2 = (xs[:, None] - xs[None, :]) ** 2 + (ys[:, None] - ys[None, :]) ** 2
    c = int(np.argmax((d2 <= 30.0 ** 2).sum(axis=1)))
    near = d2[c] <= 150.0 ** 2
    xs, ys = xs[near], ys[near]
    r0, r1 = int(xs.min()) - pad, int(xs.max()) + pad + 1
    c0, c1 = int(ys.min()) - pad, int(ys.max()) + pad + 1
    if r1 - r0 < min_side:
        r0 -= (min_side - (r1 - r0)) // 2; r1 = r0 + min_side
    if c1 - c0 < min_side:
        c0 -= (min_side - (c1 - c0)) // 2; c1 = c0 + min_side
    # square-ish window so pixel footprints are drawn to scale
    h, w = r1 - r0, c1 - c0
    if h > w:
        c0 -= (h - w) // 2; c1 = c0 + h
    else:
        r0 -= (w - h) // 2; r1 = r0 + w
    if n_rows is not None:
        r0, r1 = max(r0, 0), min(r1, n_rows)
        c0, c1 = max(c0, 0), min(c1, n_cols)
    return r0, r1, c0, c1


def _scalebar(ax, n_px_extent, pixel_m=30.0):
    km_choices = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20)
    target = n_px_extent * pixel_m / 1000.0 / 4.0
    km = min(km_choices, key=lambda k: abs(k - target))
    L = km * 1000.0 / pixel_m
    from matplotlib.offsetbox import AnchoredOffsetbox, AuxTransformBox, TextArea, VPacker
    from matplotlib.patches import Rectangle
    bar = AuxTransformBox(ax.transData)
    h = abs(ax.get_ylim()[0] - ax.get_ylim()[1]) * 0.012
    bar.add_artist(Rectangle((0, 0), L, h, fc='k', ec='k', lw=0))
    box = VPacker(children=[TextArea(f'{km:g} km', textprops=dict(fontsize=6.5)), bar],
                  align='center', pad=0, sep=1.5)
    ab = AnchoredOffsetbox(loc='lower left', child=box, pad=0.35, borderpad=0.6, frameon=True)
    ab.patch.set(fc='white', ec='none', alpha=0.8)
    ax.add_artist(ab)


def fig_overview(out, fig_dir):
    ctx = out['context']
    methods = out['methods']
    base = next((methods[k] for k in ('ProfileML', 'Amici2022') if k in methods), next(iter(methods.values())))
    raw_a = out['raw'].get('Amici2022')
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))
    if ctx is not None and 'rgb' in ctx:
        rgb = ctx['rgb'].astype(float) / 255.0
        step = int(ctx.get('step', np.array(1)))
        axes[0].imshow(rgb, extent=(-0.5, rgb.shape[1] * step - 0.5, rgb.shape[0] * step - 0.5, -0.5),
                       interpolation='nearest')
        n_rows, n_cols = int(ctx['shape'][0]), int(ctx['shape'][1])
    else:
        n_rows = int(np.nanmax(base['ind_x'])) + 20; n_cols = int(np.nanmax(base['ind_y'])) + 20
        axes[0].set_facecolor('#eeeeee')
        axes[0].set_xlim(-0.5, n_cols - 0.5); axes[0].set_ylim(n_rows - 0.5, -0.5)
    cls = raw_a['detection_class'].values if (raw_a is not None and 'detection_class' in raw_a) else None
    if cls is not None:
        for c_, col, lab in (('fire', '#D55E00', 'fire'), ('candidate', '#56B4E9', 'candidate')):
            s = cls == c_
            if s.any():
                axes[0].scatter(raw_a['ind_y'].values[s], raw_a['ind_x'].values[s], s=4, c=col,
                                lw=0, label=f'{lab} ({int(s.sum())})')
        axes[0].legend(loc='upper right', markerscale=2.5, frameon=True, framealpha=0.85)
    else:
        axes[0].scatter(base['ind_y'], base['ind_x'], s=4, c='#D55E00', lw=0)
    r0, r1, c0, c1 = _window(methods, n_rows, n_cols)
    axes[0].add_patch(plt.Rectangle((c0 - 0.5, r0 - 0.5), c1 - c0, r1 - r0, fill=False, ec='k', lw=0.8))
    axes[0].set_title('Scene: detected fire pixels' + (' (SWIR2-SWIR1-NIR composite)' if ctx is not None else ''))
    axes[0].set_xlabel('column (30 m pixels)'); axes[0].set_ylabel('row (30 m pixels)')
    _scalebar(axes[0], n_cols)
    _panel(axes[0], 'a')
    if ctx is not None and 'rgb_window' in ctx:
        w = ctx['rgb_window'].astype(float) / 255.0
        wr0, wr1, wc0, wc1 = [int(v) for v in ctx['window']]
        axes[1].imshow(w, extent=(wc0 - 0.5, wc1 - 0.5, wr1 - 0.5, wr0 - 0.5), interpolation='nearest')
        axes[1].set_xlim(c0 - 0.5, c1 - 0.5); axes[1].set_ylim(r1 - 0.5, r0 - 0.5)
        axes[1].set_title('Fire area, local contrast stretch')
    else:
        axes[1].set_facecolor('#eeeeee')
        axes[1].scatter(base['ind_y'], base['ind_x'], s=6, c='#D55E00', marker='s', lw=0)
        axes[1].set_xlim(c0 - 0.5, c1 - 0.5); axes[1].set_ylim(r1 - 0.5, r0 - 0.5)
        axes[1].set_title('Fire area')
    axes[1].set_xlabel('column'); axes[1].set_aspect('equal')
    _scalebar(axes[1], c1 - c0)
    _panel(axes[1], 'b')
    fig.tight_layout()
    return _save(fig, fig_dir, 'R1_overview'), (r0, r1, c0, c1)


def _tmap(ax, m, win, norm, ctx, value='T'):
    r0, r1, c0, c1 = win
    img = np.full((r1 - r0, c1 - c0), np.nan)
    v = m['valid'].values
    xi = m['ind_x'].values[v].astype(int) - r0; yi = m['ind_y'].values[v].astype(int) - c0
    inside = (xi >= 0) & (xi < r1 - r0) & (yi >= 0) & (yi < c1 - c0)
    vals = m[value].values[v] if value in m else None
    img[xi[inside], yi[inside]] = vals[inside]
    if ctx is not None and 'grey_window' in ctx:
        g = ctx['grey_window'].astype(float) / 255.0
        wr0, wr1, wc0, wc1 = [int(t) for t in ctx['window']]
        ax.imshow(g, cmap='gray', vmin=0, vmax=1, extent=(wc0 - 0.5, wc1 - 0.5, wr1 - 0.5, wr0 - 0.5),
                  interpolation='nearest')
    else:
        ax.set_facecolor('#e9e9e9')
    # pixels detected but without a usable retrieval of THIS method
    nv = ~v
    xn = m['ind_x'].values[nv].astype(int); yn = m['ind_y'].values[nv].astype(int)
    if nv.any():
        ax.scatter(yn, xn, s=2.0, marker='s', c='#1f78b4', lw=0, alpha=0.7)
    im = ax.imshow(img, cmap='inferno', norm=norm, extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5),
                   interpolation='nearest')
    ax.set_xlim(c0 - 0.5, c1 - 0.5); ax.set_ylim(r1 - 0.5, r0 - 0.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(True)
    return im


def fig_maps(out, win, fig_dir, keys):
    methods = out['methods']
    # one shared scale; its top is the MEDIAN over methods of their 98th percentile, so one
    # method with a very different distribution saturates (arrow) instead of compressing the rest
    p2 = [np.percentile(methods[k]['T'].values[methods[k]['valid'].values], 2) for k in keys if methods[k]['valid'].sum()]
    p98 = [np.percentile(methods[k]['T'].values[methods[k]['valid'].values], 98) for k in keys if methods[k]['valid'].sum()]
    vmin, vmax = (min(p2), float(np.median(p98))) if p2 else (500.0, 1400.0)
    norm = Normalize(vmin=float(vmin), vmax=float(vmax))
    n = len(keys)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.45 * nrow + 0.35), squeeze=False)
    im = None
    for i, k in enumerate(keys):
        ax = axes[i // ncol][i % ncol]
        im = _tmap(ax, methods[k], win, norm, out['context'])
        v = methods[k]['valid'].values
        ax.set_title(f'{LABELS[k]}\n{int(v.sum())} of {len(v)} pixels retrieved', fontsize=7.2)
        _panel(ax, 'abcdefghij'[i])
        if i == 0:
            _scalebar(ax, win[3] - win[2])
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_axis_off()
    fig.subplots_adjust(right=0.88, hspace=0.25, wspace=0.06)
    cax = fig.add_axes([0.90, 0.15, 0.018, 0.7])
    cb = fig.colorbar(im, cax=cax, extend='both')
    cb.set_label('Retrieved fire temperature (K)')
    return _save(fig, fig_dir, 'R2_temperature_maps'), (vmin, vmax)


def fig_uncertainty_map(out, win, fig_dir, key='ProfileML'):
    m = out['methods'][key].copy()
    fin = m['valid'] & ~m['lo_unb'] & ~m['hi_unb'] & np.isfinite(m['lo']) & np.isfinite(m['hi'])
    m['W'] = np.where(fin, m['hi'] - m['lo'], np.nan)
    unb = m['valid'] & ~fin
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    m2 = m.copy(); m2['valid'] = fin.values
    Wv = m['W'].values[fin.values]
    norm = Normalize(0, float(np.percentile(Wv, 98)) if len(Wv) else 300)
    r0, r1, c0, c1 = win
    img = np.full((r1 - r0, c1 - c0), np.nan)
    xi = m['ind_x'].values[fin.values].astype(int) - r0; yi = m['ind_y'].values[fin.values].astype(int) - c0
    ok = (xi >= 0) & (xi < r1 - r0) & (yi >= 0) & (yi < c1 - c0)
    img[xi[ok], yi[ok]] = Wv[ok]
    if out['context'] is not None and 'grey_window' in out['context']:
        g = out['context']['grey_window'].astype(float) / 255.0
        wr0, wr1, wc0, wc1 = [int(t) for t in out['context']['window']]
        axes[0].imshow(g, cmap='gray', vmin=0, vmax=1, extent=(wc0 - 0.5, wc1 - 0.5, wr1 - 0.5, wr0 - 0.5))
    else:
        axes[0].set_facecolor('#e9e9e9')
    im = axes[0].imshow(img, cmap='viridis', norm=norm, extent=(c0 - 0.5, c1 - 0.5, r1 - 0.5, r0 - 0.5),
                        interpolation='nearest')
    if unb.any():
        axes[0].scatter(m['ind_y'].values[unb.values], m['ind_x'].values[unb.values], s=3, marker='x',
                        c='#D55E00', lw=0.5, label=f'interval reaches search bound ({int(unb.sum())})')
        axes[0].legend(loc='lower right', frameon=True, framealpha=0.85)
    axes[0].set_xlim(c0 - 0.5, c1 - 0.5); axes[0].set_ylim(r1 - 0.5, r0 - 0.5)
    axes[0].set_xticks([]); axes[0].set_yticks([])
    fig.colorbar(im, ax=axes[0], shrink=0.85, label='95% interval width (K)')
    axes[0].set_title(f'{LABELS[key]}: 95% interval width')
    _panel(axes[0], 'a')
    v = m['valid'].values
    T = m['T'].values[v]; lo = m['lo'].values[v]; hi = m['hi'].values[v]
    lu = m['lo_unb'].values[v]; hu = m['hi_unb'].values[v]
    o = np.argsort(T)
    x = np.arange(len(T))
    axes[1].vlines(x, np.where(lu[o], np.nan, lo[o]), np.where(hu[o], np.nan, hi[o]), color='#BBBBBB', lw=0.6)
    if (lu | hu).any():
        axes[1].vlines(x[(lu | hu)[o]], np.where(lu[o], 300, lo[o])[(lu | hu)[o]],
                       np.where(hu[o], 2000, hi[o])[(lu | hu)[o]], color='#F4B183', lw=0.35, alpha=0.7,
                       label='open-ended interval (to the search bound)')
    axes[1].plot(x, T[o], '.', ms=1.6, color=COLORS[key], label='point estimate')
    axes[1].set_xlabel('pixel (sorted by temperature)')
    axes[1].set_ylabel('Temperature (K)')
    axes[1].set_title('Point estimates and 95% intervals')
    axes[1].legend(loc='upper left')
    _panel(axes[1], 'b')
    fig.tight_layout()
    return _save(fig, fig_dir, f'R4_uncertainty_{key}')


def fig_distributions(out, fig_dir, keys):
    methods = out['methods']
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.2, 1.0, 1.0]))
    allT = np.concatenate([methods[k]['T'].values[methods[k]['valid'].values] for k in keys])
    lo, hi = (np.percentile(allT, [0.5, 99.5]) if len(allT) else (300, 2000))
    bins = np.linspace(lo, hi, 36)
    for k in keys:
        T = methods[k]['T'].values[methods[k]['valid'].values]
        if len(T) < 2:
            continue
        axes[0].hist(T, bins=bins, histtype='step', lw=1.1, color=COLORS[k], label=f'{LABELS[k]} (n={len(T)})')
        s = np.sort(T)
        axes[1].plot(s, np.arange(1, len(s) + 1) / len(s), color=COLORS[k], lw=1.1)
    axes[0].set_xlabel('Temperature (K)'); axes[0].set_ylabel('pixels'); axes[0].legend(loc='upper right')
    axes[0].set_title('Distribution'); _panel(axes[0], 'a')
    axes[1].set_xlabel('Temperature (K)'); axes[1].set_ylabel('cumulative fraction')
    axes[1].set_title('Empirical CDF'); axes[1].set_xlim(lo, hi); _panel(axes[1], 'b')
    data = [methods[k]['T'].values[methods[k]['valid'].values] for k in keys]
    keep = [i for i, d in enumerate(data) if len(d) >= 2]
    if keep:
        bp = axes[2].boxplot([data[i] for i in keep], whis=(5, 95), widths=0.55,
                             patch_artist=True, showfliers=False, medianprops=dict(color='k', lw=1))
        for patch, i in zip(bp['boxes'], keep):
            patch.set_facecolor(COLORS[keys[i]]); patch.set_alpha(0.55); patch.set_linewidth(0.6)
        axes[2].set_xticks(range(1, len(keep) + 1))
        axes[2].set_xticklabels([LABELS[keys[i]].replace('This work', 'This\nwork').replace(' et al.', '\net al.').replace(' (', '\n(')
                                 for i in keep], fontsize=6)
    axes[2].set_ylabel('Temperature (K)'); axes[2].set_title('Quartiles, 5-95% whiskers')
    _panel(axes[2], 'c')
    fig.tight_layout()
    return _save(fig, fig_dir, 'R3_distributions')


def fig_agreement(out, fig_dir, ref, others):
    methods = out['methods']
    others = [k for k in others if agreement(methods[ref], methods[k]).get('n', 0) >= 3]
    if not others:
        return None, {}
    n = len(others)
    fig, axes = plt.subplots(2, n, figsize=(max(2.4 * n, 4.6), 4.8), squeeze=False)
    stats = {}
    for i, k in enumerate(others):
        a = agreement(methods[ref], methods[k])
        stats[k] = a
        x, y = a['x'], a['y']
        ax = axes[0][i]
        lo = min(x.min(), y.min()); hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], color='k', lw=0.6, ls='--')
        ax.plot(x, y, '.', ms=2.0, color=COLORS[k], alpha=0.6)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect('equal')
        ax.set_xlabel(f'{LABELS[ref]} (K)'); ax.set_ylabel(f'{LABELS[k]} (K)')
        ax.set_title(f'n={a["n"]}, Spearman {a["spearman"]:.2f}', fontsize=7)
        _panel(ax, 'abcdefgh'[i])
        bx = axes[1][i]
        mean = 0.5 * (x + y); d = y - x
        bx.plot(mean, d, '.', ms=2.0, color=COLORS[k], alpha=0.6)
        for yv, ls in ((a['mean_diff'], '-'), (a['loa_lo'], ':'), (a['loa_hi'], ':')):
            bx.axhline(yv, color='k', lw=0.7, ls=ls)
        bx.axhline(0, color='#999999', lw=0.5)
        bx.set_xlabel('mean of the two (K)'); bx.set_ylabel(f'{LABELS[k].split(" (")[0]} minus {LABELS[ref]} (K)', fontsize=6.5)
        bx.set_title(f'mean {a["mean_diff"]:+.0f} K, 95% LoA [{a["loa_lo"]:+.0f}, {a["loa_hi"]:+.0f}]', fontsize=7)
        _panel(bx, 'abcdefgh'[n + i])
    fig.tight_layout()
    return _save(fig, fig_dir, 'R5_agreement'), stats


def fig_diagnostics(out, fig_dir, key='ProfileML'):
    raw = out['raw'][key]
    m = out['methods'][key]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4))
    rc = raw['reduced_chi2'].values.astype(float) if 'reduced_chi2' in raw else np.array([])
    rc = rc[np.isfinite(rc) & (rc > 0)]
    if len(rc):
        b = np.logspace(np.log10(max(rc.min(), 1e-2)), np.log10(rc.max() * 1.01), 40)
        axes[0].hist(rc, bins=b, color='#999999')
        axes[0].axvline(5.0, color='#D55E00', lw=1, ls='--', label='adequacy limit (5)')
        axes[0].axvline(1.0, color='k', lw=0.6, ls=':', label='calibration target (median 1)')
        axes[0].set_xscale('log'); axes[0].legend(loc='upper right')
    axes[0].set_xlabel('reduced chi-square'); axes[0].set_ylabel('pixels'); axes[0].set_title('Fit adequacy')
    _panel(axes[0], 'a')
    q = raw['fit_quality'].astype(str).value_counts() if 'fit_quality' in raw else pd.Series(dtype=int)
    cats = [c for c in ('ok', 'poor', 'failed') if c in q.index]
    axes[1].bar(cats, [q[c] for c in cats], color=['#009E73', '#E69F00', '#999999'][:len(cats)])
    for i, c in enumerate(cats):
        axes[1].text(i, q[c], str(int(q[c])), ha='center', va='bottom', fontsize=7)
    if 'n_bands_censored' in raw:
        nc = int((raw['n_bands_censored'] > 0).sum())
        axes[1].set_title(f'Fit quality\n({nc} px with saturated bands)')
    else:
        axes[1].set_title('Fit quality')
    axes[1].set_ylabel('pixels'); _panel(axes[1], 'b')
    v = m['valid'].values
    f = m['f'].values.astype(float)
    wv = (m['hi'] - m['lo']).values[v]
    sc = axes[2].scatter(m['T'].values[v], f[v], s=3, c=wv, cmap='viridis', lw=0,
                         vmin=0, vmax=float(np.nanpercentile(wv, 95)) if np.isfinite(wv).any() else 1)
    axes[2].set_yscale('log'); axes[2].set_xlabel('Temperature (K)'); axes[2].set_ylabel('fire fraction f')
    axes[2].set_title('Temperature vs fire fraction')
    fig.colorbar(sc, ax=axes[2], label='95% width (K)', shrink=0.85, extend='max')
    _panel(axes[2], 'c')
    fig.tight_layout()
    return _save(fig, fig_dir, f'R6_diagnostics_{key}')


def fig_transmittance(out, fig_dir):
    taus = out['tau']
    if not taus:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    for tag, t in taus.items():
        w = t['wavelength_nm'].values
        inw = _bool(t, 'in_fit_window') if 'in_fit_window' in t else np.isfinite(t['tau'].values)
        tau = np.where(inw, t['tau'].values, np.nan)
        ax.plot(w, tau, color=COLORS[tag], lw=1.1, label=f'{LABELS[tag]}: scene estimate')
        for h in ('tau_half_A', 'tau_half_B'):
            if h in t:
                ax.plot(w, np.where(inw, t[h].values, np.nan), color=COLORS[tag], lw=0.6, ls=':',
                        label=f'{LABELS[tag]}: split-half draws' if h.endswith('A') else None)
    ax.set_xlabel('Wavelength (nm)'); ax.set_ylabel('fire-term transmittance')
    ax.set_ylim(0, 1.05); ax.legend(loc='lower left', ncol=2)
    ax.set_title('Scene fire-term transmittance (gaps: bands outside the fit window)')
    fig.tight_layout()
    return _save(fig, fig_dir, 'R7_transmittance')


def fig_v1_v2(out, fig_dir):
    a, b = out['methods'].get('ProfileML'), out['methods'].get('ProfileMLv2')
    if a is None or b is None:
        return None, None
    st = agreement(a, b)
    if st.get('n', 0) < 3:
        return None, st
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    x, y = st['x'], st['y']
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    axes[0].plot([lo, hi], [lo, hi], 'k--', lw=0.6)
    axes[0].plot(x, y, '.', ms=2, color=COLORS['ProfileMLv2'], alpha=0.6)
    axes[0].set_xlabel('This work v1 (K)'); axes[0].set_ylabel('This work v2 (K)')
    axes[0].set_aspect('equal'); axes[0].set_title(f'Per pixel, n={st["n"]}'); _panel(axes[0], 'a')
    d = y - x
    axes[1].hist(d, bins=40, color=COLORS['ProfileMLv2'])
    axes[1].axvline(np.median(d), color='k', lw=0.8)
    axes[1].set_xlabel('v2 - v1 (K)'); axes[1].set_ylabel('pixels')
    axes[1].set_title(f'median {np.median(d):+.0f} K, quartiles {np.percentile(d, 25):+.0f}/{np.percentile(d, 75):+.0f} K')
    _panel(axes[1], 'b')
    fig.tight_layout()
    return _save(fig, fig_dir, 'R8_v1_v2'), st


def saturated_summary(raw, valid, emissivity=1.0):
    """Scene FRP from this work: retrieved (valid pixels) and the model-free
    lower bounds of the pixels without a usable retrieval (saturation_bounds)."""
    if raw is None or 'FRP_min_MW_saturated' not in raw:
        return None
    # the joint (all-band, linear-programming) bound where the run computed
    # it - never lower than the single-band bound, equally model-free
    col = 'FRP_min_joint_MW_saturated' if 'FRP_min_joint_MW_saturated' in raw else 'FRP_min_MW_saturated'
    lb = raw[col].values.astype(float)
    th = raw['T_hot_min_K_saturated'].values.astype(float) if 'T_hot_min_K_saturated' in raw else np.full(len(raw), np.nan)
    T = raw['T_K'].values.astype(float)
    f = raw['fire_fraction'].values.astype(float)
    frp = np.where(valid, emissivity * SIGMA_SB * T ** 4 * f * PIXEL_AREA_M2 / 1e6, 0.0)
    unretr = ~valid
    has = unretr & np.isfinite(lb)
    q = raw['fit_quality'].astype(str).values if 'fit_quality' in raw else np.full(len(raw), '')
    single = raw['FRP_min_MW_saturated'].values.astype(float)
    return {'frp_retrieved': float(np.nansum(frp)), 'frp_bounds': float(np.nansum(lb[has])),
            'frp_bounds_single': float(np.nansum(single[has])), 'joint': col != 'FRP_min_MW_saturated',
            'n_unretrieved': int(unretr.sum()), 'n_bounded': int(has.sum()),
            'n_failed_bounded': int((has & (q == 'failed')).sum()), 'n_failed': int((q == 'failed').sum()),
            'T_hot_min': th[has], 'lb': lb[has], 'mask_bounded': has,
            'n_censored_any': int(np.isfinite(lb).sum())}


def fig_saturated(out, win, fig_dir, key):
    raw, m = out['raw'][key], out['methods'][key]
    s = saturated_summary(raw, m['valid'].values)
    if s is None or s['n_bounded'] == 0:
        return None, s
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), gridspec_kw=dict(width_ratios=[1.25, 1, 0.8]))
    r0, r1, c0, c1 = win
    ax = axes[0]
    if out['context'] is not None and 'grey_window' in out['context']:
        g = out['context']['grey_window'].astype(float) / 255.0
        wr0, wr1, wc0, wc1 = [int(t) for t in out['context']['window']]
        ax.imshow(g, cmap='gray', vmin=0, vmax=1, extent=(wc0 - 0.5, wc1 - 0.5, wr1 - 0.5, wr0 - 0.5))
    else:
        ax.set_facecolor('#e9e9e9')
    v = m['valid'].values
    ax.scatter(m['ind_y'].values[v], m['ind_x'].values[v], s=2, marker='s', c='#9a9a9a', lw=0, label='retrieved')
    b = s['mask_bounded']
    sc = ax.scatter(m['ind_y'].values[b], m['ind_x'].values[b], s=5, marker='s', c=raw['T_hot_min_K_saturated'].values[b],
                    cmap='magma', lw=0, label='no retrieval: bound')
    ax.set_xlim(c0 - 0.5, c1 - 0.5); ax.set_ylim(r1 - 0.5, r0 - 0.5); ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(sc, ax=ax, shrink=0.8, label='hottest component >= (K)')
    ax.legend(loc='lower right', fontsize=6, frameon=True, framealpha=0.85, markerscale=2)
    ax.set_title('Pixels without a retrieval:\nlower bound on the hottest component', fontsize=7.2)
    _panel(ax, 'a')
    axes[1].hist(s['lb'], bins=30, color='#7a2a00')
    axes[1].set_xlabel('FRP lower bound per pixel (MW)'); axes[1].set_ylabel('pixels')
    axes[1].set_title(f'{s["n_bounded"]} bounded pixels', fontsize=7.2)
    _panel(axes[1], 'b')
    axes[2].bar([0, 1], [s['frp_retrieved'], s['frp_bounds']], color=[COLORS.get(key, '#D55E00'), '#7a2a00'])
    axes[2].bar([2], [s['frp_retrieved'] + s['frp_bounds']], color='#555555')
    axes[2].set_xticks([0, 1, 2]); axes[2].set_xticklabels(['retrieved', 'bounds of\nunretrieved', 'scene\n>= total'], fontsize=6)
    axes[2].set_ylabel('FRP (MW)'); axes[2].set_title('Scene FRP (lower bound)', fontsize=7.2)
    _panel(axes[2], 'c')
    fig.tight_layout()
    return _save(fig, fig_dir, 'R9_saturated'), s


def fig_frp_partners(out, fig_dir):
    summ, pairs = out['frp']['summary'], out['frp']['pairs']
    if summ is None or not len(summ):
        return None
    summ = summ[summ['n_detections_on_enmap_fire'] > 0]
    if not len(summ):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    cols = {}
    palette = ['#0072B2', '#009E73', '#CC79A7', '#56B4E9', '#E69F00', '#000000']
    for i, (_, r) in enumerate(summ.iterrows()):
        lab = f"{r['source']} {r['platform']}"
        c = cols.setdefault(lab, palette[len(cols) % len(palette)])
        yerr = r['partner_frp_unc_MW'] if np.isfinite(r.get('partner_frp_unc_MW', np.nan)) else None
        axes[0].errorbar(r['dt_minutes'], r['partner_frp_MW'], yerr=yerr, fmt='o', color=c, ms=4,
                         label=lab if lab not in [h.get_label() for h in axes[0].lines] else None)
        axes[0].plot([r['dt_minutes']] * 2, [r['enmap_frp_retrieved_MW'], r['enmap_frp_with_bounds_MW']],
                     color='#D55E00', lw=1.4)
        axes[0].plot(r['dt_minutes'], r['enmap_frp_with_bounds_MW'], marker='_', color='#D55E00', ms=8)
    axes[0].axvline(0, color='k', lw=0.5, ls=':')
    axes[0].set_xlabel('partner time - EnMAP time (min)'); axes[0].set_ylabel('FRP on the matched footprints (MW)')
    axes[0].set_title('Partner FRP (points) and EnMAP\n(orange bar: retrieved -> + saturated bounds)', fontsize=7)
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, fontsize=5.8, loc='upper left')
    axes[0].set_ylim(bottom=0)
    _panel(axes[0], 'a')
    if pairs is not None and len(pairs):
        p = pairs.copy()
        p['dt'] = (pd.to_datetime(p['datetime_utc']) - pd.to_datetime(out['run_info'].get('acquisition_utc') or
                                                                       p['datetime_utc'].min())).dt.total_seconds() / 60
        near = p[p['dt'].abs() <= 120]
        if len(near):
            x = near['partner_frp_MW'].values; y = near['enmap_frp_lower_bound_MW'].values
            ok = (x > 0) & (y > 0)
            lo, hi = max(min(x[ok].min(), y[ok].min()) * 0.5, 0.1), max(x[ok].max(), y[ok].max()) * 2
            axes[1].plot([lo, hi], [lo, hi], 'k--', lw=0.6)
            for (lab, g) in near[ok].groupby(['source', 'platform']):
                axes[1].plot(g['partner_frp_MW'], g['enmap_frp_lower_bound_MW'], 'o', ms=3,
                             color=cols.get(f'{lab[0]} {lab[1]}', '#555555'), label=f'{lab[0]} {lab[1]}')
            axes[1].set_xscale('log'); axes[1].set_yscale('log'); axes[1].set_xlim(lo, hi); axes[1].set_ylim(lo, hi)
            axes[1].set_xlabel('partner FRP per detection (MW)'); axes[1].set_ylabel('EnMAP FRP lower bound (MW)')
            axes[1].set_title('Per detection, partners within 2 h', fontsize=7)
            axes[1].legend(fontsize=5.8)
        else:
            axes[1].text(0.5, 0.5, 'no partner within 2 h', ha='center', va='center', transform=axes[1].transAxes)
            axes[1].set_axis_off()
    _panel(axes[1], 'b')
    fig.tight_layout()
    return _save(fig, fig_dir, 'R10_frp_partners')


def _coreg_extrapolation_note(out, ref, cinfo):
    """Sentence on how many fire pixels lie outside the area the tie points
    cover: AROSICS rejects windows in the burn scar (it changes between the
    two dates), so the shift at the fire is often extrapolated."""
    m = cinfo.get('model') or {}
    raw = out['raw'].get(ref)
    b = m.get('tiepoint_bounds')
    if raw is None or b is None or not cinfo.get('crs_wkt') or 'lon_wgs84_nominal' not in raw:
        return ''
    try:
        from pyproj import CRS, Transformer
        tr = Transformer.from_crs('EPSG:4326', CRS.from_wkt(cinfo['crs_wkt']), always_xy=True)
        X, Y = tr.transform(raw['lon_wgs84_nominal'].values, raw['lat_wgs84_nominal'].values)
        X, Y = np.asarray(X), np.asarray(Y)
        inside = (X >= b[0]) & (X <= b[2]) & (Y >= b[1]) & (Y <= b[3])
        dist = np.hypot(np.clip(np.maximum(b[0] - X, X - b[2]), 0, None),
                        np.clip(np.maximum(b[1] - Y, Y - b[3]), 0, None))
    except Exception:
        return ''
    n_out = int((~inside).sum())
    if not n_out:
        return 'All fire pixels lie within the area spanned by the tie points. '
    return (f'<b>{n_out} of {len(inside)} fire pixels lie outside the rectangle spanned by the tie points</b> '
            f'(up to {dist.max() / 1000:.1f} km, median {np.median(dist[~inside]) / 1000:.1f} km beyond it): AROSICS '
            'finds no reliable match in the burn scar, which changed between the two acquisitions, so the shift '
            'there is extrapolated from the affine model. The independent check below tests the correction over '
            'the whole scene, not locally at the fire. ')


def fig_coregistration(out, fig_dir, ref):
    """(a) AROSICS tie-point shifts (exaggerated) and the fire pixels at the
    L1B and the co-registered positions; (b) independent check: offset each
    LST product still asks of EnMAP, before and after."""
    info = (out.get('coreg') or {}).get('info') or {}
    tp = (out.get('coreg') or {}).get('tiepoints')
    checks = info.get('thermal_check') or []
    if (tp is None or not len(tp)) and not checks:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1), gridspec_kw={'width_ratios': [1.25, 1]})
    ax = axes[0]
    raw = out['raw'].get(ref)
    if raw is not None and 'lat_wgs84_nominal' in raw:
        ax.plot(raw['lon_wgs84_nominal'], raw['lat_wgs84_nominal'], '.', ms=2.2, color='#999999',
                label='fire pixels, L1B positions')
        ax.plot(raw['lon_wgs84'], raw['lat_wgs84'], '.', ms=2.2, color='#D55E00',
                label='fire pixels, co-registered')
    if tp is not None and len(tp) and info.get('crs_wkt'):
        try:
            from pyproj import CRS, Transformer
            tr = Transformer.from_crs(CRS.from_wkt(info['crs_wkt']), 'EPSG:4326', always_xy=True)
            X, Y = tp['X_MAP'].values.astype(float), tp['Y_MAP'].values.astype(float)
            dx, dy = tp['X_SHIFT_M'].values.astype(float), tp['Y_SHIFT_M'].values.astype(float)
            k = 5.0
            lo0, la0 = tr.transform(X, Y)
            lo1, la1 = tr.transform(X + k * dx, Y + k * dy)
            rel = tp['RELIABILITY'].values.astype(float) if 'RELIABILITY' in tp else np.full(len(tp), np.nan)
            q = ax.quiver(lo0, la0, np.asarray(lo1) - lo0, np.asarray(la1) - la0, rel, cmap='viridis',
                          angles='xy', scale_units='xy', scale=1, width=0.006, clim=(40, 90))
            cb = fig.colorbar(q, ax=ax, shrink=0.8, pad=0.02)
            cb.set_label('tie-point reliability (%)', fontsize=6); cb.ax.tick_params(labelsize=5.5)
            ax.plot([], [], color='#444444', lw=1, label=f'AROSICS tie-point shift (x{k:.0f})')
        except Exception as exc:
            warnings.warn(f'scene_report: tie points not drawn ({exc})')
    lat_m = float(np.nanmean(raw['lat_wgs84'])) if raw is not None and 'lat_wgs84' in raw else 40.0
    ax.set_aspect(1.0 / max(np.cos(np.radians(lat_m)), 1e-3))
    ax.set_xlabel('longitude (deg)', fontsize=6.5); ax.set_ylabel('latitude (deg)', fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.legend(fontsize=5.5, loc='best', framealpha=0.85)
    ax.set_title('EnMAP to HLS (AROSICS)', fontsize=7)
    _panel(ax, 'a')
    ax = axes[1]
    palette = ['#0072B2', '#009E73', '#CC79A7', '#E69F00']
    lim = 300.0
    for i, c in enumerate(checks):
        col = palette[i % len(palette)]
        for stage, mk in (('nominal', 'o'), ('coregistered', 's')):
            s = c.get(stage)
            if not s:
                continue
            ax.errorbar(s['peak_dx_m'], s['peak_dy_m'], xerr=s['peak_sd_m'][0], yerr=s['peak_sd_m'][1],
                        fmt=mk, mfc='none' if stage == 'nominal' else col, color=col, ms=4, lw=0.7,
                        label=f"{c['label']}, {'L1B' if stage == 'nominal' else 'co-registered'}")
            lim = max(lim, abs(s['peak_dx_m']) + 300, abs(s['peak_dy_m']) + 300)
        if c.get('nominal') and c.get('coregistered'):
            ax.annotate('', xy=(c['coregistered']['peak_dx_m'], c['coregistered']['peak_dy_m']),
                        xytext=(c['nominal']['peak_dx_m'], c['nominal']['peak_dy_m']),
                        arrowprops=dict(arrowstyle='->', color=col, lw=0.7))
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
    ax.set_xlabel('offset east still needed (m)', fontsize=6.5)
    ax.set_ylabel('offset north still needed (m)', fontsize=6.5)
    ax.tick_params(labelsize=5.5)
    ax.set_title('Independent check: EnMAP -NDVI vs LST', fontsize=7)
    if checks:
        ax.legend(fontsize=5.2, loc='best', framealpha=0.85)
    _panel(ax, 'b')
    fig.tight_layout()
    return _save(fig, fig_dir, 'R8_coregistration')


# ----------------------------------------------------------------- PDF assembly
def _fmt(v, nd=0, unit=''):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return '-'
    return f'{v:.{nd}f}{unit}'


def _sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(block), b''):
            h.update(chunk)
    return h.hexdigest()


def _register_fonts():
    """Embed DejaVu Sans (shipped with matplotlib) so Greek letters and
    math symbols render; the 14 standard PDF fonts lack them. Falls back to
    Helvetica."""
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.lib.fonts import addMapping
        d = os.path.join(matplotlib.get_data_path(), 'fonts', 'ttf')
        if 'DejaVuSans' not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont('DejaVuSans', os.path.join(d, 'DejaVuSans.ttf')))
            pdfmetrics.registerFont(TTFont('DejaVuSans-Bold', os.path.join(d, 'DejaVuSans-Bold.ttf')))
            pdfmetrics.registerFont(TTFont('DejaVuSans-Oblique', os.path.join(d, 'DejaVuSans-Oblique.ttf')))
            pdfmetrics.registerFont(TTFont('DejaVuSans-BoldOblique', os.path.join(d, 'DejaVuSans-BoldOblique.ttf')))
            addMapping('DejaVuSans', 0, 0, 'DejaVuSans')
            addMapping('DejaVuSans', 1, 0, 'DejaVuSans-Bold')
            addMapping('DejaVuSans', 0, 1, 'DejaVuSans-Oblique')
            addMapping('DejaVuSans', 1, 1, 'DejaVuSans-BoldOblique')
        return 'DejaVuSans', 'DejaVuSans-Bold'
    except Exception as exc:
        warnings.warn(f'scene_report: DejaVu Sans not available ({exc}); Greek letters may not render')
        return 'Helvetica', 'Helvetica-Bold'


# captions for the pipeline's own figures (Appendix A), by file-name prefix
APPENDIX_CAPTIONS = (
    ('Fig1_spectra', 'At-sensor and peak-normalised radiance of the hottest fire pixel against the scene '
                     'background classes; shaded: opaque water/CO2 bands removed before fitting.'),
    ('Fig2_temperature_maps', 'Amici et al. (2022) retrieval over the scene: temperature, temperature >= 600 K, '
                              'fire-fraction coefficient and the trial-to-trial precision.'),
    ('Fig3_hotspot_index', 'Detection: the scene-wide likelihood-ratio statistic and the SWIR hotspot index, '
                           'with the detected pixels.'),
    ('Fig4_multi_sensor', 'Thermal partner products over the scene (different footprint and time; context only).'),
    ('Fig_MethodAgreement', 'Method comparison through the common driver: per-pixel agreement of each published '
                            'method with Amici et al. (2022).'),
    ('Fig_MethodMaps', 'Method comparison through the common driver: temperature maps of each method.'),
    ('Fig_MethodSpectralFit', 'Method comparison through the common driver: fitted spectra of example pixels.'),
    ('Fig_ProfileMLv2', 'This work (v2): retrieval summary figure of the chain (maps, intervals, transmittance).'),
    ('Fig_ProfileML', 'This work (v1): retrieval summary figure of the chain (maps, intervals, transmittance).'),
)


def _appendix_image(path, fig_dir, max_px=1800):
    """A copy of a pipeline figure reduced to `max_px` wide for embedding
    (the originals are 300 dpi and would make the PDF tens of MB)."""
    try:
        from PIL import Image as _PIL
        os.makedirs(fig_dir, exist_ok=True)
        out = os.path.join(fig_dir, 'appendix_' + os.path.basename(path))
        with _PIL.open(path) as im:
            if im.size[0] <= max_px:
                return path
            h = int(round(im.size[1] * max_px / im.size[0]))
            im.convert('RGB').resize((max_px, h), _PIL.LANCZOS).save(out, optimize=True)
        return out
    except Exception:
        return path


def build_scene_report(results_dir, scene_id=None, output_path=None, verbose=True):
    """Typeset the PDF report of the scene whose outputs are in `results_dir`.
    Returns the path of the PDF."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_JUSTIFY
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                        Spacer, Table, TableStyle)
    except ImportError as exc:
        raise ImportError('scene_report needs reportlab (it is in the emit environment)') from exc

    _style()
    F, FB = _register_fonts()
    out = collect_scene_outputs(results_dir, scene_id)
    sid = out['scene_id']
    methods = out['methods']
    if not methods:
        raise FileNotFoundError(f'No Results_/MethodCompare_ files for a scene in {results_dir}')
    info = out['run_info']
    emissivity = float(info.get('settings', {}).get('emissivity', 1.0) or 1.0)
    fig_dir = os.path.join(results_dir, 'report_figures')
    output_path = output_path or os.path.join(results_dir, f'Report_{sid}.pdf')
    # this work's primary variant (the run record names it; v2 since 13 Oct 2026) is the
    # reference of every comparison and is listed first
    ref = info.get('primary') if info.get('primary') in methods else next(
        (k for k in ('ProfileMLv2', 'ProfileML') if k in methods), None)
    primary = [k for k in PRIMARY_ORDER if k in methods]
    if ref is None:
        ref = primary[0]
    primary = [ref] + [k for k in primary if k != ref]

    # ---- figures
    figs = {}
    figs['overview'], win = fig_overview(out, fig_dir)
    _m0 = next(iter(methods.values()))
    _x, _y = _m0['ind_x'].values, _m0['ind_y'].values
    n_outside = int(np.sum((_x < win[0]) | (_x >= win[1]) | (_y < win[2]) | (_y >= win[3])))
    outside_txt = (f' {n_outside} scattered detection(s) lie outside this window (Figure 1a) and are '
                   'included in every statistic.' if n_outside else '')
    figs['maps'], (vmin, vmax) = fig_maps(out, win, fig_dir, primary)
    figs['dist'] = fig_distributions(out, fig_dir, primary)
    figs['agree'], agree = fig_agreement(out, fig_dir, ref, [k for k in primary if k != ref])
    for k in ('ProfileML', 'ProfileMLv2'):
        if k in methods:
            figs[f'unc_{k}'] = fig_uncertainty_map(out, win, fig_dir, k)
            figs[f'diag_{k}'] = fig_diagnostics(out, fig_dir, k)
    figs['tau'] = fig_transmittance(out, fig_dir)
    figs['sat'], sat = (fig_saturated(out, win, fig_dir, ref) if ref in ('ProfileML', 'ProfileMLv2')
                        else (None, None))
    figs['frp'] = fig_frp_partners(out, fig_dir)
    figs['coreg'] = fig_coregistration(out, fig_dir, ref)
    figs['v12'], st12 = fig_v1_v2(out, fig_dir)
    stats = {k: method_statistics(m, emissivity) for k, m in methods.items()}

    # ---- styles
    ss = getSampleStyleSheet()
    body = ParagraphStyle('body', parent=ss['BodyText'], fontName=F, fontSize=8.8, leading=11.6,
                          alignment=TA_JUSTIFY, spaceAfter=4)
    small = ParagraphStyle('small', parent=body, fontSize=7.4, leading=9.4)
    cap = ParagraphStyle('cap', parent=body, fontSize=7.6, leading=9.6, spaceBefore=2, spaceAfter=10)
    h1 = ParagraphStyle('h1', parent=ss['Heading1'], fontName=FB, fontSize=13.5, spaceAfter=6,
                        textColor=colors.HexColor('#1a1a1a'))
    h2 = ParagraphStyle('h2', parent=ss['Heading2'], fontName=FB, fontSize=10.8, spaceBefore=8,
                        spaceAfter=4, textColor=colors.HexColor('#7a2a00'))
    title = ParagraphStyle('title', parent=h1, fontSize=17, leading=21, spaceAfter=4)
    cellS = ParagraphStyle('cell', parent=body, fontSize=7.0, leading=8.4, alignment=0, spaceAfter=0)
    W = A4[0] - 36 * mm

    def P(t, st=body):
        return Paragraph(t, st)

    def img(path, width=W):
        if not path or not os.path.exists(path):
            return Spacer(1, 1)
        from PIL import Image as _PIL
        with _PIL.open(path) as im_:
            w_, h_ = im_.size
        h = width * h_ / w_
        maxh = 210 * mm
        if h > maxh:
            width, h = width * maxh / h, maxh
        return Image(path, width=width, height=h)

    fig_no = [0]

    def figure(path, caption, width=W):
        if not path:
            return []
        fig_no[0] += 1
        return [KeepTogether([img(path, width), P(f'<b>Figure {fig_no[0]}.</b> {caption}', cap)])]

    tab_no = [0]

    def table(rows, caption, col_widths=None, header_rows=1, font=7.0):
        tab_no[0] += 1
        data = [[Paragraph(str(c), ParagraphStyle('c', parent=cellS, fontSize=font,
                                                  fontName=FB if r < header_rows else F))
                 for c in row] for r, row in enumerate(rows)]
        t = Table(data, colWidths=col_widths, repeatRows=header_rows)
        t.setStyle(TableStyle([
            ('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.black),
            ('LINEBELOW', (0, header_rows - 1), (-1, header_rows - 1), 0.5, colors.black),
            ('LINEBELOW', (0, -1), (-1, -1), 0.8, colors.black),
            ('ROWBACKGROUNDS', (0, header_rows), (-1, -1), [colors.white, colors.HexColor('#f4f4f4')]),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 1.6), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
            ('LEFTPADDING', (0, 0), (-1, -1), 2.5), ('RIGHTPADDING', (0, 0), (-1, -1), 2.5)]))
        return [KeepTogether([P(f'<b>Table {tab_no[0]}.</b> {caption}', cap), t, Spacer(1, 8)])]

    story = []
    acq = info.get('acquisition_utc') or 'not recorded'
    centre = ''
    a_raw = out['raw'].get('Amici2022')
    if a_raw is not None and 'lat_wgs84' in a_raw:
        centre = f'{np.nanmedian(a_raw["lat_wgs84"]):.4f}&deg; N, {np.nanmedian(a_raw["lon_wgs84"]):.4f}&deg; E'
    story += [P('EnMAP wildfire temperature report', title),
              P(f'<b>Scene</b> {_esc(sid)} &nbsp; <b>Acquisition</b> {_esc(str(acq))} UTC'
                + (f' &nbsp; <b>Fire-area centre</b> {centre}' if centre else ''), small),
              P(f'<b>Report generated</b> {_dt.datetime.now().strftime("%Y-%m-%d %H:%M")} from '
                f'{_esc(os.path.abspath(results_dir))}', small), Spacer(1, 6)]

    # ---- summary
    story.append(P('1 Summary', h2))
    n_det = len(next(iter(methods.values())))
    det = info.get('detection', {})
    det_txt = (f'{n_det} pixels were detected as fire'
               + (f' ({det.get("n_fire")} fire class, {det.get("n_candidate")} candidates of the '
                  f'scene-wide likelihood-ratio detector at a false-alarm rate of {det.get("p_fa")} per pixel)'
                  if det.get('n_fire') is not None else '') + '.')
    summ = [det_txt]
    if ref in ('ProfileML', 'ProfileMLv2') and stats[ref].get('n_retrieved'):
        s = stats[ref]
        summ.append(
            f'{LABELS[ref]}, the primary method of this work, retrieved an adequate fit for {s["n_retrieved"]} of them: median fire temperature '
            f'<b>{s["median"]:.0f} K</b> (interquartile range {s["p25"]:.0f}-{s["p75"]:.0f} K, 5-95% '
            f'{s["p5"]:.0f}-{s["p95"]:.0f} K), with a median 95% interval width of '
            f'{_fmt(s.get("width_median"))} K over the {s.get("n_bounded", 0)} pixels whose interval is closed.')
        if s.get('n_unconstrained'):
            summ.append(f'{s["n_unconstrained"]} further adequately fitted pixels have a 95% interval spanning '
                        'the whole 300-2000 K search range - the data do not constrain their temperature - and are '
                        'counted as unconstrained rather than as retrievals.')
        raw = out['raw'][ref]
        nf = int((raw['fit_quality'] == 'failed').sum()) if 'fit_quality' in raw else 0
        npoor = int((raw['fit_quality'] == 'poor').sum()) if 'fit_quality' in raw else 0
        if nf or npoor:
            summ.append(f'{npoor} pixels were fitted but the single-temperature model does not describe them '
                        f'(reduced chi-square above 5) and {nf} returned no temperature at all; neither group '
                        'enters any statistic of this report. Pixels without a retrieval are typically the '
                        'brightest ones, whose usable bands are saturated, so every scene statistic here '
                        'describes the retrievable part of the fire.')
    for k in primary:
        if k == ref or not stats[k].get('n_retrieved'):
            continue
        s = stats[k]
        summ.append(f'{LABELS[k]}: median {s["median"]:.0f} K over {s["n_retrieved"]} pixels'
                    + (f'; median difference from this work on the common pixels '
                       f'{agree[k]["median_diff"]:+.0f} K' if k in agree else '') + '.')
    if st12 and st12.get('n', 0) >= 3:
        summ.append(f'The two variants of this work differ by a median {st12["median_diff"]:+.0f} K per pixel (v2 minus v1) '
                    f'on the {st12["n"]} pixels both retrieved (Section 7). The difference arises in the scene '
                    'transmittance: in v1 it is also estimated from poorly fitted pixels, which on real scenes are '
                    'the brightest ones and dominate it; v2 uses adequately fitted pixels only. On the reference '
                    'tile this made the scene level independent of the numerical details of the chain (2 K instead '
                    'of 46 K), and on synthetic fires the two variants are equally accurate (VERIFICATION.md '
                    'Section 23). The difference is not contained in the per-pixel intervals.')
    story.append(P(' '.join(summ)))

    # ---- statistics table
    story.append(P('2 Temperature statistics by method', h2))
    hdr = ['Method', 'n / detected', 'median', 'mean', 'IQR', '5-95%', 'min-max', 'f-weighted mean',
           'median 95% width', 'FRP (MW)']
    rows = [hdr]
    for k in list(primary) + [k for k in ('SmokeSTCR', 'Amici2022_driver', 'ProfileML_driver') if k in stats]:
        s = stats[k]
        if not s.get('n_retrieved'):
            rows.append([LABELS[k], f'0 / {s["n_detected"]}'] + ['-'] * 8)
            continue
        rows.append([LABELS[k], f'{s["n_retrieved"]} / {s["n_detected"]}', _fmt(s['median']), _fmt(s['mean']),
                     f'{s["p25"]:.0f}-{s["p75"]:.0f}', f'{s["p5"]:.0f}-{s["p95"]:.0f}',
                     f'{s["min"]:.0f}-{s["max"]:.0f}', _fmt(s.get('T_fweighted')),
                     _fmt(s.get('width_median')), _fmt(s.get('FRP_MW'), 1)])
    story += table(rows, 'Retrieved fire temperature (K) per method, over the pixels each method retrieved '
                   '(statistics policy in Section 11). "f-weighted mean" weights each pixel by its retrieved fire '
                   'fraction. FRP is derived from each method\'s own (T, f) as &epsilon;&sigma;T<super>4</super>fA '
                   f'(&epsilon; = {emissivity:g}, A = 900 m<super>2</super>): a model quantity, and a lower bound '
                   'for the scene because unretrieved pixels contribute nothing. Interval widths: this work - 95% '
                   'profile-likelihood interval (closed intervals only); Amici et al. - trial spread, not a '
                   'confidence interval.',
                   col_widths=[32 * mm, 16 * mm, 14 * mm, 12 * mm, 17 * mm, 17 * mm, 18 * mm, 17 * mm, 16 * mm, 14 * mm])

    # temperature classes
    hdr = (['Method'] + [f'{a}-{b - 1} K' if b < 2001 else f'&ge; {a} K' for a, b in zip(T_CLASSES[:-1], T_CLASSES[1:])]
           + ['> 900 K (flaming-dominated)'])
    rows = [hdr]
    for k in primary:
        m = methods[k]
        T = m['T'].values[m['valid'].values]
        if not len(T):
            continue
        cnt = np.histogram(T, bins=T_CLASSES)[0]
        rows.append([LABELS[k]] + [f'{c} ({100 * c / len(T):.0f}%)' for c in cnt]
                    + [f'{int((T > 900).sum())} ({100 * (T > 900).mean():.0f}%)'])
    story += table(rows, 'Number (share) of retrieved pixels in each temperature class. The last column uses the '
                   '900 K split between smouldering- and flaming-dominated pixels of Keely et al. (2026, arXiv:2608.07580), '
                   'the midpoint between reported smouldering ceilings (~850 K) and flaming onset (~950 K); it is an '
                   'interpretive convention, and a pixel-effective temperature below it does not exclude flames.',
                   col_widths=[36 * mm] + [22 * mm] * (len(T_CLASSES) - 1) + [26 * mm])

    # ---- maps
    story.append(PageBreak())
    story.append(P('3 Maps', h2))
    story += figure(figs['overview'], '(a) Detected fire pixels over the scene (orange: fire class; blue: '
                    'candidates the detector could not separate from saturated or specular targets; the box is '
                    'the window of panel b and of Figure 2). (b) The fire area in a local contrast stretch. '
                    'Axes are image rows and columns of 30 m pixels.')
    story += figure(figs['maps'], f'Retrieved fire temperature of each method on the same pixel grid and colour '
                    f'scale ({vmin:.0f}-{vmax:.0f} K, from the lowest 2nd percentile of any method to the median over methods of their 98th percentiles, so a method whose values lie beyond the scale saturates - arrow on the colour bar - instead of compressing the others). Each square is one 30 m '
                    'pixel drawn to scale; small blue squares are detected pixels for which that method returned no '
                    'usable temperature (this work: poor or failed fit; others: not converged or at the search bound).'
                    + outside_txt)

    # ---- distributions and agreement
    story.append(PageBreak())
    story.append(P('4 Distributions and agreement between methods', h2))
    story += figure(figs['dist'], '(a) Histograms, (b) empirical cumulative distributions and (c) quartiles with '
                    '5-95% whiskers of the retrieved temperatures of each method, each over its own retrieved pixels.')
    if figs['agree']:
        story += figure(figs['agree'], f'Agreement with {LABELS[ref]} on the pixels both methods retrieved. Top: '
                        'per-pixel scatter with the 1:1 line. Bottom: Bland-Altman plots - difference against mean, '
                        'with the mean difference (solid) and the 95% limits of agreement (dotted).', )
        rows = [['Method', 'n common', 'median diff', 'IQR of diff', 'median |diff|', 'RMSD', '95% LoA',
                 'Spearman &rho;']]
        for k, a in agree.items():
            rows.append([LABELS[k], a['n'], f'{a["median_diff"]:+.0f}', f'{a["q25"]:+.0f} / {a["q75"]:+.0f}',
                         f'{a["mad_abs"]:.0f}', f'{a["rmsd"]:.0f}', f'{a["loa_lo"]:+.0f} / {a["loa_hi"]:+.0f}',
                         f'{a["spearman"]:.2f}'])
        story += table(rows, f'Pairwise agreement with {LABELS[ref]} (differences in K, method minus reference). '
                       'A difference between methods is not an error of either: there is no independent temperature '
                       'truth for a real scene. The synthetic-truth evaluation of the methods is in '
                       'VERIFICATION.md Sections 18 and 22.',
                       col_widths=[40 * mm, 15 * mm, 17 * mm, 21 * mm, 19 * mm, 13 * mm, 25 * mm, 16 * mm])

    # ---- this work: uncertainty, diagnostics, transmittance
    story.append(PageBreak())
    story.append(P('5 This work: uncertainty and fit diagnostics', h2))
    for k in ('ProfileML', 'ProfileMLv2'):
        if f'unc_{k}' in figs:
            story += figure(figs[f'unc_{k}'], f'{LABELS[k]}. (a) Width of the 95% profile-likelihood interval per '
                            'pixel; crosses mark intervals that reach the 300 or 2000 K search bound (open-ended: the '
                            'data constrain only one side). (b) Point estimates sorted by temperature with their '
                            'intervals.')
            story += figure(figs[f'diag_{k}'], f'{LABELS[k]}. (a) Reduced chi-square of every converged fit; the '
                            'error model is calibrated so that its scene median is 1, and fits above 5 are reported '
                            'as poor and excluded. (b) Fit-quality classes. (c) Temperature against retrieved fire '
                            'fraction, coloured by interval width: hot, small fires and cool, large ones are the '
                            'weakly separable directions of the problem.')
    story.append(P('6 Scene fire-term transmittance', h2))
    if figs['tau']:
        story += figure(figs['tau'], 'Fire-term transmittance estimated from the scene\'s own fire pixels (solid) '
                        'and from two random halves of them (dotted); the halves bound the uncertainty of the '
                        'estimate and are propagated into the per-pixel intervals.')
    k_info = info.get('profile_ml', {})
    if k_info:
        rows = [['Variant', 'transmittance from', 'model error &epsilon;<sub>m</sub>', 'kernel length (nm)',
                 'nugget', 'scene estimates', 'run time (s)']]
        for tag, d in k_info.items():
            rows.append([LABELS.get(tag, tag), _fmt(d.get('n_tau_pixels'), 0) + ' px' if d.get('n_tau_pixels') else '-',
                         _fmt(d.get('eps'), 3), _fmt(d.get('kernel_length_nm')), _fmt(d.get('kernel_nugget'), 2),
                         _esc(str(d.get('scene_estimates', 'all'))), _fmt(d.get('seconds'))])
        story += table(rows, 'Scene-level quantities of this work\'s chain, estimated from the fire pixels '
                       'themselves.', col_widths=[30 * mm, 24 * mm, 22 * mm, 22 * mm, 14 * mm, 22 * mm, 20 * mm])

    if figs['v12']:
        story.append(P('7 Sensitivity: this work v1 against v2', h2))
        story += figure(figs['v12'], 'Per-pixel temperatures of the two variants of this work (v2: fixed-point '
                        'reweighting, and scene-level estimates from adequately fitted pixels only). Their difference '
                        'comes from the scene transmittance, which is weakly separable from the overall temperature '
                        'level and to which poorly fitted bright pixels contribute in v1 only. On synthetic fires '
                        'with the real error structure the two are equally accurate; on the reference tile v2\'s '
                        'scene level moved by -2/+9 K when its adequacy cap was changed from 5 to 3/10 '
                        '(VERIFICATION.md Sections 22-23).')

    # ---- references
    ref_cols = []
    for k in (ref, 'Amici2022'):
        r = out['raw'].get(k)
        if r is not None:
            ref_cols += [c for c in r.columns if c.endswith('_LST_K') or c == 'field_temperature_K']
    ref_cols = sorted(set(ref_cols))
    story.append(P('8 Reference data', h2))

    # ---- co-registration (coregistration.py)
    cinfo = (out.get('coreg') or {}).get('info')
    coreg_applied = bool(cinfo) and cinfo.get('status') == 'applied'
    if cinfo:
        story.append(P('<b>Geometric co-registration.</b>', body))
        m = cinfo.get('model') or {}
        refd = cinfo.get('reference') or {}
        txt = ('EnMAP L1B is not orthorectified: its pixel positions come from the product\'s affine, which ignores '
               'terrain relief and the off-nadir view, and every comparison with another sensor inherits that error. '
               'Before comparing, the scene was matched with AROSICS (Scheffler et al. 2017, Remote Sens. 9, 676; '
               'local tie-point grid, its three outlier filters) against the Harmonized Landsat Sentinel-2 product '
               '(HLS v2.0, 30 m, geolocated to the Sentinel-2 global reference image), NIR band, nearest '
               'cloud-free acquisition. ')
        if refd:
            txt += (f"Reference: {_esc(str(refd.get('product', '')))} of {_esc(str(refd.get('datetime', ''))[:16])} UTC "
                    f"({refd.get('dt_days', float('nan')):+.1f} days from EnMAP, cloud {refd.get('cloud_pct', float('nan')):.0f}%). ")
        if coreg_applied:
            rm = m.get('loo_rmse_m') or [np.nan, np.nan]
            rc = m.get('loo_rmse_constant_m') or [np.nan, np.nan]
            txt += (f"{m.get('n_tiepoints')} of {cinfo.get('n_grid_points')} grid points gave valid tie points. "
                    + (f"The shift varies across the scene, so it is modelled as an affine function of position (the "
                       f"first-order polynomial AROSICS itself uses to correct an image); held-out tie points are "
                       f"predicted to {rm[0]:.0f} m east / {rm[1]:.0f} m north RMSE, against {rc[0]:.0f} / {rc[1]:.0f} m "
                       f"for a constant shift. " if m.get('kind') == 'affine' else
                       f"Too few or too clustered tie points for an affine model: the median shift is applied "
                       f"(held-out RMSE {rm[0]:.0f} m east / {rm[1]:.0f} m north). ")
                    + _coreg_extrapolation_note(out, ref, cinfo)
                    + 'From here on, every position compared with another sensor - the LST sampled at the fire '
                      'pixels, the FRP detections matched to the fire (Section 10), the multi-sensor maps - is the '
                      'co-registered one; the L1B positions are kept beside them (lat/lon_wgs84_nominal). The '
                      'retrieval does not use positions and is unchanged.')
        else:
            txt += f"Co-registration was <b>{_esc(str(cinfo.get('status')))}</b>; the comparisons use the L1B positions."
        story.append(P(txt, body))
        checks = cinfo.get('thermal_check') or []
        if checks:
            story.append(P(
                'Independent check, with data AROSICS never saw: vegetation is cooler than bare and burnt ground, so '
                'the EnMAP -NDVI image, averaged onto each LST product\'s grid, correlates with that product\'s LST; '
                'shifting the EnMAP positions over a +/-3 km grid of trial offsets, the correlation peaks at the '
                'offset EnMAP still needs (spread: 4x4-cell block bootstrap). After a correct co-registration '
                'that offset is near zero, up to the LST product\'s own geolocation error and its 1 km pixel.', body))
            rows_c = [['LST product', 'offset still needed, L1B (m E / m N)', 'after co-registration',
                       'r at zero offset, L1B -> co-reg.']]
            for c in checks:
                n_, k_ = c.get('nominal') or {}, c.get('coregistered') or {}
                f_ = lambda s: (f"{s['peak_dx_m']:+.0f} / {s['peak_dy_m']:+.0f} "
                                f"(&plusmn;{s['peak_sd_m'][0]:.0f} / {s['peak_sd_m'][1]:.0f})") if s else '-'
                rows_c.append([_esc(c['label']), f_(n_), f_(k_),
                               f"{n_.get('r_zero', np.nan):.3f}" + (f" -> {k_['r_zero']:.3f}" if k_ else '')])
            story += table(rows_c, 'Offset of the EnMAP positions at which EnMAP -NDVI best matches each LST product '
                           '(peak of the correlation over trial offsets; &plusmn; one bootstrap standard deviation).',
                           col_widths=[38 * mm, 46 * mm, 46 * mm, 40 * mm])
        if figs.get('coreg'):
            story += figure(figs['coreg'], '(a) AROSICS tie-point shifts (arrows, exaggerated five-fold, coloured '
                            'by AROSICS reliability) and the fire pixels at their L1B (grey) and co-registered '
                            '(orange) positions. (b) Offset each LST product still asks of the EnMAP positions, '
                            'at the L1B positions (open) and after co-registration (filled), with one bootstrap '
                            'standard deviation.')

    if ref_cols:
        r = out['raw'].get(ref) if out['raw'].get(ref) is not None else out['raw']['Amici2022']
        has_nom = any(c + '_nominal' in r for c in ref_cols)
        head = ['Reference', 'pixels with a value', 'median (K)', 'max (K)', 'Spearman &rho; with this work']
        if has_nom:
            head = ['Reference', 'pixels with a value', 'median (K)', 'Spearman &rho; with this work',
                    'median at L1B positions (K)', 'Spearman &rho; at L1B positions']
        rows = [head]
        from scipy.stats import spearmanr

        def _rho(v):
            ok = np.isfinite(v)
            if ref in ('ProfileML', 'ProfileMLv2') and ok.sum() >= 5:
                mv = methods[ref]['valid'].values & ok
                if mv.sum() >= 5:
                    return f'{spearmanr(methods[ref]["T"].values[mv], v[mv]).correlation:.2f} (n={int(mv.sum())})'
            return '-'
        for c in ref_cols:
            if c not in r:
                continue
            v = r[c].values.astype(float)
            ok = np.isfinite(v)
            name = c.replace('_LST_K', ' LST').replace('_', ' ')
            med = _fmt(np.nanmedian(v) if ok.any() else np.nan)
            if has_nom:
                vn = r[c + '_nominal'].values.astype(float) if c + '_nominal' in r else np.full(len(r), np.nan)
                rows.append([name, int(ok.sum()), med, _rho(v),
                             _fmt(np.nanmedian(vn) if np.isfinite(vn).any() else np.nan), _rho(vn)])
            else:
                rows.append([name, int(ok.sum()), med, _fmt(np.nanmax(v) if ok.any() else np.nan), _rho(v)])
        story += table(rows, 'Reference temperatures sampled at the fire pixels'
                       + (' (co-registered positions; the last two columns at the L1B positions)' if has_nom else '')
                       + '. Thermal-infrared land-surface '
                       'temperature products average a 70-1000 m footprint and are retrieved at a different time; '
                       'they are a different physical quantity from a sub-pixel flame temperature and cannot validate '
                       'it directly - only a rank correlation is shown. Field measurements, when given, are matched '
                       'to the nearest pixel.',
                       col_widths=([34 * mm, 20 * mm, 18 * mm, 34 * mm, 26 * mm, 34 * mm] if has_nom else
                                   [40 * mm, 26 * mm, 22 * mm, 22 * mm, 40 * mm]))
    else:
        story.append(P('No reference data (field measurements or thermal partners) were available for this run.'))

    # ---- saturated pixels
    story.append(PageBreak())
    story.append(P('9 Saturated pixels: what they still establish', h2))
    if sat is not None and sat.get('n_bounded'):
        story.append(P(
            f'{sat["n_unretrieved"]} detected pixels have no usable temperature from {LABELS[ref]}; '
            f'{sat["n_bounded"]} of them (including {sat["n_failed_bounded"]} of the {sat["n_failed"]} pixels with no '
            'retrieval at all) have saturated bands. A saturated band states only that the radiance is at least '
            'the detector ceiling; with emissivity and transmittance at most one and an upper bound on the '
            'background radiance (the maximum of the pixel\'s background library and the 99.9th percentile of the '
            'scene background), that bounds two quantities from below <i>for any distribution of temperatures '
            'inside the pixel</i>: the fire radiative power, FRP &ge; A (L<sub>sat</sub> - L<sub>bkg</sub>) '
            'min<sub>T</sub> &sigma;T<super>4</super>/B(&lambda;,T), and the temperature of the hottest fire '
            'component, T<sub>hot</sub> &ge; B<super>-1</super>(&lambda;, L<sub>sat</sub> - L<sub>bkg</sub>) '
            '(saturation_bounds.py). Checked on 17 061 synthetic saturated pixels with one to three '
            'temperatures, emissivity 0.5-1 and an absorbing atmosphere: no bound was ever violated; the FRP bound '
            'is a median 14% of the true FRP and the temperature bound a median 408 K below the hottest '
            'component - they are conservative by construction (VERIFICATION.md Section 24).'))
        if sat.get('joint'):
            story.append(P(
                'The FRP bound used here is the <i>joint</i> bound: all bands must hold at once for the same '
                'temperature distribution, so the smallest FRP of any non-negative distribution of emitting areas '
                'that reaches every saturated ceiling and every linear, unflagged band (less three noise standard '
                'deviations) is a linear programme; its grid is corrected so the result remains a strict lower '
                'bound (VERIFICATION.md Section 29.4). It is never below the single-band bound '
                f'({sat["frp_bounds_single"]:.0f} MW for the same pixels).'))
        story.append(P(
            f'Here the bounds add at least <b>{sat["frp_bounds"]:.0f} MW</b> to the {sat["frp_retrieved"]:.0f} MW '
            f'retrieved, so the scene FRP is at least <b>{sat["frp_retrieved"] + sat["frp_bounds"]:.0f} MW</b>; '
            f'the hottest component of the bounded pixels is at least {np.median(sat["T_hot_min"]):.0f} K '
            f'(median; 10-90%: {np.percentile(sat["T_hot_min"], 10):.0f}-{np.percentile(sat["T_hot_min"], 90):.0f} K).'))
        story += figure(figs['sat'], '(a) Pixels without a usable retrieval, coloured by the lower bound on the '
                        'temperature of their hottest fire component (grey: retrieved pixels). (b) Lower bounds on '
                        'their FRP. (c) Scene FRP: retrieved, plus the bounds of the unretrieved pixels - itself a '
                        'lower bound, since every bound is.')
    else:
        story.append(P('No pixel without a usable retrieval has saturated bands in this scene, or the saturation '
                       'bounds were not computed for this run.'))

    # ---- uncertainty budget
    ut = out.get('uncertainty')
    if ut is not None and len(ut):
        story.append(P('Uncertainty budget (this work)', h2))
        story.append(P(
            'Per pixel, the temperature carries its 95% profile-likelihood interval, widened to cover the two '
            'half-sample transmittance estimates and the background-library draws, and the fire radiative power '
            'the matching interval along the temperature profile (uncertainty.py). Dozier (1981) and Dennison et '
            'al. (2006) publish no uncertainty; their columns T_mc_p2_5 / T_mc_p97_5 are a Monte Carlo '
            'propagation of the instrument noise (and, for Dozier, of the background) added by this work - their '
            'temperatures are unchanged. Scene totals below: the per-pixel errors share the transmittance and the '
            'calibration, so the truth lies between the independent and the fully correlated combination; the '
            'absolute radiometric calibration scales every fraction and FRP, not the temperature.'))
        rows_u = [['quantity', 'value', '95% low', '95% high', 'unit']]
        for _, r_ in ut.iterrows():
            rows_u.append([str(r_['quantity']), _fmt(r_['value'], 1), _fmt(r_['low'], 1),
                           ('open' if np.isinf(float(r_['high'])) else _fmt(r_['high'], 1)), str(r_['unit'])])
        story += table(rows_u, 'Scene-level uncertainty budget (Uncertainty_*.csv; notes on each row there).',
                       col_widths=[W * 0.52, W * 0.12, W * 0.12, W * 0.12, W * 0.12])

    # ---- other satellites: FRP
    story.append(P('10 Fire radiative power from other satellites', h2))
    fsum = out['frp']['summary']
    if fsum is not None and len(fsum):
        story.append(P(
            'Active-fire products of other sensors report fire radiative power from the mid-infrared (4 &mu;m), '
            'the quantity this work derives as &epsilon;&sigma;T<super>4</super>fA. A partner detection is matched '
            'to the EnMAP fire if its footprint (its pixel area, widened by 1 km for the geolocation error of the two '
            'products) contains EnMAP fire pixels. EnMAP FRP is shown as an interval from the retrieved value to '
            'the retrieved value plus the saturated-pixel lower bounds (Section 9); both ends are lower bounds. '
            'Fire radiative power changes within minutes, so only partners within about 15 min of EnMAP '
            '(geostationary sensors) compare the same moment; polar-orbiter passes hours apart compare only the '
            'order of magnitude. Coarse sensors miss small and cool fires, and the SWIR misses the cool tail.'
            + (' The EnMAP fire pixels are at their co-registered positions (Section 8).' if coreg_applied else '')))
        rows = [['Product', 'time (UTC)', '&Delta;t (min)', 'detections on EnMAP fire / in area',
                 'partner FRP (MW)', 'EnMAP FRP, retrieved -> with bounds (MW)']]
        for _, r in fsum.iterrows():
            unc = r.get('partner_frp_unc_MW', np.nan)
            rows.append([f"{r['source']} {r['platform']}", str(r['datetime_utc'])[:16], f"{r['dt_minutes']:+.0f}",
                         f"{int(r['n_detections_on_enmap_fire'])} / {int(r['n_detections_in_bbox'])}",
                         f"{r['partner_frp_MW']:.0f}" + (f' &plusmn; {unc:.0f}' if np.isfinite(unc) else ''),
                         f"{r['enmap_frp_retrieved_MW']:.0f} -> {r['enmap_frp_with_bounds_MW']:.0f}"])
        story += table(rows, 'Fire radiative power of the partner products on the footprints they share with the '
                       'EnMAP fire, against this work over the same footprints, ordered by time difference.',
                       col_widths=[40 * mm, 26 * mm, 15 * mm, 30 * mm, 26 * mm, 37 * mm])
        if figs['frp']:
            story += figure(figs['frp'], '(a) Partner FRP on the matched footprints against the time difference to '
                            'EnMAP (points, with the product\'s uncertainty) and EnMAP over the same footprints (orange bars: '
                            'from the retrieved FRP to the retrieved FRP plus the saturated-pixel bounds). (b) Per '
                            'partner detection within 2 h, EnMAP lower bound against the partner FRP, with the 1:1 line.')
    else:
        log = out['run_info'].get('fire_partner_log') or []
        story.append(P('No active-fire product of another satellite was used for this run'
                       + (': ' + '; '.join(_esc(f'{a}: {b}') for a, b in log) if log else
                          ' (Retrieval page of the GUI, or --fire-partners / --fire-partner-file).')))

    # ---- methods and caveats
    story.append(PageBreak())
    story.append(P('11 Methods, statistics policy and interpretation', h2))
    for k in primary + [k for k in ('SmokeSTCR',) if k in methods]:
        story.append(P(f'<b>{LABELS[k]}.</b> {CITATIONS[k]}', body))
    story.append(P('<b>Which pixels count.</b> A pixel enters a method\'s statistics only if that method returned a '
                   'usable temperature: this work - a converged fit with reduced chi-square &le; 5; Amici et al. - a '
                   'finite trial mean not at the search bound; Dozier and Dennison et al. - converged and not at the '
                   'search bound; a pixel of this work whose interval spans the whole 300-2000 K search range is '
                   'counted as unconstrained, not as a retrieval. Pairwise statistics use the pixels both methods '
                   'retrieved. The methods therefore '
                   'describe slightly different pixel sets; Table 1 gives the counts.'))
    story.append(P('<b>Intervals.</b> This work reports a 95% profile-likelihood interval per pixel with the '
                   'transmittance uncertainty (split-half draws) folded in; measured coverage on synthetic fires with '
                   'the real error structure is 0.94-0.95 (VERIFICATION.md Section 22). An interval that reaches the '
                   'search bound is open-ended and is reported as such, not as a width. The Amici et al. spread over '
                   'random background draws is not a confidence interval.'))
    story.append(P('<b>Limitations.</b> No real scene has an independent per-pixel temperature truth; differences '
                   'between methods are not errors of either. Pixels whose usable bands are saturated or flagged '
                   'by the L1B saturation-crosstalk mask are not retrieved by this work (crosstalk has no known '
                   'sign, so those bands cannot be used even as bounds); scene totals such as FRP are lower bounds. '
                   'Temperatures are fire-term effective temperatures of a single-temperature model; a pixel holding '
                   'flaming and smouldering fire is fitted as poor rather than forced to one value.'))

    # ---- appendix figures
    if out['figures']:
        story.append(PageBreak())
        story.append(P('Appendix A. Figures written by the pipeline', h2))
        for p in out['figures']:
            cap_ = next((c for pre, c in APPENDIX_CAPTIONS if os.path.basename(p).startswith(pre)), '')
            story += figure(_appendix_image(p, fig_dir), f'{_esc(cap_)} <i>({_esc(os.path.basename(p))})</i>')

    # ---- reproducibility
    story.append(PageBreak())
    story.append(P('Appendix B. Reproducibility record', h2))
    settings = info.get('settings', {})
    if settings:
        rows = [['Setting', 'Value']] + [[_esc(str(k)), _esc(str(v))] for k, v in sorted(settings.items())]
        story += table(rows, 'Run settings recorded by the pipeline.', col_widths=[60 * mm, 110 * mm])
    sw = info.get('software', {})
    sw = dict(sw, report_python=platform.python_version(), report_platform=platform.platform())
    try:
        import reportlab
        sw['reportlab'] = reportlab.Version
    except Exception:
        pass
    sw['matplotlib'] = matplotlib.__version__; sw['numpy'] = np.__version__; sw['pandas'] = pd.__version__
    rows = [['Component', 'Version']] + [[_esc(k), _esc(str(v))] for k, v in sorted(sw.items())]
    story += table(rows, 'Software.', col_widths=[60 * mm, 110 * mm])
    files = sorted(p for p in glob.glob(os.path.join(results_dir, '*')) if os.path.isfile(p)
                   and not os.path.basename(p).startswith('Report_'))
    rows = [['File', 'Size (kB)', 'SHA-256']]
    for p in files:
        rows.append([_esc(os.path.basename(p)), f'{os.path.getsize(p) / 1024:.0f}', _sha256(p)[:32] + '...'])
    story += table(rows, 'Output files of this scene with their checksums (first 128 bits of SHA-256), so a '
                   'later rerun can be checked for bit-identity.', col_widths=[92 * mm, 16 * mm, 62 * mm], font=6.2)

    def _page(canvas, doc):
        canvas.saveState()
        canvas.setFont(F, 7)
        canvas.setFillColor(colors.HexColor('#666666'))
        canvas.drawString(18 * mm, 10 * mm, f'EnMAP wildfire temperature report - {sid}')
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f'page {doc.page}')
        canvas.restoreState()

    doc = SimpleDocTemplate(output_path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title=f'EnMAP wildfire temperature report {sid}', author='EnMAP fire pipeline',
                            subject='Fire temperature retrieval: comparison of methods')
    # Image streams binary (Flate) rather than ASCII85-wrapped: without the
    # optional C accelerator reportlab encodes ASCII85 in pure Python, ~8 s
    # per report; the PDF content is the same and the file smaller.
    # invariant mode: fixed creation date and document ID, so the same run
    # inputs give a byte-identical report (the run date printed in the text
    # still changes from day to day)
    from reportlab import rl_config
    _a85, _inv = rl_config.useA85, rl_config.invariant
    rl_config.useA85 = 0
    rl_config.invariant = 1
    try:
        doc.build(story, onFirstPage=_page, onLaterPages=_page)
    finally:
        rl_config.useA85 = _a85
        rl_config.invariant = _inv
    # machine-readable copy of the tables
    rows = []
    for k, s in stats.items():
        rows.append(dict(method=k, label=LABELS.get(k, k), **{a: b for a, b in s.items()}))
    pd.DataFrame(rows).to_csv(os.path.join(results_dir, f'ReportSummary_{sid}.csv'), index=False)
    if verbose:
        print(f'Saved {output_path}')
    return output_path


# ----------------------------------------------------------------- pipeline hooks
def write_report_context(scene, fire_points, results_dir, scene_id, max_side=900, pad=12):
    """Save the small images the report draws its maps on (a scene composite,
    downsampled to at most `max_side` pixels, and the fire window at full
    resolution), so the report can be rebuilt from the folder alone."""
    import Fire_analysis_with_EnMAP as fa
    rgb = fa._build_rgb_composite(scene)
    n_rows, n_cols = rgb.shape[:2]
    step = max(1, int(np.ceil(max(n_rows, n_cols) / max_side)))
    fp = np.asarray(fire_points)
    r0, r1 = max(int(fp[:, 0].min()) - 3 * pad, 0), min(int(fp[:, 0].max()) + 3 * pad + 1, n_rows)
    c0, c1 = max(int(fp[:, 1].min()) - 3 * pad, 0), min(int(fp[:, 1].max()) + 3 * pad + 1, n_cols)
    try:
        win_rgb = fa._fire_enhanced_rgb(scene, r0, r1, c0, c1)
    except Exception:
        win_rgb = rgb[r0:r1, c0:c1]
    grey = fa._muted_background(rgb[r0:r1, c0:c1]) @ np.array([0.2126, 0.7152, 0.0722])
    p = os.path.join(results_dir, f'ReportContext_{scene_id}.npz')
    np.savez_compressed(p, rgb=(rgb[::step, ::step] * 255).astype(np.uint8), step=np.array(step),
                        shape=np.array([n_rows, n_cols]), window=np.array([r0, r1, c0, c1]),
                        rgb_window=(np.clip(win_rgb, 0, 1) * 255).astype(np.uint8),
                        grey_window=(np.clip(grey, 0, 1) * 255).astype(np.uint8))
    return p


def write_run_info(results_dir, scene_id, info):
    p = os.path.join(results_dir, f'RunInfo_{scene_id}.json')

    def _clean(o):
        if isinstance(o, dict):
            return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o) if np.isfinite(o) else None
        if isinstance(o, float) and not np.isfinite(o):
            return None
        if isinstance(o, (str, int, float, bool)) or o is None:
            return o
        return str(o)
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(_clean(info), fh, indent=1)
    return p


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser(description='Build the PDF report of one processed EnMAP scene.')
    ap.add_argument('results_dir', help="the scene's results folder (results_temperature/<scene id>)")
    ap.add_argument('--scene-id', default=None)
    ap.add_argument('--output', default=None)
    a = ap.parse_args()
    build_scene_report(a.results_dir, scene_id=a.scene_id, output_path=a.output)

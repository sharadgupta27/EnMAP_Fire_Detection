# coding: utf-8
"""Stage 3 of the coincident-data search: score the stage-2 tiles, choose
one entry per datatake, and draw the overlap figure of each chosen
datatake (map of footprints + detections, and the +/-3 h timeline).

  conda.exe run -n emit python candidate_sites/ds_stage3.py [--top N]

Outputs: results_audit_check/data_search/ranked_tiles.csv, ranked_datatakes.csv
         paper/figures/ds_<fire_key>_<datatake>.pdf/.png
"""
import argparse
import glob
import json
import os
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from shapely.geometry import shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'paper'))
import basemap                                     # noqa: E402

OUT = os.path.join(ROOT, 'results_audit_check', 'data_search')
CACHE = os.path.join(OUT, 'stage2_cache')
FIG = os.path.join(ROOT, 'paper', 'figures')
os.makedirs(FIG, exist_ok=True)
warnings.filterwarnings('ignore')

HELD = {('DT0000149658', 14), ('DT0000149658', 15), ('DT0000149654', 4),
        ('DT0000001755', 3), ('DT0000001755', 4)}
C = {'enmap': '#E69F00', 'viirs': '#D55E00', 'modis': '#CC79A7', 'slstr': '#009E73',
     'eco': '#56B4E9', 'emit': '#F0E442', 'landsat': '#0072B2', 's2': '#999999', 'geo': '#555555'}
PLAT_MK = {'Suomi-NPP': 'o', 'NOAA-20': 's', 'NOAA-21': 'D', 'Terra': '^', 'Aqua': 'v'}


def load_tables():
    d = pd.read_csv(os.path.join(OUT, 'stage2_tiles.csv'))
    geo = {}
    for tag in ('stage1', 'stage1_sweep'):
        p = os.path.join(OUT, tag + '_tiles.geojson')
        if os.path.exists(p):
            for f in json.load(open(p))['features']:
                geo.setdefault(f['properties']['scene_id'], shape(f['geometry']))
    return d, geo


def score(d):
    d = d.copy()
    d['has_viirs'] = d['n_viirs'] > 0
    d['has_modis'] = d['n_modis'] > 0
    d['has_slstr'] = d['slstr_best_dt_min'].notna()
    d['has_geo'] = (d['geo_slots'].fillna(0) > 0) & ~d['geo'].str.contains('not in pipeline')
    d['has_eco'] = d['ecostress_n'] > 0
    d['has_emit'] = d['emit_n'] > 0
    d['has_landsat'] = d['landsat_n'] > 0
    d['has_s2'] = d['s2_n'] > 0
    d['core'] = d[['has_viirs', 'has_modis', 'has_slstr', 'has_geo']].sum(axis=1)
    d['hires'] = d[['has_eco', 'has_emit', 'has_landsat', 'has_s2']].sum(axis=1)
    d['thermal_hr'] = d['has_eco'] | d['has_landsat']
    d['held'] = [(a, int(b)) in HELD for a, b in zip(d['datatake'], d['tile_no'])]
    lsh = (pd.to_datetime(d['acq_datetime_utc']).dt.hour + pd.to_datetime(d['acq_datetime_utc']).dt.minute / 60
           + d['clon'] / 15.0) % 24
    d['local_solar_h'] = lsh.round(2)
    d['night'] = (lsh < 6) | (lsh > 18)
    d['tier'] = np.where((d['core'] == 4) & d['thermal_hr'], 'A',
                         np.where((d['core'] >= 3) & (d['hires'] >= 1), 'B',
                                  np.where(d['core'] >= 3, 'C', 'D')))
    d = d.sort_values(['tier', 'hires', 'n_viirs', 'cloud_pct'], ascending=[True, False, False, True])
    return d


def cached(name):
    p = os.path.join(CACHE, re.sub(r'[^A-Za-z0-9_.-]', '_', name) + '.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as fh:
            return pickle.load(fh)
    return []


def firms(fkey, dtk, date):
    p = os.path.join(OUT, 'firms', f'{fkey}_{dtk}_{date}.pkl')
    if os.path.exists(p):
        with open(p, 'rb') as fh:
            return pickle.load(fh)
    return None


def draw(grp, geo, all_tiles, out_stem):
    """grp: stage-2 rows of one (fire_key, datatake)."""
    fkey, dtk = grp['fire_key'].iloc[0], grp['datatake'].iloc[0]
    t0 = pd.Timestamp(grp['acq_datetime_utc'].iloc[0]).to_pydatetime()
    date = grp['acq_datetime_utc'].iloc[0][:10]
    sel = [geo[s] for s in grp['scene_id']]
    dt_tiles = all_tiles[(all_tiles['datatake'] == dtk) & (all_tiles['acq_datetime_utc'].str[:10] == date)]
    other = [geo[s] for s in dt_tiles['scene_id'] if s in geo and s not in set(grp['scene_id'])]
    u = sel[0]
    for g in sel[1:]:
        u = u.union(g)
    w, s, e, n = u.bounds
    pad = max(e - w, n - s) * 0.35 + 0.05
    W_, S_, E_, N_ = w - pad, s - pad, e + pad, n + pad
    lat0 = 0.5 * (S_ + N_)

    fig = plt.figure(figsize=(11.5, 5.4))
    ax = fig.add_axes([0.04, 0.08, 0.50, 0.86])
    tl = fig.add_axes([0.62, 0.12, 0.36, 0.80])
    try:
        bm = basemap.fetch_basemap((W_, S_, E_, N_), basemap.layer_for_acquisition(t0))
        bw, be, bs, bn = bm['extent']
        ax.imshow(bm['img'], extent=(bw, be, bs, bn), origin='upper', zorder=0)
        attribution = bm['attribution']
    except Exception as exc:
        ax.set_facecolor('#eeeeee'); attribution = 'basemap unavailable (%s)' % type(exc).__name__

    def outline(g, **kw):
        geoms = getattr(g, 'geoms', [g])
        for gg in geoms:
            if gg.geom_type == 'Polygon':
                x, y = gg.exterior.xy
                ax.plot(x, y, **kw)

    view = shape({'type': 'Polygon', 'coordinates': [[(W_, S_), (E_, S_), (E_, N_), (W_, N_), (W_, S_)]]})
    present = {}
    for key, col, lw, ls, lab in (('s2', C['s2'], 0.8, ':', 'Sentinel-2 granule'),
                                  ('landsat', C['landsat'], 1.3, '-', 'Landsat 8/9 scene'),
                                  ('eco', C['eco'], 1.3, '--', 'ECOSTRESS tile'),
                                  ('emit', C['emit'], 1.6, '-', 'EMIT scene')):
        gs = [g[2] for g in cached(f'{fkey}_{dtk}_{key}') if g[2] is not None and g[2].intersects(u)]
        for gg in gs:
            outline(gg, color=col, lw=lw, ls=ls, zorder=3)
        if gs:
            covers = any(gg.contains(view) for gg in gs)
            present[key] = Line2D([], [], color=col, lw=lw, ls=ls,
                                  label=lab + (' (covers whole map)' if covers else ''))
    for g in other:
        outline(g, color=C['enmap'], lw=0.8, ls='--', zorder=4)
    nsel = len(sel)
    for i_, (g, tn) in enumerate(zip(sel, grp['tile_no'])):
        outline(g, color=C['enmap'], lw=2.2, zorder=5)
        if nsel > 4 and i_ not in (0, nsel - 1):
            continue
        xs_, ys_ = g.exterior.xy
        k_ = int(np.argmax(ys_))
        ax.text(xs_[k_], ys_[k_], f'{int(tn):03d}', color='white', fontsize=8, ha='center',
                va='bottom', weight='bold', zorder=6,
                bbox=dict(boxstyle='round,pad=0.15', fc=C['enmap'], ec='none', alpha=0.85))
    det = firms(fkey, dtk, date)
    if det is not None and len(det):
        dd = det.copy()
        dd['dt'] = [(x - t0).total_seconds() / 60 for x in pd.to_datetime(dd['datetime_utc'])]
        dd = dd[(dd['dt'].abs() <= 180) & dd['lon'].between(W_, E_) & dd['lat'].between(S_, N_)]
        for plat, gg in dd.groupby('platform'):
            ax.scatter(gg['lon'], gg['lat'], s=10, marker=PLAT_MK.get(plat, 'o'),
                       c=C['modis'] if plat in ('Terra', 'Aqua') else C['viirs'],
                       edgecolors='k', linewidths=0.25, zorder=7)
    ax.set_xlim(W_, E_); ax.set_ylim(S_, N_)
    ax.set_aspect(1 / np.cos(np.radians(lat0)))
    ax.tick_params(labelsize=7)
    ax.set_xlabel('Longitude (°)', fontsize=8); ax.set_ylabel('Latitude (°)', fontsize=8)
    import textwrap
    ax.text(0.01, 0.01, chr(10).join(textwrap.wrap(attribution, 70)), transform=ax.transAxes, fontsize=4.5, color='white',
            va='bottom', bbox=dict(fc='black', alpha=0.4, ec='none', pad=1))
    h = [Line2D([], [], color=C['enmap'], lw=2.2, label='EnMAP tile (this entry)'),
         Line2D([], [], color=C['enmap'], lw=0.8, ls='--', label='other tiles, same datatake'),
         Line2D([], [], color='none', marker='o', mfc=C['viirs'], mec='k', label='VIIRS fire ±3 h'),
         Line2D([], [], color='none', marker='^', mfc=C['modis'], mec='k', label='MODIS fire ±3 h'),
         ] + list(present.values())
    ax.legend(handles=h, fontsize=6, loc='upper left', framealpha=0.85)
    ax.set_title('a  Footprints and fire detections', fontsize=9, loc='left')

    # ---- timeline (relative to the first chosen tile)
    r0 = grp.iloc[0]
    rows = []
    for p in str(r0.get('viirs_passes') or '').split(';'):
        if p and p != 'nan':
            plat, m, k = p.split(':'); rows.append(('VIIRS', float(m.replace('min', '')), plat, C['viirs']))
    for p in str(r0.get('modis_passes') or '').split(';'):
        if p and p != 'nan':
            plat, m, k = p.split(':'); rows.append(('MODIS', float(m.replace('min', '')), plat, C['modis']))
    for p in str(r0.get('slstr') or '').split(';'):
        if p and p != 'nan':
            plat, m = p.split(':'); rows.append(('SLSTR FRP', float(m.replace('min', '')), plat, C['slstr']))
    for nm, key, col in (('ECOSTRESS', 'ecostress', C['eco']), ('EMIT', 'emit', C['emit']),
                         ('Landsat', 'landsat', C['landsat']), ('Sentinel-2', 's2', C['s2'])):
        if r0[f'{key}_n'] > 0:
            rows.append((nm, float(r0[f'{key}_dt_min']), '', col))
    labels = ['EnMAP', 'VIIRS', 'MODIS', 'SLSTR FRP', 'Geostationary', 'ECOSTRESS', 'EMIT', 'Landsat', 'Sentinel-2']
    ypos = {l: len(labels) - 1 - i for i, l in enumerate(labels)}
    tl.axvspan(-180, 180, color='#f3f3f3', zorder=0)
    tl.axvline(0, color=C['enmap'], lw=1.5)
    tl.scatter([0], [ypos['EnMAP']], s=80, marker='*', color=C['enmap'], zorder=5)
    seen_lab = {}
    for lab, m, plat, col in sorted(rows, key=lambda x: (x[0], x[1])):
        k = seen_lab.get(lab, 0); seen_lab[lab] = k + 1
        tl.scatter([m], [ypos[lab]], s=40, color=col, edgecolors='k', linewidths=0.4, zorder=5,
                   marker=PLAT_MK.get(plat, 'o'))
        if plat:
            tl.annotate(plat.replace('Sentinel-', 'S'), (m, ypos[lab]), xytext=(0, 6 if k % 2 == 0 else -11),
                        textcoords='offset points', fontsize=5.5, ha='center')
    if r0['geo_slots'] and r0['geo_slots'] == r0['geo_slots'] and r0['geo_slots'] > 0:
        step = 10 if ('MTG' in r0['geo'] or 'GOES' in r0['geo']) else 15
        xs = np.arange(-180, 181, step)
        tl.scatter(xs, np.full(len(xs), ypos['Geostationary']), s=6, color=C['geo'], marker='|')
        tl.annotate(r0['geo'].split(' (')[0], (0, ypos['Geostationary']), xytext=(0, 6),
                    textcoords='offset points', fontsize=5.5, ha='center')
    elif 'Himawari' in str(r0['geo']):
        tl.annotate('Himawari AHI: coverage, not in pipeline', (0, ypos['Geostationary']), fontsize=5.5,
                    ha='center', va='center', color='0.4')
    tl.set_yticks(list(ypos.values())); tl.set_yticklabels(list(ypos.keys()), fontsize=7)
    tl.set_xlim(-190, 190); tl.set_ylim(-0.7, len(labels) - 0.3)
    tl.set_xticks(range(-180, 181, 60)); tl.tick_params(labelsize=7)
    tl.set_xlabel('Minutes from EnMAP tile %03d (%s UTC)' % (int(r0['tile_no']), r0['acq_datetime_utc'][11:19]),
                  fontsize=8)
    tl.set_title('b  Acquisitions within ±3 h', fontsize=9, loc='left')
    tl.grid(axis='x', color='white', lw=0.8)
    for sp in ('top', 'right'):
        tl.spines[sp].set_visible(False)
    fig.savefig(out_stem + '.pdf'); fig.savefig(out_stem + '.png', dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--top', type=int, default=14)
    ap.add_argument('--keys', default='', help='comma list fire_key:datatake to force')
    a = ap.parse_args()
    d, geo = load_tables()
    d = score(d)
    d.to_csv(os.path.join(OUT, 'ranked_tiles.csv'), index=False)
    all_tiles = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(OUT, 'stage1*_tiles.csv'))])
    # one entry per datatake: the tiles of the best tier with fire
    ents = []
    for (fk, dtk), g in d.groupby(['fire_key', 'datatake'], sort=False):
        best = g['tier'].min()
        gg = g[g['tier'] == best].sort_values('tile_no')
        ents.append({'fire_key': fk, 'datatake': dtk, 'tier': best, 'n_tiles': len(gg),
                     'tiles': ','.join('%03d' % int(t) for t in gg['tile_no']),
                     'n_viirs': int(gg['n_viirs'].sum()), 'hires': int(gg['hires'].max()),
                     'held_all': bool(gg['held'].all())})
    E = pd.DataFrame(ents).sort_values(['tier', 'hires', 'n_viirs'], ascending=[True, False, False])
    E = E.drop_duplicates(['datatake', 'tiles'], keep='first')
    E.to_csv(os.path.join(OUT, 'ranked_datatakes.csv'), index=False)
    print(E.to_string())
    if a.keys:
        for spec in a.keys.split(','):
            dtk, _, tl = spec.partition(':')
            g = d[(d['datatake'] == dtk) & ~d['held'] & (d['n_viirs'] > 0)]
            if tl:
                g = g[g['tile_no'].astype(int).isin([int(x) for x in tl.split('+')])]
            if not len(g):
                print('nothing for', spec); continue
            g = g.sort_values('tile_no')
            stem = os.path.join(FIG, f'ds_{dtk}' + (f'_{tl.replace("+", "-")}' if tl else ''))
            draw(g, geo, all_tiles, stem)
            print('fig', stem)


if __name__ == '__main__':
    main()

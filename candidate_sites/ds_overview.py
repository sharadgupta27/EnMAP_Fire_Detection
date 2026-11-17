# coding: utf-8
"""Overview map of the coincident-data search (paper/figures/ds_overview)."""
import glob
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import ds_tables                                   # noqa: E402
OUT = os.path.join(ROOT, 'results_audit_check', 'data_search')
FIG = os.path.join(ROOT, 'paper', 'figures')

s1 = pd.concat([pd.read_csv(p) for p in glob.glob(os.path.join(OUT, 'stage1*_tiles.csv'))]).drop_duplicates('scene_id')
d = pd.read_csv(os.path.join(OUT, 'ranked_tiles.csv'))
fig = plt.figure(figsize=(11, 5.2))
ax = fig.add_axes([0.01, 0.02, 0.98, 0.93], projection=ccrs.Robinson())
ax.set_global()
ax.add_feature(cfeature.LAND, facecolor='#ececec', edgecolor='none')
ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor='#888888')
pc = ccrs.PlateCarree()
ax.scatter(s1['clon'], s1['clat'], s=2, color='#bbbbbb', transform=pc, label='EnMAP tile tested (%d)' % len(s1),
           rasterized=True)
f = s1[s1['n_viirs'] > 0]
ax.scatter(f['clon'], f['clat'], s=4, color='#E69F00', transform=pc,
           label='VIIRS fire in tile within ±3 h (%d)' % len(f), rasterized=True)
ab = d[d['tier'].isin(['A', 'B'])]
ax.scatter(ab['clon'], ab['clat'], s=7, color='#D55E00', transform=pc,
           label='tier A/B: other data within ±3 h (%d)' % len(ab), rasterized=True)
for e, name, dtk, tiles, _ in ds_tables.ENTRIES:
    g = ds_tables.entry_rows(d, dtk, tiles)
    if not len(g):
        continue
    x, y = g['clon'].mean(), g['clat'].mean()
    ax.scatter([x], [y], s=60, marker='*', color='#0072B2', edgecolors='k', linewidths=0.4, transform=pc, zorder=5)
    if e.endswith('b'):
        continue
    ax.annotate(e.rstrip('a'), (x, y), xycoords=pc._as_mpl_transform(ax), xytext=(4, 3),
                textcoords='offset points', fontsize=7, weight='bold', color='#0072B2')
ax.scatter([], [], s=60, marker='*', color='#0072B2', edgecolors='k', label='recommended entry (Table 1)')
ax.legend(loc='lower left', fontsize=7, frameon=True, markerscale=1.5)
fig.savefig(os.path.join(FIG, 'ds_overview.pdf'), dpi=200)
fig.savefig(os.path.join(FIG, 'ds_overview.png'), dpi=160)
print('ok')

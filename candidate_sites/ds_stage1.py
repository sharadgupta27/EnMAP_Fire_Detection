# coding: utf-8
"""Stage 1 of the coincident-data search: EnMAP tiles over the catalogued
fires, and whether VIIRS/MODIS saw fire INSIDE each tile footprint within
+/-WINDOW_H of the EnMAP acquisition (FIRMS).

  conda.exe run -n emit python candidate_sites/ds_stage1.py

Outputs (results_audit_check/data_search/):
  stage1_tiles.csv      one row per EnMAP L2A tile (cloud <= MAX_CLOUD)
  stage1_tiles.geojson  footprints
  firms/*.pkl           cached FIRMS answers, one per (fire, datatake)
"""
import datetime as dt
import json
import os
import pickle
import sys
import time
import warnings

import numpy as np
import pandas as pd
from shapely.geometry import shape, Point
from shapely import prepared

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import ds_fires                      # noqa: E402
import megafire_enmap as mf          # noqa: E402
import fire_partners as fp           # noqa: E402

OUT = os.path.join(ROOT, 'results_audit_check', 'data_search')
FIRMS_DIR = os.path.join(OUT, 'firms')
os.makedirs(FIRMS_DIR, exist_ok=True)
WINDOW_H = 3.0
MAX_CLOUD = 70.0
warnings.filterwarnings('ignore')


def naive(s):
    return pd.Timestamp(s).tz_convert(None).to_pydatetime() if pd.Timestamp(s).tzinfo else pd.Timestamp(s).to_pydatetime()


SCREEN = ('VIIRS_SNPP_SP', 'VIIRS_SNPP_NRT', 'VIIRS_NOAA20_SP', 'VIIRS_NOAA20_NRT')


def firms_cached(key, bbox, when, screen=False):
    if screen:
        sp = os.path.join(FIRMS_DIR, key + '.screen.pkl')
        if os.path.exists(sp):
            with open(sp, 'rb') as fh:
                d0 = pickle.load(fh)
        else:
            try:
                d0 = fp.fetch_firms(bbox, when, window_hours=WINDOW_H, sources=SCREEN)
            except Exception as exc:
                print('   screen failed', exc); return None
            with open(sp, 'wb') as fh:
                pickle.dump(d0, fh)
        if not len(d0):
            return d0
    path = os.path.join(FIRMS_DIR, key + '.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    for attempt in range(3):
        try:
            df = fp.fetch_firms(bbox, when, window_hours=WINDOW_H)
            break
        except Exception as exc:
            print('   FIRMS retry', exc); time.sleep(10)
    else:
        return None
    with open(path, 'wb') as fh:
        pickle.dump(df, fh)
    return df


def main():
    sweep = 'sweep' in sys.argv
    fires = ds_fires.sweep_regions() if sweep else ds_fires.all_fires()
    tag = 'stage1_sweep' if sweep else 'stage1'
    rows, feats_out = [], []
    for fkey, name, region, start, end, bbox, ref in fires:
        feats = mf.stac_search(bbox, start, end)
        feats = [f for f in feats if (mf._num(f['properties'].get('eo:cloud_cover')) or 0) <= MAX_CLOUD]
        print(f'{fkey:16s} {name[:40]:40s} {len(feats):4d} tiles', flush=True)
        if not feats:
            continue
        by_dt = {}
        for f in feats:
            sid = f['id']
            dtk = 'DT' + sid.split('DT')[1][:10]
            by_dt.setdefault(dtk + '_' + f['properties']['datetime'][:10], []).append(f)
        for dkey, fs in sorted(by_dt.items()):
            geoms = [shape(f['geometry']) for f in fs]
            u = geoms[0]
            for g in geoms[1:]:
                u = u.union(g)
            w, s, e, n = u.bounds
            times = sorted(naive(f['properties']['datetime']) for f in fs)
            tmid = times[len(times) // 2]
            det = firms_cached(f'{fkey}_{dkey}', (w - 0.02, s - 0.02, e + 0.02, n + 0.02), tmid, screen=sweep)
            for f, g in zip(fs, geoms):
                p = f['properties']
                t = naive(p['datetime'])
                r = {'fire_key': fkey, 'fire': name, 'region': region, 'ref': ref,
                     'scene_id': f['id'], 'datatake': dkey.split('_')[0],
                     'tile_no': f['id'].split('_')[-3] if f['id'].count('_') >= 3 else '',
                     'acq_datetime_utc': t.strftime('%Y-%m-%dT%H:%M:%S'),
                     'cloud_pct': mf._num(p.get('eo:cloud_cover')),
                     'sun_elev_deg': mf._num(p.get('view:sun_elevation')),
                     'west': g.bounds[0], 'south': g.bounds[1], 'east': g.bounds[2], 'north': g.bounds[3],
                     'clon': g.centroid.x, 'clat': g.centroid.y, 'firms_ok': det is not None}
                if det is not None and len(det):
                    pg = prepared.prep(g)
                    inside = np.array([pg.contains(Point(x, y)) for x, y in zip(det['lon'], det['lat'])])
                    d = det[inside].copy()
                    d['dt_min'] = [(x - t).total_seconds() / 60 for x in pd.to_datetime(d['datetime_utc'])]
                    d = d[d['dt_min'].abs() <= WINDOW_H * 60]
                    for fam in ('VIIRS', 'MODIS'):
                        dd = d[d['source'].str.contains(fam)]
                        r[f'n_{fam.lower()}'] = len(dd)
                        r[f'frp_{fam.lower()}_MW'] = float(dd['frp_MW'].sum()) if len(dd) else 0.0
                        passes = []
                        for (plat, ), grp in dd.groupby(['platform']):
                            # one overpass per platform within the window (scans > 30 min apart split)
                            tt = np.sort(grp['dt_min'].values)
                            cuts = np.flatnonzero(np.diff(tt) > 30) + 1
                            for chunk in np.split(tt, cuts):
                                passes.append((plat, float(np.median(chunk)), len(chunk)))
                        passes.sort(key=lambda x: abs(x[1]))
                        r[f'{fam.lower()}_passes'] = ';'.join(f'{p}:{m:+.0f}min:{k}' for p, m, k in passes)
                        r[f'{fam.lower()}_best_dt_min'] = passes[0][1] if passes else np.nan
                else:
                    for fam in ('viirs', 'modis'):
                        r[f'n_{fam}'] = 0; r[f'frp_{fam}_MW'] = 0.0
                        r[f'{fam}_passes'] = ''; r[f'{fam}_best_dt_min'] = np.nan
                rows.append(r)
                feats_out.append({'type': 'Feature', 'geometry': f['geometry'],
                                  'properties': {'scene_id': f['id'], 'fire_key': fkey}})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, tag + '_tiles.csv'), index=False)
    with open(os.path.join(OUT, tag + '_tiles.geojson'), 'w') as fh:
        json.dump({'type': 'FeatureCollection', 'features': feats_out}, fh)
    df = pd.DataFrame(rows)
    print('\ntiles:', len(df), ' with VIIRS fire in footprint +/-3 h:', int((df['n_viirs'] > 0).sum()))


if __name__ == '__main__':
    main()

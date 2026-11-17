# coding: utf-8
"""Do the high-resolution partners (ECOSTRESS, EMIT, Landsat, Sentinel-2)
cover the contemporaneous VIIRS fire detections themselves, not just part
of the EnMAP tile? Share of the tile's VIIRS detections (+/-3 h) inside
each partner granule footprint."""
import json, os, pickle, re, sys
import numpy as np, pandas as pd
from shapely.geometry import shape, Point
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_audit_check', 'data_search')
CACHE = os.path.join(OUT, 'stage2_cache')
d = pd.read_csv(os.path.join(OUT, 'ranked_tiles.csv'))
geo = {}
for tag in ('stage1', 'stage1_sweep'):
    for f in json.load(open(os.path.join(OUT, tag + '_tiles.geojson')))['features']:
        geo.setdefault(f['properties']['scene_id'], shape(f['geometry']))


def load(name):
    p = os.path.join(CACHE, re.sub(r'[^A-Za-z0-9_.-]', '_', name) + '.pkl')
    return pickle.load(open(p, 'rb')) if os.path.exists(p) else []


rows = []
for _, r in d[(d.n_viirs > 0) & (d.hires > 0)].iterrows():
    det = pickle.load(open(os.path.join(OUT, 'firms', f"{r.fire_key}_{r.datatake}_{r.acq_datetime_utc[:10]}.pkl"), 'rb'))
    t = pd.Timestamp(r.acq_datetime_utc); g = geo[r.scene_id]
    det = det[[g.contains(Point(x, y)) for x, y in zip(det.lon, det.lat)]]
    det = det[((pd.to_datetime(det.datetime_utc) - t).abs() <= pd.Timedelta(hours=3)) & det.source.str.contains('VIIRS')]
    pts = [Point(x, y) for x, y in zip(det.lon, det.lat)]
    out = {'scene_id': r.scene_id}
    for key, nm in (('eco', 'ecostress'), ('emit', 'emit'), ('landsat', 'landsat'), ('s2', 's2')):
        best = 0.0
        for item in load(f'{r.fire_key}_{r.datatake}_{key}'):
            gg = item[2]
            if gg is None or not pts:
                continue
            tt = item[1]
            if abs((pd.Timestamp(tt) - t).total_seconds()) > 3 * 3600:
                continue
            best = max(best, float(np.mean([gg.contains(p) for p in pts])))
        out[f'{nm}_fire_cover'] = round(best, 3)
    rows.append(out)
c = pd.DataFrame(rows)
d = d.merge(c, on='scene_id', how='left')
for nm in ('ecostress', 'emit', 'landsat', 's2'):
    d[f'{nm}_fire_cover'] = d[f'{nm}_fire_cover'].fillna(0.0)
d.to_csv(os.path.join(OUT, 'ranked_tiles.csv'), index=False)
pd.set_option('display.width', 250)
keys = sys.argv[1].split(',') if len(sys.argv) > 1 else []
x = d[d.datatake.isin(keys) & (d.n_viirs > 0)]
print(x[['datatake', 'tile_no', 'n_viirs', 'ecostress_n', 'ecostress_fire_cover', 'landsat_n', 'landsat_fire_cover', 's2_n', 's2_fire_cover']].to_string())

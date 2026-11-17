# coding: utf-8
"""Where the contemporaneous VIIRS detections sit inside each shortlisted
EnMAP tile (for the media correlation in data_search.tex)."""
import json, os, pickle, sys
import numpy as np, pandas as pd
from shapely.geometry import shape, Point
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results_audit_check', 'data_search')
d = pd.read_csv(os.path.join(OUT, 'ranked_tiles.csv'))
geo = {}
for tag in ('stage1', 'stage1_sweep'):
    for f in json.load(open(os.path.join(OUT, tag + '_tiles.geojson')))['features']:
        geo.setdefault(f['properties']['scene_id'], shape(f['geometry']))
keys = sys.argv[1].split(',')
rows = []
for _, r in d[d.datatake.isin(keys) & (d.n_viirs > 0)].iterrows():
    p = os.path.join(OUT, 'firms', f"{r.fire_key}_{r.datatake}_{r.acq_datetime_utc[:10]}.pkl")
    det = pickle.load(open(p, 'rb'))
    t = pd.Timestamp(r.acq_datetime_utc)
    g = geo[r.scene_id]
    det = det[[g.contains(Point(x, y)) for x, y in zip(det.lon, det.lat)]]
    det = det[(pd.to_datetime(det.datetime_utc) - t).abs() <= pd.Timedelta(hours=3)]
    v = det[det.source.str.contains('VIIRS')]
    rows.append(dict(datatake=r.datatake, tile=int(r.tile_no), n=len(v), lat=v.lat.mean(), lon=v.lon.mean(),
                     lat_min=v.lat.min(), lat_max=v.lat.max(), lon_min=v.lon.min(), lon_max=v.lon.max(),
                     frp_MW=v.frp_MW.sum(), frp_max=v.frp_MW.max()))
pd.set_option('display.width', 250)
print(pd.DataFrame(rows).round(3).sort_values(['datatake', 'tile']).to_string())

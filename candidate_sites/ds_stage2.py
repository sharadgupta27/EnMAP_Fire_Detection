# coding: utf-8
"""Stage 2 of the coincident-data search: for every EnMAP tile that stage 1
found with VIIRS or MODIS fire inside its footprint within +/-3 h, list the
other products acquired within +/-3 h that cover it:

  SLSTR FRP (SL_2_FRP___, CDSE catalogue)        - any granule intersecting the tile
  geostationary FRP: MTG-FCI / SEVIRI (LSA SAF),  - archive listing for the date
                     GOES ABI FDC (AWS)           - satellite by longitude/date
                     Himawari AHI                 - coverage only (not in pipeline)
  ECOSTRESS ECO_L2T_LSTE v002 (CMR)               - tile overlap fraction
  EMIT L1B radiance EMITL1BRAD v001 (CMR)          - swath polygon overlap fraction
  Landsat 8/9 C2 L2 ST (USGS LandsatLook STAC)     - scene overlap fraction
  Sentinel-2 L2A (Earth Search STAC)               - granule overlap fraction

Catalogue searches only; nothing is downloaded.
  conda.exe run -n emit python candidate_sites/ds_stage2.py
"""
import datetime as dt
import json
import os
import pickle
import re
import sys
import warnings

import numpy as np
import pandas as pd
import requests
from shapely.geometry import box, shape, Polygon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, ROOT)
import thermal_partners as tp        # noqa: E402
import fire_partners as fp           # noqa: E402
import credentials                   # noqa: E402

OUT = os.path.join(ROOT, 'results_audit_check', 'data_search')
CACHE = os.path.join(OUT, 'stage2_cache')
os.makedirs(CACHE, exist_ok=True)
W = 3.0
EARTH_SEARCH = 'https://earth-search.aws.element84.com/v1/search'
S = requests.Session()
warnings.filterwarnings('ignore')


def cached(name, fn):
    path = os.path.join(CACHE, re.sub(r'[^A-Za-z0-9_.-]', '_', name) + '.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    v = fn()
    with open(path, 'wb') as fh:
        pickle.dump(v, fh)
    return v


def frac(tile, geom):
    try:
        return float(tile.intersection(geom).area / tile.area)
    except Exception:
        return np.nan


def dtmin(t, t0):
    return (t - t0).total_seconds() / 60.0


# ------------------------------------------------------------------ searches
def slstr(bbox, t0):
    return [(r['granule'], r['platform'], r['datetime_utc'])
            for r in tp.search_slstr(bbox, t0, window_hours=W, product_types=('SL_2_FRP___',))]


def cmr_items(short, version, bbox, t0):
    return tp._cmr_bbox_search(short, version, bbox, t0 - dt.timedelta(hours=W), t0 + dt.timedelta(hours=W))


def cmr_geom(umm):
    g = umm['SpatialExtent']['HorizontalSpatialDomain']['Geometry']
    if g.get('GPolygons'):
        pts = g['GPolygons'][0]['Boundary']['Points']
        return Polygon([(p['Longitude'], p['Latitude']) for p in pts])
    b = tp._cmr_bbox_of(umm)
    return box(*b) if b else None


def cmr_list(short, versions, bbox, t0):
    out, seen = [], set()
    for v in versions:
        for it in cmr_items(short, v, bbox, t0):
            u = it['umm']
            if u['GranuleUR'] in seen:
                continue
            seen.add(u['GranuleUR'])
            out.append((u['GranuleUR'], tp._as_naive_utc(u['TemporalExtent']['RangeDateTime']['BeginningDateTime']),
                        cmr_geom(u)))
    return out


def landsat(bbox, t0):
    out = []
    body = {'collections': ['landsat-c2l2-st'], 'bbox': list(bbox),
            'datetime': '%s/%s' % (tp._iso(t0 - dt.timedelta(hours=W)), tp._iso(t0 + dt.timedelta(hours=W))),
            'limit': 100}
    r = S.post(tp.LANDSATLOOK_STAC, json=body, timeout=120)
    r.raise_for_status()
    for f in r.json().get('features', []):
        p = f['properties']
        if (p.get('platform') or '').upper() not in ('LANDSAT_8', 'LANDSAT_9'):
            continue
        out.append((f['id'], tp._as_naive_utc(p['datetime']), shape(f['geometry']), p.get('eo:cloud_cover')))
    return out


def sentinel2(bbox, t0):
    out = []
    body = {'collections': ['sentinel-2-l2a'], 'bbox': list(bbox),
            'datetime': '%s/%s' % (tp._iso(t0 - dt.timedelta(hours=W)), tp._iso(t0 + dt.timedelta(hours=W))),
            'limit': 100}
    r = S.post(EARTH_SEARCH, json=body, timeout=120)
    r.raise_for_status()
    for f in r.json().get('features', []):
        p = f['properties']
        out.append((f['id'], tp._as_naive_utc(p['datetime']), shape(f['geometry']), p.get('eo:cloud_cover')))
    return out


def lsasaf_slots(day, mtg):
    """Slot times (UTC) in the LSA SAF archive for `day`."""
    c = credentials.get('lsasaf')
    auth = (c['username'], c['password']) if c.get('username') else None
    if mtg:
        r = S.get(fp.MTG_FRP_URL.format(d=day), auth=auth, timeout=60)
        if not r.ok:
            return []
        return sorted({dt.datetime.strptime(s, '%Y%m%d%H%M') for _, s in fp.MTG_FRP_NAME.findall(r.text)})
    r = S.get(fp.LSASAF_URL.format(d=day), auth=auth, timeout=60)
    if not r.ok:
        return []
    return sorted({dt.datetime.strptime(s, '%Y%m%d%H%M')
                   for _, s in re.findall(r'(HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_[A-Za-z-]+_(\d{12}))', r.text)})


def goes_slots(sat, t0):
    """Start times of ABI-L2-FDCF scans in NOAA's AWS bucket within +/-W h."""
    lo, hi = t0 - dt.timedelta(hours=W), t0 + dt.timedelta(hours=W)
    out, h = [], lo.replace(minute=0, second=0, microsecond=0)
    while h <= hi:
        pre = f'ABI-L2-FDCF/{h:%Y}/{h.timetuple().tm_yday:03d}/{h:%H}/'
        try:
            keys = fp._s3_keys(fp.GOES_BUCKET.format(sat=sat), pre, S)
        except Exception:
            keys = []
        for k in keys:
            m = re.search(r'_s(\d{13})', k)
            if m:
                t = dt.datetime.strptime(m.group(1)[:11], '%Y%j%H%M')
                if lo <= t <= hi:
                    out.append(t)
        h += dt.timedelta(hours=1)
    return sorted(set(out))


def geostationary(lon, lat, t0):
    """(name, n_slots_within_window or None, note)."""
    lo, hi = t0 - dt.timedelta(hours=W), t0 + dt.timedelta(hours=W)
    sat = fp.goes_satellite_for(lon, t0)
    if sat and lon < -30:                       # the Americas: GOES views far better than MSG/MTG
        sl = cached(f'goes_{sat}_{t0:%Y%m%d%H%M}', lambda: goes_slots(sat, t0))
        return f'{sat.upper()} ABI FDC (AWS)', len(sl), 'FDCF scans listed in the AWS bucket'
    if -65 <= lon <= 65 and abs(lat) <= 70:
        out = []
        if t0 >= dt.datetime(2025, 1, 1):
            sl = cached(f'mtg_{t0:%Y%m%d}', lambda: lsasaf_slots(t0.date(), True))
            n = sum(lo <= s <= hi for s in sl)
            out.append(('MTG-FCI FRP (LSA SAF)', n))
        sl = cached(f'sev_{t0:%Y%m%d}', lambda: lsasaf_slots(t0.date(), False))
        n = sum(lo <= s <= hi for s in sl)
        out.append(('MSG-SEVIRI FRP (LSA SAF)', n))
        best = max(out, key=lambda x: x[1])
        return best[0], best[1], '; '.join(f'{a}: {b}' for a, b in out)
    if sat:
        sl = cached(f'goes_{sat}_{t0:%Y%m%d%H%M}', lambda: goes_slots(sat, t0))
        return f'{sat.upper()} ABI FDC (AWS)', len(sl), 'FDCF scans listed in the AWS bucket'
    if lon >= 75 or lon <= -170:
        return 'Himawari-8/9 AHI (JAXA, not in pipeline)', None, 'coverage only'
    return 'none', 0, ''


# ------------------------------------------------------------------ main
def main():
    parts, geo = [], {}
    for tag in ('stage1', 'stage1_sweep'):
        p = os.path.join(OUT, tag + '_tiles.csv')
        if not os.path.exists(p) or (tag == 'stage1_sweep' and 'named' in sys.argv):
            continue
        part = pd.read_csv(p)
        if tag == 'stage1_sweep':           # systematic sweep: only substantial, mostly clear fires
            part = part[(part['n_viirs'] >= 10) & (part['cloud_pct'] <= 30)]
        parts.append(part)
        for f in json.load(open(os.path.join(OUT, tag + '_tiles.geojson')))['features']:
            geo[f['properties']['scene_id']] = shape(f['geometry'])
    t1 = pd.concat(parts, ignore_index=True).drop_duplicates('scene_id', keep='first')
    t1 = t1[(t1['n_viirs'] > 0) | (t1['n_modis'] > 0)].copy()
    print('tiles with fire at +/-3 h:', len(t1), flush=True)
    rows = []
    for (fkey, dtk), grp in t1.groupby(['fire_key', 'datatake']):
        tiles = [geo[s] for s in grp['scene_id']]
        u = tiles[0]
        for g in tiles[1:]:
            u = u.union(g)
        bbox = tuple(np.round(u.bounds, 4))
        t0 = pd.Timestamp(sorted(grp['acq_datetime_utc'])[len(grp) // 2]).to_pydatetime()
        tag = f'{fkey}_{dtk}'
        print(f'{tag:40s} {len(grp)} tiles', flush=True)
        try:
            sl = cached(tag + '_slstr', lambda: slstr(bbox, t0))
        except Exception as e:
            sl = []; print('  slstr', e)
        eco = cached(tag + '_eco', lambda: cmr_list('ECO_L2T_LSTE', ('002', '003'), bbox, t0))
        emit = cached(tag + '_emit', lambda: cmr_list('EMITL1BRAD', ('001',), bbox, t0))
        try:
            ls = cached(tag + '_landsat', lambda: landsat(bbox, t0))
        except Exception as e:
            ls = []; print('  landsat', e)
        try:
            s2 = cached(tag + '_s2', lambda: sentinel2(bbox, t0))
        except Exception as e:
            s2 = []; print('  s2', e)
        geo_name, geo_n, geo_note = geostationary(u.centroid.x, u.centroid.y, t0)
        for _, r in grp.iterrows():
            g = geo[r['scene_id']]
            t = pd.Timestamp(r['acq_datetime_utc']).to_pydatetime()
            d = r.to_dict()
            d['slstr'] = ';'.join(f'{n[:3]}:{dtmin(tt, t):+.0f}min' for n, _p, tt in sorted(sl, key=lambda x: abs(dtmin(x[2], t))))
            d['slstr_granules'] = ';'.join(sorted({n for n, _p, _t in sl}))
            d['slstr_best_dt_min'] = min((dtmin(tt, t) for _n, _p, tt in sl), key=abs) if sl else np.nan

            def best(lst, extra=None):
                cand = []
                for item in lst:
                    gid, tt, gg = item[0], item[1], item[2]
                    f = frac(g, gg) if gg is not None else np.nan
                    if not (f > 0.05):
                        continue
                    cand.append((gid, dtmin(tt, t), f, item[3] if len(item) > 3 else None))
                cand.sort(key=lambda x: (abs(x[1]), -x[2]))
                return cand
            for nm, lst in (('ecostress', eco), ('emit', emit), ('landsat', ls), ('s2', s2)):
                c = best(lst)
                d[f'{nm}_n'] = len(c)
                d[f'{nm}_id'] = c[0][0] if c else ''
                d[f'{nm}_dt_min'] = c[0][1] if c else np.nan
                d[f'{nm}_overlap'] = round(max(x[2] for x in c), 3) if c else 0.0
                d[f'{nm}_cloud'] = c[0][3] if c else np.nan
                d[f'{nm}_all'] = ';'.join(x[0] for x in c)
            d['geo'] = geo_name; d['geo_slots'] = geo_n; d['geo_note'] = geo_note
            rows.append(d)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, 'stage2_tiles.csv'), index=False)
    df = pd.DataFrame(rows)
    print('done', len(df))


if __name__ == '__main__':
    main()

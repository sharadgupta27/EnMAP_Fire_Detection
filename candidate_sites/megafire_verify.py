#!/usr/bin/env python
# coding: utf-8
"""Second pass on megafire_enmap.py: is the fire actually IN the scene?

megafire_enmap.py matches an EnMAP scene against a padded perimeter envelope,
which is necessary but not sufficient - a 30 km tile can clip the corner of
the box and see nothing but unburnt rangeland. So for every EnMAP tile of the
selected datatakes this reads open Sentinel-2 L2A COGs (Element84 Earth
Search) inside that tile's own footprint and reports, per tile:

  fire_px  Murphy et al. (2016) SWIR test: B12/B11 >= 1.4 and B12/B8A >= 1.4
           and B12 >= 0.15                                  -> flame in the tile
  burn_%   NBR(B8A,B12) < -0.10                             -> scar in the tile

Both come from the Sentinel-2 pass nearest the EnMAP overpass, up to +/-5 d
away, so fire_px is evidence the tile was burning around the acquisition, not
a count at the EnMAP overpass instant. burn_% is the more stable number and is
what separates "EnMAP flew past the fire" from "EnMAP flew over the fire".

    python megafire_verify.py            # all datatakes in TARGETS
    python megafire_verify.py --fire Park
"""

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
import warnings

warnings.filterwarnings('ignore')
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'

import numpy as np
import rasterio
import requests
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES_JSON = os.path.join(HERE, 'megafire_enmap_scenes.json')
OUT_CSV = os.path.join(HERE, 'megafire_enmap_verified.csv')
EARTH_SEARCH = 'https://earth-search.aws.element84.com/v1/search'

# datatakes worth the read: every "during" one over a >=30,000 ha fire, plus
# the post-fire datatakes for the fires EnMAP missed while they were burning.
TARGETS = {
    'Park Fire': ['DT0000095449', 'DT0000097032', 'DT0000097578'],
    'Smokehouse Creek': ['DT0000064959', 'DT0000064963'],
    'Shoe Fire / Candle Lake': ['DT0000128782', 'DT0000148179', 'DT0000151064'],
    'Chiquitania / Santa Cruz': ['DT0000088201'],
    'Sierra Oeste (Avila-Madrid)': ['DT0000206450', 'DT0000207273'],
    'Arganil / Piodao': ['DT0000149658'],
    'Molezuelas de la Carballeda': ['DT0000149654'],
    'Larouco / Ourense': ['DT0000149658'],
    'Uiseong-Andong': ['DT0000120726', 'DT0000126451'],
    'Dragon Bravo': ['DT0000152948'],
    'Jasper Wildfire Complex': ['DT0000088452'],
    'Nopiming': ['DT0000137566'],
    'Morrill Fire': ['DT0000188137', 'DT0000213914'],
    'Grampians (Victoria)': ['DT0000130607'],
    'Gironde': ['DT0000208994', 'DT0000211127'],
    'Pantanal': ['DT0000117655'],
}

S = requests.Session()
S.headers['User-Agent'] = 'enmap-megafire-search/1.0'


def iso(s):
    return dt.datetime.fromisoformat(s.replace('Z', '+00:00'))


def s2_scenes(bbox, when, days=5):
    """Sentinel-2 L2A scenes over bbox within +/-days, nearest first."""
    lo, hi = when - dt.timedelta(days=days), when + dt.timedelta(days=days)
    r = S.post(EARTH_SEARCH, json={
        'collections': ['sentinel-2-l2a'], 'bbox': list(bbox),
        'datetime': lo.strftime('%Y-%m-%dT%H:%M:%SZ') + '/' + hi.strftime('%Y-%m-%dT%H:%M:%SZ'),
        'limit': 100}, timeout=120)
    r.raise_for_status()
    feats = r.json().get('features', [])
    feats.sort(key=lambda f: abs((iso(f['properties']['datetime']) - when).total_seconds()))
    return feats


def read_win(href, bbox):
    with rasterio.open(href) as src:
        b = transform_bounds('EPSG:4326', src.crs, *bbox, densify_pts=21)
        w = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
        if w.width < 2 or w.height < 2:
            return None
        arr = src.read(1, window=w, boundless=True, fill_value=0).astype('float32')
    # L2A COGs are int16 reflectance x 10000
    return arr / 10000.0


def fire_test(bbox, when, days=5, max_cloud=40):
    """(fire_px, burn_pct, s2_id, dt_hours, cloud) for the nearest usable pass.

    An overcast partner returns zeros for both tests, which reads as "nothing
    burned here" when it means "nothing was visible". So take the nearest pass
    under max_cloud first and only fall back to the cloudy ones if there is
    no such scene in the window.
    """
    cand = s2_scenes(bbox, when, days)
    clear = [f for f in cand if (f['properties'].get('eo:cloud_cover') or 100) <= max_cloud]
    for f in (clear + [f for f in cand if f not in clear])[:8]:
        a = f['assets']
        try:
            b12 = read_win(a['swir22']['href'], bbox)
            b11 = read_win(a['swir16']['href'], bbox)
            b8a = read_win(a['nir08']['href'], bbox)
            if b12 is None or b11 is None or b8a is None:
                continue
        except Exception as exc:
            sys.stderr.write('    read-fail %s %s\n' % (f['id'][:30], type(exc).__name__))
            continue
        n = min(b12.shape[0], b11.shape[0], b8a.shape[0])
        m = min(b12.shape[1], b11.shape[1], b8a.shape[1])
        b12, b11, b8a = b12[:n, :m], b11[:n, :m], b8a[:n, :m]
        valid = (b12 > 0) & (b11 > 0) & (b8a > 0)
        if valid.sum() < 1000:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            r1 = np.where(b11 > 0, b12 / b11, 0)
            r2 = np.where(b8a > 0, b12 / b8a, 0)
        fire = (r1 >= 1.4) & (r2 >= 1.4) & (b12 >= 0.15) & valid
        nbr = np.where((b8a + b12) > 0, (b8a - b12) / (b8a + b12), np.nan)
        burn = (nbr < -0.10) & valid
        dh = (iso(f['properties']['datetime']) - when).total_seconds() / 3600.0
        return (int(fire.sum()), 100.0 * burn.sum() / valid.sum(), f['id'],
                dh, f['properties'].get('eo:cloud_cover'))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fire', help='substring filter on the fire name')
    ap.add_argument('--days', type=int, default=5, help='max S2 offset')
    ap.add_argument('--max-cloud', type=int, default=40,
                    help='prefer Sentinel-2 partners at or below this cloud cover')
    args = ap.parse_args()

    with open(SCENES_JSON, encoding='utf-8') as _fh:
        data = json.load(_fh)
    rows = []

    for fire, dtks in TARGETS.items():
        if args.fire and args.fire.lower() not in fire.lower():
            continue
        info = data[fire]
        scenes = info['during'] + info['post']
        for dtk in dtks:
            tiles = [s for s in scenes if s['datatake'] == dtk]
            if not tiles:
                sys.stderr.write('%s %s: not in JSON\n' % (fire, dtk))
                continue
            print('\n%s  %s  (%s, %d ha)' % (fire, dtk, info['region'], info['area_ha']))
            print('  %-4s %-19s %5s %6s %7s %7s  %s'
                  % ('tile', 'enmap_utc', 'cld%', 'fire', 'burn%', 's2_dt_h', 's2_scene'))
            for s in tiles:
                # the ENMAP01-____L2A- prefix has literal underscores, so match
                # the tile number by shape instead of splitting on '_'
                m = re.search(r'_(\d{3})_V\d', s['scene_id'])
                tno = m.group(1) if m else '?'
                when = iso(s['acq_datetime_utc'])
                res = fire_test(s['bbox'], when, args.days, args.max_cloud)
                if res is None:
                    print('  %-4s %-19s %5s   no usable Sentinel-2 pass' % (tno, s['acq_datetime_utc'][:19], s['cloud_pct']))
                    continue
                fpx, bpct, sid, dh, scc = res
                print('  %-4s %-19s %5s %6d %7.1f %+7.1f  %s (cc %.0f%%)'
                      % (tno, s['acq_datetime_utc'][:19], s['cloud_pct'], fpx, bpct, dh, sid, scc or 0))
                rows.append({
                    'fire': fire, 'region': info['region'], 'area_ha': info['area_ha'],
                    'phase': s['phase'], 'datatake': dtk, 'tile_no': tno,
                    'scene_id': s['scene_id'], 'acq_datetime_utc': s['acq_datetime_utc'],
                    'enmap_cloud_pct': s['cloud_pct'],
                    'west_lon': s['bbox'][0], 'south_lat': s['bbox'][1],
                    'east_lon': s['bbox'][2], 'north_lat': s['bbox'][3],
                    's2_active_fire_px': fpx, 's2_burn_scar_pct': round(bpct, 2),
                    's2_scene': sid, 's2_dt_h': round(dh, 2), 's2_cloud_pct': scc,
                })

    if rows:
        with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print('\n-> %s  (%d tiles)' % (OUT_CSV, len(rows)))


if __name__ == '__main__':
    main()

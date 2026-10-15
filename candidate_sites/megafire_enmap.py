#!/usr/bin/env python
# coding: utf-8
"""Which very large wildfires since 2024 does EnMAP actually cover?

Complements wildfire_4sensor.py, which ranked sites by three-sensor
coincidence. Here the ranking criterion is the SIZE of the fire: start from
the megafires themselves (final burned area from the national/EFFIS/CIFFC
record) and ask the DLR STAC whether EnMAP flew the perimeter at all.

Two windows are searched per fire:
  during = ignition-2 d .. containment+2 d    -> active fire in the scene
  post   = containment   .. containment+180 d -> burn scar / post-fire recovery

L2A is what the public STAC exposes; L1B for the same datatake is ordered
through EOWEB (see README.md, "How to order").

    python megafire_enmap.py             # writes megafire_enmap_coverage.csv
    python megafire_enmap.py --days 365  # longer post-fire window
"""

import argparse
import csv
import datetime as dt
import json
import os
import sys
import time

import requests

STAC = 'https://geoservice.dlr.de/eoc/ogc/stac/v1/search'
COLLECTION = 'ENMAP_HSI_L2A'
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'megafire_enmap_coverage.csv')
OUT_JSON = os.path.join(HERE, 'megafire_enmap_scenes.json')

# name, region, final burned area (ha), ignition, containment, bbox W,S,E,N
# Areas are the reported final burned area (national agency / EFFIS / CIFFC).
# Bboxes are the perimeter envelope padded to ~0.1 deg, so a scene can
# intersect the box and still miss the scar - check per scene before ordering.
FIRES = [
    # ---- 2024 ----
    ("Smokehouse Creek", "Texas Panhandle, USA", 428352,
     "2024-02-26", "2024-03-16", (-101.60, 35.45, -100.00, 36.30)),
    ("Windy Deuce", "Texas Panhandle, USA", 58300,
     "2024-02-26", "2024-03-10", (-101.90, 35.55, -101.20, 36.05)),
    ("Park Fire", "N Sierra Nevada, CA, USA", 173900,
     "2024-07-24", "2024-10-15", (-122.05, 39.55, -121.10, 40.50)),
    ("Jasper Wildfire Complex", "Alberta, Canada", 32722,
     "2024-07-22", "2024-09-07", (-118.60, 52.55, -117.40, 53.30)),
    ("Parker Lake (Fort Nelson)", "British Columbia, Canada", 13000,
     "2024-05-10", "2024-07-31", (-123.20, 58.50, -122.20, 59.10)),
    ("Chiquitania / Santa Cruz", "Bolivia", 3000000,
     "2024-08-01", "2024-10-15", (-63.60, -18.20, -60.80, -15.30)),
    ("Pantanal", "Mato Grosso do Sul, Brazil", 1000000,
     "2024-06-01", "2024-09-30", (-58.20, -20.20, -56.00, -17.00)),
    ("Grampians (Victoria)", "Victoria, Australia", 76000,
     "2024-12-16", "2025-01-25", (142.00, -37.70, 142.80, -36.75)),

    # ---- 2025 ----
    ("Uiseong-Andong", "North Gyeongsang, South Korea", 45000,
     "2025-03-22", "2025-03-30", (128.45, 36.05, 129.35, 36.75)),
    ("Palisades", "Los Angeles, CA, USA", 9490,
     "2025-01-07", "2025-01-31", (-118.65, 34.02, -118.45, 34.15)),
    ("Eaton", "Los Angeles, CA, USA", 5675,
     "2025-01-07", "2025-01-31", (-118.20, 34.15, -118.00, 34.30)),
    ("Shoe Fire / Candle Lake", "Saskatchewan, Canada", 565701,
     "2025-05-07", "2025-08-31", (-106.60, 53.40, -103.80, 55.10)),
    ("Nopiming", "Manitoba, Canada", 130000,
     "2025-05-13", "2025-08-31", (-96.10, 50.20, -94.90, 51.30)),
    ("Dragon Bravo", "Grand Canyon N Rim, AZ, USA", 58700,
     "2025-07-04", "2025-09-30", (-112.60, 36.10, -111.80, 36.75)),
    ("White Sage", "Kaibab Plateau, AZ, USA", 32000,
     "2025-07-09", "2025-08-31", (-112.40, 36.55, -111.90, 37.00)),
    ("Arganil / Piodao", "Coimbra, Portugal", 64451,
     "2025-08-13", "2025-08-24", (-8.25, 39.95, -7.45, 40.55)),
    ("Molezuelas de la Carballeda", "Zamora, Spain", 37000,
     "2025-08-10", "2025-08-25", (-6.70, 41.75, -5.85, 42.45)),
    ("Larouco / Ourense", "Galicia, Spain", 30000,
     "2025-08-12", "2025-08-25", (-7.60, 41.85, -6.80, 42.45)),
    ("Los Alerces", "Chubut, Argentina", 41000,
     "2025-01-30", "2025-03-15", (-71.90, -43.10, -71.30, -42.40)),

    # ---- 2026 ----
    ("Morrill Fire", "W Nebraska Sandhills, USA", 260359,
     "2026-03-12", "2026-03-25", (-103.40, 40.95, -101.20, 42.20)),
    ("Sierra Oeste (Avila-Madrid)", "Castilla y Leon / Madrid, Spain", 60000,
     "2026-07-22", "2026-08-10", (-5.30, 40.05, -4.05, 40.75)),
    ("Gironde", "Nouvelle-Aquitaine, France", 37000,
     "2026-07-20", "2026-08-05", (-1.30, 44.25, -0.10, 45.05)),
]

SESSION = requests.Session()


def stac_search(bbox, start, end, limit=200):
    """All L2A scenes intersecting bbox in [start, end], paging through."""
    feats, token = [], None
    while True:
        body = {
            'collections': [COLLECTION],
            'bbox': list(bbox),
            'datetime': start + 'T00:00:00Z/' + end + 'T23:59:59Z',
            'limit': limit,
        }
        if token:
            body['token'] = token
        for attempt in range(4):
            try:
                r = SESSION.post(STAC, json=body, timeout=120)
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 3:
                    raise
                sys.stderr.write('  retry (%s)\n' % exc)
                time.sleep(3 * (attempt + 1))
        doc = r.json()
        page = doc.get('features', [])
        feats.extend(page)
        token = None
        for link in doc.get('links', []):
            if link.get('rel') == 'next':
                token = (link.get('body') or {}).get('token')
        if not token or not page:
            return feats


def _num(v):
    """eo:cloud_cover comes back as an int on some scenes and a string on others."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def scene_rows(feats, phase):
    rows = []
    for f in feats:
        p = f['properties']
        sid = f['id']
        dtk = 'DT' + sid.split('DT')[1][:10] if 'DT' in sid else ''
        rows.append({
            'phase': phase,
            'scene_id': sid,
            'datatake': dtk,
            'acq_datetime_utc': p.get('datetime'),
            'cloud_pct': _num(p.get('eo:cloud_cover')),
            'sun_elev_deg': round(_num(p.get('view:sun_elevation')) or 0, 1),
            'orbit_state': p.get('sat:orbit_state'),
            'biome': p.get('enmap:biomeType'),
            'bbox': [round(v, 4) for v in f.get('bbox', [])[:4]],
        })
    rows.sort(key=lambda r: r['acq_datetime_utc'] or '')
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=180,
                    help='length of the post-fire window')
    args = ap.parse_args()

    all_scenes, summary = {}, []

    for name, region, ha, ign, out, bbox in FIRES:
        ign_d = dt.date.fromisoformat(ign)
        end_d = dt.date.fromisoformat(out)
        during = (str(ign_d - dt.timedelta(days=2)), str(end_d + dt.timedelta(days=2)))
        post = (out, str(end_d + dt.timedelta(days=args.days)))

        sys.stderr.write('%-32s %s .. %s\n' % (name, during[0], post[1]))
        d_rows = scene_rows(stac_search(bbox, *during), 'during')
        p_rows = scene_rows(stac_search(bbox, *post), 'post')
        # the windows touch, so drop what the during query already returned
        seen = set(r['scene_id'] for r in d_rows)
        p_rows = [r for r in p_rows if r['scene_id'] not in seen]

        all_scenes[name] = {'region': region, 'area_ha': ha, 'bbox': bbox,
                            'ignition': ign, 'end': out,
                            'during': d_rows, 'post': p_rows}

        best = min(d_rows, key=lambda r: r['cloud_pct'] if r['cloud_pct'] is not None else 999, default=None)
        best_p = min(p_rows, key=lambda r: r['cloud_pct'] if r['cloud_pct'] is not None else 999, default=None)
        summary.append({
            'fire': name, 'region': region, 'area_ha': ha,
            'ignition': ign, 'end': out,
            'bbox': '%.2f,%.2f,%.2f,%.2f' % bbox,
            'n_during': len(d_rows), 'n_post': len(p_rows),
            'n_during_clear20': len([r for r in d_rows if (r['cloud_pct'] if r['cloud_pct'] is not None else 100) <= 20]),
            'n_post_clear20': len([r for r in p_rows if (r['cloud_pct'] if r['cloud_pct'] is not None else 100) <= 20]),
            'best_during_scene': best['scene_id'] if best else '',
            'best_during_date': best['acq_datetime_utc'] if best else '',
            'best_during_cloud': best['cloud_pct'] if best else '',
            'best_post_scene': best_p['scene_id'] if best_p else '',
            'best_post_date': best_p['acq_datetime_utc'] if best_p else '',
            'best_post_cloud': best_p['cloud_pct'] if best_p else '',
        })

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    with open(OUT_JSON, 'w', encoding='utf-8') as fh:
        json.dump(all_scenes, fh, indent=1)

    hdr = '%-30s %9s  %5s %6s  %5s %6s' % ('fire', 'ha', 'dur', '<=20%', 'post', '<=20%')
    print(hdr)
    print('-' * len(hdr))
    for s in sorted(summary, key=lambda x: -x['area_ha']):
        print('%-30s %9d  %5d %6d  %5d %6d' % (
            s['fire'][:30], s['area_ha'], s['n_during'], s['n_during_clear20'],
            s['n_post'], s['n_post_clear20']))
    print('\n-> %s\n-> %s' % (OUT_CSV, OUT_JSON))


if __name__ == '__main__':
    main()

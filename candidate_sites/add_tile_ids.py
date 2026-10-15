#!/usr/bin/env python
# coding: utf-8
"""Append EnMAP tile identifiers to sites_ordering.csv, taken from enmap_scenes.csv.

A datatake is ordered tile by tile in EOWEB, so the site-level table is only
half the story: knowing that S1 is 'DT0000198082' still leaves you to work
out which of its 25 tiles to order. These columns answer that directly.

Added columns:
    all_tile_nos          every tile in the datatake
    validated_tile_nos    tiles where EnMAP, Landsat 8/9 and ECOSTRESS all
                          overlap AND the acquisitions fall within ~1 h -
                          i.e. the tiles actually worth ordering
    best_tile_no          the validated tile with the most Sentinel-2
                          active-fire pixels (ties broken by lower cloud)
    best_tile_scene_id    that tile's full EnMAP L2A scene identifier, which
                          is the string EOWEB matches on
    best_tile_fire_px     its Sentinel-2 active-fire pixel count

best_tile_* is left empty for the tier-2 sites: none of their tiles passes
the three-sensor test, and there is no per-tile fire measurement for them,
so any "best" would be invented rather than measured.

Existing columns and their values are preserved untouched; this only
appends. Run from this directory:  python add_tile_ids.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = os.path.join(HERE, 'sites_ordering.csv')
SCENES = os.path.join(HERE, 'enmap_scenes.csv')

NEW_COLS = ['all_tile_nos', 'validated_tile_nos', 'best_tile_no',
            'best_tile_scene_id', 'best_tile_fire_px']


def to_int(v, default=None):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def main():
    with open(SCENES, newline='', encoding='utf-8-sig') as fh:
        scenes = list(csv.DictReader(fh))
    with open(SITES, newline='', encoding='utf-8-sig') as fh:
        rdr = csv.DictReader(fh)
        fields = list(rdr.fieldnames)
        sites = [r for r in rdr if r.get('site_id')]

    by_site = {}
    for r in scenes:
        by_site.setdefault(r['site_id'], []).append(r)

    for s in sites:
        rows = by_site.get(s['site_id'], [])
        rows.sort(key=lambda r: r['tile_no'])
        validated = [r for r in rows if (r.get('three_sensor_validated') or '').strip().upper() == 'YES']

        s['all_tile_nos'] = ','.join(r['tile_no'] for r in rows)
        s['validated_tile_nos'] = ','.join(r['tile_no'] for r in validated)

        best = None
        if validated:
            # Most active-fire pixels wins; a tile with no fire measurement
            # cannot win on it, and cloud breaks ties.
            scored = [(to_int(r.get('s2_active_fire_px'), -1),
                       -(to_int(r.get('cloud_pct'), 100)), r) for r in validated]
            top = max(scored, key=lambda t: (t[0], t[1]))
            if top[0] >= 0:
                best = top[2]
        s['best_tile_no'] = best['tile_no'] if best else ''
        s['best_tile_scene_id'] = best['enmap_scene_id'] if best else ''
        s['best_tile_fire_px'] = (best.get('s2_active_fire_px') or '') if best else ''

        # Cross-check against the count the table already carries, so a
        # mismatch surfaces here rather than in an order.
        declared = to_int(s.get('n_scenes_3sensor_validated'), None)
        if declared is not None and declared != len(validated):
            print(f"  ! {s['site_id']}: n_scenes_3sensor_validated={declared} but "
                  f"{len(validated)} tiles are flagged YES in enmap_scenes.csv")
        declared_n = to_int(s.get('n_scenes_in_datatake'), None)
        if declared_n is not None and declared_n != len(rows):
            print(f"  ! {s['site_id']}: n_scenes_in_datatake={declared_n} but "
                  f"{len(rows)} tiles listed in enmap_scenes.csv")

    out_fields = fields + [c for c in NEW_COLS if c not in fields]
    with open(SITES, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=out_fields)
        wr.writeheader()
        for s in sites:
            wr.writerow({k: s.get(k, '') for k in out_fields})

    for s in sites:
        print(f"{s['site_id']:>4}  tiles={len(by_site.get(s['site_id'], [])):>2}  "
              f"validated=[{s['validated_tile_nos'] or '-'}]  "
              f"best={s['best_tile_no'] or '-'}"
              f"{'  fire_px=' + s['best_tile_fire_px'] if s['best_tile_fire_px'] else ''}")
    print(f"\nappended {len(NEW_COLS)} columns to {os.path.basename(SITES)}")


if __name__ == '__main__':
    main()

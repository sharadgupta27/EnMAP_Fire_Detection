#!/usr/bin/env python
# coding: utf-8
"""Re-run the current fire test over the original S1-S10 tiles.

The S* sites in enmap_scenes.csv were scored by an earlier version of the
Sentinel-2 test whose weaknesses are written up in sites/README.md: the
partner scene was allowed to be up to ~1 day away with no offset recorded,
and cloud cover on that partner was not considered, so an overcast pass
could read as "no fire" and a stale one as "fire".

This applies megafire_verify.fire_test to every S* tile - nearest Sentinel-2
pass at <=40 % cloud within +/-5 d, offset recorded - so the old sites and
the new megafire sites are ranked by the same measurement, which is what the
keep/drop decision needs.

    python retest_legacy_sites.py     # writes legacy_sites_retest.csv
"""

import csv
import os
import sys

from megafire_verify import fire_test, iso

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, 'enmap_scenes.csv')
OUT_CSV = os.path.join(HERE, 'legacy_sites_retest.csv')


def main():
    with open(IN_CSV, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))

    out = []
    for r in rows:
        bbox = [float(r['west_lon']), float(r['south_lat']),
                float(r['east_lon']), float(r['north_lat'])]
        when = iso(r['acq_datetime_utc'])
        res = fire_test(bbox, when, days=5, max_cloud=40)
        if res is None:
            fpx, bpct, sid, dh, scc = 0, 0.0, '', None, None
            note = 'no usable Sentinel-2 pass'
        else:
            fpx, bpct, sid, dh, scc = res
            note = ''
        print('%-4s %-12s t%-4s %-24s fire=%5d burn=%5.1f%% dt=%+7.1fh  %s'
              % (r['site_id'], r['datatake'], r['tile_no'],
                 r['acq_datetime_utc'][:19], fpx, bpct, dh or 0, note or sid))
        sys.stdout.flush()
        out.append({
            'site_id': r['site_id'], 'site_name': r['site_name'],
            'datatake': r['datatake'], 'tile_no': r['tile_no'],
            'acq_datetime_utc': r['acq_datetime_utc'],
            'enmap_cloud_pct': r['cloud_pct'],
            'west_lon': bbox[0], 'south_lat': bbox[1],
            'east_lon': bbox[2], 'north_lat': bbox[3],
            'old_fire_px': r['s2_active_fire_px'],
            'old_burn_pct': r['s2_burn_scar_pct'],
            'new_fire_px': fpx, 'new_burn_pct': round(bpct, 2),
            's2_scene': sid, 's2_dt_h': round(dh, 2) if dh is not None else '',
            's2_cloud_pct': scc, 'note': note,
        })

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print('\n-> %s  (%d tiles)' % (OUT_CSV, len(out)))


if __name__ == '__main__':
    main()

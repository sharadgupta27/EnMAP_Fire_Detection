"""SoCal Jan 2025 - pass 2: ECOSTRESS data check + Sentinel-2 fire test +
widened Landsat search, per EnMAP tile that contains an active fire."""
import json, os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
from datetime import timedelta

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import requests
import earthaccess
import fire_check as fc
from pass2_eco import granule_valid_pixels

LL = "https://landsatlook.usgs.gov/stac-server/search"
S = requests.Session(); S.headers['User-Agent'] = 'enmap-fire-search/1.0'
SEARCH_H = 6.0          # ECOSTRESS: hours either side


def landsat(bbox, t, days=4):
    lo = (t - timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    hi = (t + timedelta(days=days)).strftime('%Y-%m-%dT%H:%M:%SZ')
    r = S.post(LL, json={'collections': ['landsat-c2l2-st'], 'bbox': list(bbox),
                         'datetime': '%s/%s' % (lo, hi), 'limit': 200}, timeout=120)
    out = []
    for f in r.json().get('features', []):
        p = f['properties']; pl = (p.get('platform') or '').upper()
        if pl not in ('LANDSAT_8', 'LANDSAT_9'):
            continue
        dh = (fc.iso(p['datetime']) - t).total_seconds() / 3600.0
        out.append({'id': f['id'], 'plat': pl, 'dh': round(dh, 2),
                    'cloud': p.get('eo:cloud_cover')})
    out.sort(key=lambda x: abs(x['dh']))
    return out


def main():
    with open(os.path.join(SP, 'socal_pass1.json')) as _fh:
        rows = [x for x in json.load(_fh) if x['fires']]
    earthaccess.login(strategy='netrc')
    sess = earthaccess.get_requests_https_session()

    for x in rows:
        t = fc.iso(x['dt'])
        tag = x['id'].split('-DT')[1][:26]
        night = (x['sun_el'] or 0) <= 0.0
        print('\n=== DT%s  %s  %s  [%s] cloud=%s ==='
              % (tag, x['dt'][:19], 'NIGHT' if night else 'day ',
                 ','.join(x['fires']), x['cloud']), flush=True)

        # --- Landsat 8/9, widened -------------------------------------------
        ls = landsat(x['bbox'], t)
        x['landsat_wide'] = ls[:4]
        if not ls:
            print('  Landsat  none within +/-4 d')
        for e in ls[:4]:
            print('  Landsat  %-44s %-10s dt=%+7.2f h  cloud=%s'
                  % (e['id'], e['plat'], e['dh'], e['cloud']))

        # --- ECOSTRESS, every granule, real pixel count ----------------------
        lo = (t - timedelta(hours=SEARCH_H)).strftime('%Y-%m-%dT%H:%M:%SZ')
        hi = (t + timedelta(hours=SEARCH_H)).strftime('%Y-%m-%dT%H:%M:%SZ')
        try:
            grans = earthaccess.search_data(short_name='ECO_L2T_LSTE',
                                            bounding_box=tuple(x['bbox']),
                                            temporal=(lo, hi), count=40)
        except Exception as exc:
            print('  ECOSTRESS CMR failed: %s' % exc); grans = []
        best = None
        for g in grans:
            links = [l for l in g.data_links() if l.endswith('_LST.tif')]
            if not links:
                continue
            nm = os.path.basename(links[0])
            res = granule_valid_pixels(links[0], x['bbox'], sess)
            if 'error' in res:
                print('  ECOSTRESS %-52s %s' % (nm[:52], res['error'][:50])); continue
            print('  ECOSTRESS %-52s valid_px=%7d aoi=%.2f LST=%s..%s'
                  % (nm[:52], res['valid_px'], res['aoi_frac'],
                     res.get('lst_min'), res.get('lst_max')), flush=True)
            if best is None or res['valid_px'] > best[1]['valid_px']:
                best = (nm, res)
        x['eco'] = {'granule': best[0], **best[1]} if best else None
        if not grans:
            print('  ECOSTRESS no granule in +/-%.0f h' % SEARCH_H)

        # --- Sentinel-2 active fire + burn scar ------------------------------
        x['s2'] = fc.check(x['bbox'], t, tag, days=3)
        with open(os.path.join(SP, 'socal_pass2.json'), 'w') as _fh:
            json.dump(rows, _fh, indent=1)

    print('\nwrote socal_pass2.json')


if __name__ == '__main__':
    main()

"""The concurrent active-fire test that the site search was missing.

ECOSTRESS is the only partner selected to be within ~3 h of EnMAP, so it is
the only one that can say whether anything was burning AT THE OVERPASS.
The search used it purely as a data-availability check (valid_px) and never
looked at the temperatures it recorded.

A wildfire inside a 70 m LST pixel raises it far above the surrounding
ground. Report the tile's own distribution so "hot" is measured against the
local background rather than an absolute guess.
"""
import json, os, sys, tempfile, warnings
warnings.filterwarnings('ignore')
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, rasterio, earthaccess
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds


def lst_stats(url, bbox, session):
    r = session.get(url, stream=True, timeout=300); r.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as fh:
        for c in r.iter_content(1 << 20):
            fh.write(c)
        path = fh.name
    try:
        with rasterio.open(path) as src:
            gb = transform_bounds('EPSG:4326', src.crs, *bbox, densify_pts=21)
            sb = src.bounds
            if min(gb[2], sb.right) <= max(gb[0], sb.left) or min(gb[3], sb.top) <= max(gb[1], sb.bottom):
                return None
            w = from_bounds(max(gb[0], sb.left), max(gb[1], sb.bottom),
                            min(gb[2], sb.right), min(gb[3], sb.top),
                            transform=src.transform).round_offsets().round_lengths()
            a = src.read(1, window=w).astype('float64')
        g = np.isfinite(a) & (a > 150.0) & (a < 1000.0)
        if g.sum() < 100:
            return None
        v = a[g]
        med = float(np.median(v))
        # anomaly threshold: 15 K over the tile's own median, floored at 330 K
        # so a hot desert background cannot masquerade as fire
        thr = max(330.0, med + 15.0)
        return {'n': int(g.sum()), 'median': round(med, 1),
                'p99': round(float(np.percentile(v, 99)), 1),
                'p99_9': round(float(np.percentile(v, 99.9)), 1),
                'max': round(float(v.max()), 1),
                'thr': round(thr, 1),
                'hot_px': int((v > thr).sum()),
                'hot_330': int((v > 330).sum()), 'hot_340': int((v > 340).sum()),
                'hot_350': int((v > 350).sum())}
    finally:
        try: os.unlink(path)
        except OSError: pass


with open('pertile_pass4.json') as _fh:
    p4 = json.load(_fh)
earthaccess.login(strategy='netrc')
sess = earthaccess.get_requests_https_session()

# rebuild the granule URL from the name recorded in pass 4
import fire_check as fc
from datetime import timedelta
out = []
print('%-32s %7s %7s %7s %7s %7s %7s %7s  %s'
      % ('tile', 'median', 'p99', 'p99.9', 'max', '>thr', '>340K', '>350K', 'verdict'))
print('-' * 120)
for e in p4:
    eco = e.get('eco') or {}
    if not eco.get('granule'):
        print('%-32s  no ECOSTRESS granule' % e['tile'].split('-DT')[1][:30]); continue
    t = fc.iso(e['enmap_dt'])
    lo = (t - timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
    hi = (t + timedelta(hours=6)).strftime('%Y-%m-%dT%H:%M:%SZ')
    grans = earthaccess.search_data(short_name='ECO_L2T_LSTE', bounding_box=tuple(e['bbox']),
                                    temporal=(lo, hi), count=40)
    url = None
    for g in grans:
        for l in g.data_links():
            if os.path.basename(l) == eco['granule']:
                url = l
    if url is None:
        print('%-32s  granule %s not relocated' % (e['tile'].split('-DT')[1][:30], eco['granule'][:40]))
        continue
    st = lst_stats(url, e['bbox'], sess)
    if st is None:
        print('%-32s  unreadable' % e['tile'].split('-DT')[1][:30]); continue
    verdict = 'ACTIVE FIRE' if st['hot_px'] > 0 else 'no thermal anomaly'
    print('%-32s %7.1f %7.1f %7.1f %7.1f %7d %7d %7d  %s'
          % (e['tile'].split('-DT')[1][:30], st['median'], st['p99'], st['p99_9'],
             st['max'], st['hot_px'], st['hot_340'], st['hot_350'], verdict))
    e['eco_thermal'] = st
    out.append(e)

with open('pertile_pass5.json', 'w') as _fh:
    json.dump(p4, _fh, indent=1)
print('\nwrote pertile_pass5.json')

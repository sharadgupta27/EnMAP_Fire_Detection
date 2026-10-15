"""W01 Colorado: was anything actually burning at the EnMAP overpass?

Three independent thermal / spectral checks over the W01 tile bbox:
  1. Landsat 8 surface temperature, same day, ~1.4 h from EnMAP
  2. Sentinel-2 Murphy fire test, EVERY scene in +/-6 d with its own dt
  3. where the flagged pixels actually are
"""
import os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, rasterio, requests
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds
import fire_check as fc

BBOX = [-106.2625057, 39.7540361, -105.8213041, 40.0784885]
T = fc.iso('2026-08-20T18:36:14.113166Z')
S = requests.Session(); S.headers['User-Agent'] = 'enmap-fire-search/1.0'

# ---- 1. Landsat 8 surface temperature ------------------------------------
print('=== Landsat 8 ST_B10 over the W01 tile, same day ===')
r = S.post('https://landsatlook.usgs.gov/stac-server/search',
           json={'collections': ['landsat-c2l2-st'], 'bbox': BBOX,
                 'datetime': '2026-08-19T00:00:00Z/2026-08-21T23:59:59Z', 'limit': 20}, timeout=120)
for f in r.json().get('features', []):
    p = f['properties']
    if (p.get('platform') or '').upper() not in ('LANDSAT_8', 'LANDSAT_9'):
        continue
    a = f['assets']
    key = next((k for k in a if k.lower() in ('st_b10', 'lwir11', 'lwir')), None)
    if key is None:
        print('   %s: no ST band asset (%s)' % (f['id'], list(a)[:8])); continue
    href = a[key]['href'].replace('s3://usgs-landsat/', 'https://landsatlook.usgs.gov/data/')
    try:
        with rasterio.open(href) as src:
            b = transform_bounds('EPSG:4326', src.crs, *BBOX, densify_pts=21)
            w = from_bounds(*b, transform=src.transform).round_offsets().round_lengths()
            arr = src.read(1, window=w, boundless=True, fill_value=0).astype('float64')
        good = arr > 0
        # Collection-2 ST scaling: K = DN*0.00341802 + 149.0
        k = arr[good] * 0.00341802 + 149.0
        dh = (fc.iso(p['datetime']) - T).total_seconds() / 3600
        print('   %s  dt=%+.2f h' % (f['id'], dh))
        print('      valid=%d  ST median=%.1f K  p99=%.1f K  p99.99=%.1f K  MAX=%.1f K'
              % (good.sum(), np.median(k), np.percentile(k, 99),
                 np.percentile(k, 99.99), k.max()))
        print('      pixels >330 K: %d      >350 K: %d      >400 K: %d'
              % ((k > 330).sum(), (k > 350).sum(), (k > 400).sum()))
    except Exception as ex:
        print('   %s read failed: %s' % (f['id'], ex))

# ---- 2. Sentinel-2, every scene, with its own dt --------------------------
print('\n=== Sentinel-2 Murphy fire test, every scene within +/-6 d ===')
for f in fc.s2_scenes(BBOX, T, days=6)[:12]:
    p = f['properties']; dt = fc.iso(p['datetime'])
    try:
        b12, tr = fc.read_win(f['assets']['swir22']['href'], BBOX)
        if b12 is None:
            continue
        b11, _ = fc.read_win(f['assets']['swir16']['href'], BBOX)
        b8a, _ = fc.read_win(f['assets']['nir08']['href'], BBOX)
        n = min(b12.shape[0], b11.shape[0], b8a.shape[0])
        m = min(b12.shape[1], b11.shape[1], b8a.shape[1])
        b12, b11, b8a = b12[:n, :m], b11[:n, :m], b8a[:n, :m]
        valid = (b12 > 0) & (b11 > 0) & (b8a > 0)
        if valid.sum() < 100:
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            r1 = np.where(b11 > 0, b12 / b11, 0); r2 = np.where(b8a > 0, b12 / b8a, 0)
        fire = (r1 >= 1.4) & (r2 >= 1.4) & (b12 >= 0.15) & valid
        nbr = np.where((b8a + b12) > 0, (b8a - b12) / (b8a + b12), np.nan)
        burn = (nbr < -0.10) & valid
        dh = (dt - T).total_seconds() / 3600
        print('   %-28s %s dt=%+7.2fh cc=%5.1f  FIRE=%4d  burn=%5.1f%%  b12max=%.3f'
              % (f['id'][:28], p['datetime'][:16], dh, p.get('eo:cloud_cover', 0),
                 int(fire.sum()), 100 * burn.sum() / max(valid.sum(), 1), float(b12[valid].max())))
        if fire.sum() and abs(dh) < 30:
            ys, xs = np.nonzero(fire)
            print('        flagged b12 range %.3f..%.3f   r1 %.2f..%.2f'
                  % (b12[fire].min(), b12[fire].max(), r1[fire].min(), r1[fire].max()))
            # how clustered? a real fire front is contiguous
            from scipy import ndimage
            lab, nlab = ndimage.label(fire)
            sizes = sorted(ndimage.sum_labels(fire, lab, range(1, nlab + 1)), reverse=True)
            print('        %d clusters, largest %s px' % (nlab, [int(s) for s in sizes[:6]]))
    except Exception as ex:
        print('   %-28s read-fail %s' % (f['id'][:28], type(ex).__name__))

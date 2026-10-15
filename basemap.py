"""Sentinel-2 cloudless (EOX) basemap for the map preview.

EOX publishes one cloud-free Sentinel-2 mosaic per year as Web-Mercator
tiles (WMTS, https://tiles.maps.eox.at). fetch_basemap() downloads the tiles
covering a lon/lat box, mosaics them, and reprojects the mosaic to a regular
lon/lat grid (EPSG:4326) so it can sit under the scene's per-pixel lon/lat
directly, with no change to how the fire pixels are drawn.

Licences, as stated in EOX's own WMTSCapabilities.xml (checked 15 Oct 2026):
the 2016 and 2017 mosaics are CC BY 4.0; 2018 and later are CC BY-NC-SA 4.0.
Both need the attribution in ATTRIBUTION on any figure that shows them.

A yearly mosaic is a composite of many acquisitions, mostly from the growing
season. It is context only: it does not show the fire, the smoke or the burn
scar at the EnMAP overpass, and the mosaic of the fire year may already show
the scar. layer_for_acquisition() therefore picks the last year BEFORE the
acquisition (a pre-fire landscape) when the acquisition date is known.

The finished mosaic is cached on disk (the scene's downloads folder when
the caller gives one, else CACHE_DIR), so a scene that has been viewed once
can be viewed again offline.
"""
import hashlib
import math
import os

import numpy as np

YEARS = (2025, 2024, 2023, 2022, 2021, 2020, 2019, 2018, 2017, 2016)
CC_BY_YEARS = (2016, 2017)
CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'enmap_fire', 'basemap')
# Zoom 13 is 19 m * cos(lat) per pixel: finer than EnMAP's 30 m, and a 30 km scene
# stays near 2000 x 2500 px (~15 MB as uint8). Zoom 14 (the 10 m Sentinel-2 limit)
# quadruples memory and download for detail the 30 m fire pixels cannot use.
# MAX_TILES bounds the download for a large box.
MAX_ZOOM = 13
MAX_TILES = 324
_TILE_URL = 'https://tiles.maps.eox.at/wmts/1.0.0/{layer}/default/g/{{z}}/{{y}}/{{x}}.jpg'


def layer_id(year):
    """WMTS layer identifier. The 2016 mosaic is the un-suffixed layer."""
    return 's2cloudless_3857' if int(year) == 2016 else f's2cloudless-{int(year)}_3857'


def tile_url(year):
    return _TILE_URL.format(layer=layer_id(year))


def licence(year):
    return 'CC BY 4.0' if int(year) in CC_BY_YEARS else 'CC BY-NC-SA 4.0'


def attribution(year):
    """Credit line EOX asks for (WMTSCapabilities abstract), shortened."""
    return (f'Sentinel-2 cloudless {int(year)} by EOX IT Services GmbH '
            f'(contains modified Copernicus Sentinel data {int(year)}), {licence(year)}')


def layer_for_acquisition(acq_dt):
    """Last mosaic year before the acquisition year (pre-fire landscape);
    the newest mosaic when the date is unknown or predates every mosaic."""
    if acq_dt is None:
        return YEARS[0]
    before = [y for y in YEARS if y < acq_dt.year]
    return max(before) if before else min(YEARS)


def _zoom_for(lon0, lat0, lon1, lat1):
    """Finest zoom <= MAX_ZOOM whose tile count over the box is <= MAX_TILES."""
    def tile_xy(lon, lat, z):
        n = 2 ** z
        lat = max(min(lat, 85.0511), -85.0511)
        x = (lon + 180.0) / 360.0 * n
        r = math.radians(lat)
        y = (1.0 - math.log(math.tan(r) + 1.0 / math.cos(r)) / math.pi) / 2.0 * n
        return x, y

    for z in range(MAX_ZOOM, 1, -1):
        x0, y1 = tile_xy(lon0, lat0, z)
        x1, y0 = tile_xy(lon1, lat1, z)
        n = (math.floor(x1) - math.floor(x0) + 1) * (math.floor(y1) - math.floor(y0) + 1)
        if n <= MAX_TILES:
            return z
    return 2


def scene_bounds(lon, lat, pad=0.05):
    """lon/lat box of the scene's finite coordinates, padded by `pad` of its
    size on every side so a slightly zoomed-out view is still covered."""
    lon = np.asarray(lon, dtype='float64')
    lat = np.asarray(lat, dtype='float64')
    ok = np.isfinite(lon) & np.isfinite(lat)
    if not ok.any():
        raise ValueError('the scene has no finite longitude/latitude')
    w, e = float(lon[ok].min()), float(lon[ok].max())
    s, n = float(lat[ok].min()), float(lat[ok].max())
    dw, dh = max(e - w, 1e-3) * pad, max(n - s, 1e-3) * pad
    return w - dw, s - dh, e + dw, n + dh


def fetch_basemap(bounds, year, cache_dir=None):
    """Sentinel-2 cloudless mosaic of `year` over bounds = (west, south, east,
    north) in degrees. Returns {'img': HxWx3 uint8, 'extent': (west,
    east, south, north) of the lon/lat grid, 'year', 'zoom', 'attribution'}.
    The finished mosaic is saved as one .npz in cache_dir (the scene's
    downloads folder in the GUI; CACHE_DIR when None) and reused from there.
    Network errors propagate; the caller falls back to the EnMAP composite."""
    import contextily as cx
    from rasterio.enums import Resampling

    w, s, e, n = (float(v) for v in bounds)
    zoom = _zoom_for(w, s, e, n)
    key = hashlib.sha1(f'{layer_id(year)}|{zoom}|{w:.5f}|{s:.5f}|{e:.5f}|{n:.5f}'.encode()).hexdigest()
    cache_dir = cache_dir or CACHE_DIR
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f'S2cloudless_{int(year)}_z{zoom}_{key[:10]}.npz')
    if os.path.exists(path):
        with np.load(path) as z:
            img, extent = z['img'], tuple(float(v) for v in z['extent'])
    else:
        # no separate raw-tile cache: the .npz above is the only file kept
        merc, merc_ext = cx.bounds2img(w, s, e, n, zoom=zoom, source=tile_url(year), ll=True,
                                       max_retries=3, n_connections=4, use_cache=False)
        img, extent = cx.warp_tiles(merc, merc_ext, t_crs='EPSG:4326', resampling=Resampling.bilinear)
        img = np.ascontiguousarray(img[..., :3])
        np.savez_compressed(path, img=img, extent=np.asarray(extent, dtype='float64'))
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return {'img': img, 'extent': tuple(extent), 'year': int(year), 'zoom': zoom,
            'attribution': attribution(year)}


def sample(bm, lon, lat):
    """Nearest-neighbour basemap colour (0-1) at each lon/lat (any shape) -> (..., 3).
    Points outside the mosaic get NaN."""
    img = bm['img']
    w, e, s, n = bm['extent']
    h, wd = img.shape[:2]
    lon = np.asarray(lon, dtype='float64')
    lat = np.asarray(lat, dtype='float64')
    col = np.floor((lon - w) / (e - w) * wd)
    row = np.floor((n - lat) / (n - s) * h)
    ok = np.isfinite(col) & np.isfinite(row) & (col >= 0) & (col < wd) & (row >= 0) & (row < h)
    out = np.full(lon.shape + (3,), np.nan, dtype='float32')
    out[ok] = img[row[ok].astype(int), col[ok].astype(int)] / 255.0
    return out

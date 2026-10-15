#!/usr/bin/env python
# coding: utf-8
"""
Geometric co-registration of an EnMAP L1B scene with AROSICS, before it is
compared with other satellites.

Why the EnMAP side is the one corrected, and against what
---------------------------------------------------------
EnMAP L1B is not orthorectified: its per-pixel positions come from the
product's (rotated) affine, an approximation that ignores terrain relief and
the off-nadir view. On the reference tile t014 (Serra da Estrela, mountainous,
off-nadir) the EnMAP -NDVI image correlates best with VIIRS LST at an offset of
~1.0 km and with Sentinel-3 SLSTR LST at ~1.7 km, in the SAME direction for
both - two independently geolocated sensors agreeing means the offset is
EnMAP's. Every comparison with another sensor (LST sampled at the fire
pixels, FRP detections matched to the fire, the multi-sensor maps) inherits
that error, so it is removed once, on the EnMAP side, before any comparison.

AROSICS (Scheffler et al. 2017, Remote Sens. 9, 676) estimates shifts by
phase correlation in moving windows. It needs a reference with resolvable
texture: matched directly against the ~1 km LST products it finds no peak on
t014 (the iteration jumps between -5 and +8 px at window sizes 16-32), so the
reference is instead the Harmonized Landsat Sentinel-2 product (HLS v2.0,
30 m, NASA LP DAAC; geolocated to the Sentinel-2 global reference image), NIR
band (S30 B8A / L30 B05) - the same 30 m pixel and a band in which EnMAP and
HLS see the same surface. The acquisition nearest in time is used (cloud
cover gated, clouds/shadow/water masked with Fmask), because the burn scar
changes the NIR between dates.

How the shifts are used
-----------------------
COREG_LOCAL computes a tie-point grid; AROSICS's own three outlier filters
(tieP_filter_level=3) remove bad matches. AROSICS corrects an image with a
first-order polynomial through its tie points; the same model is fitted here
to the shift vectors (east and north shift as affine functions of position),
because what the pipeline needs is not a resampled image but corrected
coordinates for every EnMAP pixel. On t014 the affine model predicts
held-out tie points to 52 m (east) / 89 m (north) RMSE, against 261 / 226 m
for a constant shift. Gates: at least MIN_TIEPOINTS valid tie points spanning
at least MIN_SPAN_M in both directions; otherwise a constant (median) shift
if at least 5 points; otherwise no correction (stated in the report).

Independent check (thermal_offset_scan): the EnMAP -NDVI image, binned onto
each LST product's grid at the nominal and at the corrected positions,
against that product's LST over trial offsets. If the correction is right,
the correlation peak moves to ~zero offset. This uses data AROSICS never saw.

AROSICS runs in a subprocess, with the current interpreter when arosics is
importable there (the emit env has it since 14 Oct 2026), else with the
.venv-arosics environment next to this file (a uv venv layered on emit); both
give identical tie points on t014. The pipeline keeps running unchanged,
without co-registration, when neither is available.
"""
import datetime as _dt
import json
import os
import subprocess
import sys
import tempfile
import warnings

import numpy as np
import pandas as pd

HLS_PRODUCTS = {'HLSS30': 'B8A', 'HLSL30': 'B05'}       # NIR band of each HLS product (both 30 m)
FMASK_BAD_BITS = 0b101010                               # bits 1 cloud, 3 cloud shadow, 5 water
MIN_TIEPOINTS = 12
MIN_SPAN_M = 5000.0
AROSICS_PARAMS = dict(grid_res=40, window_size=(128, 128), max_shift=100, tieP_filter_level=3)
_HERE = os.path.dirname(os.path.abspath(__file__))
VENV_PYTHON = os.path.join(_HERE, '.venv-arosics', 'Scripts' if os.name == 'nt' else 'bin',
                           'python.exe' if os.name == 'nt' else 'python')


# ----------------------------------------------------------------- reference
def find_hls_reference(bbox, acq_dt, max_days=20, max_cloud=30.0):
    """HLS granules of the single acquisition nearest `acq_dt` (within
    max_days) whose tiles' mean cloud cover is at most max_cloud %.
    Returns (product, date, [granules]) or None."""
    import earthaccess
    earthaccess.login(strategy='netrc')
    t0 = acq_dt.replace(tzinfo=None)
    groups = {}
    for product in HLS_PRODUCTS:
        found = earthaccess.search_data(
            short_name=product, version='2.0', bounding_box=tuple(bbox),
            temporal=((t0 - _dt.timedelta(days=max_days)).isoformat(),
                      (t0 + _dt.timedelta(days=max_days)).isoformat()))
        for g in found:
            u = g['umm']
            t = _dt.datetime.fromisoformat(u['TemporalExtent']['RangeDateTime']['BeginningDateTime'][:19])
            cc = [float(a['Values'][0]) for a in u.get('AdditionalAttributes', [])
                  if a['Name'] == 'CLOUD_COVERAGE']
            groups.setdefault((product, t.date()), []).append((g, t, cc[0] if cc else np.nan))
    best = None
    for (product, day), items in groups.items():
        cloud = float(np.nanmean([c for _, _, c in items])) if items else np.nan
        if not np.isfinite(cloud) or cloud > max_cloud:
            continue
        dt_h = abs((items[0][1] - t0).total_seconds()) / 3600.0
        key = (dt_h, cloud)
        if best is None or key < best[0]:
            best = (key, product, items[0][1], [g for g, _, _ in items], cloud)
    if best is None:
        return None
    _, product, when, grans, cloud = best
    return {'product': product, 'datetime': when, 'granules': grans, 'cloud_pct': cloud,
            'dt_hours': (when - t0).total_seconds() / 3600.0}


def download_hls_reference(ref, download_dir):
    """NIR band and Fmask of every tile of the chosen acquisition."""
    import earthaccess
    band = HLS_PRODUCTS[ref['product']]
    links = [l for g in ref['granules'] for l in g.data_links()
             if l.endswith((f'.{band}.tif', '.Fmask.tif'))]
    out = os.path.join(download_dir, 'hls_reference')
    os.makedirs(out, exist_ok=True)
    have = [os.path.join(out, os.path.basename(l)) for l in links]
    if not all(os.path.exists(p) for p in have):
        earthaccess.download([l for l, p in zip(links, have) if not os.path.exists(p)], out)
    nir = sorted(p for p in have if p.endswith(f'.{band}.tif') and os.path.exists(p))
    fmask = sorted(p for p in have if p.endswith('.Fmask.tif') and os.path.exists(p))
    return nir, fmask


# ----------------------------------------------------------------- AROSICS
def write_enmap_band(scene, path, nm=865.0):
    """One EnMAP band as a GeoTIFF on the product's own (rotated) affine."""
    from osgeo import gdal
    wl = np.asarray(scene['wavelength'])
    i = int(np.argmin(np.abs(wl - nm)))
    arr = np.asarray(scene['cube'][:, i, :], dtype='float32')
    ds = gdal.GetDriverByName('GTiff').Create(path, arr.shape[1], arr.shape[0], 1, gdal.GDT_Float32)
    ds.SetGeoTransform(scene['geotransform']); ds.SetProjection(scene['projection_wkt'])
    b = ds.GetRasterBand(1); b.SetNoDataValue(np.nan); b.WriteArray(arr)
    ds.FlushCache(); ds = None
    return float(wl[i])


def _arosics_tiepoints(enmap_tif, nir_paths, fmask_paths, out_csv, params):
    """Runs INSIDE the AROSICS interpreter (see main()). EnMAP band warped
    north-up to the HLS UTM grid at 30 m = target; HLS NIR mosaic, Fmask-
    masked = reference."""
    from osgeo import gdal
    from geoarray import GeoArray
    from arosics import COREG_LOCAL
    utm = gdal.Open(nir_paths[0]).GetProjection()
    tgt = gdal.Warp('', enmap_tif, format='MEM', dstSRS=utm, xRes=30, yRes=30, resampleAlg='bilinear',
                    srcNodata=np.nan, dstNodata=-9999, outputType=gdal.GDT_Float32, targetAlignedPixels=True)
    t_arr, t_gt = tgt.ReadAsArray(), tgt.GetGeoTransform()
    xmin, ymax = t_gt[0], t_gt[3]
    xmax, ymin = xmin + t_arr.shape[1] * 30, ymax - t_arr.shape[0] * 30
    pad = 3000

    def mosaic(paths):
        return gdal.Warp('', paths, format='MEM', dstSRS=utm, xRes=30, yRes=30, resampleAlg='near',
                         outputBounds=(xmin - pad, ymin - pad, xmax + pad, ymax + pad), targetAlignedPixels=True)
    ref = mosaic(nir_paths)
    r_arr = ref.ReadAsArray().astype('float32')
    if fmask_paths:
        fm = mosaic(fmask_paths).ReadAsArray().astype(np.uint16)
        r_arr[((fm & FMASK_BAD_BITS) != 0) | (fm == 255)] = -9999
    CRL = COREG_LOCAL(GeoArray(r_arr, ref.GetGeoTransform(), utm, nodata=-9999),
                      GeoArray(t_arr, t_gt, utm, nodata=-9999),
                      grid_res=params['grid_res'], window_size=tuple(params['window_size']),
                      max_shift=params['max_shift'], tieP_filter_level=params['tieP_filter_level'],
                      q=True, progress=False)
    tab = CRL.CoRegPoints_table.drop(columns=['geometry'], errors='ignore')
    tab['crs_wkt'] = ''
    tab.loc[tab.index[:1], 'crs_wkt'] = utm
    tab.to_csv(out_csv, index=False)


def run_arosics(enmap_tif, nir_paths, fmask_paths, out_csv, python=None, timeout=1800):
    """Tie-point table from AROSICS COREG_LOCAL, computed in a subprocess."""
    if python is None:
        try:
            import arosics  # noqa: F401
            python = sys.executable
        except ImportError:
            python = VENV_PYTHON
    if not os.path.exists(python):
        raise RuntimeError(f'AROSICS is not installed: no importable arosics and no {VENV_PYTHON}')
    job = {'enmap_tif': enmap_tif, 'nir_paths': list(nir_paths), 'fmask_paths': list(fmask_paths),
           'out_csv': out_csv, 'params': {k: list(v) if isinstance(v, tuple) else v
                                          for k, v in AROSICS_PARAMS.items()}}
    with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
        json.dump(job, f)
    try:
        p = subprocess.run([python, os.path.abspath(__file__), f.name], capture_output=True, text=True,
                           timeout=timeout)
    finally:
        os.unlink(f.name)
    if p.returncode != 0 or not os.path.exists(out_csv):
        tail = (p.stderr or p.stdout or '').strip().splitlines()[-3:]
        raise RuntimeError('AROSICS failed: ' + ' | '.join(tail))
    tab = pd.read_csv(out_csv)
    wkt = tab['crs_wkt'].dropna()
    return tab.drop(columns=['crs_wkt']), (wkt.iloc[0] if len(wkt) else None)


# ----------------------------------------------------------------- shift model
def valid_tiepoints(tab):
    ok = (tab['ABS_SHIFT'] != -9999)
    if 'OUTLIER' in tab:
        ok &= ~tab['OUTLIER'].astype(str).str.lower().isin(('true', '1'))
    return tab[ok].reset_index(drop=True)


def fit_shift_model(tp):
    """Affine (first-order polynomial) model of the east/north shift over the
    scene, with leave-one-out RMSE; constant (median) model as the fallback.
    Coordinates in the tie points' UTM metres."""
    X, Y = tp['X_MAP'].values.astype(float), tp['Y_MAP'].values.astype(float)
    sh = np.c_[tp['X_SHIFT_M'].values.astype(float), tp['Y_SHIFT_M'].values.astype(float)]
    n = len(tp)
    out = {'n_tiepoints': int(n)}
    if n < 5:
        out['kind'] = 'none'
        return out
    x0, y0 = float(X.mean()), float(Y.mean())
    A = np.c_[np.ones(n), (X - x0) / 1e4, (Y - y0) / 1e4]
    loo_a, loo_c = np.zeros((n, 2)), np.zeros((n, 2))
    for i in range(n):
        m = np.arange(n) != i
        loo_c[i] = sh[i] - np.median(sh[m], axis=0)
        if n >= 4:
            c, *_ = np.linalg.lstsq(A[m], sh[m], rcond=None)
            loo_a[i] = sh[i] - A[i] @ c
    coef, *_ = np.linalg.lstsq(A, sh, rcond=None)
    span = (float(X.max() - X.min()), float(Y.max() - Y.min()))
    rmse_a = np.sqrt(np.mean(loo_a ** 2, axis=0))
    rmse_c = np.sqrt(np.mean(loo_c ** 2, axis=0))
    out.update({'x0': x0, 'y0': y0, 'span_m': span, 'median_shift_m': np.median(sh, axis=0).tolist(),
                'affine_coef': coef.tolist(), 'loo_rmse_affine_m': rmse_a.tolist(),
                'loo_rmse_constant_m': rmse_c.tolist(),
                'tiepoint_bounds': [float(X.min()), float(Y.min()), float(X.max()), float(Y.max())]})
    affine_ok = (n >= MIN_TIEPOINTS and min(span) >= MIN_SPAN_M
                 and np.hypot(*rmse_a) < np.hypot(*rmse_c))
    out['kind'] = 'affine' if affine_ok else 'constant'
    out['loo_rmse_m'] = (rmse_a if affine_ok else rmse_c).tolist()
    return out


def shift_at(model, X, Y):
    """East/north shift (m) the model assigns at UTM positions X, Y."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    if model.get('kind') == 'affine':
        c = np.asarray(model['affine_coef'])
        A = np.stack([np.ones_like(X), (X - model['x0']) / 1e4, (Y - model['y0']) / 1e4], axis=-1)
        s = A @ c
        return s[..., 0], s[..., 1]
    if model.get('kind') == 'constant':
        dx, dy = model['median_shift_m']
        return np.full(X.shape, dx), np.full(X.shape, dy)
    return np.zeros(X.shape), np.zeros(X.shape)


def apply_shift_model(lat, lon, model, crs_wkt):
    """Corrected lat/lon arrays: each EnMAP pixel moved by the model's shift
    at its nominal position (AROSICS shifts are what the target must move)."""
    from pyproj import CRS, Transformer
    fwd = Transformer.from_crs('EPSG:4326', CRS.from_wkt(crs_wkt), always_xy=True)
    inv = Transformer.from_crs(CRS.from_wkt(crs_wkt), 'EPSG:4326', always_xy=True)
    X, Y = fwd.transform(lon, lat)
    dx, dy = shift_at(model, X, Y)
    lon2, lat2 = inv.transform(X + dx, Y + dy)
    return np.asarray(lat2).reshape(lat.shape), np.asarray(lon2).reshape(lon.shape), dx, dy


# ----------------------------------------------------------------- independent check
def proxy_neg_ndvi(scene):
    wl = np.asarray(scene['wavelength'])
    b = lambda nm: np.asarray(scene['cube'][:, int(np.argmin(np.abs(wl - nm))), :], dtype='float32')
    red, nir = b(660.0), b(860.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        return -(nir - red) / (nir + red)


def thermal_offset_scan(proxy, lat, lon, partner, half=3000.0, step=250.0, n_boot=200, seed=0):
    """Pearson r between the EnMAP proxy (binned onto the partner's grid at
    the given positions) and the partner LST, over trial offsets of the EnMAP
    positions; the peak offset is what EnMAP's positions would still need.
    4x4-cell block bootstrap for its spread. Returns a dict, or None.
    Coarse products only (cells >= 250 m, the scan step): a 30-70 m grid
    would need a finer scan and ~1e6 cells x every offset in memory."""
    from pyproj import CRS, Transformer
    lst = np.asarray(partner['lst_K'], dtype=float)
    gt = partner['geotransform']
    if abs(gt[2]) > 1e-12 or abs(gt[4]) > 1e-12 or abs(gt[1]) < step:
        return None
    tr = Transformer.from_crs('EPSG:4326', CRS.from_wkt(partner['projection_wkt']), always_xy=True)
    ok = np.isfinite(proxy) & np.isfinite(lat) & np.isfinite(lon)
    x, y = tr.transform(lon[ok], lat[ok])
    v = proxy[ok].astype(float)
    offs = np.arange(-half, half + 0.5 * step, step)
    # crop the partner to the scene footprint plus the scan margin (VIIRS
    # arrives as a whole 1200 x 1200 km tile)
    c_lo = int(np.floor((x.min() - half - gt[0]) / gt[1])) - 1
    c_hi = int(np.floor((x.max() + half - gt[0]) / gt[1])) + 2
    r_lo = int(np.floor((y.max() + half - gt[3]) / gt[5])) - 1
    r_hi = int(np.floor((y.min() - half - gt[3]) / gt[5])) + 2
    c_lo, r_lo = max(c_lo, 0), max(r_lo, 0)
    c_hi, r_hi = min(c_hi, lst.shape[1]), min(r_hi, lst.shape[0])
    if c_hi - c_lo < 3 or r_hi - r_lo < 3:
        return None
    lst = lst[r_lo:r_hi, c_lo:c_hi]
    gt = (gt[0] + c_lo * gt[1], gt[1], 0.0, gt[3] + r_lo * gt[5], 0.0, gt[5])
    nr, nc = lst.shape

    def binned(dx, dy):
        c = np.floor((x + dx - gt[0]) / gt[1]).astype(np.int64)
        r = np.floor((y + dy - gt[3]) / gt[5]).astype(np.int64)
        m = (c >= 0) & (c < nc) & (r >= 0) & (r < nr)
        idx = r[m] * nc + c[m]
        s = np.bincount(idx, weights=v[m], minlength=nr * nc)
        k = np.bincount(idx, minlength=nr * nc)
        return s, k

    stack, counts = [], []
    for dy in offs:
        for dx in offs:
            s, k = binned(dx, dy)
            stack.append(s); counts.append(k)
    k = np.array(counts)
    full = np.median(k[k > 0]) if (k > 0).any() else 0
    common = np.all(k >= 0.5 * full, axis=0) & np.isfinite(lst.ravel())
    if common.sum() < 50:
        return None
    Xs = np.array(stack)[:, common] / k[:, common]
    yv = lst.ravel()[common]
    rr, cc = np.divmod(np.flatnonzero(common), nc)
    blocks = np.unique((rr // 4) * 100000 + cc // 4, return_inverse=True)[1]

    def corr_grid(w):
        sw = w.sum(); my = (w * yv).sum() / sw
        mx = (Xs * w).sum(axis=1) / sw
        cov = ((Xs - mx[:, None]) * w * (yv - my)).sum(axis=1)
        vx = ((Xs - mx[:, None]) ** 2 * w).sum(axis=1)
        vy = (w * (yv - my) ** 2).sum()
        return (cov / np.sqrt(vx * vy)).reshape(len(offs), len(offs))

    def peak(R):
        i, j = np.unravel_index(np.nanargmax(R), R.shape)
        def vert(a, q):
            if 0 < q < len(a) - 1:
                den = a[q - 1] - 2 * a[q] + a[q + 1]
                if den < 0:
                    return q + 0.5 * (a[q - 1] - a[q + 1]) / den
            return float(q)
        return offs[0] + vert(R[i, :], j) * step, offs[0] + vert(R[:, j], i) * step, R[i, j]

    R = corr_grid(np.ones_like(yv))
    dx, dy, rmax = peak(R)
    rng = np.random.default_rng(seed)
    nb = blocks.max() + 1
    boots = np.array([peak(corr_grid(np.bincount(rng.integers(0, nb, nb), minlength=nb)[blocks]
                                     .astype(float)))[:2] for _ in range(n_boot)])
    z = len(offs) // 2
    return {'n_cells': int(common.sum()), 'cell_m': float(abs(gt[1])),
            'r_zero': float(R[z, z]), 'r_peak': float(rmax), 'peak_dx_m': float(dx), 'peak_dy_m': float(dy),
            'peak_sd_m': boots.std(axis=0).tolist(), 'at_scan_edge': bool(max(abs(dx), abs(dy)) >= half - 1)}


# ----------------------------------------------------------------- driver
def coregister_scene(scene, acq_dt, download_dir, max_days=20, max_cloud=30.0, verbose=True):
    """Co-register the scene's geolocation to HLS. Never raises: returns
    {'status': 'applied' | 'not applied: <reason>', ...}; when applied, also
    'lat'/'lon' (corrected), 'shift_east_m'/'shift_north_m' (per pixel),
    'model', 'tiepoints' (DataFrame), 'reference'."""
    say = print if verbose else (lambda *a, **k: None)
    if acq_dt is None:
        return {'status': 'not applied: acquisition time unknown, no reference can be searched'}
    lat, lon = scene['lat'], scene['lon']
    if not np.isfinite(lat).any():
        return {'status': 'not applied: the scene carries no geolocation'}
    bbox = (float(np.nanmin(lon)), float(np.nanmin(lat)), float(np.nanmax(lon)), float(np.nanmax(lat)))
    try:
        say(f'Co-registration: searching HLS (30 m, Sentinel-2-referenced) within +/-{max_days} days ...')
        ref = find_hls_reference(bbox, acq_dt, max_days=max_days, max_cloud=max_cloud)
        if ref is None:
            return {'status': f'not applied: no HLS acquisition within {max_days} days with cloud <= {max_cloud:.0f}%'}
        say(f"  reference {ref['product']} {ref['datetime']:%Y-%m-%d %H:%M} UTC "
            f"({ref['dt_hours'] / 24:+.1f} days, cloud {ref['cloud_pct']:.0f}%, {len(ref['granules'])} tile(s))")
        nir, fmask = download_hls_reference(ref, download_dir)
        if not nir:
            return {'status': 'not applied: the HLS download failed'}
        work = os.path.join(download_dir, 'hls_reference')
        enmap_tif = os.path.join(work, 'enmap_nir_for_coregistration.tif')
        band_nm = write_enmap_band(scene, enmap_tif)
        say(f'  AROSICS COREG_LOCAL: EnMAP {band_nm:.0f} nm against HLS {HLS_PRODUCTS[ref["product"]]} ...')
        tab, crs_wkt = run_arosics(enmap_tif, nir, fmask, os.path.join(work, 'arosics_tiepoints.csv'))
    except Exception as exc:
        return {'status': f'not applied: {exc.__class__.__name__}: {exc}'}
    tp = valid_tiepoints(tab)
    model = fit_shift_model(tp)
    info = {'reference': {'product': ref['product'], 'datetime': ref['datetime'].isoformat(),
                          'dt_days': ref['dt_hours'] / 24.0, 'cloud_pct': ref['cloud_pct'],
                          'files': [os.path.basename(p) for p in nir],
                          'enmap_band_nm': band_nm},
            'arosics_params': {k: (list(v) if isinstance(v, tuple) else v) for k, v in AROSICS_PARAMS.items()},
            'n_grid_points': int(len(tab)), 'model': model, 'tiepoints': tp, 'crs_wkt': crs_wkt}
    if model['kind'] == 'none' or crs_wkt is None:
        info['status'] = f"not applied: only {model['n_tiepoints']} valid AROSICS tie points (need >= 5)"
        return info
    lat2, lon2, dx, dy = apply_shift_model(lat, lon, model, crs_wkt)
    info.update({'status': 'applied', 'lat': lat2, 'lon': lon2, 'shift_east_m': dx, 'shift_north_m': dy})
    say(f"  {model['n_tiepoints']} valid tie points of {len(tab)}; {model['kind']} shift model, "
        f"held-out RMSE {np.hypot(*model['loo_rmse_m']):.0f} m; scene shift median "
        f"{np.nanmedian(dx):+.0f} m east, {np.nanmedian(dy):+.0f} m north "
        f"(range {np.nanmin(np.hypot(dx, dy)):.0f}-{np.nanmax(np.hypot(dx, dy)):.0f} m)")
    return info


def summary_table(info):
    """One-row-per-quantity table for the results folder and the report."""
    m = info.get('model') or {}
    ref = info.get('reference') or {}
    rows = [('status', info.get('status'), ''),
            ('reference', f"{ref.get('product', '')} {str(ref.get('datetime', ''))[:16]}", ''),
            ('reference time - EnMAP time', ref.get('dt_days'), 'days'),
            ('reference cloud cover', ref.get('cloud_pct'), '%'),
            ('AROSICS grid points', info.get('n_grid_points'), ''),
            ('valid tie points', m.get('n_tiepoints'), ''),
            ('shift model', m.get('kind'), ''),
            ('held-out RMSE east', (m.get('loo_rmse_m') or [np.nan, np.nan])[0], 'm'),
            ('held-out RMSE north', (m.get('loo_rmse_m') or [np.nan, np.nan])[1], 'm'),
            ('constant-model held-out RMSE east', (m.get('loo_rmse_constant_m') or [np.nan, np.nan])[0], 'm'),
            ('constant-model held-out RMSE north', (m.get('loo_rmse_constant_m') or [np.nan, np.nan])[1], 'm')]
    if info.get('status') == 'applied':
        dx, dy = info['shift_east_m'], info['shift_north_m']
        rows += [('scene shift east, median', float(np.nanmedian(dx)), 'm'),
                 ('scene shift north, median', float(np.nanmedian(dy)), 'm'),
                 ('scene shift magnitude, min', float(np.nanmin(np.hypot(dx, dy))), 'm'),
                 ('scene shift magnitude, max', float(np.nanmax(np.hypot(dx, dy))), 'm')]
    for chk in info.get('thermal_check', []):
        for stage in ('nominal', 'coregistered'):
            s = chk.get(stage)
            if s:
                rows.append((f"{chk['label']} LST: residual offset, {stage} positions",
                             f"{s['peak_dx_m']:+.0f} E / {s['peak_dy_m']:+.0f} N "
                             f"(sd {s['peak_sd_m'][0]:.0f}/{s['peak_sd_m'][1]:.0f}); "
                             f"r {s['r_zero']:.3f} at zero, {s['r_peak']:.3f} at peak", 'm'))
    return pd.DataFrame(rows, columns=['quantity', 'value', 'unit'])


def main():
    """Subprocess entry: python coregistration.py <job.json>."""
    with open(sys.argv[1]) as f:
        job = json.load(f)
    warnings.filterwarnings('ignore')
    _arosics_tiepoints(job['enmap_tif'], job['nir_paths'], job['fmask_paths'], job['out_csv'], job['params'])


if __name__ == '__main__':
    main()

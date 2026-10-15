#!/usr/bin/env python
# coding: utf-8
"""
Active-fire products of other satellites, for comparison with an EnMAP scene.

thermal_partners.py brings land-surface TEMPERATURE products (Landsat,
ECOSTRESS, VIIRS LST, SLSTR LST): footprint-averaged surface temperatures, a
different physical quantity from a sub-pixel flame temperature. This module
brings the products that measure the SAME quantity this work can derive -
fire radiative power (FRP) - and, for GOES, a sub-pixel fire temperature
and area:

    source          sensor / pixel           quantity                 access
    FIRMS           VIIRS 375 m, MODIS 1 km  FRP per detection        free MAP_KEY (FIRMS_MAP_KEY)
    GOES ABI FDC    2 km, 5-10 min           FRP, fire T, fire area   public AWS bucket, no login
    SLSTR SL_2_FRP  1 km (MWIR), 500 m SWIR  FRP per detection        Copernicus Data Space login
    LSA SAF FRP     SEVIRI 3 km, 15 min      FRP per detection        LSA SAF login (fetch_lsasaf_frp);
                    (FCI 1 km: files only)                            MTG-FCI files loaded from disk

Keys and logins are entered once and remembered (credentials.py).

Every loader returns one table with the columns of DETECTION_COLUMNS, and
compare_frp() sets them against the EnMAP scene.

How the comparison is made, and why (stated in the report):
  * the EnMAP side is this work's FRP, eps sigma T^4 f A per retrieved pixel,
    plus the model-free lower bounds of the pixels without a retrieval
    (saturation_bounds.py), so the EnMAP total is reported as
    [retrieved, retrieved + saturated lower bounds] - itself a LOWER bound;
  * a partner detection is matched to the EnMAP fire if its pixel footprint
    (approximated by a square of the product's pixel area) contains EnMAP
    fire pixels; totals are compared over the EnMAP fire extent;
  * the time difference is reported for every row: fire FRP changes within
    minutes, so only geostationary slots within ~15 min of EnMAP are close in
    time; polar-orbiter passes are hours apart and compare only the order of
    magnitude;
  * MIR (4 um) FRP of the partners and SWIR-derived FRP of EnMAP have
    different sensitivities: coarse sensors miss small/cool fires (detection
    limits ~1-10 MW), and the SWIR misses the cool tail, so agreement is
    expected to the factor level, not the percent level.
"""

import datetime as _dt
import glob
import io
import os
import re
import warnings

import numpy as np
import pandas as pd

DETECTION_COLUMNS = ['source', 'platform', 'datetime_utc', 'lat', 'lon', 'frp_MW', 'frp_unc_MW',
                     'fire_T_K', 'fire_area_m2', 'pixel_area_km2', 'confidence', 'day_night', 'file']

FIRMS_URL = 'https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{area}/{days}/{date}'
FIRMS_SOURCES = ('VIIRS_SNPP_SP', 'VIIRS_SNPP_NRT', 'VIIRS_NOAA20_SP', 'VIIRS_NOAA20_NRT', 'VIIRS_NOAA21_NRT',
                 'MODIS_SP', 'MODIS_NRT')
GOES_BUCKET = 'https://noaa-{sat}.s3.amazonaws.com'
GOES_FIRE_CLASSES = (10, 11, 12, 13, 14, 15, 30, 31, 32, 33, 34, 35)   # FDC Mask: fire pixel codes
SLSTR_EPOCH = _dt.datetime(2000, 1, 1)                                  # time: microseconds since
GOES_EPOCH = _dt.datetime(2000, 1, 1, 12, 0, 0)                         # t: seconds since (J2000)


def _empty():
    return pd.DataFrame({c: pd.Series(dtype=float if c not in ('source', 'platform', 'confidence',
                                                                 'day_night', 'file', 'datetime_utc') else object)
                         for c in DETECTION_COLUMNS})


def _frame(rows):
    df = pd.DataFrame(rows)
    for c in DETECTION_COLUMNS:
        if c not in df:
            df[c] = np.nan
    return df[DETECTION_COLUMNS]


def _in_bbox(lat, lon, bbox):
    w, s, e, n = bbox
    return (lon >= w) & (lon <= e) & (lat >= s) & (lat <= n)


# --------------------------------------------------------------------------- FIRMS
def parse_firms_csv(text, source):
    """FIRMS area-API / archive CSV -> detection table. VIIRS rows carry
    scan/track of the 375 m I-band pixel (km), MODIS of the 1 km pixel."""
    if not text.strip() or text.lstrip().lower().startswith(('invalid', '<!doctype', '<html')):
        raise ValueError(f'FIRMS returned no CSV for {source}: {text[:120]!r}')
    df = pd.read_csv(io.StringIO(text))
    if df.empty:
        return _empty()
    need = {'latitude', 'longitude', 'frp', 'acq_date', 'acq_time'}
    if not need <= set(df.columns):
        raise ValueError(f'FIRMS CSV for {source} lacks {sorted(need - set(df.columns))}')
    hhmm = df['acq_time'].astype(int).astype(str).str.zfill(4)
    when = pd.to_datetime(df['acq_date'].astype(str) + ' ' + hhmm, format='%Y-%m-%d %H%M')
    plat = df['satellite'].astype(str) if 'satellite' in df else source
    sat_names = {'N': 'Suomi-NPP', 'N20': 'NOAA-20', '1': 'NOAA-20', 'N21': 'NOAA-21', '2': 'NOAA-21',
                 'T': 'Terra', 'A': 'Aqua', 'Terra': 'Terra', 'Aqua': 'Aqua'}
    area = (df['scan'] * df['track']).values if {'scan', 'track'} <= set(df.columns) else np.nan
    return _frame({'source': 'FIRMS ' + ('VIIRS' if 'VIIRS' in source else 'MODIS'),
                   'platform': [sat_names.get(str(p), str(p)) for p in plat],
                   'datetime_utc': when.dt.to_pydatetime(), 'lat': df['latitude'].values,
                   'lon': df['longitude'].values, 'frp_MW': df['frp'].values.astype(float),
                   'pixel_area_km2': area, 'confidence': df.get('confidence', pd.Series([None] * len(df))).astype(str).values,
                   'day_night': df.get('daynight', pd.Series([None] * len(df))).astype(str).values,
                   'file': source})


def fetch_firms(bbox, when, window_hours=6.0, map_key=None, sources=FIRMS_SOURCES, session=None):
    """FIRMS detections within `bbox` (west, south, east, north) and
    +/-window_hours of `when` (naive UTC). Standard-processing (_SP) sources
    are tried first; NRT ones fill in recent dates. Needs a free MAP_KEY
    (argument, FIRMS_MAP_KEY, or the key remembered by credentials.py;
    https://firms.modaps.eosdis.nasa.gov/api/map_key)."""
    import requests
    import credentials
    key = credentials.get('firms', map_key=map_key)['map_key']
    if not key:
        raise RuntimeError('FIRMS needs a free MAP_KEY: enter it once in the GUI (it is remembered), set '
                           'FIRMS_MAP_KEY, or pass map_key (https://firms.modaps.eosdis.nasa.gov/api/map_key).')
    lo, hi = when - _dt.timedelta(hours=window_hours), when + _dt.timedelta(hours=window_hours)
    days = (hi.date() - lo.date()).days + 1
    if days > 5:
        raise ValueError('FIRMS serves at most 5 days per request; use window_hours <= 48')
    area = '%.4f,%.4f,%.4f,%.4f' % tuple(bbox)
    s = session or requests.Session()
    frames, got = [], set()
    for src in sources:
        family = src.rsplit('_', 1)[0]                 # SP and NRT of one sensor: keep the first that has data
        if family in got:
            continue
        url = FIRMS_URL.format(key=key, source=src, area=area, days=days, date=lo.strftime('%Y-%m-%d'))
        try:
            r = s.get(url, timeout=120)
            r.raise_for_status()
            df = parse_firms_csv(r.text, src)
        except Exception as exc:
            warnings.warn(f'FIRMS {src}: {exc}')
            continue
        if len(df):
            df = df[(df['datetime_utc'] >= lo) & (df['datetime_utc'] <= hi)]
            if len(df):
                frames.append(df)
                got.add(family)
    return pd.concat(frames, ignore_index=True) if frames else _empty()


# --------------------------------------------------------------------------- GOES ABI FDC
def goes_satellite_for(lon, when):
    """The GOES that sees longitude `lon` best at `when` (East at 75.2 W,
    West at 137.2 W; GOES-19 replaced GOES-16 as East on 7 Apr 2025, GOES-18
    replaced GOES-17 as West on 4 Jan 2023). None outside both disks' usable
    part (more than ~65 deg of longitude from the sub-satellite point)."""
    east = 'goes19' if when >= _dt.datetime(2025, 4, 7) else 'goes16'
    west = 'goes18' if when >= _dt.datetime(2023, 1, 4) else 'goes17'
    d_e, d_w = abs(lon + 75.2), abs(lon + 137.2)
    if min(d_e, d_w) > 65:
        return None
    return east if d_e <= d_w else west


def _goes_lonlat(x, y, proj):
    """GOES-R fixed grid (scan angles x, y in rad, arrays of one shape) -> lon, lat
    (GOES-R Product User Guide, sec. 4.2.8). NaN off the Earth disk."""
    r_eq, r_pol = proj['semi_major_axis'], proj['semi_minor_axis']
    H = proj['perspective_point_height'] + r_eq
    lon0 = np.radians(proj['longitude_of_projection_origin'])
    X, Y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
    a = np.sin(X) ** 2 + np.cos(X) ** 2 * (np.cos(Y) ** 2 + (r_eq / r_pol) ** 2 * np.sin(Y) ** 2)
    b = -2 * H * np.cos(X) * np.cos(Y)
    c = H ** 2 - r_eq ** 2
    disc = b ** 2 - 4 * a * c
    with np.errstate(invalid='ignore'):
        rs = (-b - np.sqrt(disc)) / (2 * a)
        sx, sy, sz = rs * np.cos(X) * np.cos(Y), -rs * np.sin(X), rs * np.cos(X) * np.sin(Y)
        lat = np.degrees(np.arctan((r_eq / r_pol) ** 2 * sz / np.sqrt((H - sx) ** 2 + sy ** 2)))
        lon = np.degrees(lon0 - np.arctan(sy / (H - sx)))
    lat = np.where(disc < 0, np.nan, lat)
    lon = np.where(disc < 0, np.nan, lon)
    return lon, lat


def load_goes_fdc(path, bbox=None):
    """Fire pixels of one ABI-L2-FDC file (full disk FDCF, CONUS FDCC or
    mesoscale FDCM): FRP (MW), fire temperature (K) and area (m2) of the
    modified-Dozier retrieval where the product has them, pixel class from
    the Mask, restricted to `bbox` if given."""
    import netCDF4
    with netCDF4.Dataset(path) as d:
        mask = np.asarray(d['Mask'][:].filled(-99) if np.ma.isMaskedArray(d['Mask'][:]) else d['Mask'][:])
        fire = np.isin(mask, GOES_FIRE_CLASSES)
        if not fire.any():
            return _empty()
        iy, ix = np.nonzero(fire)
        proj = {k: float(d['goes_imager_projection'].getncattr(k)) for k in
                ('semi_major_axis', 'semi_minor_axis', 'perspective_point_height', 'longitude_of_projection_origin')}
        x, y = np.asarray(d['x'][:], float), np.asarray(d['y'][:], float)     # scale/offset applied by netCDF4
        lon, lat = _goes_lonlat(x[ix], y[iy], proj)

        def take(name):
            v = d[name][:]
            v = v.filled(np.nan) if np.ma.isMaskedArray(v) else np.asarray(v, float)
            return np.asarray(v, float)[iy, ix]
        power, temp, area = take('Power'), take('Temp'), take('Area')
        when = GOES_EPOCH + _dt.timedelta(seconds=float(d['t'][:]))
        if 'platform_ID' in d.ncattrs():
            sat = str(d.getncattr('platform_ID'))
        else:
            m = re.search(r'_(G\d\d)_', os.path.basename(path))
            sat = m.group(1) if m else 'GOES'
    keep = np.isfinite(lat) & (np.ones_like(lat, bool) if bbox is None else _in_bbox(lat, lon, bbox))
    codes = mask[iy, ix]
    return _frame({'source': 'GOES ABI FDC', 'platform': sat, 'datetime_utc': [when] * int(keep.sum()),
                   'lat': lat[keep], 'lon': lon[keep], 'frp_MW': power[keep], 'fire_T_K': temp[keep],
                   'fire_area_m2': area[keep], 'pixel_area_km2': 4.0,
                   'confidence': [str(c) for c in codes[keep]], 'file': os.path.basename(path)})


def _s3_keys(bucket, prefix, session):
    keys, token = [], None
    while True:
        params = {'list-type': '2', 'prefix': prefix}
        if token:
            params['continuation-token'] = token
        r = session.get(bucket + '/', params=params, timeout=60)
        r.raise_for_status()
        keys += re.findall(r'<Key>([^<]+)</Key>', r.text)
        m = re.search(r'<NextContinuationToken>([^<]+)</NextContinuationToken>', r.text)
        if not m:
            return keys
        token = m.group(1)


def fetch_goes_fdc(bbox, when, window_minutes=20.0, product='FDCF', satellite=None, cache_dir='.',
                   session=None, max_files=6):
    """Download (to cache_dir) and load the ABI-L2-FDC scans whose start is
    within +/-window_minutes of `when` (naive UTC), from NOAA's public AWS
    bucket - no login. Returns (detections, files). Outside both GOES disks
    (Europe, Africa, Asia) returns an empty table and says so."""
    import requests
    lon_c = 0.5 * (bbox[0] + bbox[2])
    sat = satellite or goes_satellite_for(lon_c, when)
    if sat is None:
        warnings.warn(f'GOES: longitude {lon_c:.1f} is outside the usable GOES disks; use LSA SAF '
                      '(SEVIRI/FCI) over Europe and Africa, or Himawari over Asia.')
        return _empty(), []
    s = session or requests.Session()
    bucket = GOES_BUCKET.format(sat=sat)
    lo, hi = when - _dt.timedelta(minutes=window_minutes), when + _dt.timedelta(minutes=window_minutes)
    files = []
    t = lo.replace(minute=0, second=0, microsecond=0)
    while t <= hi:
        prefix = f'ABI-L2-{product}/{t:%Y}/{t.timetuple().tm_yday:03d}/{t:%H}/'
        for k in _s3_keys(bucket, prefix, s):
            m = re.search(r'_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})', k)
            if not m:
                continue
            start = (_dt.datetime(int(m.group(1)), 1, 1) + _dt.timedelta(days=int(m.group(2)) - 1,
                     hours=int(m.group(3)), minutes=int(m.group(4)), seconds=int(m.group(5))))
            if lo <= start <= hi:
                files.append((abs((start - when).total_seconds()), k))
        t += _dt.timedelta(hours=1)
    files = [k for _, k in sorted(files)[:max_files]]
    os.makedirs(cache_dir, exist_ok=True)
    frames, paths = [], []
    for k in files:
        p = os.path.join(cache_dir, os.path.basename(k))
        if not os.path.exists(p):
            r = s.get(f'{bucket}/{k}', timeout=300)
            r.raise_for_status()
            with open(p + '.part', 'wb') as fh:
                fh.write(r.content)
            os.replace(p + '.part', p)
        paths.append(p)
        frames.append(load_goes_fdc(p, bbox))
    det = pd.concat(frames, ignore_index=True) if frames else _empty()
    return det, paths


# --------------------------------------------------------------------------- Sentinel-3 SLSTR FRP
def load_slstr_frp(sen3_dir, bbox=None):
    """SL_2_FRP___ product (S3MPC.ACR.FRP.003 i1r4): FRP_in.nc (MWIR, every
    fire) and, at night, FRP_an/bn.nc (SWIR). time is microseconds since
    2000-01-01; FRP in MW; IFOV_area in m2."""
    import netCDF4
    rows = []
    for name, frp_var, unc_var in (('FRP_in.nc', 'FRP_MWIR', 'FRP_uncertainty_MWIR'),
                                   ('FRP_an.nc', 'FRP_SWIR', 'FRP_uncertainty_SWIR'),
                                   ('FRP_bn.nc', 'FRP_SWIR', 'FRP_uncertainty_SWIR')):
        p = os.path.join(sen3_dir, name)
        if not os.path.exists(p):
            continue
        with netCDF4.Dataset(p) as d:
            if 'latitude' not in d.variables or d.dimensions.get('fires') is None or len(d.dimensions['fires']) == 0:
                continue
            lat, lon = np.asarray(d['latitude'][:], float), np.asarray(d['longitude'][:], float)
            frp = np.asarray(d[frp_var][:], float)
            unc = np.asarray(d[unc_var][:], float) if unc_var in d.variables else np.full(len(frp), np.nan)
            us = np.asarray(d['time'][:], np.int64)
            area = np.asarray(d['IFOV_area'][:], float) / 1e6 if 'IFOV_area' in d.variables else np.full(len(frp), np.nan)
            cls = np.asarray(d['classification'][:]) if 'classification' in d.variables else np.zeros(len(frp), int)
        when = [SLSTR_EPOCH + _dt.timedelta(microseconds=int(u)) for u in us]
        keep = np.isfinite(lat) & (np.ones_like(lat, bool) if bbox is None else _in_bbox(lat, lon, bbox))
        for i in np.flatnonzero(keep):
            rows.append({'source': 'SLSTR FRP ' + ('MWIR' if frp_var == 'FRP_MWIR' else 'SWIR'),
                         'platform': os.path.basename(os.path.normpath(sen3_dir))[:3], 'datetime_utc': when[i],
                         'lat': lat[i], 'lon': lon[i], 'frp_MW': frp[i], 'frp_unc_MW': unc[i],
                         'pixel_area_km2': area[i],
                         'confidence': 'vegetation_fire' if int(cls[i]) & 1 else f'class_{int(cls[i])}',
                         'file': os.path.basename(os.path.normpath(sen3_dir)) + '/' + name})
    return _frame(rows) if rows else _empty()


def fetch_slstr_frp(bbox, when, window_hours=6.0, cache_dir='.', token=None):
    """Search (no login) and download (Copernicus Data Space login,
    CDSE_USERNAME / CDSE_PASSWORD) the SL_2_FRP___ products within
    +/-window_hours, and load their detections in `bbox`."""
    import thermal_partners as tp
    recs = tp.search_slstr(bbox, when, window_hours=window_hours, product_types=('SL_2_FRP___',))
    # one product per overpass: the NT (non-time-critical) reprocessing if both exist
    by_pass = {}
    for r in recs:
        key = (r['platform'], r['datetime_utc'].replace(second=0, microsecond=0))
        if key not in by_pass or '_NT_' in r['granule']:
            by_pass[key] = r
    frames, paths = [], []
    for r in sorted(by_pass.values(), key=lambda r: abs((r['datetime_utc'] - when).total_seconds())):
        try:
            d = tp.download_slstr(r, cache_dir, token=token)
        except Exception as exc:
            warnings.warn(f'SLSTR FRP download failed for {r["granule"]}: {exc}')
            continue
        paths.append(d)
        frames.append(load_slstr_frp(d, bbox))
    det = pd.concat(frames, ignore_index=True) if frames else _empty()
    return det, paths


# --------------------------------------------------------------------------- LSA SAF (SEVIRI / FCI)
def load_lsasaf_frp(path, bbox=None):
    """LSA SAF FRP-PIXEL list product (MSG SEVIRI, HDF5) or the MTG-FCI FRP
    product (NetCDF). The datasets are located by name (LATITUDE/latitude,
    LONGITUDE/longitude, FRP, FRP_UNCERTAINTY), scaled by the HDF5
    attribute SCALING_FACTOR where present (LSA SAF convention: physical =
    stored / SCALING_FACTOR), and the result is checked for plausibility
    (latitudes and longitudes in range, FRP >= 0); a file that fails the
    check is rejected rather than read wrongly. NOTE: written from the
    product documentation and checked on synthetic files only - no LSA SAF
    file was available when it was written; the first real file is checked
    by the plausibility test. The slot time is taken from the file name
    (..._YYYYMMDDHHMM)."""
    m = re.search(r'(\d{12})(?!.*\d{12})', os.path.basename(path))
    when = _dt.datetime.strptime(m.group(1), '%Y%m%d%H%M') if m else None
    names = {'lat': ('LATITUDE', 'latitude', 'lat'), 'lon': ('LONGITUDE', 'longitude', 'lon'),
             'frp': ('FRP', 'frp'), 'unc': ('FRP_UNCERTAINTY', 'frp_uncertainty', 'FRP_UNC')}
    data = {}
    if path.lower().endswith(('.nc', '.nc4')):
        import netCDF4
        with netCDF4.Dataset(path) as d:
            for k, cands in names.items():
                v = next((d[c] for c in cands if c in d.variables), None)
                data[k] = None if v is None else np.asarray(v[:], float)
    else:
        import h5py
        with h5py.File(path, 'r') as h:
            flat = {}

            def _collect(name, obj):                 # must return None: h5py stops on any other value
                if isinstance(obj, h5py.Dataset):
                    flat.setdefault(name.split('/')[-1], obj)
            h.visititems(_collect)
            for k, cands in names.items():
                ds = next((flat[c] for c in cands if c in flat), None)
                if ds is None:
                    data[k] = None
                    continue
                v = np.asarray(ds[()], float)
                sf = ds.attrs.get('SCALING_FACTOR')
                off = ds.attrs.get('OFFSET')
                if off is not None and float(np.ravel(off)[0]) != 0.0:
                    raise ValueError(f'{path}: dataset {ds.name} has a non-zero OFFSET; its convention '
                                     'is not verified here - refusing to guess')
                if sf is not None and float(np.ravel(sf)[0]) not in (0.0, 1.0):
                    v = v / float(np.ravel(sf)[0])
                data[k] = v
    if data['lat'] is None or data['lon'] is None or data['frp'] is None:
        raise ValueError(f'{path}: latitude, longitude or FRP dataset not found')
    lat, lon, frp = (np.ravel(data[k]) for k in ('lat', 'lon', 'frp'))
    if not (np.nanmax(np.abs(lat)) <= 90 and np.nanmax(np.abs(lon)) <= 180 and np.nanmin(frp) >= 0):
        raise ValueError(f'{path}: implausible values (|lat| max {np.nanmax(np.abs(lat)):.3g}, |lon| max '
                         f'{np.nanmax(np.abs(lon)):.3g}, FRP min {np.nanmin(frp):.3g}) - format not as expected')
    unc = np.ravel(data['unc']) if data['unc'] is not None else np.full(len(frp), np.nan)
    keep = np.ones(len(lat), bool) if bbox is None else _in_bbox(lat, lon, bbox)
    src = 'LSA SAF FRP (MTG-FCI)' if ('MTG' in os.path.basename(path).upper() or 'FCI' in os.path.basename(path).upper()) \
        else 'LSA SAF FRP (MSG-SEVIRI)'
    return _frame({'source': src, 'platform': src.split('(')[1].rstrip(')'),
                   'datetime_utc': [when] * int(keep.sum()), 'lat': lat[keep], 'lon': lon[keep],
                   'frp_MW': frp[keep], 'frp_unc_MW': unc[keep],
                   'pixel_area_km2': 1.0 if 'FCI' in src else 9.0, 'file': os.path.basename(path)})


LSASAF_URL = 'https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/FRP-PIXEL/HDF5/{d:%Y/%m/%d}/'


def fetch_lsasaf_frp(bbox, when, window_minutes=20.0, cache_dir='.', username=None, password=None,
                     session=None):
    """MSG-SEVIRI FRP-PIXEL list products (every 15 min, Europe/Africa) within
    +/-window_minutes of `when`, from the LSA SAF data server. The directory
    listing is public; the files need an LSA SAF login (HTTP Basic: argument,
    LSASAF_USERNAME/LSASAF_PASSWORD, or the login remembered by
    credentials.py). Returns (detections in bbox, paths)."""
    import bz2
    import requests
    import credentials
    c = credentials.get('lsasaf', username=username, password=password)
    if not (c['username'] and c['password']):
        raise RuntimeError('LSA SAF needs a login: sign in once in the GUI (it is remembered) or set '
                           'LSASAF_USERNAME / LSASAF_PASSWORD (register at https://lsa-saf.eumetsat.int).')
    s = session or requests.Session()
    lo, hi = when - _dt.timedelta(minutes=window_minutes), when + _dt.timedelta(minutes=window_minutes)
    urls = []
    for day in sorted({lo.date(), hi.date()}):
        base = LSASAF_URL.format(d=day)
        r = s.get(base, timeout=60)
        r.raise_for_status()
        for name in sorted(set(re.findall(r'(HDF5_LSASAF_MSG_FRP-PIXEL-ListProduct_[A-Za-z-]+_(\d{12})[^"/<]*)',
                                          r.text))):
            t = _dt.datetime.strptime(name[1], '%Y%m%d%H%M')
            if lo <= t <= hi:
                urls.append((abs((t - when).total_seconds()), base + name[0]))
    os.makedirs(cache_dir, exist_ok=True)
    frames, paths = [], []
    for _, url in sorted(urls):
        out = os.path.join(cache_dir, os.path.basename(url).replace('.bz2', ''))
        if not os.path.exists(out):
            r = s.get(url, auth=(c['username'], c['password']), timeout=300)
            if r.status_code == 401:
                raise RuntimeError('LSA SAF rejected the login (HTTP 401).')
            r.raise_for_status()
            blob = r.content
            if blob[:3] == b'BZh':
                blob = bz2.decompress(blob)
            if blob[:8] != b'\x89HDF\r\n\x1a\n':
                raise ValueError(f'{os.path.basename(url)}: not an HDF5 file after download '
                                 f'(starts {blob[:8]!r}) - the server may have returned a login page')
            with open(out, 'wb') as fh:
                fh.write(blob)
        paths.append(out)
        frames.append(load_lsasaf_frp(out, bbox))
    det = pd.concat(frames, ignore_index=True) if frames else _empty()
    return det, paths


def load_partner_file(path, bbox=None):
    """Dispatch a user-supplied product file or directory to its loader."""
    b = os.path.basename(os.path.normpath(path)).upper()
    if b.endswith('.SEN3') and 'FRP' in b:
        return load_slstr_frp(path, bbox)
    if 'ABI-L2-FDC' in b:
        return load_goes_fdc(path, bbox)
    if 'LSASAF' in b or 'FRP-PIXEL' in b or ('FRP' in b and ('MTG' in b or 'MSG' in b)):
        return load_lsasaf_frp(path, bbox)
    if b.endswith('.CSV'):
        with open(path, encoding='utf-8') as fh:
            return parse_firms_csv(fh.read(), os.path.basename(path))
    raise ValueError(f'unrecognised active-fire product: {path}')


# --------------------------------------------------------------------------- comparison
def enmap_frp_table(df, lat, lon, emissivity=1.0, pixel_area_m2=900.0):
    """Per-pixel EnMAP FRP from this work's results frame: eps sigma T^4 f A
    where the retrieval is usable (fit 'ok', interval not open at both ends),
    and the model-free saturated lower bound (FRP_min_MW_saturated) where it
    is not. lat/lon: per-pixel coordinates (aligned with df rows)."""
    sigma = 5.670374419e-8
    T = df['T_K'].values.astype(float)
    f = df['fire_fraction'].values.astype(float)
    q = df['fit_quality'].astype(str).values
    unc = (df['T_lower_unbounded'].astype(bool).values & df['T_upper_unbounded'].astype(bool).values
           if {'T_lower_unbounded', 'T_upper_unbounded'} <= set(df.columns) else np.zeros(len(df), bool))
    ok = (q == 'ok') & np.isfinite(T) & np.isfinite(f) & ~unc
    frp = np.where(ok, emissivity * sigma * T ** 4 * f * pixel_area_m2 / 1e6, 0.0)
    lb = df['FRP_min_MW_saturated'].values.astype(float) if 'FRP_min_MW_saturated' in df else np.full(len(df), np.nan)
    bound = np.where(~ok & np.isfinite(lb), lb, 0.0)
    return pd.DataFrame({'lat': lat, 'lon': lon, 'frp_retrieved_MW': frp, 'frp_bound_MW': bound,
                         'retrieved': ok, 'T_K': np.where(ok, T, np.nan)})


def compare_frp(enmap, partners, enmap_time, tolerance_km=1.0):
    """One row per partner product slot (source, platform, time): partner
    detections whose footprint (a square of pixel_area_km2 around the
    detection, widened by `tolerance_km` on every side for the geolocation
    error of the two products - SLSTR/VIIRS/MODIS ~0.3-1 km, EnMAP L1B
    geolocation of the same order) contains EnMAP fire pixels, their FRP
    total, and the EnMAP FRP inside those footprints - retrieved, and
    retrieved + saturated lower bounds - as well as the partner FRP total
    over the whole search box (the EnMAP fire extent plus a margin). Plus a
    per-detection pairing table for scatter plots."""
    if partners is None or not len(partners) or enmap is None or not len(enmap):
        return pd.DataFrame(), pd.DataFrame()
    km_lat = 111.32
    rows, pairs = [], []
    # one row per product file (an overpass or a geostationary slot); swath
    # products stamp each detection with its own scan time, so the slot time
    # is the median over the file's detections
    for (src, plat, _file), g in partners.groupby(['source', 'platform', 'file'], dropna=False):
        t = pd.Series(pd.to_datetime(g['datetime_utc'])).median() if g['datetime_utc'].notna().any() else None
        t = None if t is None or pd.isna(t) else t.to_pydatetime()
        e_ret = e_lb = 0.0
        n_det, frp_p, frp_unc2, temps = 0, 0.0, 0.0, []
        used = np.zeros(len(enmap), bool)
        for _, r in g.iterrows():
            half = 0.5 * np.sqrt(r['pixel_area_km2'] if np.isfinite(r['pixel_area_km2']) else 1.0) + tolerance_km
            dlat = half / km_lat
            dlon = half / (km_lat * max(np.cos(np.radians(r['lat'])), 1e-3))
            inside = ((np.abs(enmap['lat'].values - r['lat']) <= dlat)
                      & (np.abs(enmap['lon'].values - r['lon']) <= dlon))
            if not inside.any():
                continue
            n_det += 1
            frp_p += r['frp_MW'] if np.isfinite(r['frp_MW']) else 0.0
            frp_unc2 += r['frp_unc_MW'] ** 2 if np.isfinite(r['frp_unc_MW']) else 0.0
            if np.isfinite(r['fire_T_K']):
                temps.append(r['fire_T_K'])
            new = inside & ~used
            used |= inside
            er = float(enmap['frp_retrieved_MW'].values[inside].sum())
            eb = float(enmap['frp_bound_MW'].values[inside].sum())
            e_ret += float(enmap['frp_retrieved_MW'].values[new].sum())
            e_lb += float(enmap['frp_bound_MW'].values[new].sum())
            Tw = enmap['T_K'].values[inside]
            wts = enmap['frp_retrieved_MW'].values[inside]
            pairs.append({'source': src, 'platform': plat, 'datetime_utc': t, 'lat': r['lat'], 'lon': r['lon'],
                          'partner_frp_MW': r['frp_MW'], 'partner_fire_T_K': r['fire_T_K'],
                          'enmap_frp_retrieved_MW': er, 'enmap_frp_lower_bound_MW': er + eb,
                          'enmap_n_pixels': int(inside.sum()),
                          'enmap_T_frp_weighted_K': (float(np.nansum(Tw * wts) / np.nansum(wts))
                                                     if np.nansum(wts) > 0 else np.nan)})
        dt_min = (pd.Timestamp(t) - pd.Timestamp(enmap_time)).total_seconds() / 60.0 if t is not None else np.nan
        rows.append({'source': src, 'platform': plat, 'datetime_utc': t, 'dt_minutes': dt_min,
                     'n_detections_in_bbox': len(g), 'partner_frp_in_bbox_MW': float(np.nansum(g['frp_MW'].values)),
                     'n_detections_on_enmap_fire': n_det,
                     'partner_frp_MW': frp_p, 'partner_frp_unc_MW': np.sqrt(frp_unc2) if frp_unc2 else np.nan,
                     'enmap_frp_retrieved_MW': e_ret, 'enmap_frp_with_bounds_MW': e_ret + e_lb,
                     'enmap_frp_retrieved_scene_MW': float(enmap['frp_retrieved_MW'].sum()),
                     'enmap_frp_with_bounds_scene_MW': float(enmap['frp_retrieved_MW'].sum() + enmap['frp_bound_MW'].sum()),
                     'partner_fire_T_median_K': float(np.median(temps)) if temps else np.nan})
    summ = pd.DataFrame(rows).sort_values('dt_minutes', key=lambda s: s.abs()) if rows else pd.DataFrame()
    return summ, pd.DataFrame(pairs)


def gather_fire_partners(bbox, when, sources=('FIRMS', 'GOES', 'SLSTR'), files=(), cache_dir='.',
                         firms_map_key=None, window_hours=6.0, geo_window_minutes=20.0, verbose=True):
    """Every requested source, with failures reported, not raised. Returns
    (detections, log) where log is a list of (source, message)."""
    frames, log = [], []
    def note(src, msg):
        log.append((src, msg))
        if verbose:
            print(f'  [{src}] {msg}')
    if 'FIRMS' in sources:
        try:
            d = fetch_firms(bbox, when, window_hours=min(window_hours, 48.0), map_key=firms_map_key)
            frames.append(d); note('FIRMS', f'{len(d)} VIIRS/MODIS detections within +/-{window_hours:g} h')
        except Exception as exc:
            note('FIRMS', f'not used: {exc}')
    if 'GOES' in sources:
        try:
            # all sources share cache_dir (the scene's downloads folder); the product
            # filenames already name the sensor, so no per-source subfolders
            d, paths = fetch_goes_fdc(bbox, when, window_minutes=geo_window_minutes,
                                      cache_dir=cache_dir)
            frames.append(d)
            note('GOES', f'{len(d)} fire pixels in {len(paths)} scans within +/-{geo_window_minutes:g} min'
                 if paths else 'no scan (outside the GOES disks or no data)')
        except Exception as exc:
            note('GOES', f'not used: {exc}')
    if 'SLSTR' in sources:
        try:
            d, paths = fetch_slstr_frp(bbox, when, window_hours=window_hours,
                                       cache_dir=cache_dir)
            frames.append(d); note('SLSTR', f'{len(d)} detections in {len(paths)} products within +/-{window_hours:g} h')
        except Exception as exc:
            note('SLSTR', f'not used: {exc}')
    if 'LSASAF' in sources:
        try:
            d, paths = fetch_lsasaf_frp(bbox, when, window_minutes=geo_window_minutes,
                                        cache_dir=cache_dir)
            frames.append(d)
            note('LSASAF', f'{len(d)} SEVIRI detections in {len(paths)} slots within +/-{geo_window_minutes:g} min'
                 if paths else 'no slot in the window')
        except Exception as exc:
            note('LSASAF', f'not used: {exc}')
    for p in files or ():
        try:
            d = load_partner_file(p, bbox)
            frames.append(d); note('file', f'{len(d)} detections from {os.path.basename(os.path.normpath(p))}')
        except Exception as exc:
            note('file', f'{os.path.basename(os.path.normpath(p))} not used: {exc}')
    frames = [f for f in frames if f is not None and len(f)]
    return (pd.concat(frames, ignore_index=True) if frames else _empty()), log

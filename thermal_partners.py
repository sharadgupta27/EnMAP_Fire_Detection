#!/usr/bin/env python
# coding: utf-8
"""Thermal companion sensors for EnMAP fire-temperature retrieval.

WHY THIS MODULE EXISTS
----------------------
Fire_analysis_with_EnMAP.py already pairs an EnMAP scene with ECOSTRESS
(ECO_L2T_LSTE) and Landsat 8/9 Collection-2 surface temperature. Those two
are the partners used by Amici et al. (2022) for PRISMA, and they remain
the best-matched pair for EnMAP. They are not always available:

  * ECOSTRESS flies on the ISS between 52 deg N and 52 deg S. Above that
    latitude - boreal Canada, Scandinavia, most of Russia, i.e. a large
    share of the world's burned area - it does not exist at all, and its
    ISS orbit gives an irregular revisit that frequently exceeds a day even
    inside the covered belt.
  * Landsat 8/9 give an 8-day combined revisit, so the nearest scene to an
    arbitrary EnMAP overpass is on average ~2 days away.

This module adds the sensors that fill those gaps, with one rule
throughout: every access route here was exercised against the real archive
before it was written down. Where a route could not be verified it is NOT
implemented - see UNIMPLEMENTED below - because a loader written from a
format specification that was never opened is a silent-error generator,
not a feature.

WHAT IS HERE
------------
  VIIRS LST      VNP21A1D/N (Suomi-NPP), VJ121A1D/N (NOAA-20),
                 VJ221A1D/N (NOAA-21). 1 km, TES retrieval - the same
                 temperature/emissivity separation family as ECOSTRESS, so
                 the retrieval physics is comparable rather than merely
                 co-located. Three spacecraft, day and night, from 2012
                 (SNPP), 2018 (NOAA-20) and 2023 (NOAA-21): there is a
                 VIIRS overpass within a few hours of any EnMAP scene
                 anywhere on Earth, including above 52 deg N.

  VIIRS fire     VNP14IMG, 375 m active fire / FRP, a swath product whose
                 granule name carries the true overpass time in minutes.

  Landsat ST     Collection-2 Level-2 ST via the USGS LandsatLook STAC.

  ECOSTRESS      ECO_L2T_LSTE v002 and v003 via NASA CMR. Both versions are
                 searched: they coexist and do not cover identical date
                 ranges, so a miss in one is not evidence of no coverage.

  Sentinel-3     SLSTR SL_2_LST___ and SL_2_FRP___ via the Copernicus Data
                 Space OData catalogue. 1 km, dual-view, <1 day revisit
                 from S3A/S3B (S3C launched 2025), and the 3.7 um F1/F2
                 fire channels that no sensor above carries.

THE TIME-MATCHING TRAP THIS MODULE EXISTS TO AVOID
--------------------------------------------------
The gridded daily VIIRS LST products (VNP21A1D and its siblings) are
labelled with a DATE, not an overpass time: CMR reports their temporal
extent as 00:00:00 to 23:59:59 of that day. Matching on that label is
meaningless - it would call every granule a perfect match.

The actual observation time is carried per pixel in the product's own
`View_Time` layer, in hours of LOCAL SOLAR time. This module reads it and
converts it back to UTC using each pixel's own longitude
(utc = view_time - lon/15), so the reported offset from the EnMAP overpass
is the real one.

That distinction is not academic. For the scene shipped with this
repository (EnMAP DT0000099384, 2024-10-19T19:06:13Z over the Mojave), the
granule's own metadata lists five separate overpass windows that day,
including one at 19:18-19:24 UTC - 12 minutes from EnMAP. The composite did
not use it: View_Time over the scene is 13.2 h local solar, i.e. ~21:02
UTC, nearly two hours later and much closer to the afternoon temperature
peak. Reporting that granule as a 12-minute match would have been wrong by
a factor of ten, and in the direction that matters most, since land surface
temperature changes fastest either side of local noon.

UNIMPLEMENTED, DELIBERATELY
---------------------------
  SDGSAT-1 TIS (30 m, 3 TIR bands, day and night) is on paper the best
  spatial match to EnMAP's 30 m grid of any thermal sensor flying. It is
  not implemented because it has no reachable machine interface: the CAS
  distribution portals (sdg.casearth.cn and the international mirror) both
  failed to connect from this machine - TLS failure and connection timeout
  respectively - and there is no STAC, OData or CMR route to the archive.
  SDGSAT-1 scenes must be ordered by hand through the portal web UI after
  registering. load_generic_lst_geotiff() below will read the LST GeoTIFF
  once you have exported one, so the rest of the pipeline is ready for it;
  only search and download are missing, and they are missing because they
  cannot be built, not because they were skipped.

  constellr (SkyBee-1/-2, 30 m native LST, <2 K, launched Jan and Jun 2025)
  is commercial. Research access runs through an ESA Third Party Missions
  proposal, which is a paperwork route, not an API. Nothing to implement.

  NASA FIRMS active-fire CSV would be a natural addition, but it requires a
  per-user MAP_KEY that this machine does not have, so the client could not
  be exercised against the live service. Not shipped rather than shipped
  untested.

Everything returned by the loaders here uses the dict contract the rest of
the pipeline already speaks - {'lst_K', 'geotransform', 'projection_wkt'} -
so a VIIRS result drops straight into validate_temperature() and
_sample_raster_at_lonlat() with no adaptation.

Self-test:
    python algorithm_verification/test_thermal_partners.py
"""

import datetime as _dt
import math
import netrc
import os
import re
import sys
import warnings

import numpy as np
import requests

# solar_time.py sits beside this file. Importing it by name works whenever
# this directory is on sys.path, but the verification scripts load this module
# by PATH (importlib.util.spec_from_file_location), which does not put its
# directory on sys.path - so bootstrap it rather than making how this file was
# loaded decide whether it imports.
try:
    from solar_time import hours_of_day, solar_hour_from_utc
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from solar_time import hours_of_day, solar_hour_from_utc


__all__ = [
    'enmap_scene_reference',
    'search_ecostress_lste', 'search_viirs_lst', 'search_viirs_fire',
    'search_landsat_st', 'search_slstr',
    'match_thermal_partners',
    'download_earthdata_granule', 'download_viirs_lst',
    'cdse_access_token', 'download_slstr',
    'load_viirs_lst', 'load_generic_lst_geotiff',
    'local_solar_hour', 'VIIRS_LST_PRODUCTS', 'VIIRS_LST_CEILING_K',
]

# --------------------------------------------------------------------------
# service endpoints - every one exercised against the live service during
# development
# --------------------------------------------------------------------------

CMR_GRANULES = 'https://cmr.earthdata.nasa.gov/search/granules.umm_json'
LANDSATLOOK_STAC = 'https://landsatlook.usgs.gov/stac-server/search'
CDSE_ODATA = 'https://catalogue.dataspace.copernicus.eu/odata/v1/Products'
CDSE_ZIPPER = 'https://zipper.dataspace.copernicus.eu/odata/v1/Products(%s)/$value'
CDSE_TOKEN_URL = ('https://identity.dataspace.copernicus.eu/auth/realms/CDSE'
                  '/protocol/openid-connect/token')
URS_HOST = 'urs.earthdata.nasa.gov'

_USER_AGENT = 'enmap-thermal-partners/1.0 (EnMAP fire temperature retrieval)'

# VIIRS land-surface-temperature products, all Collection 002 on LP DAAC.
#
# The three spacecraft matter for a reason beyond redundancy: their equator
# crossings are staggered (SNPP and NOAA-20 half an orbit apart in the same
# 13:30 plane, NOAA-21 joining in 2023), so together they give four LST
# observations per day at any point - two day, two night - which is what
# makes a VIIRS partner findable within a few hours of an arbitrary EnMAP
# overpass when ECOSTRESS is not.
VIIRS_LST_PRODUCTS = {
    'VNP21A1D': {'platform': 'Suomi-NPP', 'day_night': 'day',   'since': '2012-01-19'},
    'VNP21A1N': {'platform': 'Suomi-NPP', 'day_night': 'night', 'since': '2012-01-19'},
    'VJ121A1D': {'platform': 'NOAA-20',   'day_night': 'day',   'since': '2018-01-05'},
    'VJ121A1N': {'platform': 'NOAA-20',   'day_night': 'night', 'since': '2018-01-05'},
    'VJ221A1D': {'platform': 'NOAA-21',   'day_night': 'day',   'since': '2023-02-10'},
    'VJ221A1N': {'platform': 'NOAA-21',   'day_night': 'night', 'since': '2023-02-10'},
}

# The MODIS/VIIRS sinusoidal grid. The tile corners are read from each
# granule's own StructMetadata.0 rather than hard-coded (see
# load_viirs_lst()), but the sphere radius is fixed by the product
# specification and is needed to build the CRS string.
SINUSOIDAL_PROJ4 = ('+proj=sinu +lon_0=0 +x_0=0 +y_0=0 '
                    '+a=6371007.181 +b=6371007.181 +units=m +no_defs')

# Upper end of the VIIRS TES land-surface-temperature retrieval. The
# product's own valid_range tops out at 65535 DN = 1310 K, which is an
# encoding limit, not a physical one: VNP21 solves temperature/emissivity
# separation on M14/M15/M16 (8.6/10.8/12.0 um), channels designed for
# ambient land surfaces that depart from linearity well below a flaming
# front. Used exactly as _ECOSTRESS_CEILING_K is used in the main module -
# to FLAG a comparison pixel as saturated, never to mask data.
VIIRS_LST_CEILING_K = 380.0


def _session():
    s = requests.Session()
    s.headers['User-Agent'] = _USER_AGENT
    return s


def _as_naive_utc(value):
    """Return `value` as a timezone-naive UTC datetime.

    The catalogues disagree about timezones - CMR and CDSE both return
    tz-aware UTC strings, while EnMAP's own acquisition time arrives naive
    from _scene_acquisition_datetime(). Subtracting one from the other
    raises TypeError, so every timestamp in this module is normalised here
    on the way in and compared naive throughout."""
    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip().replace('Z', '+00:00')
        try:
            value = _dt.datetime.fromisoformat(text)
        except ValueError:
            m = re.search(r'(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2}:\d{2})', value)
            if not m:
                return None
            value = _dt.datetime.strptime(m.group(1) + 'T' + m.group(2),
                                          '%Y-%m-%dT%H:%M:%S')
    if getattr(value, 'tzinfo', None) is not None:
        value = value.astimezone(_dt.timezone.utc).replace(tzinfo=None)
    return value


def _iso(dt):
    return _as_naive_utc(dt).strftime('%Y-%m-%dT%H:%M:%SZ')


def local_solar_hour(dt_utc, lon_deg):
    """Mean local solar time, in hours, at longitude `lon_deg`.

    Mean rather than apparent: the equation of time reaches ~16 minutes,
    which matters for sun-angle work but not for the day/night and
    hours-apart judgements made here, and pulling in a solar ephemeris to
    shave 16 minutes off a number that is only ever printed would be false
    precision."""
    # Arithmetic shared with Fire_analysis_with_EnMAP._local_solar_hour() via
    # solar_time.py; normalising whatever the catalogues returned is what
    # distinguishes this variant.
    dt_utc = _as_naive_utc(dt_utc)
    return solar_hour_from_utc(hours_of_day(dt_utc), lon_deg)


# --------------------------------------------------------------------------
# EnMAP scene reference
# --------------------------------------------------------------------------

def enmap_scene_reference(path, acquisition_datetime=None):
    """Footprint and acquisition time of an EnMAP scene, as the searches
    below need them.

    `path` may be the METADATA.XML itself, or any product file sitting next
    to one (the SPECTRAL_IMAGE .TIF/.dat), or a directory containing one.

    Returns {'scene_id', 'datetime_utc', 'bbox', 'polygon', 'centre'} where
    bbox is (west, south, east, north) and polygon is the scene's own
    bounding polygon as [(lon, lat), ...], closed.

    The polygon is kept alongside the bbox because an EnMAP footprint is a
    tilted quadrilateral, not a north-up rectangle: its bounding box can be
    ~30% larger in area than the scene. The catalogue searches all use the
    bbox (every service accepts one, and winding-order rules for polygon
    queries differ between CMR and STAC in ways that fail silently), but
    footprint_overlap_fraction() uses the true polygon so a granule that
    only clips the bbox corner is not reported as covering the scene."""
    xml_path = _resolve_enmap_metadata(path)
    root = _parse_xml(xml_path)

    points = []
    for el in root.iter():
        if el.tag.rsplit('}', 1)[-1] != 'point':
            continue
        lat = lon = None
        for child in el:
            tag = child.tag.rsplit('}', 1)[-1].lower()
            if tag == 'latitude':
                lat = float(child.text)
            elif tag == 'longitude':
                lon = float(child.text)
        if lat is not None and lon is not None:
            points.append((lon, lat))
        # EnMAP writes the footprint ring first (closed: last point repeats
        # the first) and then a single centre point. Stop at closure so the
        # centre is not appended as a fifth vertex, which would fold the
        # polygon in on itself and make any area/overlap test nonsense.
        if len(points) > 2 and points[-1] == points[0]:
            break

    if len(points) < 4:
        raise ValueError('no usable boundingPolygon found in %s' % xml_path)
    if points[-1] != points[0]:
        points.append(points[0])

    lons = [p[0] for p in points]
    lats = [p[1] for p in points]
    bbox = (min(lons), min(lats), max(lons), max(lats))

    dt = _as_naive_utc(acquisition_datetime) if acquisition_datetime is not None else None
    if dt is None:
        dt = _enmap_datetime(root, xml_path)
    if dt is None:
        raise ValueError(
            'could not determine the acquisition time of %s - pass '
            'acquisition_datetime= explicitly' % xml_path)

    scene_id = os.path.basename(xml_path)
    m = re.search(r'(DT\d+_\d{8}T\d{6}Z_\d+)', scene_id)
    if m:
        scene_id = m.group(1)

    return {
        'scene_id': scene_id,
        'datetime_utc': dt,
        'bbox': bbox,
        'polygon': points,
        'centre': (0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])),
        'metadata_xml': xml_path,
    }


def _parse_xml(path):
    import xml.etree.ElementTree as ET
    return ET.parse(path).getroot()


def _resolve_enmap_metadata(path):
    """Locate the EnMAP METADATA.XML for `path`.

    Reuses the main module's _find_enmap_metadata_xml() when `path` is a
    product raster, so the two agree about which sidecar belongs to which
    cube rather than each having its own idea."""
    path = str(path)
    if os.path.isfile(path) and path.upper().endswith('METADATA.XML'):
        return path
    if os.path.isdir(path):
        for root_dir, _dirs, files in os.walk(path):
            for name in sorted(files):
                if name.upper().endswith('METADATA.XML'):
                    return os.path.join(root_dir, name)
        raise FileNotFoundError('no METADATA.XML under %s' % path)
    try:
        from Fire_analysis_with_EnMAP import _find_enmap_metadata_xml
    except ImportError:
        _find_enmap_metadata_xml = None
    if _find_enmap_metadata_xml is not None:
        found = _find_enmap_metadata_xml(path)
        if found:
            return found
    sibling = os.path.join(os.path.dirname(path) or '.', 'METADATA.XML')
    if os.path.isfile(sibling):
        return sibling
    raise FileNotFoundError('no EnMAP METADATA.XML found for %s' % path)


def _enmap_datetime(root, xml_path):
    """Scene start time from the parsed metadata, falling back to the main
    module's filename-convention parser."""
    for el in root.iter():
        if el.tag.rsplit('}', 1)[-1] == 'startTime' and el.text:
            dt = _as_naive_utc(el.text.strip())
            if dt is not None:
                return dt
    try:
        from Fire_analysis_with_EnMAP import _scene_acquisition_datetime
    except ImportError:
        return None
    return _as_naive_utc(_scene_acquisition_datetime(xml_path))


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------

def _polygon_area(points, lat0_deg=None):
    """Shoelace area of a closed lon/lat ring, with longitudes scaled by
    cos(lat0_deg) so the result is proportional to area on the ground
    rather than to raw degrees.

    `lat0_deg` MUST be supplied by the caller when two areas are going to
    be divided by one another, and it must be the same value for both.
    Letting each polygon pick its own mean latitude looks harmless and is
    not: a clipped sub-polygon generally has a different mean latitude from
    the polygon it was clipped from, so the two scale factors no longer
    cancel and the ratio comes out slightly wrong. The self-test caught
    exactly that - a quarter-cover case returned 0.25000001 instead of
    0.25. Small here, but it is a bias, not noise, and it grows with
    latitude."""
    if len(points) < 4:
        return 0.0
    if lat0_deg is None:
        lat0_deg = sum(p[1] for p in points) / len(points)
    k = math.cos(math.radians(lat0_deg))
    total = 0.0
    for (x1, y1), (x2, y2) in zip(points[:-1], points[1:]):
        total += (x1 * k) * y2 - (x2 * k) * y1
    return abs(total) / 2.0


def _clip_polygon_to_bbox(points, bbox):
    """Sutherland-Hodgman clip of a closed ring against an axis-aligned
    box. Used to measure how much of the EnMAP footprint a candidate
    granule's bbox actually covers."""
    west, south, east, north = bbox
    edges = (('x>=', west), ('x<=', east), ('y>=', south), ('y<=', north))
    poly = list(points[:-1]) if points[0] == points[-1] else list(points)

    def inside(pt, edge):
        op, val = edge
        if op == 'x>=':
            return pt[0] >= val
        if op == 'x<=':
            return pt[0] <= val
        if op == 'y>=':
            return pt[1] >= val
        return pt[1] <= val

    def intersect(p, q, edge):
        op, val = edge
        px, py = p
        qx, qy = q
        if op in ('x>=', 'x<='):
            if qx == px:
                return (val, py)
            t = (val - px) / (qx - px)
            return (val, py + t * (qy - py))
        if qy == py:
            return (px, val)
        t = (val - py) / (qy - py)
        return (px + t * (qx - px), val)

    for edge in edges:
        if not poly:
            return []
        out = []
        for i, cur in enumerate(poly):
            prev = poly[i - 1]
            cur_in, prev_in = inside(cur, edge), inside(prev, edge)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur, edge))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur, edge))
        poly = out
    if poly:
        poly.append(poly[0])
    return poly


def footprint_overlap_fraction(scene_polygon, granule_bbox):
    """Fraction of the EnMAP footprint covered by a granule's bounding box,
    in [0, 1].

    This is what separates a granule that genuinely covers the scene from
    one that clips a corner of its bounding box. wildfire_4sensor.py in
    candidate_sites/ learned the same lesson the expensive way: a granule
    can match on time and place in the catalogue and still contain nothing
    useful over the tile that has the fire in it."""
    # One latitude reference for both areas, so the cos(lat) factor cancels
    # exactly - see _polygon_area().
    lat0 = sum(p[1] for p in scene_polygon) / len(scene_polygon)
    scene_area = _polygon_area(scene_polygon, lat0)
    if scene_area <= 0:
        return 0.0
    clipped = _clip_polygon_to_bbox(scene_polygon, granule_bbox)
    return min(1.0, _polygon_area(clipped, lat0) / scene_area) if clipped else 0.0


# --------------------------------------------------------------------------
# catalogue search
#
# Every search returns a list of dicts sharing one schema, so
# match_thermal_partners() can rank them together:
#
#   sensor          'ECOSTRESS' | 'VIIRS-LST' | 'VIIRS-FIRE' | 'Landsat' | 'SLSTR'
#   product         short name / product type as the archive calls it
#   platform        spacecraft
#   granule         granule or scene identifier
#   datetime_utc    naive UTC datetime, or None when the archive only
#                   labels the product with a date (see time_is_nominal)
#   time_is_nominal True when datetime_utc is a placeholder rather than an
#                   overpass time. Ranking must not trust it.
#   bbox            (west, south, east, north) or None when not published
#   overlap         fraction of the EnMAP footprint covered, or None
#   cloud_pct       scene cloud cover where the archive publishes one
#   urls            list of asset URLs
#   archive         which service answered
# --------------------------------------------------------------------------

def _granule_record(**kw):
    rec = {
        'sensor': None, 'product': None, 'platform': None, 'granule': None,
        'datetime_utc': None, 'time_is_nominal': False, 'bbox': None,
        'overlap': None, 'cloud_pct': None, 'urls': [], 'archive': None,
    }
    rec.update(kw)
    return rec


def _cmr_bbox_search(short_name, version, bbox, start, end, page_size=200,
                     session=None, timeout=120):
    """One CMR granule query, returning the raw umm_json items.

    Deliberately unauthenticated: CMR search is open, and attaching
    Earthdata basic-auth credentials to it makes the request fail rather
    than succeed (confirmed - the search returns an error payload with no
    'items' key). Only the subsequent asset download is authenticated."""
    s = session or _session()
    params = {
        'short_name': short_name,
        'bounding_box': '%.6f,%.6f,%.6f,%.6f' % tuple(bbox),
        'temporal': '%s,%s' % (_iso(start), _iso(end)),
        'page_size': page_size,
    }
    if version:
        params['version'] = version
    try:
        r = s.get(CMR_GRANULES, params=params, timeout=timeout)
    except requests.RequestException as exc:
        warnings.warn('CMR unreachable for %s: %s' % (short_name, exc))
        return []
    if not r.ok:
        warnings.warn('CMR query for %s v%s failed: HTTP %s'
                      % (short_name, version, r.status_code))
        return []
    try:
        return r.json().get('items', [])
    except ValueError:
        warnings.warn('CMR returned non-JSON for %s v%s' % (short_name, version))
        return []


def _cmr_bbox_of(umm):
    """Bounding box of a CMR granule, from whichever geometry it publishes.

    Handles the two shapes LP DAAC actually uses - an explicit
    BoundingRectangle (ECOSTRESS tiles, VIIRS grids) and a GPolygon point
    list (swath products such as VNP14IMG) - and returns None rather than
    guessing when neither is present, so a granule with no published
    geometry is reported as unknown-overlap instead of silently scoring
    zero and being dropped."""
    try:
        geom = umm['SpatialExtent']['HorizontalSpatialDomain']['Geometry']
    except (KeyError, TypeError):
        return None
    rects = geom.get('BoundingRectangles') or []
    if rects:
        r = rects[0]
        return (float(r['WestBoundingCoordinate']), float(r['SouthBoundingCoordinate']),
                float(r['EastBoundingCoordinate']), float(r['NorthBoundingCoordinate']))
    polys = geom.get('GPolygons') or []
    if polys:
        pts = polys[0].get('Boundary', {}).get('Points') or []
        if pts:
            lons = [float(p['Longitude']) for p in pts]
            lats = [float(p['Latitude']) for p in pts]
            return (min(lons), min(lats), max(lons), max(lats))
    return None


def _cmr_data_urls(umm, suffixes=None):
    urls = []
    for u in umm.get('RelatedUrls') or []:
        if u.get('Type') != 'GET DATA':
            continue
        url = u.get('URL', '')
        if suffixes and not url.lower().endswith(tuple(suffixes)):
            continue
        urls.append(url)
    return urls


def search_ecostress_lste(bbox, when, window_hours=24.0, versions=('002', '003'),
                          scene_polygon=None, session=None):
    """ECOSTRESS ECO_L2T_LSTE granules intersecting `bbox` within
    +/-window_hours of `when`.

    Both v002 and v003 are searched and the results merged by granule name.
    They coexist in CMR with different date coverage, so querying only one
    can report "no coverage" for a location the other covers."""
    when = _as_naive_utc(when)
    lo = when - _dt.timedelta(hours=window_hours)
    hi = when + _dt.timedelta(hours=window_hours)
    s = session or _session()
    out, seen = [], set()
    for version in versions:
        for item in _cmr_bbox_search('ECO_L2T_LSTE', version, bbox, lo, hi, session=s):
            umm = item['umm']
            name = umm['GranuleUR']
            if name in seen:
                continue
            seen.add(name)
            gbox = _cmr_bbox_of(umm)
            out.append(_granule_record(
                sensor='ECOSTRESS', product='ECO_L2T_LSTE.%s' % version,
                platform='ISS', granule=name,
                datetime_utc=_as_naive_utc(
                    umm['TemporalExtent']['RangeDateTime']['BeginningDateTime']),
                bbox=gbox,
                overlap=(footprint_overlap_fraction(scene_polygon, gbox)
                         if (scene_polygon and gbox) else None),
                urls=_cmr_data_urls(umm, ('.tif',)),
                archive='NASA CMR / LP DAAC'))
    return out


def search_viirs_lst(bbox, when, window_hours=24.0, products=None,
                     scene_polygon=None, session=None):
    """VIIRS daily gridded LST granules (VNP21A1D/N, VJ121A1D/N,
    VJ221A1D/N) whose day overlaps the search window.

    IMPORTANT - these records come back with time_is_nominal=True, and
    datetime_utc set to local solar noon (day products) or midnight (night
    products) of the granule's day. That is a PLACEHOLDER, not an overpass
    time. These are daily composites on the sinusoidal grid; CMR labels
    their temporal extent 00:00:00-23:59:59, and the true observation time
    varies across the tile and lives per pixel in the granule's View_Time
    layer.

    Nothing may be ranked on that placeholder. Download the granule and
    call load_viirs_lst(), which returns the true per-pixel observation
    time; match_thermal_partners() keeps these rows visible but refuses to
    rank them against sensors carrying real timestamps, precisely so a
    placeholder can never be mistaken for a close match.

    Solar noon is used rather than midnight for day products because it is
    the least-wrong single instant for one, and because midnight would make
    a granule look like an implausibly good match to an EnMAP scene that is
    itself always acquired near local mid-morning."""
    when = _as_naive_utc(when)
    lo = when - _dt.timedelta(hours=window_hours)
    hi = when + _dt.timedelta(hours=window_hours)
    products = list(products) if products else list(VIIRS_LST_PRODUCTS)
    centre_lon = 0.5 * (bbox[0] + bbox[2])
    s = session or _session()
    out = []
    for short_name in products:
        meta = VIIRS_LST_PRODUCTS.get(short_name, {})
        for item in _cmr_bbox_search(short_name, '002', bbox, lo, hi, session=s):
            umm = item['umm']
            day = _as_naive_utc(
                umm['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
            if day is None:
                continue
            target_local = 12.0 if meta.get('day_night') == 'day' else 0.0
            nominal = (day.replace(hour=0, minute=0, second=0, microsecond=0)
                       + _dt.timedelta(hours=target_local - centre_lon / 15.0))
            gbox = _cmr_bbox_of(umm)
            out.append(_granule_record(
                sensor='VIIRS-LST', product=short_name,
                platform=meta.get('platform', 'VIIRS'),
                granule=umm['GranuleUR'],
                datetime_utc=nominal, time_is_nominal=True,
                bbox=gbox,
                overlap=(footprint_overlap_fraction(scene_polygon, gbox)
                         if (scene_polygon and gbox) else None),
                urls=_cmr_data_urls(umm, ('.h5',)),
                archive='NASA CMR / LP DAAC'))
    return out


def search_viirs_fire(bbox, when, window_hours=24.0, scene_polygon=None,
                      session=None):
    """VNP14IMG 375 m active-fire / FRP granules within +/-window_hours.

    Unlike the gridded LST products above, this is a 6-minute swath granule
    whose timestamp is real to the minute, so these records carry
    time_is_nominal=False and rank honestly beside ECOSTRESS and Landsat.

    Its value to this pipeline is the 4 um channel. Every other thermal
    partner here - ECOSTRESS, Landsat TIRS, VIIRS M15/M16, SLSTR - measures
    in the 8-12 um window, where a flaming front drives the retrieval past
    the top of its design range. VIIRS M13 at 3.99 um has a dual-gain
    design that stays linear to ~634 K, which is why VNP14IMG can report
    fire radiative power for pixels whose 11 um LST retrieval is pinned at
    its ceiling."""
    when = _as_naive_utc(when)
    lo = when - _dt.timedelta(hours=window_hours)
    hi = when + _dt.timedelta(hours=window_hours)
    s = session or _session()
    out = []
    for item in _cmr_bbox_search('VNP14IMG', '002', bbox, lo, hi, session=s):
        umm = item['umm']
        gbox = _cmr_bbox_of(umm)
        out.append(_granule_record(
            sensor='VIIRS-FIRE', product='VNP14IMG', platform='Suomi-NPP',
            granule=umm['GranuleUR'],
            datetime_utc=_as_naive_utc(
                umm['TemporalExtent']['RangeDateTime']['BeginningDateTime']),
            bbox=gbox,
            overlap=(footprint_overlap_fraction(scene_polygon, gbox)
                     if (scene_polygon and gbox) else None),
            urls=_cmr_data_urls(umm, ('.nc',)),
            archive='NASA CMR / LP DAAC'))
    return out


def search_landsat_st(bbox, when, window_hours=24.0, scene_polygon=None,
                      session=None, collections=('landsat-c2l2-st',)):
    """Landsat Collection-2 Level-2 surface-temperature scenes from the
    USGS LandsatLook STAC.

    Restricted to Landsat 8 and 9. Landsat 7 is excluded deliberately: its
    ETM+ scan-line corrector failed in 2003, and the satellite was
    decommissioned in 2024 after drifting far off its nominal overpass
    time, so a late L7 scene is neither spatially complete nor acquired
    when its timestamp implies."""
    when = _as_naive_utc(when)
    lo = when - _dt.timedelta(hours=window_hours)
    hi = when + _dt.timedelta(hours=window_hours)
    s = session or _session()
    body = {'collections': list(collections),
            'bbox': [bbox[0], bbox[1], bbox[2], bbox[3]],
            'datetime': '%s/%s' % (_iso(lo), _iso(hi)),
            'limit': 100}
    try:
        r = s.post(LANDSATLOOK_STAC, json=body, timeout=120)
    except requests.RequestException as exc:
        warnings.warn('LandsatLook STAC unreachable: %s' % exc)
        return []
    if not r.ok:
        warnings.warn('LandsatLook STAC failed: HTTP %s' % r.status_code)
        return []
    out = []
    for f in r.json().get('features', []):
        p = f.get('properties', {})
        platform = (p.get('platform') or '').upper()
        if platform not in ('LANDSAT_8', 'LANDSAT_9'):
            continue
        gbox = tuple(f['bbox'][:4]) if f.get('bbox') else None
        urls = []
        for key, asset in (f.get('assets') or {}).items():
            if key.lower() in ('st_b10', 'lwir11', 'st_qa', 'qa_pixel') and asset.get('href'):
                urls.append(asset['href'])
        out.append(_granule_record(
            sensor='Landsat', product='landsat-c2l2-st',
            platform=platform.replace('_', ' ').title(),
            granule=f.get('id'),
            datetime_utc=_as_naive_utc(p.get('datetime')),
            bbox=gbox,
            overlap=(footprint_overlap_fraction(scene_polygon, gbox)
                     if (scene_polygon and gbox) else None),
            cloud_pct=p.get('eo:cloud_cover'),
            urls=urls, archive='USGS LandsatLook STAC'))
    return out


def search_slstr(bbox, when, window_hours=24.0,
                 product_types=('SL_2_LST___', 'SL_2_FRP___'),
                 scene_polygon=None, session=None):
    """Sentinel-3 SLSTR Level-2 LST and FRP products from the Copernicus
    Data Space OData catalogue.

    Catalogue search needs no credentials; downloading does - see
    download_slstr().

    The spatial filter is an OData intersection against the scene bounding
    box rather than a point, so a granule clipping the edge of the scene is
    still found. SLSTR swaths are ~1400 km wide, so in practice a returned
    granule covers the whole EnMAP scene many times over."""
    when = _as_naive_utc(when)
    lo = when - _dt.timedelta(hours=window_hours)
    hi = when + _dt.timedelta(hours=window_hours)
    west, south, east, north = bbox
    ring = '%.6f %.6f,%.6f %.6f,%.6f %.6f,%.6f %.6f,%.6f %.6f' % (
        west, south, east, south, east, north, west, north, west, south)
    s = session or _session()
    out = []
    for ptype in product_types:
        flt = ("Collection/Name eq 'SENTINEL-3' and "
               "contains(Name,'" + ptype + "') and "
               "OData.CSC.Intersects(area=geography'SRID=4326;POLYGON((" + ring + "))') and "
               "ContentDate/Start gt " + lo.strftime('%Y-%m-%dT%H:%M:%S.000Z') + " and "
               "ContentDate/Start lt " + hi.strftime('%Y-%m-%dT%H:%M:%S.000Z'))
        try:
            r = s.get(CDSE_ODATA, params={'$filter': flt, '$top': 100}, timeout=180)
        except requests.RequestException as exc:
            warnings.warn('CDSE catalogue unreachable: %s' % exc)
            continue
        if not r.ok:
            warnings.warn('CDSE query for %s failed: HTTP %s - %s'
                          % (ptype, r.status_code, r.text[:200]))
            continue
        for p in r.json().get('value', []):
            name = p.get('Name', '')
            platform = {'S3A': 'Sentinel-3A', 'S3B': 'Sentinel-3B',
                        'S3C': 'Sentinel-3C'}.get(name[:3], 'Sentinel-3')
            out.append(_granule_record(
                sensor='SLSTR', product=ptype, platform=platform,
                granule=name,
                datetime_utc=_as_naive_utc(p.get('ContentDate', {}).get('Start')),
                bbox=None, overlap=None,
                urls=[CDSE_ZIPPER % p['Id']],
                archive='Copernicus Data Space (OData)'))
    return out


# --------------------------------------------------------------------------
# the unified matcher
# --------------------------------------------------------------------------

def match_thermal_partners(scene, window_hours=24.0, sensors=None,
                           min_overlap=0.0, include_nominal_time=True,
                           verbose=True):
    """Search every implemented archive for thermal partners to one EnMAP
    scene and return a single ranked pandas DataFrame.

    `scene` is either the dict from enmap_scene_reference() or any path
    that function accepts.

    Ranking rule, and the reason this function exists rather than a loop
    over the searches: rows whose timestamp is a placeholder
    (time_is_nominal, i.e. the gridded daily VIIRS LST products) are sorted
    BELOW every row with a real overpass time, regardless of the apparent
    offset, and their dt_hours is reported as NaN rather than as a number.
    Sorting them together would put a granule whose only known time is
    "some time that day" above a Landsat scene 40 minutes away, purely
    because its placeholder happened to land nearer the EnMAP overpass.
    The VIIRS rows are still listed - they are often the only partner
    available above 52 deg N - but they are labelled 'day-composite' so the
    real time has to be resolved by loading the granule.

    min_overlap drops rows covering less than that fraction of the EnMAP
    footprint. It is 0.0 by default because two of the five searches
    (SLSTR, and any CMR granule publishing no geometry) legitimately return
    no footprint, and those rows carry overlap=None; a non-zero threshold
    keeps them, since None means unknown, not zero."""
    import pandas as pd

    if not isinstance(scene, dict):
        scene = enmap_scene_reference(scene)
    bbox, poly = scene['bbox'], scene['polygon']
    when = scene['datetime_utc']
    lon_c = scene['centre'][0]

    all_sensors = ('ECOSTRESS', 'Landsat', 'VIIRS-LST', 'VIIRS-FIRE', 'SLSTR')
    wanted = set(sensors) if sensors else set(all_sensors)
    session = _session()

    if verbose:
        print('[match] EnMAP %s  %s UTC  (%.2f h local solar)'
              % (scene['scene_id'], when.strftime('%Y-%m-%d %H:%M:%S'),
                 local_solar_hour(when, lon_c)))
        print('[match] footprint %.4f,%.4f,%.4f,%.4f   window +/-%.1f h'
              % (bbox + (window_hours,)))

    rows = []
    searches = (
        ('ECOSTRESS', search_ecostress_lste),
        ('Landsat', search_landsat_st),
        ('VIIRS-LST', search_viirs_lst),
        ('VIIRS-FIRE', search_viirs_fire),
        ('SLSTR', search_slstr),
    )
    for name, fn in searches:
        if name not in wanted:
            continue
        try:
            found = fn(bbox, when, window_hours=window_hours,
                       scene_polygon=poly, session=session)
        except Exception as exc:                       # one archive down
            # must not lose the other four
            warnings.warn('%s search failed: %s: %s'
                          % (name, exc.__class__.__name__, exc))
            found = []
        if verbose:
            print('[match]   %-11s %d granule(s)' % (name, len(found)))
        rows.extend(found)

    if not rows:
        if verbose:
            print('[match] no thermal partner found in any archive')
        return pd.DataFrame(columns=[
            'sensor', 'platform', 'product', 'granule', 'datetime_utc',
            'dt_hours', 'time_quality', 'local_solar_h', 'day_night',
            'overlap', 'cloud_pct', 'archive', 'urls'])

    for r in rows:
        dt_utc = r['datetime_utc']
        nominal = r['time_is_nominal']
        r['dt_hours'] = (np.nan if (dt_utc is None or nominal)
                         else (dt_utc - when).total_seconds() / 3600.0)
        r['time_quality'] = ('day-composite' if nominal
                             else ('exact' if dt_utc is not None else 'unknown'))
        lsh = local_solar_hour(dt_utc, lon_c) if dt_utc is not None else np.nan
        r['local_solar_h'] = lsh
        r['day_night'] = ('night' if (lsh < 6.0 or lsh >= 18.0) else 'day') \
            if np.isfinite(lsh) else 'unknown'
        r['urls'] = list(r['urls'])

    df = pd.DataFrame(rows)
    if min_overlap > 0:
        keep = df['overlap'].isna() | (df['overlap'] >= min_overlap)
        df = df[keep]
    if not include_nominal_time:
        df = df[~df['time_is_nominal']]

    # placeholders last, then by |offset|
    df = df.assign(_nom=df['time_is_nominal'].astype(int),
                   _abs=df['dt_hours'].abs())
    df = df.sort_values(['_nom', '_abs'], kind='mergesort').drop(columns=['_nom', '_abs'])
    df = df.reset_index(drop=True)

    if verbose and len(df):
        real = df[~df['time_is_nominal']]
        if len(real):
            b = real.iloc[0]
            # search_landsat_st() builds granule from f.get('id'), which a STAC
            # feature may omit; slicing None would lose all five archive results
            # to a TypeError in the summary line.
            print('[match] closest real-time partner: %s %s  %+.2f h  (%s)'
                  % (b['sensor'], str(b['granule'] or 'unnamed')[:52],
                     b['dt_hours'], b['day_night']))
        else:
            print('[match] every partner found is a day-composite; '
                  'load one to resolve its true observation time')
    cols = ['sensor', 'platform', 'product', 'granule', 'datetime_utc',
            'dt_hours', 'time_quality', 'local_solar_h', 'day_night',
            'overlap', 'cloud_pct', 'archive', 'time_is_nominal', 'urls']
    return df[[c for c in cols if c in df.columns]]


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------

def _earthdata_credentials():
    """(username, password) for NASA Earthdata, from ~/.netrc or the
    EARTHDATA_USERNAME/EARTHDATA_PASSWORD environment variables.

    Never prompts. The main module explains at length why an interactive
    prompt is unacceptable here (it can run inside a GUI worker thread with
    no console, where input() blocks forever); the same applies to this
    module, which is also called from batch scripts."""
    user = os.environ.get('EARTHDATA_USERNAME')
    pwd = os.environ.get('EARTHDATA_PASSWORD')
    if user and pwd:
        return user, pwd
    for path in (os.environ.get('NETRC'), os.path.expanduser('~/.netrc'),
                 os.path.expanduser('~/_netrc')):
        if not path or not os.path.isfile(path):
            continue
        try:
            auth = netrc.netrc(path).authenticators(URS_HOST)
        except (netrc.NetrcParseError, OSError):
            continue
        if auth:
            return auth[0], auth[2]
    raise RuntimeError(
        'No NASA Earthdata credentials found. Either set EARTHDATA_USERNAME '
        'and EARTHDATA_PASSWORD, or add a ~/.netrc entry:\n'
        '    machine %s login <user> password <pass>\n'
        'Register free at https://urs.earthdata.nasa.gov' % URS_HOST)


def download_earthdata_granule(url, out_dir, overwrite=False, timeout=1800,
                               chunk=1 << 20, verbose=True):
    """Download one LP DAAC asset, following the URS redirect chain.

    Returns the local path. Skips the transfer when the file is already
    present and non-empty unless overwrite=True.

    The download is written to a '.part' file and renamed only on success,
    so an interrupted transfer can never leave behind a truncated file that
    a later run would mistake for a complete one - which for an HDF5
    granule surfaces much later as an opaque library error, not as an
    obviously missing file."""
    user, pwd = _earthdata_credentials()
    os.makedirs(out_dir, exist_ok=True)
    dest = os.path.join(out_dir, url.rstrip('/').split('/')[-1])
    if os.path.isfile(dest) and os.path.getsize(dest) > 0 and not overwrite:
        if verbose:
            print('[download] have %s' % os.path.basename(dest))
        return dest

    s = _session()
    s.auth = (user, pwd)
    r = s.get(url, timeout=timeout, stream=True, allow_redirects=True)
    if r.status_code in (401, 403):
        raise RuntimeError(
            'Earthdata rejected the credentials for %s (HTTP %d). Check the '
            '~/.netrc entry, and that the account has accepted the LP DAAC '
            'EULA at https://urs.earthdata.nasa.gov' % (url, r.status_code))
    r.raise_for_status()
    tmp = dest + '.part'
    total = 0
    with open(tmp, 'wb') as fh:
        for block in r.iter_content(chunk):
            if block:
                fh.write(block)
                total += len(block)
    if total == 0:
        os.remove(tmp)
        raise RuntimeError('empty download from %s' % url)
    os.replace(tmp, dest)
    if verbose:
        print('[download] %s (%.1f MB)' % (os.path.basename(dest), total / 1e6))
    return dest


def download_viirs_lst(record_or_url, out_dir, **kw):
    """Download a VIIRS LST granule from a match_thermal_partners() row (or
    a bare URL) and return the local .h5 path."""
    if isinstance(record_or_url, str):
        url = record_or_url
    else:
        urls = list(record_or_url['urls'])
        h5 = [u for u in urls if u.lower().endswith('.h5')]
        if not h5:
            raise ValueError('no .h5 asset in record %r'
                             % (record_or_url.get('granule'),))
        url = h5[0]
    return download_earthdata_granule(url, out_dir, **kw)


def cdse_access_token(username=None, password=None, timeout=90):
    """Exchange Copernicus Data Space credentials for an OAuth access token.

    Credentials come from the arguments, CDSE_USERNAME/CDSE_PASSWORD, or the
    login remembered by credentials.py. The token is short-lived (the service
    issues 1800 s), which is why it is fetched per download session rather
    than cached to disk."""
    import credentials
    c = credentials.get('cdse', username=username, password=password)
    username, password = c['username'], c['password']
    if not username or not password:
        raise RuntimeError(
            'No Copernicus Data Space login. Sign in once in the GUI (it is remembered) or set '
            'CDSE_USERNAME and CDSE_PASSWORD (register free at https://dataspace.copernicus.eu). '
            'Catalogue search via search_slstr() works without them; only the '
            'download needs an account.')
    r = requests.post(CDSE_TOKEN_URL, timeout=timeout,
                      data={'client_id': 'cdse-public', 'grant_type': 'password',
                            'username': username, 'password': password})
    if r.status_code == 401:
        raise RuntimeError('Copernicus Data Space rejected the login (HTTP 401).')
    r.raise_for_status()
    return r.json()['access_token']


def download_slstr(record_or_url, out_dir, token=None, unzip=True,
                   overwrite=False, timeout=3600, verbose=True):
    """Download one Sentinel-3 SLSTR product from Copernicus Data Space.

    Returns the path to the unpacked .SEN3 directory (unzip=True, the
    default, since every loader here wants the directory) or to the .zip.

    An SLSTR L2 product is ~75 MB, so as with the Earthdata downloader the
    transfer lands in a '.part' file and is renamed only once complete."""
    if isinstance(record_or_url, str):
        url = record_or_url
        name = 'slstr_product'
    else:
        url = record_or_url['urls'][0]
        name = record_or_url['granule']
    os.makedirs(out_dir, exist_ok=True)
    stem = name[:-5] if name.endswith('.SEN3') else name
    zip_path = os.path.join(out_dir, stem + '.zip')
    sen3_dir = os.path.join(out_dir, stem + '.SEN3')

    if unzip and os.path.isdir(sen3_dir) and not overwrite:
        if verbose:
            print('[SLSTR] have %s' % os.path.basename(sen3_dir))
        return sen3_dir
    if not (os.path.isfile(zip_path) and os.path.getsize(zip_path) > 0 and not overwrite):
        token = token or cdse_access_token()
        r = requests.get(url, headers={'Authorization': 'Bearer ' + token},
                         timeout=timeout, stream=True, allow_redirects=True)
        if r.status_code in (401, 403):
            raise RuntimeError('Copernicus Data Space refused the download '
                               '(HTTP %d) - the token may have expired.' % r.status_code)
        r.raise_for_status()
        tmp = zip_path + '.part'
        total = 0
        with open(tmp, 'wb') as fh:
            for block in r.iter_content(1 << 20):
                if block:
                    fh.write(block)
                    total += len(block)
        os.replace(tmp, zip_path)
        if verbose:
            print('[SLSTR] %s (%.1f MB)' % (os.path.basename(zip_path), total / 1e6))
    if not unzip:
        return zip_path

    import zipfile
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
        roots = {n.split('/')[0] for n in z.namelist() if '/' in n}
    if os.path.isdir(sen3_dir):
        return sen3_dir
    for root in roots:                     # archive names it slightly differently
        cand = os.path.join(out_dir, root)
        if os.path.isdir(cand):
            return cand
    raise RuntimeError('could not find the unpacked .SEN3 directory in %s' % out_dir)


# --------------------------------------------------------------------------
# loaders
#
# All of them return the dict contract the rest of the pipeline speaks:
#   {'lst_K', 'geotransform', 'projection_wkt'}
# plus sensor-specific extras. That is what lets a VIIRS or SLSTR result be
# passed straight to validate_temperature() and _sample_raster_at_lonlat()
# in Fire_analysis_with_EnMAP.py with no adaptation.
# --------------------------------------------------------------------------

def _sinusoidal_wkt():
    from pyproj import CRS
    return CRS.from_proj4(SINUSOIDAL_PROJ4).to_wkt()


def _viirs_grid_name(h5):
    grids = h5['HDFEOS/GRIDS']
    names = list(grids.keys())
    if not names:
        raise ValueError('no HDFEOS grid in this file')
    return names[0]


def _struct_metadata_corners(h5):
    """Tile corners in sinusoidal metres, read from the granule's own
    StructMetadata.0.

    Read rather than computed from the tile's h/v index. The h/v arithmetic
    is standard and would give the same answer for a nominal tile, but it
    silently encodes an assumption about the global grid definition, and
    the file states the corners outright - so there is no reason to
    reconstruct what is written down."""
    text = h5['HDFEOS INFORMATION/StructMetadata.0'][()]
    if isinstance(text, bytes):
        text = text.decode('ascii', 'replace')
    ul = re.search(r'UpperLeftPointMtrs=\(([-\d.,eE+]+)\)', text)
    lr = re.search(r'LowerRightMtrs=\(([-\d.,eE+]+)\)', text)
    if not (ul and lr):
        raise ValueError('StructMetadata.0 has no UpperLeft/LowerRight corners')
    ulx, uly = [float(v) for v in ul.group(1).split(',')]
    lrx, lry = [float(v) for v in lr.group(1).split(',')]
    return ulx, uly, lrx, lry


def load_viirs_lst(path, apply_quality_mask=True, strict_quality=False,
                   bbox=None, verbose=True):
    """Load a VIIRS daily gridded LST granule (VNP21A1D/N, VJ121A1D/N,
    VJ221A1D/N) and return it in the pipeline dict contract.

    Returns, in addition to 'lst_K'/'geotransform'/'projection_wkt':

      'observation_datetime_utc'  per-pixel UTC datetime of the observation
                                  actually used in the composite, as a
                                  float array of UTC hours since the
                                  granule day began (NaN where no
                                  observation)
      'view_time_local_h'         the product's own View_Time layer, hours
                                  of local solar time
      'observation_utc_median'    a single naive datetime, the median over
                                  valid pixels - what to compare against
                                  the EnMAP acquisition time
      'emissivity'                dict of the Emis_14/15/16 layers, unpacked
      'qc'                        the raw QC layer
      'platform', 'product'

    WHY THE TIME LAYER MATTERS (this is the whole point of the function)
    --------------------------------------------------------------------
    These products are daily composites. CMR labels them 00:00:00-23:59:59
    of their day, so any time-matching done on catalogue metadata alone is
    meaningless. The observation actually used is recorded per pixel in
    View_Time, in hours of LOCAL SOLAR time; converting it back to UTC needs
    each pixel's own longitude:

        utc_hours = view_time_local - longitude_deg / 15

    For the scene shipped with this repository the difference is nearly two
    hours - see the module docstring. A validation statistic computed
    against the wrong one of those two times is not the statistic it
    appears to be, because land surface temperature changes fastest either
    side of local noon.

    QUALITY
    -------
    apply_quality_mask (default True) drops pixels the granule's own QC
    layer reports as not produced (mandatory QA bits 0-1 equal to 2 or 3 -
    cloud, or other failure). strict_quality additionally drops
    'produced, check other QA' (QA == 1) and any pixel whose cloud field
    (bits 4-5) is not 'cloud free'. Strict is off by default because on a
    fire scene the smoke plume frequently trips the cloud test, and
    discarding every flagged pixel can empty the scene; the QC layer is
    returned in full so a caller can make its own decision.

    Scale factors and fill values are read from each dataset's own HDF5
    attributes, never hard-coded - the same defensive stance
    load_ecostress_lst() takes in the main module, and for the same reason:
    guessing the packing wrong is a silent factor-of-fifty error, not a
    crash."""
    import h5py
    from pyproj import CRS, Transformer

    with h5py.File(str(path), 'r') as h5:
        grid = _viirs_grid_name(h5)
        fields = h5['HDFEOS/GRIDS/%s/Data Fields' % grid]
        if 'LST_1KM' not in fields:
            raise ValueError('%s carries no LST_1KM layer (grid %r)' % (path, grid))

        def unpack(name):
            """Apply the layer's own scale_factor/add_offset, blanking fill."""
            ds = fields[name]
            raw = ds[:].astype('float64')
            attrs = ds.attrs
            scale = float(np.ravel(attrs.get('scale_factor', [1.0]))[0])
            offset = float(np.ravel(attrs.get('add_offset', [0.0]))[0])
            fill = attrs.get('_Fillvalue', attrs.get('_FillValue', None))
            out = raw * scale + offset
            if fill is not None:
                fv = np.ravel(fill)[0]
                if np.issubdtype(np.asarray(fv).dtype, np.number):
                    out = np.where(raw == float(fv), np.nan, out)
            return out, scale, offset

        lst_k, lst_scale, lst_offset = unpack('LST_1KM')
        view_time, _, _ = unpack('View_Time')
        qc = fields['QC'][:].astype('uint32') if 'QC' in fields else None
        emis = {}
        for band in ('Emis_14', 'Emis_15', 'Emis_16'):
            if band in fields:
                emis[band], _, _ = unpack(band)

        ulx, uly, lrx, lry = _struct_metadata_corners(h5)
        ny, nx = lst_k.shape
        granule_day = _viirs_day_from_attrs(h5)

    px = (lrx - ulx) / nx
    py = (lry - uly) / ny
    geotransform = (ulx, px, 0.0, uly, 0.0, py)
    projection_wkt = _sinusoidal_wkt()

    # A zero LST is fill, not 0 K; unpack() has already used the declared
    # _Fillvalue, but a physically impossible value that survives it is
    # still not a temperature. 150 K is the same generous floor the main
    # module uses in _LST_PLAUSIBLE_K.
    lst_k = np.where(lst_k < 150.0, np.nan, lst_k)

    n_before = int(np.isfinite(lst_k).sum())
    if apply_quality_mask and qc is not None:
        mandatory = qc & 0x3          # bits 0-1
        bad = mandatory >= 2          # 2 = not produced (cloud), 3 = not produced
        if strict_quality:
            bad |= (mandatory == 1)
            bad |= ((qc >> 4) & 0x3) != 0     # bits 4-5: cloud state
        lst_k = np.where(bad, np.nan, lst_k)
        if verbose:
            n_after = int(np.isfinite(lst_k).sum())
            print('[VIIRS] QC masking removed %d of %d valid pixels (%d left)'
                  % (n_before - n_after, n_before, n_after))

    # ---- per-pixel longitude, to turn local solar time back into UTC ----
    cols = np.arange(nx, dtype='float64')
    rows = np.arange(ny, dtype='float64')
    xs = ulx + (cols + 0.5) * px
    ys = uly + (rows + 0.5) * py
    xg, yg = np.meshgrid(xs, ys)
    to_wgs84 = Transformer.from_crs(CRS.from_proj4(SINUSOIDAL_PROJ4),
                                    CRS.from_epsg(4326), always_xy=True)
    lon_g, lat_g = to_wgs84.transform(xg, yg)

    valid_time = np.isfinite(view_time) & np.isfinite(lst_k)
    utc_hours = np.where(valid_time, view_time - lon_g / 15.0, np.nan)

    stem = os.path.basename(str(path))
    product = _viirs_product_from_name(stem)
    meta = VIIRS_LST_PRODUCTS.get(product, {})
    # The granule's own StartTime attribute first, the filename convention
    # only as a fallback: a renamed or re-exported file still carries the
    # correct date inside, and deriving the day from the name alone breaks
    # silently, leaving observation_utc_median=None with no explanation.
    day = granule_day or _viirs_day_from_name(stem)

    observation_median = None
    if np.isfinite(utc_hours).any() and day is not None:
        med = float(np.nanmedian(utc_hours))
        # A night granule observed just after local midnight has a local
        # solar time near 0 but a UTC time that can fall on either side of
        # the granule's own day boundary. Wrapping into [0, 24) and adding
        # to the granule date keeps the result on the labelled day, which
        # is what the granule name asserts.
        observation_median = day + _dt.timedelta(hours=med % 24.0)

    if bbox is not None:
        inside = ((lon_g >= bbox[0]) & (lon_g <= bbox[2]) &
                  (lat_g >= bbox[1]) & (lat_g <= bbox[3]))
        lst_k = np.where(inside, lst_k, np.nan)

    if verbose:
        finite = np.isfinite(lst_k)
        if finite.any():
            print('[VIIRS] %s %s: %d valid pixels, %.1f-%.1f K'
                  % (product, meta.get('platform', ''), int(finite.sum()),
                     float(np.nanmin(lst_k)), float(np.nanmax(lst_k))))
            if np.isfinite(utc_hours[finite]).any():
                # Wrap the midpoint, not each end independently: taking the
                # modulo of min and max separately inverts the printed range
                # for a granule spanning UTC midnight (e.g. "22.50-0.50 h").
                raw_lo = float(np.nanmin(utc_hours[finite]))
                raw_hi = float(np.nanmax(utc_hours[finite]))
                shift = raw_lo - (raw_lo % 24.0)
                print('[VIIRS] observation time %.2f-%.2f h UTC '
                      '(scale %g, offset %g)'
                      % (raw_lo - shift, raw_hi - shift, lst_scale, lst_offset))
        else:
            print('[VIIRS] %s: no valid pixels after masking' % product)

    return {
        'lst_K': lst_k,
        'geotransform': geotransform,
        'projection_wkt': projection_wkt,
        'view_time_local_h': view_time,
        'observation_utc_hours': utc_hours,
        'observation_utc_median': observation_median,
        'emissivity': emis,
        'qc': qc,
        'lon': lon_g,
        'lat': lat_g,
        'product': product,
        'platform': meta.get('platform'),
        'ceiling_K': VIIRS_LST_CEILING_K,
    }


def _viirs_day_from_name(stem):
    """Granule day from a '<PROD>.AYYYYDDD.<tile>...' filename.

    Fallback only - _viirs_day_from_attrs() reads it from the granule
    itself, which survives renaming."""
    m = re.search(r'\.A(\d{4})(\d{3})\.', stem)
    if not m:
        return None
    year, doy = int(m.group(1)), int(m.group(2))
    return _dt.datetime(year, 1, 1) + _dt.timedelta(days=doy - 1)


def _viirs_day_from_attrs(h5):
    """Granule day from the file's own root attributes.

    VNP21A1D and its siblings carry 'StartTime' (e.g. b'2024-10-19
    00:00:00') and 'RangeBeginningDate' at the HDF5 root. Preferred over
    the filename because it keeps working for a granule that has been
    renamed, staged under a cache key, or re-exported."""
    for key in ('StartTime', 'RangeBeginningDate', 'PGE_StartTime'):
        if key not in h5.attrs:
            continue
        raw = np.ravel(h5.attrs[key])[0]
        if isinstance(raw, bytes):
            raw = raw.decode('ascii', 'replace')
        m = re.search(r'(\d{4})-(\d{2})-(\d{2})', str(raw))
        if m:
            return _dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def _viirs_product_from_name(stem):
    """Product short name for a VIIRS granule filename.

    Matches a known short name anywhere in the name rather than taking the
    text before the first dot, so a file staged as 'VNP21A1D_test.h5' or
    'cache/VJ121A1D-20241019.h5' is still recognised as its product and
    gets the right platform and day/night label attached."""
    for name in VIIRS_LST_PRODUCTS:
        if name in stem:
            return name
    return stem.split('.')[0] if '.' in stem else stem


def _utm_crs_for(lon, lat):
    from pyproj import CRS
    # Clamp rather than wrap with % 60. At exactly lon = +180 the wrap gives
    # zone 1 (a CRS on the other side of the planet) where 60 is meant;
    # attach_pixel_geolocation() in the main module clamps for this reason and
    # the two must not disagree about which zone a scene lands in.
    zone = min(max(int(math.floor((lon + 180.0) / 6.0)) + 1, 1), 60)
    return CRS.from_epsg((32600 if lat >= 0 else 32700) + zone)


def load_slstr_lst(sen3_dir, target_bbox, resolution_m=1000.0, target_crs=None,
                   apply_quality_mask=True, max_distance_m=None, verbose=True):
    """Load a Sentinel-3 SLSTR SL_2_LST product and resample its nadir-view
    swath onto a regular projected grid.

    `target_bbox` is (west, south, east, north) in degrees - normally the
    EnMAP scene bbox from enmap_scene_reference(). `target_crs` defaults to
    the UTM zone containing the bbox centre, which is also what ECOSTRESS
    L2T tiles and Landsat scenes use, so all four sensors land in the same
    projection family.

    RESAMPLING - nearest neighbour, deliberately
    --------------------------------------------
    SLSTR L2 LST is a swath product: every pixel carries its own latitude
    and longitude and the grid is not regular. Getting it into the
    pipeline's raster contract therefore requires a resampling step, and
    the choice here is nearest neighbour with a hard distance cutoff.

    Bilinear or inverse-distance weighting would produce a smoother-looking
    field, but it would do so by averaging retrieved land surface
    temperatures across pixels that may straddle a land-cover or fire
    boundary - inventing values that no instrument measured, in exactly the
    places this pipeline cares about most. Nearest neighbour moves each
    retrieval without altering it, and `max_distance_m` (default 1.5x the
    SLSTR nadir sampling of 1 km) guarantees no output cell is filled from
    an observation further away than one sensor footprint. Cells with no
    observation inside that radius stay NaN rather than being extrapolated.

    Returned, in addition to the standard three keys:
      'lst_uncertainty_K'   the product's own per-pixel standard error,
                            resampled identically
      'observation_utc'     per-output-cell acquisition time, taken from
                            the source scanline (time_in.nc)
      'observation_utc_median'
      'source_row'          source scanline index per output cell
      'ceiling_K'           355.5 K - see below

    THE 355.5 K CEILING
    -------------------
    SLSTR packs LST as int16 with scale 0.002 and offset 290.0, so the
    highest representable value is 0.002 * 32767 + 290 = 355.53 K. That is
    an encoding limit sitting BELOW the ~380 K ceiling of ECOSTRESS and
    Landsat, and far below any flaming front. SLSTR LST is therefore a
    background/ambient reference for this pipeline and nothing more; for
    the hot component use its companion SL_2_FRP product, whose 3.7 um
    channels are designed for exactly that. The scale and offset are read
    from the file rather than assumed, so if a future baseline repacks the
    variable this ceiling is recomputed, not stale."""
    import netCDF4 as nc
    from pyproj import CRS, Transformer
    from scipy.spatial import cKDTree

    sen3_dir = str(sen3_dir)

    def _open(name):
        p = os.path.join(sen3_dir, name)
        if not os.path.isfile(p):
            raise FileNotFoundError('%s not found in %s' % (name, sen3_dir))
        return nc.Dataset(p)

    with _open('LST_in.nc') as ds:
        var = ds.variables['LST']
        lst = np.ma.filled(var[:].astype('float64'), np.nan)
        scale = float(np.ravel(getattr(var, 'scale_factor', 1.0))[0])
        offset = float(np.ravel(getattr(var, 'add_offset', 0.0))[0])
        ceiling = scale * 32767.0 + offset
        unc = (np.ma.filled(ds.variables['LST_uncertainty'][:].astype('float64'), np.nan)
               if 'LST_uncertainty' in ds.variables else None)
        exception = (np.ma.filled(ds.variables['exception'][:].astype('int32'), 0)
                     if 'exception' in ds.variables else None)

    with _open('geodetic_in.nc') as ds:
        lat = np.ma.filled(ds.variables['latitude_in'][:].astype('float64'), np.nan)
        lon = np.ma.filled(ds.variables['longitude_in'][:].astype('float64'), np.nan)

    cloud = None
    if apply_quality_mask and os.path.isfile(os.path.join(sen3_dir, 'flags_in.nc')):
        with _open('flags_in.nc') as ds:
            if 'cloud_in' in ds.variables:
                cloud = np.ma.filled(ds.variables['cloud_in'][:].astype('uint32'), 0)

    scan_time = None
    with _open('time_in.nc') as ds:
        if 'time_stamp_i' in ds.variables:
            scan_time = np.ma.filled(
                ds.variables['time_stamp_i'][:].astype('float64'), np.nan)

    good = np.isfinite(lst) & np.isfinite(lat) & np.isfinite(lon)
    if apply_quality_mask:
        if cloud is not None:
            # cloud_in is a bitmask of the individual cloud tests; any bit
            # set means at least one test fired. Conservative by design: a
            # cloud-top temperature compared against an EnMAP fire pixel is
            # not a validation point.
            good &= (cloud == 0)
        if exception is not None:
            good &= (exception == 0)
    n_src = int(good.sum())
    if n_src == 0:
        raise ValueError('no valid SLSTR LST pixels in %s after masking' % sen3_dir)

    west, south, east, north = target_bbox
    if target_crs is None:
        target_crs = _utm_crs_for(0.5 * (west + east), 0.5 * (south + north))
    elif not hasattr(target_crs, 'to_wkt'):
        target_crs = CRS.from_user_input(target_crs)

    fwd = Transformer.from_crs(CRS.from_epsg(4326), target_crs, always_xy=True)
    corners_x, corners_y = fwd.transform([west, east, east, west],
                                         [south, south, north, north])
    minx, maxx = min(corners_x), max(corners_x)
    miny, maxy = min(corners_y), max(corners_y)
    # Snap outward to whole pixels so the grid origin is a round number and
    # the requested bbox is fully contained rather than clipped.
    minx = math.floor(minx / resolution_m) * resolution_m
    miny = math.floor(miny / resolution_m) * resolution_m
    maxx = math.ceil(maxx / resolution_m) * resolution_m
    maxy = math.ceil(maxy / resolution_m) * resolution_m
    nx = max(1, int(round((maxx - minx) / resolution_m)))
    ny = max(1, int(round((maxy - miny) / resolution_m)))
    geotransform = (minx, resolution_m, 0.0, maxy, 0.0, -resolution_m)

    src_x, src_y = fwd.transform(lon[good], lat[good])
    tree = cKDTree(np.column_stack([src_x, src_y]))

    gx = minx + (np.arange(nx) + 0.5) * resolution_m
    gy = maxy - (np.arange(ny) + 0.5) * resolution_m
    gxx, gyy = np.meshgrid(gx, gy)
    if max_distance_m is None:
        max_distance_m = 1.5 * 1000.0          # 1.5 x SLSTR nadir sampling
    dist, idx = tree.query(np.column_stack([gxx.ravel(), gyy.ravel()]),
                           k=1, distance_upper_bound=max_distance_m)
    hit = np.isfinite(dist) & (idx < len(src_x))

    def _regrid(source_2d):
        flat = source_2d[good]
        out = np.full(gxx.size, np.nan)
        out[hit] = flat[idx[hit]]
        return out.reshape(ny, nx)

    lst_grid = _regrid(lst)
    unc_grid = _regrid(unc) if unc is not None else None

    rows_src = np.broadcast_to(np.arange(lst.shape[0])[:, None], lst.shape)[good]
    row_grid = np.full(gxx.size, -1, dtype='int64')
    row_grid[hit] = rows_src[idx[hit]]
    row_grid = row_grid.reshape(ny, nx)

    obs_utc = np.full((ny, nx), np.nan)
    observation_median = None
    if scan_time is not None:
        epoch = _dt.datetime(2000, 1, 1)
        with np.errstate(invalid='ignore'):
            sel = row_grid >= 0
            # Guard the join rather than trusting that time_in.nc is indexed on
            # the same scanline axis as LST_in.nc. A baseline change there
            # should cost the observation time, not the whole granule - by this
            # point ~75 MB has already been downloaded and unpacked.
            n_scan = len(scan_time)
            over = int(np.sum(sel & (row_grid >= n_scan)))
            sel &= row_grid < n_scan
            obs_utc[sel] = scan_time[row_grid[sel]]
            if over:
                warnings.warn('time_in.nc carries %d scanlines but LST_in.nc has %d rows; '
                              'observation time is unavailable for %d output cell(s).'
                              % (n_scan, lst.shape[0], over))
        if np.isfinite(obs_utc).any():
            observation_median = epoch + _dt.timedelta(
                microseconds=float(np.nanmedian(obs_utc)))

    if verbose:
        n_hit = int(np.isfinite(lst_grid).sum())
        print('[SLSTR] %s' % os.path.basename(sen3_dir))
        print('[SLSTR] %d source pixels -> %dx%d grid at %.0f m, %d cells filled '
              '(%.1f%%)' % (n_src, ny, nx, resolution_m, n_hit,
                            100.0 * n_hit / max(1, gxx.size)))
        if n_hit:
            print('[SLSTR] %.1f-%.1f K (encoding ceiling %.1f K)'
                  % (float(np.nanmin(lst_grid)), float(np.nanmax(lst_grid)), ceiling))
        if observation_median is not None:
            print('[SLSTR] observation time %s UTC'
                  % observation_median.strftime('%Y-%m-%d %H:%M:%S'))

    return {
        'lst_K': lst_grid,
        'geotransform': geotransform,
        'projection_wkt': target_crs.to_wkt(),
        'lst_uncertainty_K': unc_grid,
        'observation_utc': obs_utc,
        'observation_utc_median': observation_median,
        'source_row': row_grid,
        'ceiling_K': ceiling,
        'product': 'SL_2_LST',
        'platform': os.path.basename(sen3_dir)[:3],
    }


def load_generic_lst_geotiff(path, scale_factor=1.0, add_offset=0.0,
                             valid_range_K=(150.0, 1400.0), verbose=True):
    """Read any single-band land-surface-temperature GeoTIFF into the
    pipeline dict contract.

    This is the entry point for the sensors that have no machine-readable
    archive and must be ordered by hand - SDGSAT-1 TIS above all, and a
    constellr delivery if one is obtained through an ESA Third Party
    Missions proposal. Neither can be searched or downloaded from here (see
    the module docstring), but once a scene has been exported as a
    georeferenced LST GeoTIFF the rest of the pipeline does not care where
    it came from.

    scale_factor/add_offset are applied as given and default to a no-op,
    i.e. the file is assumed to be in Kelvin already. Values landing
    outside valid_range_K after scaling are blanked and COUNTED, and the
    count is printed: a product delivered in Celsius, or still in packed
    DN, shows up immediately as "every pixel rejected" instead of
    propagating a silent 273 K bias into the validation statistics."""
    from osgeo import gdal
    gdal.UseExceptions()
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise ValueError('GDAL could not open %s' % path)
    band = ds.GetRasterBand(1)
    raw = band.ReadAsArray().astype('float64')
    nodata = band.GetNoDataValue()
    valid = np.isfinite(raw)
    if nodata is not None and np.isfinite(nodata):
        valid &= (raw != nodata)
    lst = np.where(valid, raw * float(scale_factor) + float(add_offset), np.nan)

    n_before = int(np.isfinite(lst).sum())
    lo, hi = valid_range_K
    lst = np.where((lst < lo) | (lst > hi), np.nan, lst)
    n_after = int(np.isfinite(lst).sum())
    if verbose:
        print('[LST] %s: %d valid pixels' % (os.path.basename(str(path)), n_after))
        if n_before and n_after == 0:
            print('[LST] WARNING: all %d pixels fell outside %.0f-%.0f K. Check '
                  'scale_factor/add_offset - a product in Celsius needs '
                  'add_offset=273.15.' % (n_before, lo, hi))
        elif n_before - n_after:
            print('[LST] %d pixels outside %.0f-%.0f K were blanked'
                  % (n_before - n_after, lo, hi))
    # Read the georeferencing out before releasing the dataset: on Windows an
    # open GDAL handle keeps the file locked for the life of the process.
    geotransform, projection_wkt = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    return {
        'lst_K': lst,
        'geotransform': geotransform,
        'projection_wkt': projection_wkt,
    }


# --------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------

def _main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        description='Find thermal companion granules for an EnMAP scene.',
        epilog='Catalogue search needs no credentials. Downloading ECOSTRESS or '
               'VIIRS needs a NASA Earthdata login (~/.netrc or '
               'EARTHDATA_USERNAME/EARTHDATA_PASSWORD); downloading Sentinel-3 '
               'needs CDSE_USERNAME/CDSE_PASSWORD.')
    ap.add_argument('scene', help='EnMAP METADATA.XML, product raster, or the '
                                  'directory holding one')
    ap.add_argument('--window', type=float, default=12.0,
                    help='search +/- this many hours (default 12)')
    ap.add_argument('--sensors', nargs='*', default=None,
                    choices=['ECOSTRESS', 'Landsat', 'VIIRS-LST', 'VIIRS-FIRE', 'SLSTR'])
    ap.add_argument('--min-overlap', type=float, default=0.0,
                    help='drop granules covering less than this fraction of the '
                         'scene (granules that publish no footprint are kept)')
    ap.add_argument('--csv', default=None, help='write the full table here')
    ap.add_argument('--download', metavar='DIR', default=None,
                    help='also download the closest real-time partner into DIR')
    args = ap.parse_args(argv)

    scene = enmap_scene_reference(args.scene)
    df = match_thermal_partners(scene, window_hours=args.window,
                                sensors=args.sensors, min_overlap=args.min_overlap)
    if not len(df):
        return 1

    show = df[['sensor', 'platform', 'product', 'granule', 'datetime_utc',
               'dt_hours', 'time_quality', 'day_night', 'overlap', 'cloud_pct']]
    print()
    print(show.to_string(index=False))
    if args.csv:
        df.to_csv(args.csv, index=False)
        print('\nwrote %s' % args.csv)

    if args.download:
        real = df[~df['time_is_nominal']]
        if not len(real):
            print('\nnothing with a real timestamp to download')
            return 0
        best = real.iloc[0]
        print('\ndownloading %s %s' % (best['sensor'], best['granule']))
        if best['sensor'] == 'SLSTR':
            path = download_slstr(best, args.download)
        elif best['sensor'] in ('ECOSTRESS', 'VIIRS-LST', 'VIIRS-FIRE'):
            paths = [download_earthdata_granule(u, args.download)
                     for u in best['urls']]
            path = paths[0] if paths else None
        else:
            print('  Landsat downloads are handled by '
                  'Fire_analysis_with_EnMAP.download_landsat_lst(), which signs '
                  'Planetary Computer URLs - use that instead.')
            path = None
        if path:
            print('  -> %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(_main())

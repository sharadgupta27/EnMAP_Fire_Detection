#!/usr/bin/env python
# coding: utf-8
"""Write the four-sensor WILDFIRE sites: table + one AOI shapefile per tile.

Input  : ../../<scratch>/pertile_pass4.json  (or --json PATH)
Output : wildfire_sites_4sensor.csv
         sites/<SITE>_<tile>_AOI.shp / .zip

Why this table exists separately from sites_ordering.csv
--------------------------------------------------------
sites_ordering.csv was built by matching partner scenes in catalogue
metadata. That is necessary but not sufficient, and it let two failure modes
through, both of which cost an order:

  1. An ECOSTRESS granule can MATCH in CMR and still contain nothing. The
     Yalu granule (51TXG, 2026-05-29) matched on time and place and is 100%
     nodata; searching +/-24 h finds no other granule over that tile at all.
     So a site is only counted here once the LST COG has been opened and its
     valid Kelvin pixels counted inside the tile footprint.

  2. A SWIR ratio test cannot tell a wildfire from a gas flare or a steel
     works. Every hot-target test - Murphy, Dennison, the pipeline's own
     hotspot index - fires on both. So each tile here also carries the land
     cover under its hot pixels, its burn-scar fraction, and the same fire
     test run ~35 days either side. An industrial source is built-up, leaves
     no scar, and burns in every scene; a wildfire is none of those.

Both checks are per TILE, not per datatake. A datatake is a ~500 km strip
and an ECOSTRESS L2T tile is ~110 km, so a granule that covers the strip
well is usually not the one covering the tile that has the fire in it -
checking at datatake level wrote off tiles that are in fact well covered.

Run from this directory with a GDAL-enabled Python:
    python wildfire_4sensor.py [--json PATH]
"""

import argparse
import csv
import json
import os
import re
import zipfile

from osgeo import ogr, osr

ogr.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'sites')
CSV_PATH = os.path.join(HERE, 'wildfire_sites_4sensor.csv')

COLUMNS = [
    'site_id', 'region', 'datatake', 'tile_no', 'enmap_scene_hint',
    'acq_datetime_utc', 'enmap_cloud_pct',
    'west_lon', 'south_lat', 'east_lon', 'north_lat',
    'landsat_scene', 'landsat_platform', 'landsat_cloud_pct', 'max_sensor_dt_h',
    'ecostress_granule', 'ecostress_valid_px', 'ecostress_tile_cover_frac',
    'eco_lst_median_K', 'eco_lst_max_K', 'eco_hot_px', 'eco_hot_thresh_K',
    's2_active_fire_px', 's2_burn_scar_pct', 's2_dt_h',
    'lc_natural_px', 'lc_builtup_px', 'lc_crop_px',
    'fire_px_minus35d', 'fire_px_plus35d', 'verdict',
]

# Shapefile .dbf field names are capped at 10 characters.
FIELDS = [
    ('site_id', ogr.OFTString, 12), ('region', ogr.OFTString, 40),
    ('datatake', ogr.OFTString, 24), ('tile_no', ogr.OFTString, 8),
    ('acq_utc', ogr.OFTString, 28), ('enmap_cld', ogr.OFTReal, 0),
    ('ls_scene', ogr.OFTString, 60), ('ls_plat', ogr.OFTString, 12),
    ('ls_cloud', ogr.OFTReal, 0), ('max_dt_h', ogr.OFTReal, 0),
    ('eco_gran', ogr.OFTString, 70), ('eco_valid', ogr.OFTInteger, 0),
    ('eco_cover', ogr.OFTReal, 0),
    ('lst_med', ogr.OFTReal, 0), ('lst_max', ogr.OFTReal, 0),
    ('lst_hotpx', ogr.OFTInteger, 0), ('lst_thr', ogr.OFTReal, 0),
    ('s2_fire', ogr.OFTInteger, 0), ('s2_burn', ogr.OFTReal, 0),
    ('s2_dt_h', ogr.OFTReal, 0),
    ('lc_nat', ogr.OFTInteger, 0), ('lc_built', ogr.OFTInteger, 0),
    ('lc_crop', ogr.OFTInteger, 0),
    ('fire_m35d', ogr.OFTInteger, 0), ('fire_p35d', ogr.OFTInteger, 0),
    ('verdict', ogr.OFTString, 40),
]


def classify(rec):
    """Wildfire, industrial, already out, or unusable - and why.

    Deliberately conservative: anything that is not clearly vegetation
    burning at the time of the EnMAP overpass is rejected rather than
    ranked, because the cost of ordering a bad scene is far higher than the
    cost of skipping a marginal good one."""
    eco = rec.get('eco') or {}
    therm = rec.get('eco_thermal') or {}
    fire = rec.get('fire_px') or 0
    nat, built, crop = rec.get('natural', 0), rec.get('builtup', 0), rec.get('crop', 0)
    before, after = rec.get('before'), rec.get('after')
    if eco.get('valid_px', 0) < 5000:
        return 'reject: no ECOSTRESS data over tile'

    # The concurrent test. Sentinel-2 revisits every 5 days, so its fire
    # pixels are almost never contemporaneous with the EnMAP overpass -
    # they say a fire burned that week, not that one was burning when EnMAP
    # looked. ECOSTRESS is the only partner the search constrains to a few
    # hours, so it is the only one that can answer the question actually
    # being asked. A wildfire inside a 70 m LST pixel lifts it far above the
    # surrounding ground; no such pixel means nothing was alight.
    if therm:
        if therm.get('hot_px', 0) < 1:
            return 'reject: no thermal anomaly at overpass (LST max %.0f K vs %.0f K bg)' % (
                therm.get('max', 0), therm.get('median', 0))
    elif fire < 5:
        return 'reject: fire already out at overpass'

    # Sentinel-2 evidence is only corroboration, and only if it is close in
    # time. Beyond a day it describes a different moment in the fire.
    s2dh = rec.get('s2_dt_h')
    if s2dh is not None and abs(s2dh) > 24.0 and not therm:
        return 'reject: fire evidence is %.0f h from the overpass' % abs(s2dh)
    if built > nat:
        return 'reject: hot pixels are built-up (industrial)'
    if crop > nat:
        return 'reject: cropland burning, not wildfire'
    # Transience. Persistence needs fire on BOTH sides: a source that burns
    # 35 days before AND 35 days after is a flare, a waste site, or a
    # landscape in its annual burning season - not the discrete event a
    # temperature retrieval should be validated against. Fire on only ONE
    # side is much weaker evidence, because a different fire nearby in the
    # same season produces it too, so that is flagged rather than rejected.
    recur = [v for v in (before, after) if v is not None]
    if recur and min(recur) >= max(3, 0.5 * fire):
        return 'reject: burns before AND after (persistent source)'
    if recur and max(recur) >= 2 * max(fire, 1):
        return 'WILDFIRE - 4 sensors, but region burns seasonally'
    return 'WILDFIRE - 4 sensors confirmed'


def slug(text, maxlen=40):
    return re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')[:maxlen].rstrip('_')


def wgs84():
    sr = osr.SpatialReference()
    sr.ImportFromEPSG(4326)
    sr.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    return sr


def rect(w, s, e, n):
    ring = ogr.Geometry(ogr.wkbLinearRing)
    for x, y in ((w, s), (e, s), (e, n), (w, n), (w, s)):
        ring.AddPoint_2D(float(x), float(y))
    poly = ogr.Geometry(ogr.wkbPolygon)
    poly.AddGeometry(ring)
    return poly


def zip_shapefile(shp_path):
    base = os.path.splitext(shp_path)[0]
    zpath = base + '.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for ext in ('.shp', '.shx', '.dbf', '.prj'):
            if os.path.exists(base + ext):
                z.write(base + ext, os.path.basename(base + ext))
    return zpath


def write_layer(path, name, rows):
    """Write `rows` as one polygon each into a fresh shapefile at `path`."""
    drv = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    ly = ds.CreateLayer(name, wgs84(), ogr.wkbPolygon)
    for fname, ftype, width in FIELDS:
        fd = ogr.FieldDefn(fname, ftype)
        if width:
            fd.SetWidth(width)
        ly.CreateField(fd)
    for row in rows:
        feat = ogr.Feature(ly.GetLayerDefn())
        feat.SetGeometry(rect(row['west_lon'], row['south_lat'],
                              row['east_lon'], row['north_lat']))
        for k, v in (('site_id', row['site_id']), ('region', row['region']),
                     ('datatake', row['datatake']), ('tile_no', row['tile_no']),
                     ('acq_utc', row['acq_datetime_utc']), ('enmap_cld', row['enmap_cloud_pct']),
                     ('ls_scene', row['landsat_scene']), ('ls_plat', row['landsat_platform']),
                     ('ls_cloud', row['landsat_cloud_pct']), ('max_dt_h', row['max_sensor_dt_h']),
                     ('eco_gran', row['ecostress_granule']), ('eco_valid', row['ecostress_valid_px']),
                     ('eco_cover', row['ecostress_tile_cover_frac']),
                     ('lst_med', row['eco_lst_median_K']), ('lst_max', row['eco_lst_max_K']),
                     ('lst_hotpx', row['eco_hot_px']), ('lst_thr', row['eco_hot_thresh_K']),
                     ('s2_fire', row['s2_active_fire_px']), ('s2_burn', row['s2_burn_scar_pct']),
                     ('s2_dt_h', row['s2_dt_h']),
                     ('lc_nat', row['lc_natural_px']), ('lc_built', row['lc_builtup_px']),
                     ('lc_crop', row['lc_crop_px']),
                     ('fire_m35d', row['fire_px_minus35d']), ('fire_p35d', row['fire_px_plus35d']),
                     ('verdict', row['verdict'])):
            if v is not None and v != '':
                feat.SetField(k, v)
        ly.CreateFeature(feat)
        feat = None
    ds = None
    return path


def tile_no(tile_id):
    m = re.search(r'_(\d{3})_V\d', tile_id)
    return m.group(1) if m else '???'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None, help='pertile_pass4.json produced by the search')
    args = ap.parse_args()
    src = args.json
    if src is None:
        raise SystemExit('pass --json PATH to pertile_pass4.json')

    with open(src, encoding='utf-8') as _fh:
        recs = json.load(_fh)
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for i, r in enumerate(recs, start=1):
        eco = r.get('eco') or {}
        therm = r.get('eco_thermal') or {}
        region, dt = r['key'].split('|', 1)
        tn = tile_no(r['tile'])
        verdict = classify(r)
        w, s, e, n = r['bbox']
        rows.append({
            'site_id': 'W%02d' % i, 'region': region, 'datatake': dt, 'tile_no': tn,
            'enmap_scene_hint': r['tile'],
            'acq_datetime_utc': r['enmap_dt'], 'enmap_cloud_pct': r.get('cloud'),
            'west_lon': round(w, 5), 'south_lat': round(s, 5),
            'east_lon': round(e, 5), 'north_lat': round(n, 5),
            'landsat_scene': r.get('ls_scene', ''), 'landsat_platform': r.get('ls_plat', ''),
            'landsat_cloud_pct': r.get('ls_cc'), 'max_sensor_dt_h': round(r.get('maxdh', -1), 2),
            'ecostress_granule': eco.get('granule', ''),
            'ecostress_valid_px': eco.get('valid_px', 0),
            'ecostress_tile_cover_frac': eco.get('aoi_frac', 0),
            'eco_lst_median_K': therm.get('median'), 'eco_lst_max_K': therm.get('max'),
            'eco_hot_px': therm.get('hot_px'), 'eco_hot_thresh_K': therm.get('thr'),
            's2_active_fire_px': r.get('fire_px'), 's2_burn_scar_pct': r.get('burn_pct'),
            's2_dt_h': r.get('s2_dt_h'),
            'lc_natural_px': r.get('natural'), 'lc_builtup_px': r.get('builtup'),
            'lc_crop_px': r.get('crop'),
            'fire_px_minus35d': r.get('before'), 'fire_px_plus35d': r.get('after'),
            'verdict': verdict,
        })

    rows.sort(key=lambda x: (not x['verdict'].startswith('WILDFIRE'),
                             -(x['eco_hot_px'] or 0),
                             -(x['eco_lst_max_K'] or 0)))
    with open(CSV_PATH, 'w', newline='', encoding='utf-8') as fh:
        wr = csv.DictWriter(fh, fieldnames=COLUMNS)
        wr.writeheader()
        wr.writerows(rows)
    print('wrote %s (%d tiles)' % (os.path.basename(CSV_PATH), len(rows)))

    made = 0
    for row in rows:
        if not row['verdict'].startswith('WILDFIRE'):
            continue
        stem = '%s_%s_%s_t%s_AOI' % (row['site_id'], slug(row['region'], 24),
                                     row['datatake'], row['tile_no'])
        path = write_layer(os.path.join(OUT_DIR, stem + '.shp'), stem, [row])
        zip_shapefile(path)
        made += 1
        print('  %-48s -> %s.zip' % (row['verdict'], stem))
    print('%d confirmed-wildfire AOI shapefiles in %s' % (made, OUT_DIR))

    # One combined layer carrying EVERY tile that was evaluated, rejected
    # ones included, each with the verdict that decided it. The per-site
    # AOIs above are what you upload to a search portal - one rectangle,
    # which is the only form every portal accepts. This layer is the audit
    # trail: it shows what was considered and why nine of ten tiles were
    # not chosen, which is the part a reviewer will ask about and which a
    # folder of three accepted AOIs cannot answer.
    all_path = write_layer(os.path.join(OUT_DIR, 'ALL_wildfire_candidates.shp'),
                           'ALL_wildfire_candidates', rows)
    zip_shapefile(all_path)
    print('combined layer: ALL_wildfire_candidates.shp (%d tiles, %d accepted)'
          % (len(rows), made))


if __name__ == '__main__':
    main()

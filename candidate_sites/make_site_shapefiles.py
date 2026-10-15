#!/usr/bin/env python
# coding: utf-8
"""Split the combined candidate-site layers into one shapefile per site, ready
to upload straight into a data-search portal as an area of interest.

Reads  : sites_ordering.csv, enmap_scenes.csv  (this directory)
Writes : sites/<SITEID>_<slug>_AOI.shp        one search polygon per site
         sites/<SITEID>_<slug>_scenes.shp     that site's EnMAP tile footprints
         sites/<SITEID>_<slug>_AOI.zip        the AOI, zipped

Everything is WGS84 (EPSG:4326) lon/lat, matching the combined layers.

Why one polygon per AOI file: EarthExplorer, EOWEB and AppEEARS all accept a
zipped shapefile as an AOI, but several of them either reject multi-part
inputs or silently use only the first feature. A single rectangle is the
form that works everywhere, so the AOI layer holds exactly one polygon and
the per-tile detail lives in the separate *_scenes layer.

Run from this directory with the emit environment, the one that has GDAL:
    C:/Users/shara/miniforge3/envs/emit/python.exe make_site_shapefiles.py
"""

import csv
import os
import re
import zipfile

from osgeo import ogr, osr

ogr.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'sites')

# Sites withdrawn on 2026-09-08 after retest_legacy_sites.py re-measured every
# S* tile with the corrected fire test (nearest Sentinel-2 pass at <=40 %
# cloud, offset recorded). These three have NO tile that reaches either
# threshold - burn >= 5 % or >= 50 active-fire px - so there is nothing in
# them for a fire study to work on. Their rows stay in sites_ordering.csv and
# enmap_scenes.csv as the audit trail; what is suppressed is the AOI, because
# an AOI is something you upload to a portal and a site with no fire must not
# be orderable by accident. Same rule that kept the rejected W* tiles out.
WITHDRAWN = {
    'S3': 'crop-residue burning: max burn 2.7 %, 34 fire px over 4 tiles',
    'S6': 'Mojave: max burn 2.1 %, 41 fire px on the single tile',
    'S7': 'max burn 1.6 %, 93 fire px over 8 tiles, and every Sentinel-2 '
          'partner is 24-48 h away; superseded by M1/M3 on the same fires',
}

# Shapefile (.dbf) field names are capped at 10 characters, so the long CSV
# column names are mapped to short ones here rather than being silently
# truncated - truncation would collide (e.g. 'val_west_lon'/'val_south_lat'
# both start the same) and produce unusable attributes.
AOI_FIELDS = [
    ('site_id', ogr.OFTString, 8), ('name', ogr.OFTString, 120),
    ('country', ogr.OFTString, 60), ('tier', ogr.OFTInteger, 0),
    ('datatake', ogr.OFTString, 24), ('acq_date', ogr.OFTString, 12),
    ('t_start', ogr.OFTString, 12), ('t_end', ogr.OFTString, 12),
    ('extent', ogr.OFTString, 20), ('n_scenes', ogr.OFTInteger, 0),
    ('n_valid', ogr.OFTInteger, 0), ('min_cloud', ogr.OFTReal, 0),
    ('fire_px', ogr.OFTInteger, 0),
    ('ls_partner', ogr.OFTString, 120), ('eco_partnr', ogr.OFTString, 120),
    ('west_lon', ogr.OFTReal, 0), ('south_lat', ogr.OFTReal, 0),
    ('east_lon', ogr.OFTReal, 0), ('north_lat', ogr.OFTReal, 0),
    # Tile identifiers, so the AOI file alone is enough to place an order:
    # a datatake is ordered tile by tile, and val_tiles says which of them
    # are worth ordering. best_scene is the full EnMAP L2A identifier EOWEB
    # matches on. (Names are at the 10-character dBase limit.)
    ('all_tiles', ogr.OFTString, 200), ('val_tiles', ogr.OFTString, 200),
    ('best_tile', ogr.OFTString, 8), ('best_scene', ogr.OFTString, 100),
    ('best_fire', ogr.OFTString, 12),
]

SCENE_FIELDS = [
    ('site_id', ogr.OFTString, 8), ('datatake', ogr.OFTString, 24),
    ('scene_id', ogr.OFTString, 90), ('tile_no', ogr.OFTString, 8),
    ('datetime', ogr.OFTString, 30), ('cloud_pct', ogr.OFTReal, 0),
    ('sun_elev', ogr.OFTReal, 0), ('validated', ogr.OFTString, 4),
    ('fire_px', ogr.OFTString, 12), ('eco_dt_h', ogr.OFTString, 12),
    ('ls_dt_h', ogr.OFTString, 12), ('ls_cloud', ogr.OFTString, 12),
]


def slug(text, maxlen=42):
    s = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')
    return s[:maxlen].rstrip('_')


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


def new_layer(path, name, fields):
    drv = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    ly = ds.CreateLayer(name, wgs84(), ogr.wkbPolygon)
    for fname, ftype, width in fields:
        fd = ogr.FieldDefn(fname, ftype)
        if width:
            fd.SetWidth(width)
        ly.CreateField(fd)
    return ds, ly


def setf(feat, name, value):
    if value is None or value == '':
        return
    feat.SetField(name, value)


def num(v, cast=float, default=None):
    try:
        return cast(float(v))
    except (TypeError, ValueError):
        return default


def zip_shapefile(shp_path):
    base = os.path.splitext(shp_path)[0]
    zpath = base + '.zip'
    with zipfile.ZipFile(zpath, 'w', zipfile.ZIP_DEFLATED) as z:
        for ext in ('.shp', '.shx', '.dbf', '.prj'):
            p = base + ext
            if os.path.exists(p):
                z.write(p, os.path.basename(p))
    return zpath


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(HERE, 'sites_ordering.csv'), newline='', encoding='utf-8-sig') as fh:
        sites = list(csv.DictReader(fh))
    with open(os.path.join(HERE, 'enmap_scenes.csv'), newline='', encoding='utf-8-sig') as fh:
        scenes = list(csv.DictReader(fh))

    by_site = {}
    for r in scenes:
        by_site.setdefault(r['site_id'], []).append(r)

    made = []
    for s in sites:
        sid = s['site_id']
        if sid in WITHDRAWN:
            print(f'{sid:>4}  withdrawn - {WITHDRAWN[sid]}')
            continue
        stem = f"{sid}_{slug(s['site_name'])}"

        # ---- AOI: one polygon --------------------------------------
        # Prefer the validated three-sensor window where one exists: that
        # is the sub-extent where EnMAP, Landsat and ECOSTRESS all actually
        # overlap, so it is the area worth ordering. Sites without one fall
        # back to the full datatake extent.
        has_val = all(s.get(k) for k in ('val_west_lon', 'val_south_lat',
                                         'val_east_lon', 'val_north_lat'))
        if has_val:
            w, so, e, n = (s['val_west_lon'], s['val_south_lat'],
                           s['val_east_lon'], s['val_north_lat'])
            extent, t0, t1 = 'validated_3sensor', s['val_time_start_utc'], s['val_time_end_utc']
        else:
            w, so, e, n = s['west_lon'], s['south_lat'], s['east_lon'], s['north_lat']
            extent, t0, t1 = 'datatake_full', s['acq_time_start_utc'], s['acq_time_end_utc']

        aoi_path = os.path.join(OUT, f'{stem}_AOI.shp')
        ds, ly = new_layer(aoi_path, f'{stem}_AOI', AOI_FIELDS)
        feat = ogr.Feature(ly.GetLayerDefn())
        feat.SetGeometry(rect(w, so, e, n))
        for k, v in (('site_id', sid), ('name', s['site_name']), ('country', s['country']),
                     ('tier', num(s['tier'], int)), ('datatake', s['datatake']),
                     ('acq_date', s['acq_date_utc']), ('t_start', t0), ('t_end', t1),
                     ('extent', extent),
                     ('n_scenes', num(s['n_scenes_in_datatake'], int)),
                     ('n_valid', num(s['n_scenes_3sensor_validated'], int)),
                     ('min_cloud', num(s['min_cloud_pct'])),
                     ('fire_px', num(s['s2_active_fire_px'], int)),
                     ('ls_partner', s['landsat_partner']),
                     ('eco_partnr', s['ecostress_partner']),
                     ('west_lon', num(w)), ('south_lat', num(so)),
                     ('east_lon', num(e)), ('north_lat', num(n)),
                     ('all_tiles', s.get('all_tile_nos', '')),
                     ('val_tiles', s.get('validated_tile_nos', '')),
                     ('best_tile', s.get('best_tile_no', '')),
                     ('best_scene', s.get('best_tile_scene_id', '')),
                     ('best_fire', s.get('best_tile_fire_px', ''))):
            setf(feat, k, v)
        ly.CreateFeature(feat)
        feat = None
        ds = None
        z = zip_shapefile(aoi_path)

        # ---- per-site EnMAP tile footprints ------------------------
        rows = by_site.get(sid, [])
        sc_path = None
        if rows:
            sc_path = os.path.join(OUT, f'{stem}_scenes.shp')
            ds, ly = new_layer(sc_path, f'{stem}_scenes', SCENE_FIELDS)
            for r in rows:
                feat = ogr.Feature(ly.GetLayerDefn())
                feat.SetGeometry(rect(r['west_lon'], r['south_lat'],
                                      r['east_lon'], r['north_lat']))
                for k, v in (('site_id', r['site_id']), ('datatake', r['datatake']),
                             ('scene_id', r['enmap_scene_id']), ('tile_no', r['tile_no']),
                             ('datetime', r['acq_datetime_utc']),
                             ('cloud_pct', num(r['cloud_pct'])),
                             ('sun_elev', num(r['sun_elev_deg'])),
                             ('validated', r['three_sensor_validated']),
                             ('fire_px', r['s2_active_fire_px']),
                             ('eco_dt_h', r['ecostress_dt_h']),
                             ('ls_dt_h', r['landsat_dt_h']),
                             ('ls_cloud', r['landsat_cloud_pct'])):
                    setf(feat, k, v)
                ly.CreateFeature(feat)
                feat = None
            ds = None

        made.append((sid, stem, extent, len(rows), os.path.basename(z)))
        print(f'{sid:>4}  {extent:<18} AOI + {len(rows):2d} scene footprints  ->  {stem}_*')

    print(f'\n{len(made)} sites written to {OUT}')
    return made


if __name__ == '__main__':
    main()

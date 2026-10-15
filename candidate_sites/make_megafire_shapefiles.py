#!/usr/bin/env python
# coding: utf-8
"""Per-site shapefiles for the megafire sites in MEGAFIRES_POST2023.md.

Reads  : megafire_enmap_verified.csv  (this directory)
Writes : sites/<SITEID>_<slug>_AOI.shp     one search polygon per site
         sites/<SITEID>_<slug>_AOI.zip     the AOI, zipped for upload
         sites/<SITEID>_<slug>_tiles.shp   that site's EnMAP tile footprints
         sites/ALL_megafire_sites.shp      every tile checked, with its verdict

Same conventions as make_site_shapefiles.py, which builds the S* layers:
WGS84 (EPSG:4326) lon/lat, one single-part polygon in each AOI layer because
EarthExplorer, EOWEB and AppEEARS variously reject multi-part AOIs or silently
use only the first feature, and the per-tile detail in a separate layer.

Two things differ from the S* builder, both because this search asks a
different question:

  * The AOI is the envelope of the tiles that have the FIRE in them, not the
    datatake extent. A megafire datatake is a ~500 km strip of which two or
    three tiles are over the burn; ordering the strip is not the intent.
  * Tier 3 sites get no AOI file, only a row in ALL_megafire_sites. They are
    fires EnMAP flew past rather than over, and an AOI is something you
    upload to a portal - a site that should not be ordered must not be
    sitting in the folder as an orderable polygon. This mirrors the rule in
    sites/README.md that kept the rejected W* tiles out of the AOI set.

Run from this directory with the emit environment, which is the one that has
GDAL:
    C:/Users/shara/miniforge3/envs/emit/python.exe make_megafire_shapefiles.py
"""

import collections
import csv
import json
import os
import re
import zipfile

from osgeo import ogr, osr

ogr.UseExceptions()

HERE = os.path.dirname(os.path.abspath(__file__))
IN_CSV = os.path.join(HERE, 'megafire_enmap_verified.csv')
# the verified CSV is per tile and carries no fire dates; they live here
IN_JSON = os.path.join(HERE, 'megafire_enmap_scenes.json')
OUT = os.path.join(HERE, 'sites')

# A tile counts as having the fire in it at the same threshold used to call
# the tier in MEGAFIRES_POST2023.md.
BURN_MIN = 5.0
FIRE_MIN = 50

# site_id -> (fire name in the CSV, datatakes, tier, one-line why)
SITES = [
    ('M1', 'Arganil / Piodao', ['DT0000149658'], 1,
     'Largest fire in Portuguese history, EnMAP overhead on day 6, 0% cloud'),
    ('M2', 'Molezuelas de la Carballeda', ['DT0000149654'], 1,
     'Record 2025 Spanish season, 2515 fire px at 0% cloud'),
    ('M3', 'Larouco / Ourense', ['DT0000149658'], 1,
     'Same Iberian strip as M1, second record fire at tiles 006/007'),
    ('M4', 'Jasper Wildfire Complex', ['DT0000088452'], 1,
     'Highest total fire-pixel count in the search, 0% cloud'),
    ('M5', 'Shoe Fire / Candle Lake', ['DT0000128782', 'DT0000148179', 'DT0000151064'], 1,
     'Largest extratropical fire with during-fire coverage; ignition-day pass'),
    ('M6', 'Chiquitania / Santa Cruz', ['DT0000088201'], 1,
     'Largest event covered while burning; 6 of 10 tiles on fire'),
    ('M7', 'Park Fire', ['DT0000095449', 'DT0000097032', 'DT0000097578'], 2,
     'Three cloud-free passes 4 and 8 d apart over one scar'),
    ('M8', 'Smokehouse Creek', ['DT0000064959', 'DT0000064963'], 2,
     'Largest US fire of 2024; 98% burn fraction, scar spectroscopy'),
    ('M9', 'Morrill Fire', ['DT0000188137', 'DT0000213914'], 2,
     'Largest US fire of 2026; post-fire only, 6 tiles at 0% cloud'),
    ('M10', 'Nopiming', ['DT0000137566'], 2,
     'Boreal scar at 0% cloud; Sentinel-2 verification loose (all passes clouded)'),
    ('M11', 'Gironde', ['DT0000208994', 'DT0000211127'], 2,
     'During-fire pass is 50-70% cloud; the 2026-08-11 post pass is clean'),
    ('M12', 'Dragon Bravo', ['DT0000152948'], 2,
     'Grand Canyon N rim, 0% cloud, eastern edge of the burn only'),
    ('M13', 'Grampians (Victoria)', ['DT0000130607'], 2,
     'Scar still strong at +4 months, 2-8% cloud'),
    # Tier 3: audit trail only, no AOI - see module docstring.
    ('M14', 'Sierra Oeste (Avila-Madrid)', ['DT0000206450', 'DT0000207273'], 3,
     'Largest Spanish fire ever but EnMAP flew either side of the perimeter'),
    ('M15', 'Uiseong-Andong', ['DT0000120726', 'DT0000126451'], 3,
     'Largest Korean fire ever; during-fire strip sits east of the burn'),
    ('M16', 'Pantanal', ['DT0000117655'], 3,
     'No EnMAP in the fire window; the one post scene is 5 months late'),
]

# .dbf field names are capped at 10 characters, so they are chosen short
# here rather than being silently truncated into collisions.
AOI_FIELDS = [
    ('site_id', ogr.OFTString, 8), ('name', ogr.OFTString, 60),
    ('region', ogr.OFTString, 60), ('tier', ogr.OFTInteger, 0),
    ('area_ha', ogr.OFTInteger, 0), ('ign_date', ogr.OFTString, 12),
    ('end_date', ogr.OFTString, 12), ('datatakes', ogr.OFTString, 120),
    ('phase', ogr.OFTString, 14), ('n_tiles', ogr.OFTInteger, 0),
    ('n_onfire', ogr.OFTInteger, 0), ('min_cloud', ogr.OFTReal, 0),
    ('max_fire', ogr.OFTInteger, 0), ('max_burn', ogr.OFTReal, 0),
    ('best_dt', ogr.OFTString, 24), ('best_tile', ogr.OFTString, 8),
    ('best_scene', ogr.OFTString, 100), ('onfire_tls', ogr.OFTString, 120),
    ('west_lon', ogr.OFTReal, 0), ('south_lat', ogr.OFTReal, 0),
    ('east_lon', ogr.OFTReal, 0), ('north_lat', ogr.OFTReal, 0),
    ('why', ogr.OFTString, 120),
]

TILE_FIELDS = [
    ('site_id', ogr.OFTString, 8), ('fire', ogr.OFTString, 60),
    ('datatake', ogr.OFTString, 24), ('tile_no', ogr.OFTString, 8),
    ('scene_id', ogr.OFTString, 100), ('datetime', ogr.OFTString, 30),
    ('cld_pct', ogr.OFTReal, 0), ('phase', ogr.OFTString, 8),
    ('area_ha', ogr.OFTInteger, 0), ('fire_px', ogr.OFTInteger, 0),
    ('burn_pct', ogr.OFTReal, 0), ('s2_scene', ogr.OFTString, 40),
    ('s2_dt_h', ogr.OFTReal, 0), ('s2_cld', ogr.OFTReal, 0),
    ('onfire', ogr.OFTString, 4), ('tier', ogr.OFTInteger, 0),
]


def slug(text, maxlen=42):
    s = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')
    return s[:maxlen].rstrip('_')


def onfire(r):
    return (float(r['s2_burn_scar_pct']) >= BURN_MIN
            or int(r['s2_active_fire_px']) >= FIRE_MIN)


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


def write_layer(path, fields, records):
    """records: list of (bbox, {field: value})."""
    drv = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(path):
        drv.DeleteDataSource(path)
    ds = drv.CreateDataSource(path)
    ly = ds.CreateLayer(os.path.splitext(os.path.basename(path))[0],
                        wgs84(), ogr.wkbPolygon)
    for fname, ftype, width in fields:
        fd = ogr.FieldDefn(fname, ftype)
        if width:
            fd.SetWidth(width)
        ly.CreateField(fd)
    for bbox, attrs in records:
        feat = ogr.Feature(ly.GetLayerDefn())
        feat.SetGeometry(rect(*bbox))
        for fname, _, _ in fields:
            v = attrs.get(fname)
            if v is not None and v != '':
                feat.SetField(fname, v)
        ly.CreateFeature(feat)
        feat = None
    ds = None


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
    with open(IN_CSV, newline='', encoding='utf-8-sig') as fh:
        rows = list(csv.DictReader(fh))
    with open(IN_JSON, encoding='utf-8') as fh:
        meta = json.load(fh)

    by_key = collections.defaultdict(list)
    for r in rows:
        by_key[(r['fire'], r['datatake'])].append(r)

    all_records = []
    made = []

    for sid, fire, dtks, tier, why in SITES:
        tiles = []
        for dtk in dtks:
            tiles.extend(by_key.get((fire, dtk), []))
        if not tiles:
            print('%-4s %-30s NO TILES - skipped' % (sid, fire))
            continue

        hot = [r for r in tiles if onfire(r)]
        stem = '%s_%s' % (sid, slug(fire))

        for r in tiles:
            all_records.append((
                [float(r['west_lon']), float(r['south_lat']),
                 float(r['east_lon']), float(r['north_lat'])],
                {'site_id': sid, 'fire': fire, 'datatake': r['datatake'],
                 'tile_no': r['tile_no'], 'scene_id': r['scene_id'],
                 'datetime': r['acq_datetime_utc'],
                 'cld_pct': float(r['enmap_cloud_pct'] or 0),
                 'phase': r['phase'], 'area_ha': int(r['area_ha']),
                 'fire_px': int(r['s2_active_fire_px']),
                 'burn_pct': float(r['s2_burn_scar_pct']),
                 's2_scene': r['s2_scene'], 's2_dt_h': float(r['s2_dt_h']),
                 's2_cld': float(r['s2_cloud_pct'] or 0),
                 'onfire': 'yes' if onfire(r) else 'no', 'tier': tier}))

        # ---- per-site tile footprints (every tile, hot or not) ----
        tile_path = os.path.join(OUT, '%s_tiles.shp' % stem)
        write_layer(tile_path, TILE_FIELDS,
                    [rec[0:2] for rec in all_records[-len(tiles):]])

        if tier == 3:
            print('%-4s %-30s tier 3: %d tiles, no AOI (audit only)'
                  % (sid, fire[:30], len(tiles)))
            made.append((sid, stem, tier, len(tiles), len(hot), None))
            continue

        # ---- AOI: envelope of the tiles that have the fire in them ----
        src = hot or tiles
        bbox = (min(float(r['west_lon']) for r in src),
                min(float(r['south_lat']) for r in src),
                max(float(r['east_lon']) for r in src),
                max(float(r['north_lat']) for r in src))
        best = max(hot or tiles, key=lambda r: (float(r['s2_burn_scar_pct']),
                                                int(r['s2_active_fire_px'])))
        phases = sorted(set(r['phase'] for r in src))
        aoi_path = os.path.join(OUT, '%s_AOI.shp' % stem)
        write_layer(aoi_path, AOI_FIELDS, [(bbox, {
            'site_id': sid, 'name': fire, 'region': tiles[0]['region'],
            'tier': tier, 'area_ha': int(tiles[0]['area_ha']),
            'ign_date': meta.get(fire, {}).get('ignition', ''),
            'end_date': meta.get(fire, {}).get('end', ''),
            'datatakes': ','.join(dtks), 'phase': '+'.join(phases),
            'n_tiles': len(tiles), 'n_onfire': len(hot),
            'min_cloud': min(float(r['enmap_cloud_pct'] or 0) for r in src),
            'max_fire': max(int(r['s2_active_fire_px']) for r in src),
            'max_burn': max(float(r['s2_burn_scar_pct']) for r in src),
            'best_dt': best['datatake'], 'best_tile': best['tile_no'],
            'best_scene': best['scene_id'],
            'onfire_tls': ','.join('%s/%s' % (r['datatake'][-6:], r['tile_no'])
                                   for r in hot),
            'west_lon': bbox[0], 'south_lat': bbox[1],
            'east_lon': bbox[2], 'north_lat': bbox[3],
            'why': why})])
        z = zip_shapefile(aoi_path)
        print('%-4s %-30s tier %d  AOI + %2d tiles (%d on fire)  -> %s_*'
              % (sid, fire[:30], tier, len(tiles), len(hot), stem))
        made.append((sid, stem, tier, len(tiles), len(hot), os.path.basename(z)))

    # ---- combined audit layer ----
    all_path = os.path.join(OUT, 'ALL_megafire_sites.shp')
    write_layer(all_path, TILE_FIELDS, all_records)
    zip_shapefile(all_path)
    print('\nALL_megafire_sites: %d tiles across %d sites'
          % (len(all_records), len(made)))
    print('%d sites written to %s' % (len(made), OUT))
    return made


if __name__ == '__main__':
    main()

# coding: utf-8
"""Fire catalogue for the coincident-data search (paper/data_search.tex).

Each entry: (key, name, region, start, end, bbox W,S,E,N, news reference key).
Dates and places come from the media / agency reports cited in
data_search.tex (key -> paper/data_search.bib). The bbox is a generous
envelope of the reported fire area: it only selects EnMAP tiles to test;
whether a tile actually holds a burning fire at the overpass is decided by
FIRMS detections inside the tile footprint within +/-3 h, never by the box.

The 22 fires of megafire_enmap.py and the S/W sites of README.md are
re-tested too, under the new time criterion.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

NEW_FIRES = [
    # ------------------------------------------------------------- 2023
    ('evros23', 'Evros / Dadia (Alexandroupolis)', 'Thrace, Greece', '2023-08-19', '2023-09-08',
     (25.55, 40.75, 26.45, 41.40), 'evros2023'),
    ('rhodes23', 'Rhodes', 'Dodecanese, Greece', '2023-07-18', '2023-07-30',
     (27.70, 35.95, 28.25, 36.35), 'rhodes2023'),
    ('tenerife23', 'Tenerife (Arafo-Candelaria)', 'Canary Islands, Spain', '2023-08-15', '2023-09-01',
     (-16.65, 28.18, -16.25, 28.52), 'tenerife2023'),
    ('quebec23', 'Lebel-sur-Quevillon / Chibougamau', 'Quebec, Canada', '2023-06-01', '2023-07-31',
     (-77.80, 48.50, -73.50, 50.60), 'canada2023'),
    ('donnie23', 'Donnie Creek', 'British Columbia, Canada', '2023-05-12', '2023-08-31',
     (-123.60, 56.80, -120.40, 58.30), 'canada2023'),
    ('nwt23', 'Yellowknife / Behchoko', 'Northwest Territories, Canada', '2023-08-01', '2023-09-15',
     (-116.90, 62.20, -113.40, 63.30), 'canada2023'),
    ('okanagan23', 'McDougall Creek / Bush Creek East', 'British Columbia, Canada', '2023-08-15', '2023-09-15',
     (-120.10, 49.80, -118.90, 51.20), 'canada2023'),
    ('chile23', 'Nuble / Biobio / Araucania', 'Chile', '2023-02-01', '2023-02-25',
     (-73.30, -38.70, -71.70, -36.50), 'chile2023'),
    ('abai23', 'Abai forest fire (Semey)', 'Abai, Kazakhstan', '2023-06-08', '2023-06-20',
     (79.80, 49.50, 81.60, 50.70), 'abai2023'),
    # ------------------------------------------------------------- 2024
    ('vina24', 'Vina del Mar / Quilpue', 'Valparaiso, Chile', '2024-02-02', '2024-02-08',
     (-71.65, -33.20, -71.25, -32.95), 'vina2024'),
    ('madeira24', 'Madeira', 'Portugal', '2024-08-14', '2024-08-28',
     (-17.30, 32.60, -16.80, 32.90), 'madeira2024'),
    ('aveiro24', 'Aveiro / Viseu', 'Portugal', '2024-09-15', '2024-09-22',
     (-8.80, 40.40, -7.60, 41.10), 'portugal2024'),
    ('attica24', 'Varnavas (NE Attica)', 'Greece', '2024-08-11', '2024-08-15',
     (23.70, 38.00, 24.10, 38.35), 'attica2024'),
    ('saopaulo24', 'Sao Paulo state fires', 'Brazil', '2024-08-22', '2024-08-30',
     (-49.60, -21.90, -47.40, -20.40), 'saopaulo2024'),
    ('line24', 'Line / Bridge / Airport fires', 'S California, USA', '2024-09-05', '2024-09-30',
     (-117.90, 33.60, -116.90, 34.40), 'socal2024'),
    # ------------------------------------------------------------- 2025
    ('ofunato25', 'Ofunato', 'Iwate, Japan', '2025-02-26', '2025-03-12',
     (141.62, 38.93, 141.88, 39.12), 'ofunato2025'),
    ('israel25', 'Eshtaol / Latrun', 'Jerusalem hills, Israel', '2025-04-30', '2025-05-03',
     (34.88, 31.72, 35.18, 31.92), 'israel2025'),
    ('flinflon25', 'Flin Flon / Creighton', 'Manitoba-Saskatchewan, Canada', '2025-05-25', '2025-06-30',
     (-102.40, 54.30, -101.40, 55.10), 'canada2025'),
    ('chios25', 'Chios', 'North Aegean, Greece', '2025-06-22', '2025-06-30',
     (25.85, 38.20, 26.20, 38.62), 'chios2025'),
    ('izmir25', 'Izmir (Odemis / Seferihisar)', 'Turkey', '2025-06-29', '2025-07-08',
     (26.60, 38.00, 28.20, 38.70), 'izmir2025'),
    ('latakia25', 'Qastal Maaf / Latakia', 'Syria', '2025-07-03', '2025-07-15',
     (35.75, 35.50, 36.25, 35.95), 'latakia2025'),
    ('gohrisch25', 'Gohrischheide', 'Saxony, Germany', '2025-07-01', '2025-07-12',
     (13.25, 51.28, 13.60, 51.46), 'gohrisch2025'),
    ('cyprus25', 'Limassol (Malia / Souni)', 'Cyprus', '2025-07-23', '2025-07-27',
     (32.70, 34.68, 33.05, 34.88), 'cyprus2025'),
    ('aude25', 'Ribaute / Lagrasse (Corbieres)', 'Aude, France', '2025-08-05', '2025-08-12',
     (2.45, 42.92, 2.90, 43.18), 'aude2025'),
    ('gifford25', 'Gifford Fire', 'California, USA', '2025-08-01', '2025-09-05',
     (-120.45, 34.80, -119.75, 35.30), 'gifford2025'),
    ('jarilla25', 'Jarilla / Valle del Jerte', 'Caceres, Spain', '2025-08-12', '2025-08-22',
     (-6.25, 40.00, -5.65, 40.40), 'spain2025'),
    ('medulas25', 'Las Medulas / El Bierzo', 'Leon, Spain', '2025-08-10', '2025-08-25',
     (-7.10, 42.25, -6.40, 42.75), 'spain2025'),
    ('dava25', 'Dava Moor / Carrbridge', 'Highlands, Scotland', '2025-06-28', '2025-07-03',
     (-3.95, 57.30, -3.45, 57.50), 'dava2025'),
    # ------------------------------------------------------------- 2026
    ('chile26', 'Penco-Lirquen and Nuble/Biobio fires', 'Chile', '2026-01-15', '2026-01-31',
     (-73.20, -37.40, -71.80, -36.20), 'chile2026'),
    ('chubut26', 'Chubut (Epuyen / Los Alerces)', 'Patagonia, Argentina', '2026-01-05', '2026-02-15',
     (-71.95, -43.25, -71.15, -41.95), 'chubut2026'),
    ('victoria26', 'Victoria bushfires (Jan 2026)', 'Victoria, Australia', '2026-01-05', '2026-01-31',
     (142.50, -38.80, 148.50, -35.90), 'victoria2026'),
    ('otsuchi26', 'Otsuchi', 'Iwate, Japan', '2026-04-22', '2026-05-10',
     (141.78, 39.28, 141.98, 39.46), 'otsuchi2026'),
    ('mourne26', 'Mourne Mountains', 'Northern Ireland, UK', '2026-04-20', '2026-04-30',
     (-6.12, 54.10, -5.85, 54.26), 'uk2026'),
    ('cottonwood26', 'Cottonwood Fire (Beaver)', 'Utah, USA', '2026-06-22', '2026-07-20',
     (-113.10, 37.95, -112.30, 38.55), 'us2026'),
    ('aspenacres26', 'Aspen Acres Fire (Beulah)', 'Colorado, USA', '2026-06-29', '2026-07-20',
     (-105.35, 37.85, -104.80, 38.30), 'us2026'),
    ('ontario26', 'NW Ontario fires (incl. Collins FN)', 'Ontario, Canada', '2026-07-05', '2026-07-31',
     (-94.00, 49.50, -86.00, 53.00), 'ontario2026'),
    ('mierla26', 'La Mierla (Sierra Norte de Guadalajara)', 'Castilla-La Mancha, Spain', '2026-07-16', '2026-07-30',
     (-3.45, 40.80, -2.85, 41.25), 'mierla2026'),
    ('attica26', 'W Attica / Boeotia', 'Greece', '2026-07-25', '2026-08-06',
     (22.95, 38.00, 23.70, 38.45), 'europe2026'),
    ('aegean26', 'Aegean Turkey (Izmir, Manisa, Balikesir)', 'Turkey', '2026-07-24', '2026-08-06',
     (26.50, 38.00, 28.60, 39.80), 'europe2026'),
    ('portugal26', 'Vouzela / Valpacos', 'N Portugal', '2026-07-15', '2026-08-20',
     (-8.35, 40.50, -7.10, 41.85), 'europe2026'),
    ('sicily26', 'Madonie / San Cataldo', 'Sicily, Italy', '2026-07-18', '2026-08-02',
     (13.55, 37.45, 14.40, 38.00), 'europe2026'),
    ('spokane26', 'Spokane-area fires', 'Washington, USA', '2026-08-01', '2026-08-20',
     (-117.75, 47.45, -117.05, 47.95), 'us2026'),
    ('bc26', 'Southern BC fires (Aug 2026 emergency)', 'British Columbia, Canada', '2026-08-01', '2026-08-31',
     (-124.00, 49.00, -117.00, 52.00), 'bc2026'),
]


def old_fires():
    """The 22 megafires of the earlier size-first search and the four
    surviving timing-first sites, re-tested here."""
    import megafire_enmap as mf
    out = []
    for name, region, _area, start, end, bbox in mf.FIRES:
        key = 'mf_' + ''.join(c for c in name.lower() if c.isalnum())[:14]
        out.append((key, name, region, start, end, bbox, 'megafire'))
    out += [
        ('s1_yalu', 'Yalu River valley', 'Liaoning/Jilin, China', '2026-05-28', '2026-05-30',
         (124.00, 39.77, 125.18, 42.52), 'sites'),
        ('s2_neuquen', 'Neuquen steppe', 'Neuquen, Argentina', '2026-03-24', '2026-03-26',
         (-69.15, -39.20, -68.54, -38.34), 'sites'),
        ('s4_laichau', 'Lai Chau / Lao Cai swidden', 'NW Vietnam', '2026-03-30', '2026-04-01',
         (102.40, 22.00, 103.20, 23.30), 'sites'),
        ('s5_bc22', 'SW British Columbia / N Washington', 'Canada/USA', '2022-08-10', '2022-08-12',
         (-123.48, 48.39, -122.67, 49.27), 'sites'),
        ('w04_myanmar', 'Myanmar swidden', 'Myanmar', '2025-03-22', '2025-03-24',
         (94.0, 20.0, 97.5, 24.5), 'sites'),
    ]
    return out


def all_fires():
    return NEW_FIRES + old_fires()


# Systematic sweep: every EnMAP tile over the main fire regions in their
# fire seasons, Jun 2022 (first EnMAP data in the archive) - Oct 2026.
# (region key, name, bbox W,S,E,N, [(start MM-DD, end MM-DD, year offset of end)])
SWEEP = [
    ('iberia', 'Iberia and S France', (-9.6, 36.0, 4.5, 44.6), [('06-01', '10-31', 0)]),
    ('emed', 'Italy, Balkans, Greece, Turkey, Levant', (8.0, 31.0, 45.0, 44.0), [('06-01', '09-30', 0)]),
    ('wus', 'Western USA', (-125.0, 31.0, -102.0, 49.0), [('06-01', '10-31', 0)]),
    ('wcan', 'Western Canada', (-130.0, 49.0, -95.0, 62.0), [('05-01', '09-15', 0)]),
    ('ecan', 'Eastern Canada', (-95.0, 46.0, -60.0, 56.0), [('05-01', '08-31', 0)]),
    ('seaus', 'SE Australia', (138.0, -39.5, 154.0, -27.0), [('11-01', '02-28', 1)]),
    ('naus', 'N Australia savanna', (115.0, -20.0, 146.0, -11.0), [('05-01', '10-31', 0)]),
    ('samz', 'Bolivia / Brazil (Chiquitania, Pantanal, Cerrado)', (-66.0, -24.0, -44.0, -8.0), [('07-15', '10-15', 0)]),
    ('chlarg', 'Central Chile / N Patagonia', (-74.0, -44.0, -68.0, -32.0), [('12-01', '03-31', 1)]),
    ('safr', 'Southern African savanna', (12.0, -20.0, 36.0, -4.0), [('06-01', '09-30', 0)]),
    ('wafr', 'Sahel / Sudanian savanna', (-17.0, 5.0, 32.0, 14.0), [('11-15', '02-15', 1)]),
    ('seasia', 'Mainland SE Asia', (94.0, 14.0, 108.0, 25.0), [('02-01', '04-30', 0)]),
]


def sweep_regions(y0=2022, y1=2026, last='2026-11-15', first='2022-06-01'):
    out = []
    for key, name, bbox, seasons in SWEEP:
        for y in range(y0 - 1, y1 + 1):
            for a, b, off in seasons:
                start, end = f'{y}-{a}', f'{y + off}-{b}'
                if end < first or start > last:
                    continue
                start, end = max(start, first), min(end, last)
                out.append((f'sw_{key}_{start[:4]}{start[5:7]}', name, name, start, end, bbox, 'sweep'))
    return out

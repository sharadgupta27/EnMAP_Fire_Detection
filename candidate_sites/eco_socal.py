"""Is ECOSTRESS really absent, or is the search wrong?
Inventory every ECOSTRESS LSTE granule over Los Angeles for all of Jan 2025.
"""
import earthaccess, requests

BBOX = (-118.9, 33.8, -117.7, 34.9)     # the burning part of LA county
earthaccess.login(strategy='netrc')

for sn in ('ECO_L2T_LSTE', 'ECO_L2G_LSTE', 'ECO2LSTE'):
    try:
        g = earthaccess.search_data(short_name=sn, bounding_box=BBOX,
                                    temporal=('2025-01-01T00:00:00Z', '2025-02-01T00:00:00Z'),
                                    count=500)
    except Exception as ex:
        print('%-14s search failed: %s' % (sn, ex)); continue
    times = sorted({x['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'] for x in g})
    print('%-14s %d granules over LA in Jan 2025' % (sn, len(g)))
    for t in times:
        print('      ', t)

# Same question straight from CMR, bypassing the earthaccess layer entirely.
print('\n--- direct CMR, ECO_L2T_LSTE, all versions ---')
for ver in ('002', '003'):
    r = requests.get('https://cmr.earthdata.nasa.gov/search/granules.json',
                     params={'short_name': 'ECO_L2T_LSTE', 'version': ver,
                             'bounding_box': '-118.9,33.8,-117.7,34.9',
                             'temporal': '2025-01-01T00:00:00Z,2025-02-01T00:00:00Z',
                             'page_size': 500}, timeout=120)
    entries = r.json().get('feed', {}).get('entry', []) if r.ok else []
    print('version %s: HTTP %s, %d granules' % (ver, r.status_code, len(entries)))
    for e in entries[:20]:
        print('      ', e['title'][:70], e['time_start'])

# Sanity check the query itself: the same box, a month we know has data.
print('\n--- control: same box, August 2025 ---')
g = earthaccess.search_data(short_name='ECO_L2T_LSTE', bounding_box=BBOX,
                            temporal=('2025-08-01T00:00:00Z', '2025-08-08T00:00:00Z'), count=200)
print('%d granules in one week of Aug 2025 (proves the query works)' % len(g))

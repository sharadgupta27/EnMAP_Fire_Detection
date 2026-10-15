"""How far is the NEAREST ECOSTRESS granule from each EnMAP fire tile, and
does that granule actually hold data? Widen to +/-48 h so the answer is a
measured gap rather than an empty result."""
import json, os, sys, warnings
warnings.filterwarnings('ignore')
os.environ['GDAL_DISABLE_READDIR_ON_OPEN'] = 'EMPTY_DIR'
from datetime import timedelta

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
import earthaccess
import fire_check as fc
from pass2_eco import granule_valid_pixels

with open(os.path.join(SP, 'socal_pass2.json')) as _fh:
    rows = [x for x in json.load(_fh) if x['fires']]
earthaccess.login(strategy='netrc')
sess = earthaccess.get_requests_https_session()

print('%-30s %-20s %-26s %8s %10s' % ('EnMAP tile', 'EnMAP time UTC', 'nearest ECOSTRESS', 'dt (h)', 'valid px'))
print('-' * 100)
for x in rows:
    t = fc.iso(x['dt'])
    lo = (t - timedelta(hours=48)).strftime('%Y-%m-%dT%H:%M:%SZ')
    hi = (t + timedelta(hours=48)).strftime('%Y-%m-%dT%H:%M:%SZ')
    grans = earthaccess.search_data(short_name='ECO_L2T_LSTE', bounding_box=tuple(x['bbox']),
                                    temporal=(lo, hi), count=200)
    cand = []
    for g in grans:
        links = [l for l in g.data_links() if l.endswith('_LST.tif')]
        if not links:
            continue
        gt = fc.iso(g['umm']['TemporalExtent']['RangeDateTime']['BeginningDateTime'])
        cand.append((abs((gt - t).total_seconds()) / 3600.0, (gt - t).total_seconds() / 3600.0,
                     os.path.basename(links[0]), links[0]))
    cand.sort()
    tag = x['id'].split('-DT')[1][:26]
    if not cand:
        print('%-30s %-20s %-26s %8s %10s' % (tag, x['dt'][:19], 'NONE within +/-48 h', '-', '-'))
        x['eco_nearest'] = None
        continue
    ad, dh, nm, url = cand[0]
    res = granule_valid_pixels(url, x['bbox'], sess)
    vp = res.get('valid_px', 'read-fail')
    print('%-30s %-20s %-26s %+8.2f %10s' % (tag, x['dt'][:19], nm.split('_LSTE_')[1][:26], dh, vp))
    x['eco_nearest'] = {'granule': nm, 'dh': round(dh, 2), **res}

with open(os.path.join(SP, 'socal_pass3.json'), 'w') as _fh:
    json.dump(rows, _fh, indent=1)
print('\nwrote socal_pass3.json')

"""SoCal January 2025 - pass 1: EnMAP tiles, sun elevation, fire containment,
Landsat 8/9 availability. All metadata, no downloads."""
import json, requests
from datetime import datetime, timedelta

DLR = "https://geoservice.dlr.de/eoc/ogc/stac/v1/search"
LL = "https://landsatlook.usgs.gov/stac-server/search"
S = requests.Session(); S.headers["User-Agent"] = "enmap-fire-search/1.0"
def iso(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))

BBOX = [-119.6, 33.3, -116.8, 34.9]

# The January 2025 Los Angeles firestorm. Ignition point and the approximate
# perimeter of each named fire, from the CAL FIRE incident record.
FIRES = [
    ('Palisades', '2025-01-07', '2025-01-31', -118.62, 34.02, -118.47, 34.13),
    ('Eaton',     '2025-01-07', '2025-01-31', -118.17, 34.16, -118.03, 34.27),
    ('Hurst',     '2025-01-07', '2025-01-16', -118.52, 34.32, -118.42, 34.40),
    ('Sunset',    '2025-01-08', '2025-01-10', -118.40, 34.08, -118.33, 34.13),
    ('Kenneth',   '2025-01-09', '2025-01-13', -118.72, 34.16, -118.62, 34.23),
    ('Hughes',    '2025-01-22', '2025-01-31', -118.75, 34.44, -118.58, 34.58),
    ('Border2',   '2025-01-23', '2025-02-06', -116.98, 32.55, -116.80, 32.72),
]

def overlaps(tb, fb):
    return not (tb[2] < fb[0] or tb[0] > fb[2] or tb[3] < fb[1] or tb[1] > fb[3])

r = S.post(DLR, json={"collections": ["ENMAP_HSI_L2A"], "bbox": BBOX,
                      "datetime": "2025-01-05T00:00:00Z/2025-02-05T23:59:59Z",
                      "limit": 200}, timeout=180)
feats = r.json()["features"]
print("%d EnMAP L2A tiles in the fire window\n" % len(feats))

rows = []
for f in sorted(feats, key=lambda x: x["properties"]["datetime"]):
    p = f["properties"]; b = f["bbox"]; t = iso(p["datetime"])
    day = p["datetime"][:10]
    hits = [nm for nm, s, e, *fb in FIRES
            if s <= day <= e and overlaps(b, fb)]
    rows.append({"id": f["id"], "dt": p["datetime"], "bbox": b,
                 "cloud": p.get("eo:cloud_cover"),
                 "sun_el": p.get("view:sun_elevation"),
                 "fires": hits})

print("%-46s %-20s %5s %6s  %s" % ("EnMAP tile", "acquired (UTC)", "cloud", "sun_el", "active fire inside"))
print("-" * 118)
for x in rows:
    tag = x["id"].split("-DT")[1][:26]
    mark = ",".join(x["fires"]) if x["fires"] else "-"
    print("DT%-44s %-20s %5s %6s  %s"
          % (tag, x["dt"][:19], x["cloud"], x["sun_el"], mark))

hot = [x for x in rows if x["fires"]]
print("\n%d of %d tiles contain an actively-burning fire\n" % (len(hot), len(rows)))

# Landsat 8/9 surface temperature within +/-24 h of each such tile
for x in hot:
    t = iso(x["dt"])
    lo = (t - timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    hi = (t + timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = S.post(LL, json={"collections": ["landsat-c2l2-st"], "bbox": x["bbox"],
                         "datetime": "%s/%s" % (lo, hi), "limit": 100}, timeout=120)
    best = []
    for f in r.json().get("features", []):
        p = f["properties"]; pl = (p.get("platform") or "").upper()
        if pl not in ("LANDSAT_8", "LANDSAT_9"):
            continue
        dh = (iso(p["datetime"]) - t).total_seconds() / 3600.0
        cc = p.get("eo:cloud_cover")
        best.append((abs(dh), dh, f["id"], pl, cc))
    best.sort()
    x["landsat"] = [{"id": i, "plat": pl, "dh": round(d, 2), "cloud": cc}
                    for _, d, i, pl, cc in best[:3]]
    print("DT%s  %s  [%s]" % (x["id"].split("-DT")[1][:26], x["dt"][:16], ",".join(x["fires"])))
    if not best:
        print("      no Landsat 8/9 within +/-24 h")
    for _, d, i, pl, cc in best[:3]:
        print("      %-42s %-10s dt=%+6.2f h  cloud=%s" % (i, pl, d, cc))

with open("socal_pass1.json", "w") as _fh:
    json.dump(rows, _fh, indent=1)
print("\nwrote socal_pass1.json")

"""Is the 06:25 UTC EnMAP datatake really a night acquisition?
Check the STAC properties AND compute the solar position independently."""
import math, requests
from datetime import datetime, timezone

S = requests.Session()
r = S.post("https://geoservice.dlr.de/eoc/ogc/stac/v1/search",
           json={"collections": ["ENMAP_HSI_L2A"], "bbox": [-119.6, 33.3, -116.8, 34.9],
                 "datetime": "2025-01-09T00:00:00Z/2025-01-21T23:59:59Z", "limit": 50},
           timeout=180)
feats = r.json()["features"]

def solar_elevation(dt, lat, lon):
    """NOAA solar position, good to ~0.1 deg - enough to tell day from night."""
    jd = dt.timestamp() / 86400.0 + 2440587.5
    n = jd - 2451545.0
    L = math.radians((280.460 + 0.9856474 * n) % 360)
    g = math.radians((357.528 + 0.9856003 * n) % 360)
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    dec = math.asin(math.sin(eps) * math.sin(lam))
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    lst = math.radians((gmst * 15 + lon) % 360)
    H = lst - ra
    la = math.radians(lat)
    return math.degrees(math.asin(math.sin(la) * math.sin(dec)
                                  + math.cos(la) * math.cos(dec) * math.cos(H)))

print("%-46s %-21s %8s %8s  %s" % ("tile", "datetime UTC", "STAC", "computed", "local solar time"))
print("-" * 108)
for f in sorted(feats, key=lambda x: x["properties"]["datetime"]):
    p = f["properties"]; b = f["bbox"]
    dt = datetime.fromisoformat(p["datetime"].replace("Z", "+00:00")).astimezone(timezone.utc)
    lat, lon = (b[1] + b[3]) / 2, (b[0] + b[2]) / 2
    el = solar_elevation(dt, lat, lon)
    lstime = (dt.hour + dt.minute / 60.0 + lon / 15.0) % 24
    print("%-46s %-21s %8s %8.2f  %02d:%02d local"
          % (f["id"].split("-DT")[1][:26], p["datetime"][:19], p.get("view:sun_elevation"),
             el, int(lstime), int(lstime % 1 * 60)))

# full property dump of one night tile, to see how DLR labels it
night = [f for f in feats if "T0625" in f["id"] or "T0628" in f["id"]]
if night:
    print("\nfull properties of %s:" % night[0]["id"])
    for k, v in sorted(night[0]["properties"].items()):
        print("   %-34s %s" % (k, v))

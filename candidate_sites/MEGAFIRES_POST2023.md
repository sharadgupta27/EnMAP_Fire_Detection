# Very large wildfires since 2024 that EnMAP actually flew over

Search run 2026-09-08. WGS84 / EPSG:4326, all times UTC.

This is the size-first counterpart to `README.md`. That search started from
EnMAP datatakes and ranked them by three-sensor coincidence; the fires it
found are mostly small (crop-residue burning, a few hundred Sentinel-2 fire
pixels). This one starts from the **megafires** - 22 of the largest
individual fire events worldwide since January 2024, taken from the national
agency / EFFIS / CIFFC record - and asks whether EnMAP flew each perimeter.

| File | Contents |
|---|---|
| `megafire_enmap.py` | The fire list, and the DLR STAC query. Two windows per fire: `during` (ignition−2 d … containment+2 d) and `post` (containment … +180 d). |
| `megafire_enmap_coverage.csv` | One row per fire - scene counts in each window, best scene in each. |
| `megafire_enmap_scenes.json` | Every matching L2A scene, with per-tile bbox, time, cloud. |
| `megafire_verify.py` | Second pass: reads Sentinel-2 L2A SWIR inside each **tile's own** footprint. |
| `megafire_enmap_verified.csv` | One row per EnMAP tile, 110 tiles - active-fire px and burn-scar % in that tile. |
| `make_megafire_shapefiles.py` | Builds the `sites/M*` layers below from that CSV. Needs GDAL, so run it with the `emit` interpreter. |
| `sites/M<n>_<name>_AOI.shp` (+`.zip`) | One search polygon per tier-1/2 site - the envelope of the tiles that have the fire in them, with the evidence as attributes. Upload the `.zip` to EOWEB / EarthExplorer. |
| `sites/M<n>_<name>_tiles.shp` | That site's EnMAP tile footprints, each carrying its own fire px / burn % / `onfire` flag. |
| `sites/ALL_megafire_sites.shp` | All 110 tiles across all 16 sites, tier-3 included - the audit trail. Do **not** upload as an AOI; several portals use only the first feature. |

The verification pass matters. A padded perimeter envelope is 100–250 km
across, an EnMAP tile is ~30 km; a datatake can intersect the box and see
nothing but unburnt rangeland. Of the 110 tiles checked, **61 have the fire
in them** (burn ≥ 5 % or ≥ 50 active-fire px); the other 49 are near misses.
The tile lists below are only the tiles that pass.

Fire evidence is the Murphy et al. (2016) SWIR test (B12/B11 ≥ 1.4 and
B12/B8A ≥ 1.4 and B12 ≥ 0.15) and burn scar is NBR(B8A,B12) < −0.10, both
from the nearest Sentinel-2 pass at ≤ 40 % cloud within ±5 d. So `fire px` is
evidence the tile was burning *around* the EnMAP overpass, not a count at the
overpass instant; `burn %` is the steadier number.

---

## Tier 1 - big fire, EnMAP overhead while it burned, cloud-free

### M1 · Arganil / Piódão, Coimbra, Portugal - **best active-fire site**
Largest wildfire in Portuguese history, 64,451 ha, 13–24 Aug 2025.
- **EnMAP 2025-08-19 12:15 UTC · `DT0000149658` · tiles 014, 015** · cloud **0 %**
- Tile 014: **2,656 fire px**, burn 22.5 % · tile 015: 410 fire px, burn 24.4 %
- Sentinel-2 partner `S2C_29TPE_20250821`, +47 h, cc 0 %
- Search bbox (W,S,E,N): `-8.25, 39.95, -7.45, 40.55`

EnMAP was overhead on day 6 of an 11-day fire, at the point of maximum spread,
with no cloud. Nothing else in this search comes close on active fire.

### M2 · Molezuelas de la Carballeda, Zamora, Spain
~37,000 ha, 10–25 Aug 2025 - the record 2025 Spanish season.
- **EnMAP 2025-08-22 11:53 UTC · `DT0000149654` · tiles 004, 005** · cloud **0 %**
- Tile 004: **2,515 fire px**, burn 15.1 % · tile 005: 509 fire px, burn 6.0 %
- Sentinel-2 `S2B_29TQG_20250823`, +23 h, cc 0 %
- Search bbox: `-6.70, 41.75, -5.85, 42.45`

This is the datatake `README.md` lists as S8; the note there ("~5,700 px,
burn 22 %") is the union over the whole strip. Per tile it is 004 and 005.

### M3 · Larouco / Ourense, Galicia, Spain - *same datatake as M1*
~30,000 ha, 12–25 Aug 2025.
- **EnMAP 2025-08-19 12:15 UTC · `DT0000149658` · tiles 006, 007** · cloud 4–34 %
- Tile 006: 59 fire px, burn **17.7 %** · tile 007: 12 fire px, burn 4.6 %
- Search bbox: `-7.60, 41.85, -6.80, 42.45`

`DT0000149658` is one ~500 km N–S Iberian strip that crosses **two** record
fires - Ourense at tiles 006/007 and Arganil at 014/015. One order covers both.

### M4 · Jasper Wildfire Complex, Alberta, Canada
32,722 ha, 22 Jul – 7 Sep 2024; destroyed a third of the Jasper townsite.
- **EnMAP 2024-08-08 06:12 UTC · `DT0000088452` · tiles 001, 002** · cloud **0 %**
- Tile 001: **2,416 fire px**, burn 12.8 % · tile 002: 2,214 fire px, burn 24.3 %
- Sentinel-2 `S2B_11UMU_20240809`, +37 h, cc 30 %
- Search bbox: `-118.60, 52.55, -117.40, 53.30`

Highest total fire-pixel count of any site here (4,645 across the strip).
Mountain terrain, so expect topographic shadow in the L1B.

### M5 · Shoe Fire / Candle Lake, Saskatchewan - **largest fire with during-fire coverage outside the tropics**
565,701 ha, 7 May – 31 Aug 2025. Three usable datatakes:

| datatake | date | phase | tiles with fire | cloud | fire px | burn % |
|---|---|---|---|---|---|---|
| `DT0000128782` | 2025-05-07 | during (**ignition day**) | 007, 008 | 0 % | 236 / 0 | 16.7 / **44.3** |
| `DT0000148179` | 2025-08-19 | during | 004–006, 008, 009 | 0–56 % | up to 95 | up to 11.2 |
| `DT0000151064` | 2025-09-07 | post | 003–006 | **0 %** | up to 356 | up to 29.4 |

- Search bbox: `-106.60, 53.40, -103.80, 55.10`

`DT0000128782` is the same day the fire started, at 0 % cloud, and tile 008
is already 44 % burnt while tile 007 still has 236 active-fire px - this is a
boreal crown fire caught on day one, front and scar in the same strip.

### M6 · Chiquitania / Santa Cruz, Bolivia - largest event in the list
~3 Mha across the 2024 Bolivian season, Aug–Oct 2024.
- **EnMAP 2024-08-12 15:02 UTC · `DT0000088201` · tiles 011–016** (6 of 10)
- Cloud 0–45 % on the fire tiles; 246 fire px and burn to 20.6 % at tile 014
- Search bbox: `-63.60, -18.20, -60.80, -15.30`

Already in `README.md` as S10, with the Landsat/ECOSTRESS partners; listed
here because it is the largest fire EnMAP covered while burning.

---

## Tier 2 - big fire, EnMAP over a fresh scar rather than flame

### M7 · Park Fire, California - 173,900 ha, 24 Jul – 15 Oct 2024
Three cloud-free datatakes over the scar, all within the containment period:

| datatake | date | tiles with fire | cloud | fire px | burn % |
|---|---|---|---|---|---|
| `DT0000095449` | 2024-09-27 | 002, 003 | 0 % | 25 | 13.2 |
| `DT0000097032` | 2024-10-01 | 002, 003, 005 | 0 % | 25 | **57.2** |
| `DT0000097578` | 2024-10-05 | 001–003, 005 | 0 % | 14 | **73.1** |

- Search bbox: `-122.05, 39.55, -121.10, 40.50`

Three passes 4 and 8 days apart over the same scar at 0 % cloud is the best
**repeat** coverage in the list - usable for post-fire severity and early
recovery, not for flame temperature.

### M8 · Smokehouse Creek, Texas Panhandle - 428,352 ha, largest US fire of 2024
- **EnMAP 2024-03-11 18:11 UTC · `DT0000064963` · tiles 001, 002** · cloud **0 %** · burn **98.2 %** / 90.1 %
- Also `DT0000064959`, 2024-03-07, tiles 001/002, cloud 25–58 %, burn 94.7 %
- Search bbox: `-101.60, 35.45, -100.00, 36.30`

Near-total burn fraction - a uniformly blackened grassland scar filling the
tile, five days before containment. Almost no active fire left by then (3 px),
so this is a scar-spectroscopy site, not a temperature-retrieval one.

### M9 · Morrill Fire, W Nebraska - 260,359 ha, largest US fire of 2026
- **No EnMAP during the fire** (12–25 Mar 2026).
- `DT0000188137`, 2026-04-14, tile 002, cloud 6 %, burn 9.8 % (3 weeks after)
- `DT0000213914`, 2026-08-27, tiles 005–010, cloud **0 %**, burn to **57.9 %**
- Search bbox: `-103.40, 40.95, -101.20, 42.20`

The August strip covers six tiles of the scar cloud-free, but at +5 months in
Sandhills grassland, so part of the low NBR is bare sand and part is scar -
separate the two before using it.

### M10 · Nopiming, Manitoba - ~130,000 ha, 13 May – 31 Aug 2025
- `DT0000137566`, 2025-06-24 17:53 UTC, tiles 002, 003, cloud **0 %**, burn 33.8 % / 12.4 %
- Search bbox: `-96.10, 50.20, -94.90, 51.30`
- Caveat: every Sentinel-2 pass within ±1 d is 100 % cloud; the numbers above
  come from 2025-06-28 (+96 h) and 2025-06-18 (−144 h). The EnMAP tiles
  themselves are 0 % cloud, so the scene is fine - the *verification* is loose.

### M11 · Gironde, France - ~37,000 ha, Jul–Aug 2026
- `DT0000208994`, 2026-07-30 (during), tiles 002–005, 313 fire px - but cloud **50–70 %**
- `DT0000211127`, 2026-08-11 (post), tiles 005, 006, cloud 0–1 %, burn 18.5 %
- Search bbox: `-1.30, 44.25, -0.10, 45.05`

The during-fire pass is real fire but too cloudy; the post pass is clean.

### M12 · Dragon Bravo, Grand Canyon North Rim - 58,700 ha, Jul–Sep 2025
- `DT0000152948`, 2025-09-15 18:47 UTC, tiles 012, 013, cloud **0 %**, burn 15.4 % / 14.0 %
- Search bbox: `-112.60, 36.10, -111.80, 36.75`
- Only two tiles, and only the eastern edge of the burn.

### M13 · Grampians, Victoria, Australia - 76,000 ha, Dec 2024 – Jan 2025
- `DT0000130607`, 2025-05-18, tiles 012, 013, 015, cloud 2–8 %, burn to 72.5 %
- Search bbox: `142.00, -37.70, 142.80, -36.75`
- +4 months, so recovery has started, but the scar is still strong.

---

## Tier 3 - large fire, EnMAP coverage too weak to order

| Fire | ha | Why not |
|---|---|---|
| Sierra Oeste (Ávila–Madrid), Spain 2026 | ~60,000 (largest Spanish fire ever) | Two cloud-free during-fire datatakes, `DT0000206450` (07-27) and `DT0000207273` (07-31), but they straddle the fire: 06450 sits east of it (only tile 015 reaches the burn, 44 fire px / 9 % burn) and 07273 sits west (max 2.7 % burn). EnMAP flew either side of the perimeter, not over it. |
| Uiseong–Andong, South Korea 2025 | ~45,000 (largest Korean fire ever) | `DT0000120726` on 2025-03-26, mid-fire, but the strip is east of the burn (max 0.8 % burn). Post-fire `DT0000126451` (2025-04-26, 0 % cloud) reaches tiles 004/005 at 7.3 % / 5.9 %. Marginal. |
| Pantanal, Brazil 2024 | ~1,000,000 | No EnMAP in the fire window at all. The one post scene, `DT0000117655` (2025-03-07), is 5 months late and shows 0.9 % burn - nothing left to see. |
| Windy Deuce, Texas 2024 | 58,300 | No EnMAP in either window. |
| Los Alerces, Chubut, Argentina 2025 | 41,000 | No EnMAP in either window. |
| White Sage, Arizona 2025 | 32,000 | One post tile only. |
| Parker Lake / Fort Nelson, BC 2024 | 13,000 | Post only, all cloudy. |
| Palisades / Eaton, Los Angeles 2025 | 9,490 / 5,675 | Well covered (see `SOCAL_JAN2025.md`) but not large fires. |

---

## How to order

Same as `README.md`. The public DLR STAC exposes **L2A only**; **L1B must be
ordered through EOWEB GeoPortal** (https://eoweb.dlr.de) with the datatake ID
and tile numbers above. Set start = end = the acquisition date - the datatake
IDs are unique.

## Caveats

1. **The fire list is curated, not exhaustive.** It covers named,
   individually-reported fire events. It deliberately does not enumerate
   African savanna burning (DR Congo alone: 13.6 Mha in 2026) or Australian
   northern savanna (11.8 Mha in 2026), which are seasonal landscape burning
   rather than discrete events. If you want those, add bboxes to `FIRES` in
   `megafire_enmap.py` and re-run - the machinery is region-agnostic.
2. **Burned areas are the reported final figures**, which for multi-fire
   complexes (Chiquitania, Pantanal) are seasonal country/region totals, not
   one perimeter. They rank the entries; they are not scene-level numbers.
3. **The SWIR test cannot separate a wildfire from a gas flare or a
   steelworks** - see the note in `wildfire_4sensor.py`. Every site above is
   a fire with an independent published record, so that failure mode is
   already excluded here, but the raw `fire px` counts carry it.
4. **Verification is per tile bbox, not per tile footprint.** EnMAP tiles are
   rotated relative to north, so the bbox is up to ~30 % larger than the tile.
   Exact footprints for the earlier search are in `enmap_footprints.geojson`;
   for these datatakes, pull them from the STAC `geometry` field.

# EnMAP + Landsat 8/9 + ECOSTRESS sample sites over fire

Search run 2026-09-05. Everything below is in **WGS84 / EPSG:4326**, all times **UTC**.

## Files

| File | Contents |
|---|---|
| `sites_ordering.csv` | **One row per site** - bounding box, exact date, time window, datatake ID, partner Landsat/ECOSTRESS granules. This is the file to order from. |
| `enmap_scenes.csv` | One row per EnMAP tile (85 tiles) - tile number, exact acquisition time, cloud %, per-tile bbox, whether all three sensors were validated on it, fire counts. |
| `site_bboxes.shp` (+`.dbf/.shx/.prj`) | 10 site rectangles as a shapefile. |
| `site_bboxes.geojson` | Same, GeoJSON. |
| `enmap_footprints.shp` (+`.dbf/.shx/.prj`) | Exact EnMAP tile footprint polygons (85), from the DLR STAC - not rectangles. |
| `enmap_footprints.geojson` | Same, GeoJSON. |
| `MEGAFIRES_POST2023.md` + `megafire_enmap_*.csv/.json` | The size-first search: the 22 largest individual wildfires worldwide since 2024, and which of them EnMAP actually flew over. Different question from this table - order from there if you want a **big** fire rather than a well-timed one. |
| `SOCAL_JAN2025.md` + `socal_jan2025_scorecard.csv` | Why the January 2025 Los Angeles firestorm is **not** a four-sensor site, and the two EnMAP *night* acquisitions over it that are worth ordering anyway. |

Two bounding boxes are given per site in `sites_ordering.csv`:
- `west_lon/south_lat/east_lon/north_lat` - the **whole datatake strip**;
- `val_west_lon/...` - only the tiles where **all three sensors were confirmed to overlap**. Use this one unless you want the full strip.

## Tier 1 - all three sensors within ~1.3 h, active fire confirmed

Order these. `dt` = partner time minus EnMAP time.

### S1 · Yalu River valley, Liaoning/Jilin (China–DPRK border) - **best site**
- **Date: 2026-05-29**, EnMAP 03:05:19–03:05:59 UTC · datatake `DT0000198082` · tiles **024–033** (10 tiles)
- **Search bbox (W,S,E,N): `124.0080, 39.7716, 125.1785, 42.5207`**  (full strip: `124.0080, 39.7716, 126.5131, 46.5629`)
- Cloud 0–1 % on every tile
- Landsat 8 `LC08_L2SP_118033_20260529_02_T1`, 02:22 UTC, **dt −0.74 h**, cloud 0.06–11.5 %
- ECOSTRESS `ECOv002_L2T_LSTE_44770_008_*` (tiles 51TXG/51TXF/51TWF/51TXE), 04:08 UTC, **dt +1.04 h**
- Sentinel-2 active fire: 7–192 px per tile, ~627 total (fire in *every* tile)

| tile | time UTC | cloud | centre lat, lon | fire px | burn % |
|---|---|---|---|---|---|
| 024 | 03:05:19 | 0 % | 42.360, 124.954 | 25 | 5.3 |
| 025 | 03:05:24 | 0 % | 42.090, 124.871 | 46 | 2.4 |
| 026 | 03:05:28 | 0 % | 41.820, 124.788 | 53 | 1.6 |
| 027 | 03:05:32 | 0 % | 41.550, 124.706 | 56 | 1.3 |
| 028 | 03:05:37 | 0 % | 41.280, 124.625 | 42 | 1.1 |
| 029 | 03:05:41 | 0 % | 41.010, 124.543 | 7 | 0.1 |
| 030 | 03:05:46 | 0 % | 40.740, 124.462 | 18 | 0.3 |
| 031 | 03:05:50 | 0 % | 40.470, 124.382 | 83 | 0.9 |
| 032 | 03:05:55 | 1 % | 40.200, 124.302 | **192** | 0.7 |
| 033 | 03:05:59 | 0 % | 39.931, 124.223 | **105** | 1.4 |

### S2 · Neuquén, Patagonian steppe (Argentina) - best active fire + fresh burn scar
- **Date: 2026-03-25**, EnMAP 15:05:54–15:06:03 UTC · datatake `DT0000185160` · tiles **001–003**
- **Search bbox: `-69.1426, -39.1990, -68.5429, -38.3421`**
- Cloud 0 % on all three tiles
- Landsat 9 `LC09_L2SP_231087_20260325_02_T1`, **dt −0.72 h**, cloud 0.05 %
- ECOSTRESS `ECOv002_L2T_LSTE_43762_022_19HDT_20260325T134701`, **dt −1.32 h**
- Fire: tile 001 = 72 px / burn 11.9 %; tile 002 = 9 px / burn 22.6 %; tile 003 = 0 px / burn 29.6 %

### ~~S3 · Vojvodina / S Hungary (crop-residue burning)~~ - WITHDRAWN 2026-09-08
- **Date: 2026-08-04**, EnMAP 10:11:29–10:11:38 UTC · datatake `DT0000207603` · tiles **006–008**
- **Search bbox: `18.6199, 45.8047, 19.2854, 46.6652`**
- Cloud 0–3 %
- Landsat 8 `LC08_L2SP_187028_20260804_02_T1`, **dt −0.74 h**, cloud 0.19 %
- ECOSTRESS `ECOv002_L2T_LSTE_45814_008_33TYM_20260804T091902`, **dt −0.88 h**
- Fire: 8 / 25 / 5 px
- **Withdrawn**: the corrected retest gives max burn 2.7 % and 34 fire px over 4 tiles. Shapefiles deleted; see `sites/README.md`.

### S4 · Lai Châu / Lào Cai, NW Vietnam (swidden burning) - same biome as the current Myanmar site
- **Date: 2026-03-31**, EnMAP 04:19:33 UTC · datatake `DT0000185878` · tile **003**
- **Search bbox: `102.7647, 22.3829, 103.1144, 22.6965`**
- Cloud 10 %
- Landsat 9 `LC09_L2SP_129045_20260331_02_T1`, **dt −0.73 h**, cloud 7.6 %
- ECOSTRESS `ECOv002_L2T_LSTE_43850_003_48QUK_20260331T031817`, **dt −1.02 h**
- Fire: 35 px. Tiles 001/002 of this datatake have more fire (≈145–151 px) but ECOSTRESS does **not** cover them.

### S5 · SW British Columbia / N Washington
- **Date: 2022-08-11**, EnMAP 19:59:24–19:59:32 UTC · datatake `DT0000002425` · tiles **001–003**
- **Search bbox: `-123.4706, 48.3938, -122.6770, 49.2611`**
- Cloud 0 %
- Landsat 8, **dt −0.86 h**, cloud 12.5 %
- ECOSTRESS `ECOv002_L2T_LSTE_*_10UDV_20220811T200835`, **dt +0.15 h** (closest of any site)
- Fire: 51 / 47 / 7 px

### ~~S6 · Mojave / Victorville, California~~ - WITHDRAWN 2026-09-08
- **Date: 2024-10-19**, EnMAP 19:06:16 UTC · datatake `DT0000099384` · tile **001**
- **Search bbox: `-117.5597, 34.4697, -117.1571, 34.7836`**
- Cloud 1 %
- Landsat 9, **dt −0.73 h**, cloud 0.41 %
- ECOSTRESS `ECOv002_L2T_LSTE_*_11SMU_20241019T192357`, **dt +0.29 h**
- Fire: 41 px
- **Withdrawn**: the corrected retest gives max burn 2.1 % and 41 fire px on the one tile. Shapefiles deleted; see `sites/README.md`.

## Tier 2 - big fire fronts, but the three sensors are days apart

Use these if you want a large, well-documented fire and can accept loose timing.

| # | Site | Date | Datatake | Search bbox (W,S,E,N) | Landsat | ECOSTRESS | S2 fire |
|---|---|---|---|---|---|---|---|
| ~~S7~~ **withdrawn** | Galicia / N Portugal | 2025-08-07 | `DT0000146122` | `-8.9262, 41.4770, -7.8527, 43.6824` | `LC09_L2SP_205029_20250807` **−0.74 h**, cc 0.17 % | `ECOv002_L2T_LSTE_40187_011_*` +3.81 h | 78 px + 196 saturated |
| S8 | Ourense / Zamora (record 2025 fires) | 2025-08-19 | `DT0000149658` | `-8.3720, 39.3440, -6.2201, 43.6859` | `LC09_L2SP_205030_20250823` +95.1 h, cc 0.28 % | `ECOv002_L2T_LSTE_40383_003_*` +17.1 h | **~5,700 px**, burn 22 % |
| S9 | León / Zamora | 2025-08-22 | `DT0000149654` | `-6.5104, 41.3365, -5.6298, 43.0016` | `LC09_L2SP_203031_20250825` +71.2 h, cc 1.53 % | `ECOv002_L2T_LSTE_40413_002_*` −8.19 h | ~2,600 px, burn 23 % |
| S10 | Chiquitania, Bolivia | 2024-08-12 | `DT0000088201` | `-63.5340, -17.9681, -62.3057, -13.5772` | `LC08_L2SP_230071_20240811` −24.9 h, cc 0.00 % | `ECOv002_L2T_LSTE_34605_013_*` +16.7 h | 506 px, burn 15–20 % |

~~S7 is the compromise: Landsat is only 44 min from EnMAP and ECOSTRESS is same-day (+3.8 h), but the fire is far smaller than S8/S9.~~

**S7 was withdrawn on 2026-09-08.** Re-measured with the corrected fire test it
has max burn 1.6 % and 93 fire px over 8 tiles, and every Sentinel-2 partner is
24-48 h from the overpass - the same non-contemporaneous evidence that sank W01.
`MEGAFIRES_POST2023.md` covers the same August-2025 Iberian fires far better as
M1 (Arganil, 3,066 fire px at 0 % cloud) and M3 (Ourense, on the same datatake).

## How to order

**EnMAP (DLR EOWEB GeoPortal, https://eoweb.dlr.de)** - the public STAC exposes L2A only; **L1B must be ordered through EOWEB** with your existing login.
1. Collection `EnMAP HSI L1B` (and/or L2A).
2. Area of interest: paste the search bbox, or import `site_bboxes.shp` / `site_bboxes.geojson`.
3. Date: set start = end = the site's date above (a single day is enough - the datatake IDs are unique).
4. Confirm you got the right datatake ID, then add the listed tile numbers to the cart.

**Landsat 8/9** - USGS EarthExplorer or M2M, `Landsat Collection 2 Level-2` (surface temperature `ST_B10` is the band the pipeline uses). Scene IDs are given above.

**ECOSTRESS** - NASA Earthdata Search / LP DAAC, product `ECO_L2T_LSTE` v002. Granule IDs above.
Reminder: the `L2T_LSTE` GeoTIFFs are float32 **already in Kelvin** - do not apply the 0.02 scale factor
(see `Fire_analysis_with_EnMAP.py:1217`, which currently does and produces ~5.8 K).

## How the sites were found

1. **EnMAP** - DLR STAC (`https://geoservice.dlr.de/eoc/ogc/stac/v1`, collection `ENMAP_HSI_L2A`), queried over 17 fire-prone regions × their fire seasons, 2022–2026 → 29,315 scenes in 3,418 datatakes.
2. **Landsat 8/9** - USGS LandsatLook STAC, `landsat-c2l2-st`.
3. **ECOSTRESS** - NASA CMR, `ECO_L2T_LSTE` v002/v003.
4. Datatakes with all three within ±1 day → **731 triples**. Top candidates then re-checked **per individual EnMAP tile**, because union-bbox matching overstates overlap (this step removed the Sahel candidates entirely and cut S4 from 3 tiles to 1).
5. **Fire confirmed independently** from open Sentinel-2 L2A COGs (Element84 Earth Search) using the Murphy et al. (2016) SWIR test: B12/B11 ≥ 1.4 **and** B12/B8A ≥ 1.4 **and** B12 ≥ 0.15; burn scar = NBR(B8A,B12) < −0.10.

Caveat: the separate saturation count (B12 ≥ 0.8) also trips on snow and bright cloud, so ranking used the ratio test only. Fire counts come from the nearest usable Sentinel-2 pass, which is up to ~1 day from the EnMAP acquisition, so treat them as evidence that the area was burning, not as an exact count at EnMAP overpass time.

## On the existing Myanmar/Mizoram site
`DT0000009864`, 2023-03-14, ~23.9 N 93.3 E is a weak triple: EnMAP 2023-03-14, ECOSTRESS 2023-03-12 at 22:45 UTC (**night**, ~04:15 local), Landsat 9 2023-03-01 - a 13-day spread with a day/night mismatch. Max LST in that ECOSTRESS granule is 299.5 K, i.e. no fire signal.

# January 2025 Southern California firestorm - four-sensor availability

Question asked: do the January 2025 SoCal wildfires have EnMAP **and**
Landsat 8/9 **and** Sentinel-2 **and** ECOSTRESS?

**Answer: no.** EnMAP, Landsat and Sentinel-2 are all there, and two of the
EnMAP acquisitions are unusually good. ECOSTRESS never comes within 12 h of
any of them, so the four-sensor conjunction this study needs does not exist
for this event.

Search run 2026-09-07. All times UTC. Table: `socal_jan2025_scorecard.csv`.

## What EnMAP holds

17 EnMAP L2A tiles fall inside `-119.6, 33.3, -116.8, 34.9` between
2025-01-05 and 2025-02-05. Eight of them contain a fire that was burning on
the day of acquisition (Palisades, Eaton, Hurst, Sunset, Kenneth; ignition
and perimeter dates from the CAL FIRE incident record).

**Two of the eight datatakes are night acquisitions**, which is the
interesting finding here. `DT0000112000` (2025-01-10 06:25) and
`DT0000111999` (2025-01-14 06:28) were taken at 22:30 local solar time on an
ASCENDING orbit - EnMAP's routine pass is descending at ~11:20 local. Solar
elevation is **-65 degrees**, confirmed two ways: the STAC reports
`view:sun_elevation = 0.0`, and an independent NOAA solar-position
calculation (`socal_night.py`) gives -65.5. DLR evidently tasked EnMAP over
the firestorm.

That matters for temperature retrieval. With no solar illumination there is
no reflected component in the SWIR, so the measured radiance *is* emission.
The two-component fire/background model loses its largest ambiguity - the
background endmember - and the retrieval becomes far better posed than any
daytime scene can be.

Two practical caveats on those night scenes:

* **Use L1B, not L2A.** Atmospheric correction at night is meaningless and
  the products say so: `sceneAOT = -999`, `sceneWV = -999`,
  `productScreeningFailedGroups = TEMPERATURE`, `productScreeningResultStatus
  = WARNING`, `enmap:status = LOW`. The pipeline reads L1B radiance anyway.
* **L1B must be ordered.** The DLR STAC `ENMAP_HSI_L1B` collection holds only
  ~800 items, a rolling recent window, against 235,294 for L2A. January 2025
  L1B is not in the STAC and has to be requested through the EnMAP portal.

## Why it fails: ECOSTRESS

ECOSTRESS is the binding constraint, as it was for every other candidate in
this study. It is not an absence of data - CMR returns **76 `ECO_L2T_LSTE`
granules over Los Angeles in January 2025**, and the granule of
2025-01-09 11:19 reaches **LST 398.6 K** over the Palisades tile, so
ECOSTRESS plainly saw the fire. The ISS overpasses simply never coincide
with an EnMAP pass:

| EnMAP tile | EnMAP time | nearest ECOSTRESS | gap | valid px |
|---|---|---|---|---|
| DT0000112000 t001-003 | 2025-01-10 06:25 | 36919_015 (01-09 11:19) | -19.1 h | 187k-268k |
| DT0000111999 t001-002 | 2025-01-14 06:28 | 37006_003 (01-15 01:32) | +19.1 h | 101k-166k |
| DT0000112557 t003 (Eaton) | 2025-01-16 19:13 | 37041_011 (01-17 08:03) | +12.8 h | **0** |
| DT0000112556 t004-005 | 2025-01-20 19:16 | 37087_013 (01-20 07:15) | -12.0 h | 61k / 35 |

The smallest gap is 12.0 h, and the single closest granule in time
(12.8 h, Eaton) is entirely nodata. A 12-hour gap spans the day-night
cycle, so it cannot validate a fire temperature.

For comparison, site `W01` (Colorado Rockies, 2026-08-20) has all four
sensors within **1.39 h**.

## The near misses, for the record

* **2025-01-20 19:16, Palisades** (`DT0000112556` tiles 004/005) pairs with
  Sentinel-2 `S2B_11SLT_20250120` at **-0.36 h - 22 minutes** - carrying 25
  active-fire pixels and 17.1 % burn scar. An excellent EnMAP/Sentinel-2
  pair. Landsat is 47 h away at 31 % cloud, ECOSTRESS 12 h.
* **2025-01-14 06:28, Palisades** (night) has Landsat 9
  `LC09_L2SP_041036_20250114` at +12.00 h and 0.21 % cloud - but that is the
  daytime pass, a different thermal state entirely, not a match.
* **2025-01-16 19:13, Eaton** has the tightest all-sensor spread on paper
  (Landsat -24.8 h at 0.03 % cloud, Sentinel-2 +23.5 h) and its ECOSTRESS
  granule is empty.

Zero of the eight tiles put all four sensors within 3 h; the best worst-case
spread is 35.7 h.

## Verdict

Do not use January 2025 SoCal as the four-sensor validation site - keep
`W01`. It is worth ordering as a **separate night-time case study**: EnMAP
L1B `DT0000112000` (2025-01-10) and `DT0000111999` (2025-01-14) over an
actively burning Palisades, where the absence of a solar background makes
the retrieval cleaner than any daytime scene, validated against ECOSTRESS
LST ~19 h either side as context rather than as coincident truth.

Nothing here was verified at the EnMAP pixel level: DLR requires an EOC
login to download, so acquisition geometry and catalogue availability are
confirmed, fire content within the EnMAP tiles is not.

## Reproducing

    python socal_pass1.py    # EnMAP tiles, solar elevation, fire containment, Landsat
    python socal_pass2.py    # ECOSTRESS pixel counts, Sentinel-2 fire test, wide Landsat
    python socal_pass3.py    # nearest ECOSTRESS within +/-48 h, with data check
    python socal_night.py    # independent solar-position check of the night acquisitions
    python eco_socal.py      # full ECOSTRESS inventory over LA for January 2025

`socal_pass2.py` and `socal_pass3.py` need Earthdata credentials in `~/.netrc`
and the GDAL environment (`miniforge3/envs/emit`).

# Thermal companion sensors for EnMAP

Everything here answers one question: **what supplies surface temperature
alongside an EnMAP scene, given that EnMAP itself has no thermal bands and
ECOSTRESS is not always available?**

Two new modules:

| module | what it does |
|---|---|
| `thermal_partners.py` | find, download and load thermal granules matching an EnMAP scene |
| `thermal_sharpening.py` | put a coarse LST field onto EnMAP's 30 m grid (DisTrad/TsHARP) |

Tests: `algorithm_verification/test_thermal_partners.py` (add `--live` for
archive queries) and `algorithm_verification/test_thermal_sharpening.py`.

---

## Run it

### From the GUI

The window is now four tabs instead of two, because the old single **Setup**
tab had grown to three unrelated concerns stacked vertically and the last of
them - *Detection & fitting* - was being pushed off the bottom edge with no
scrollbar and no way to reach it:

| tab | what is on it |
|---|---|
| 📡 **Scene** | EnMAP product, second detector, merge toggles, output directory |
| 🌡 **Thermal partners** | the five reference sensors + Find thermal partners |
| 🔬 **Detection & fitting** | detection method, sensitivity, trials, performance |
| 📊 **Results** | log, preview, summary (unchanged) |

Every tab is wrapped in a scrolling container, so no setting can become
unreachable again on a smaller screen, at a different DPI scale, or with a
larger system font. The scrollbar only appears when it is actually needed.

On **Thermal partners**, each sensor shows what it *is* under its name -
`70 m · ISS, 52°S–52°N only · NASA LP DAAC` - rather than hiding that in a
tooltip. Five rows that all read "label, entry, two buttons, checkbox" are
genuinely hard to tell apart; the resolution and archive are what
distinguish them.

- **VIIRS LST** - pick a `.h5`, or tick *Auto-download*
- **Sentinel-3 SLSTR** - pick the `.SEN3` directory, or tick *Auto-download*
- **🔍 Find thermal partners** - searches all five archives and prints the
  ranked table into the Results log. Read-only: downloads nothing, needs no
  login. Use it to see what exists before committing to a run.

Both auto-download boxes behave like the existing ones. VIIRS shares the
**same NASA Earthdata login as ECOSTRESS**, so one prompt covers both, and
skipping that prompt disables both. Sentinel-3 uses a *different* account
(Copernicus), which is checked **before** the run starts rather than failing
twenty minutes in - if `CDSE_USERNAME`/`CDSE_PASSWORD` are unset you are
offered the choice of continuing without Sentinel-3 or going back.

Results appear as extra columns (`viirs_LST_K`, `slstr_LST_K`) in the
results CSV, with their own bias/RMSE lines and their own acquisition-time
offsets in the log.

### From the command line

```bash
conda activate emit                       # see "Environment" below - this matters

# the pipeline, with the new partners
python Fire_analysis_with_EnMAP.py <scene> --viirs-auto-fetch --slstr-auto-fetch
python Fire_analysis_with_EnMAP.py <scene> --viirs granule.h5 --slstr product.SEN3

# just the discovery tool
python thermal_partners.py dims_op_oc_oc-en_703961221_1 --window 12 --csv partners.csv
```

```python
import thermal_partners as tp

scene = tp.enmap_scene_reference('dims_op_oc_oc-en_703961221_1')
df = tp.match_thermal_partners(scene, window_hours=12)   # all five archives

best = df[~df['time_is_nominal']].iloc[0]                # closest real timestamp
path = tp.download_earthdata_granule(best['urls'][0], 'results_temperature/partners')
```

Loaders return the dict contract the pipeline already speaks -
`{'lst_K', 'geotransform', 'projection_wkt'}` - so a VIIRS or SLSTR result
drops straight into `validate_temperature()` and `_sample_raster_at_lonlat()`
with no adaptation.

### How it is wired in

Three additive changes; nothing existing changed behaviour.

| where | change |
|---|---|
| `validate_temperature()` | new `extra_references=` parameter - a list of `{label, data, datetime, ceiling_K, column}` dicts. Each gets its own sampled column, bias/RMSE line and saturation caveat. |
| `run_pipeline()` | new `viirs_path`, `viirs_auto_fetch`, `slstr_path`, `slstr_auto_fetch`, `partner_max_time_diff_hours` |
| `_load_thermal_partners()` | new helper that resolves those into `extra_references` |

`_load_thermal_partners()` **returns `[]` and warns in every failure mode** -
missing `thermal_partners.py`, uninstalled `h5py`/`netCDF4`, absent
credentials, an archive being down, or simply no granule in the window. These
two sensors are an addition to the ECOSTRESS/Landsat pair, so none of them
may turn a working run into a failed one. With no partner requested the
helper returns immediately and touches no network.

`ceiling_K` is carried **per reference**, not shared: SLSTR saturates at
355.5 K, *below* ECOSTRESS's ~380 K, so a shared default would under-report
saturated comparisons.

---

## What was implemented, and what was verified

Every access route below was exercised against the live archive before it
was written. Nothing here is coded from a specification that was not opened.

| sensor | product | resolution | search | download | load |
|---|---|---|---|---|---|
| ECOSTRESS | `ECO_L2T_LSTE` v002 **and v003** | 70 m | ✅ CMR | ✅ Earthdata | already in main module |
| VIIRS LST | `VNP21A1D/N`, `VJ121A1D/N`, `VJ221A1D/N` | 1 km | ✅ CMR | ✅ Earthdata | ✅ `load_viirs_lst()` |
| VIIRS fire | `VNP14IMG` (375 m, FRP) | 375 m | ✅ CMR | ✅ Earthdata | - |
| Landsat 8/9 | `landsat-c2l2-st` | 100 m → 30 m | ✅ LandsatLook STAC | main module | already in main module |
| Sentinel-3 | SLSTR `SL_2_LST___`, `SL_2_FRP___` | 1 km | ✅ CDSE OData | ✅ CDSE OAuth | ✅ `load_slstr_lst()` |

Searching v002 **and** v003 of ECOSTRESS was not cosmetic: the second test
scene (NE China, 2026-05-29) is covered only by **v003**. Querying v002
alone would have reported no ECOSTRESS coverage for a scene that has it.

### The four-sensor stack, measured

Both test scenes found a full stack. For EnMAP `DT0000099384`
(2024-10-19T19:06:13Z, Mojave), the granules were downloaded and read:

| sensor | offset from EnMAP | local solar | mean LST over the footprint |
|---|---|---|---|
| Sentinel-3A SLSTR | −1.25 h | 10.0 h | 303.43 K |
| Landsat 9 | −0.73 h | 10.5 h | 307.71 K |
| ECOSTRESS | +0.30 h | 11.6 h | 309.18 K |
| VIIRS (SNPP) | +1.90 h | 13.2 h | 311.59 K |

The four order **monotonically by acquisition time**, warming ~2.6 K/h as a
desert surface climbs toward the afternoon peak. Nothing enforces that
ordering in the code - it is four independent instruments, three archives,
two projections and two resampling paths agreeing. It is the strongest
single check that the georeferencing, unpacking and time decoding are right.

---

## The time-matching trap

The gridded daily VIIRS LST products are labelled with a **date**, not an
overpass time - CMR reports their extent as 00:00:00–23:59:59. Matching on
that label is meaningless; it makes every granule look like a perfect match.

The real observation time is per pixel, in the product's own `View_Time`
layer, in hours of **local solar** time. `load_viirs_lst()` converts it back
to UTC using each pixel's own longitude (`utc = view_time − lon/15`).

For the Mojave scene this is not a technicality. The granule's metadata
lists five overpass windows that day, one of them **19:18–19:24 UTC - 12
minutes from EnMAP**. The composite did not use it: `View_Time` over the
scene is 13.2 h local solar, i.e. **~21:02 UTC, +1.90 h**. Reporting that
granule as a 12-minute match would have been wrong by a factor of ten, in
the direction that matters most, because LST changes fastest either side of
local noon.

So `match_thermal_partners()` marks those rows `time_is_nominal=True`,
reports `dt_hours` as `NaN` rather than a number, and **sorts them below
every granule with a real timestamp** regardless of apparent offset. They
stay listed - above 52° N they are often the only partner available - but
they can never masquerade as a close match.

---

## Saturation ceilings - read before comparing against a fire

Every partner here measures in the 8–12 µm window and departs from
linearity far below a flaming front. The ceilings differ, and one is lower
than people expect:

| sensor | ceiling | why |
|---|---|---|
| Sentinel-3 SLSTR | **355.5 K** | int16 packing, `0.002·DN + 290` - an *encoding* limit |
| ECOSTRESS | ~380 K | retrieval design (`_ECOSTRESS_CEILING_K`) |
| Landsat TIRS | ~372 K | band design |
| VIIRS M15/M16 | ~380 K | retrieval design |

SLSTR's is read from the file at load time, not hard-coded, so a future
baseline that repacks the variable recomputes it.

**None of these can measure a fire.** They supply *background* temperature.
The sub-pixel fire temperature comes from the EnMAP SWIR retrieval - that is
what `temperature_methods.py` exists for. The one partner that does not
saturate on hot targets is **VIIRS M13 at 3.99 µm** (dual-gain, linear to
~634 K), which is why `VNP14IMG` is searched: it reports fire radiative
power for pixels whose 11 µm LST is pinned at its ceiling.

---

## Thermal sharpening - and an honest negative result

`thermal_sharpening.py` implements DisTrad (Kustas et al. 2003) / TsHARP
(Agam et al. 2007) with residual correction.

**The mathematics is exact and verified.** Against synthetic truth with a
known temperature/vegetation relation it recovers the coefficients to 0.04 %
(−25.010 against −25.0) and cuts RMSE from 3.345 K to 0.754 K (77.5 %
better, against an irreducible 0.800 K noise floor). Aggregated back to the
sensor grid the sharpened field reproduces the measurement to **1.7e-13 K** -
exact conservation, which holds for nonlinear models too because the
residual is defined against the cell mean of the *fine predictions* rather
than against `f(mean predictor)`.

**On the scene in this repository it does not help.** Sharpening ECOSTRESS,
SLSTR and VIIRS onto the 30 m grid and scoring each against Landsat 9's
independent 30 m ST:

| coarse source | ratio | in-sample R² | cv R² | pattern RMSE change |
|---|---|---|---|---|
| ECOSTRESS 70 m | 2.3× | 0.315 | 0.314 | **−4.3 %** |
| SLSTR 1 km | 33× | 0.067 | −0.028 | +1.6 % |
| VIIRS 1 km | 33× | 0.346 | 0.293 | **−33.3 %** |

The cause is visible in the predictor: scene-median NDVI is −0.118 and
median fractional vegetation cover 0.10. This is **near-bare desert**, and
the temperature/vegetation relation the method depends on is not there.
This is the documented failure mode of DisTrad/TsHARP over arid ground, not
a defect in the implementation.

Three things follow, all built into the code:

1. **In-sample R² must not be used to decide.** The VIIRS fit had the
   *highest* R² of the three and gave by far the worst result. The module
   reports `cv_r2` from **spatially-blocked** cross-validation instead -
   blocked, because both fields are strongly autocorrelated and random
   k-fold measures interpolation between neighbours, not generalisation.

2. **`cv_r2` is necessary but not sufficient, and the module says so.** It
   correctly rejected SLSTR (−0.028) but passed VIIRS (0.293), which then
   failed badly. The reason is structural: `cv_r2` measures skill *between*
   coarse cells, while sharpening depends on the relation holding *within*
   them, and no coarse-scale statistic can settle that. (A scale-invariance
   test - refitting on 2× and 4× aggregations - was tried and does not
   separate these cases either; VIIRS had the *most* stable slope, 4.4 %
   spread.) So the gate is a floor, not a certificate. To establish that
   sharpening helped on a given scene, use **`validate_sharpening()`**
   against an independent 30 m thermal product. That is the only test that
   answers the question, which is why it is shipped as a function.

3. **Failure is graceful.** A predictor carrying no information flattens the
   regression, the residual term supplies the whole answer, and the output
   returns to the nearest-neighbour coarse field rather than to noise -
   confirmed numerically: three zero-R² predictors (VNIR albedo, SWIR
   brightness, a SWIR band ratio) each reproduced the unsharpened
   correlation of 0.8164 to four decimal places. A useless predictor costs
   nothing; only a spuriously correlated one does damage.

One confound, stated plainly: the −33.3 % for VIIRS is not all sharpening
error. VIIRS observed at ~21:00 UTC and the Landsat reference at 18:22 UTC,
2.6 h apart, so part of that gap is genuine diurnal change.
`validate_sharpening()` reports the mean offset separately so the size of
the time effect stays visible.

**Recommendation:** try sharpening on vegetated, heterogeneous fire scenes,
and always confirm with `validate_sharpening()` before using the output.
On arid scenes, use the coarse field (`on_low_skill='coarse'`).

---

## Deliberately NOT implemented

Per the requirement that nothing be built on an unverified assumption:

- **SDGSAT-1 TIS** (30 m, 3 TIR bands, day *and* night) - on paper the best
  spatial match to EnMAP of any thermal sensor flying. **No reachable
  machine interface**: both CAS portals failed from this machine (TLS
  failure on `sdg.casearth.cn`, connection timeout on the mirror), and there
  is no STAC/OData/CMR route. Scenes must be ordered by hand through the web
  UI after registering. `load_generic_lst_geotiff()` will read the exported
  GeoTIFF, so the pipeline is ready for it - only search and download are
  missing, because they cannot be built.
- **constellr** (SkyBee-1/-2, 30 m LST, <2 K, launched Jan/Jun 2025) -
  commercial. Research access is an ESA Third Party Missions proposal, which
  is paperwork, not an API. Worth filing; nothing to code.
- **NASA FIRMS** active-fire CSV - needs a per-user `MAP_KEY` this machine
  does not have, so the client could not be exercised. Not shipped rather
  than shipped untested.

---

## Environment

Use the **`emit`** conda environment (it has GDAL, `h5py`, `netCDF4`,
`rasterio`, `pyproj`, `earthaccess`).

> **Activate it - do not call the interpreter by path.**
> `conda activate emit`, not `.../envs/emit/python.exe`.
>
> This environment links numpy/scipy against MKL, and MKL loads its kernel
> DLLs (`mkl_def`, `mkl_avx2`, …) with a plain `LoadLibrary` that resolves
> only through `PATH`. Running the interpreter by absolute path leaves
> `envs/emit/Library/bin` off `PATH`, and **every `np.linalg` call then
> hard-crashes the process** with Windows fatal exception `0xc06d007f` - no
> Python traceback, no error message, just a dead process. This bites the
> existing pipeline too, since `temperature_methods.py` needs
> `scipy.optimize.nnls`. Activating the environment fixes it completely.

### Credentials

| service | needed for | how |
|---|---|---|
| NASA Earthdata | ECOSTRESS, VIIRS downloads | `~/.netrc` entry for `urs.earthdata.nasa.gov`, or `EARTHDATA_USERNAME`/`EARTHDATA_PASSWORD` |
| Copernicus Data Space | Sentinel-3 downloads | `CDSE_USERNAME`/`CDSE_PASSWORD` |

Catalogue **search needs no credentials anywhere** - only downloads do.
Both are already configured on this machine and were used to verify the
download paths end to end.

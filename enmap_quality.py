#!/usr/bin/env python
# coding: utf-8
"""
Per-pixel, per-band quality of EnMAP L1B spectra at the fire pixels:
which bands are invalid, which are saturated, and the radiance ceiling of
every band.

WHY THIS EXISTS. An EnMAP L1B product ships a per-band pixel mask
(QL_PIXELMASK_VNIR/SWIR, 1 = flagged) and stores radiance as 16-bit DN with
a per-band gain and offset (METADATA.XML, GainOfBand/OffsetOfBand), so the
largest radiance a band can record is

    L_ceiling,k = gain_k * 65535 + offset_k                                (1)

On the Portugal scene (DT0000149658) that ceiling is 0.043-0.048
W m^-2 sr^-1 nm^-1 across 2000-2450 nm and an active fire reaches it: the
brightest fire pixels have 5-25 SWIR bands within 4% of the DN ceiling, and
the pixel mask flags every band from ~1968 nm to the end of the SWIR array
on those pixels (56-58 bands), including bands whose DN is far below the
ceiling - the pattern of saturation CROSSTALK along the spectral axis of
the SWIR detector, which the metadata quality block reports separately
(saturationCrosstalkSWIR). Every retrieval in this repository, and every
published one, fitted those bands as measurements.

WHAT IS DONE ABOUT IT. For each fire pixel this module classifies every
band as one of

    valid      - unflagged, positive radiance: a measurement;
    censored   - DN >= `saturation_dn` (default 63000, i.e. within ~4% of
                 full scale), flagged or not: the true radiance is at least
                 the recorded value (a saturating detector compresses, it
                 does not invent signal), so the band is fitted as the
                 one-sided constraint model >= recorded radiance
                 (profile_retrieval.weighted_fcls, `censored`);
    invalid    - flagged with DN below `saturation_dn` (crosstalk-affected,
                 dead/defective, or written as 0): not used at all.

The censored class is what lets the hottest pixels keep the information
that pins their temperature from below ("this band is at least this
bright") without fitting a compressed value as if it were the radiance.

Band order: the product files and GainOfBand list VNIR bands first, then
SWIR; the merged cube the pipeline builds is sorted by centre wavelength,
and the two detectors overlap at ~900-1000 nm. Every per-band output is
therefore permuted into the cube's order (cube_band_order(); corrected 13
Oct 2026 - before, the quality of the overlap bands was attached to the
wrong bands. No retrieval was affected: the fit window starts at 1461 nm,
and from 1000 nm on the two orders coincide).
"""

import os
import re

import numpy as np

__all__ = ['band_ceilings_from_metadata', 'load_pixel_quality', 'find_quality_files']

_GAIN_RE = re.compile(r'<GainOfBand>([^<]+)</GainOfBand>')
_OFFSET_RE = re.compile(r'<OffsetOfBand>([^<]+)</OffsetOfBand>')
_WL_RE = re.compile(r'<wavelengthCenterOfBand>([^<]+)</wavelengthCenterOfBand>')


def cube_band_order(xml_path, n_bands):
    """Permutation from the product's VNIR-then-SWIR band order to the order
    of the merged cube, which Fire_analysis_with_EnMAP._merge_vnir_swir_scenes
    sorts by centre wavelength (stable, so VNIR before SWIR on ties). The two
    detectors overlap at ~900-1000 nm, so the orders differ there; without
    this permutation the quality of those bands is attached to the wrong
    bands. Returns None when the metadata does not list `n_bands` centre
    wavelengths (the caller then keeps the product order)."""
    with open(xml_path, encoding='utf-8', errors='ignore') as fh:
        wl = np.array([float(v) for v in _WL_RE.findall(fh.read())])
    if wl.size != n_bands:
        return None
    return np.argsort(wl, kind='stable')


def find_quality_files(raster_path):
    """Locate the METADATA.XML, the two QL_PIXELMASK rasters and the two
    SPECTRAL_IMAGE (DN) rasters that belong to the same L1B product as
    `raster_path` (any file of the product, e.g. the merged cube). Returns
    a dict of absolute paths (values None when a file is absent)."""
    d = os.path.dirname(os.path.abspath(raster_path))
    base = os.path.basename(raster_path)
    m = re.match(r'(.*?)-(SPECTRAL_IMAGE|QL_|METADATA)', base)
    stem = m.group(1) if m else None
    out = {'metadata': None, 'pixelmask_vnir': None, 'pixelmask_swir': None,
           'image_vnir': None, 'image_swir': None}
    if stem is None:
        return out
    for name in os.listdir(d):
        if not name.startswith(stem):
            continue
        p = os.path.join(d, name)
        up = name.upper()
        if up.endswith('METADATA.XML'):
            out['metadata'] = p
        elif 'QL_PIXELMASK_VNIR' in up:
            out['pixelmask_vnir'] = p
        elif 'QL_PIXELMASK_SWIR' in up:
            out['pixelmask_swir'] = p
        elif 'SPECTRAL_IMAGE_VNIR' in up and up.endswith(('.TIF', '.TIFF')):
            out['image_vnir'] = p
        elif 'SPECTRAL_IMAGE_SWIR' in up and up.endswith(('.TIF', '.TIFF')):
            out['image_swir'] = p
    return out


def sun_view_zenith(xml_path):
    """Scene-centre sun zenith and viewing zenith angles (degrees) from an
    EnMAP L1B METADATA.XML (sunElevationAngle / viewingZenithAngle,
    <center>). Returns (sun_zenith, view_zenith); None for an angle that is
    not found."""
    with open(xml_path, encoding='utf-8', errors='ignore') as fh:
        xml = fh.read()

    def centre(tag):
        m = re.search(rf'<{tag}>.*?<center[^>]*>([^<]+)</center>.*?</{tag}>', xml, re.S)
        return float(m.group(1)) if m else None
    elev = centre('sunElevationAngle')
    return (None if elev is None else 90.0 - elev), centre('viewingZenithAngle')


def band_ceilings_from_metadata(xml_path, dn_max=65535):
    """Per-band radiance ceiling (1) from the metadata gains and offsets.
    Returns (ceiling, gains, offsets), each (n_bands,) in the merged
    VNIR-then-SWIR order."""
    with open(xml_path, encoding='utf-8', errors='ignore') as fh:
        xml = fh.read()
    gains = np.array([float(v) for v in _GAIN_RE.findall(xml)])
    offsets = np.array([float(v) for v in _OFFSET_RE.findall(xml)])
    if gains.size == 0 or gains.size != offsets.size:
        raise ValueError(f'{xml_path}: found {gains.size} gains and {offsets.size} offsets')
    return gains * dn_max + offsets, gains, offsets


def load_pixel_quality(raster_path, points, n_bands_expected=None, saturation_dn=63000,
                       verbose=True):
    """Per-pixel band classification for the (row, col) `points`.

    Returns dict with boolean arrays valid, censored, invalid of shape
    (n_points, n_bands), censor_level (n_points, n_bands) = the recorded
    radiance where censored (NaN elsewhere), dn (n_points, n_bands) uint16,
    ceiling (n_bands,), and `available` (False when the product files
    could not be found, in which case every band is valid and nothing is
    censored - the retrieval then behaves exactly as before this module
    existed)."""
    from osgeo import gdal
    gdal.UseExceptions()
    points = np.atleast_2d(np.asarray(points, dtype=int))
    n_pts = points.shape[0]
    files = find_quality_files(raster_path)
    need = ('metadata', 'pixelmask_vnir', 'pixelmask_swir', 'image_vnir', 'image_swir')
    if any(files[k] is None for k in need):
        if verbose:
            missing = [k for k in need if files[k] is None]
            print(f'  [enmap_quality] product files not found next to {os.path.basename(raster_path)} '
                  f'({missing}); all bands treated as valid, none censored')
        nb = n_bands_expected or 0
        return {'available': False, 'valid': np.ones((n_pts, nb), bool),
                'censored': np.zeros((n_pts, nb), bool), 'invalid': np.zeros((n_pts, nb), bool),
                'censor_level': np.full((n_pts, nb), np.nan), 'dn': None, 'ceiling': None}

    ceiling, gains, offsets = band_ceilings_from_metadata(files['metadata'])
    nb = len(ceiling)
    if n_bands_expected is not None and nb != n_bands_expected:
        raise ValueError(f'metadata lists {nb} bands but the cube has {n_bands_expected}')

    def _read_points(path):
        # one read of the points' bounding window for all bands (was one
        # 1 x 1 read per point and band: 3.4e5 GDAL calls, ~10 s on t014)
        ds = gdal.Open(path)
        n = ds.RasterCount
        if n_pts == 0:
            return np.zeros((0, n), dtype=np.int64)
        r0, c0 = points.min(axis=0)
        r1, c1 = points.max(axis=0) + 1
        win = ds.ReadAsArray(int(c0), int(r0), int(c1 - c0), int(r1 - r0))
        win = win.reshape(n, int(r1 - r0), int(c1 - c0))
        return win[:, points[:, 0] - r0, points[:, 1] - c0].T.astype(np.int64)

    flag = np.concatenate([_read_points(files['pixelmask_vnir']),
                           _read_points(files['pixelmask_swir'])], axis=1) != 0
    dn = np.concatenate([_read_points(files['image_vnir']), _read_points(files['image_swir'])], axis=1)
    if flag.shape[1] != nb or dn.shape[1] != nb:
        raise ValueError(f'band count mismatch: mask {flag.shape[1]}, image {dn.shape[1]}, '
                         f'metadata {nb}')
    # Saturation is a property of the DN, not of the flag: on this scene
    # several fire pixels reach the ceiling in bands the processor did not
    # flag (e.g. 7 bands >= 63000 DN with only 2 flagged), so the censoring
    # criterion is the DN alone. A censored band is fitted as a LOWER BOUND
    # on the radiance, which is correct for any compressive response, so
    # censoring a band that was in fact still linear costs precision only,
    # never accuracy - the safe direction.
    censored = dn >= saturation_dn
    invalid = flag & ~censored
    valid = ~flag & ~censored & (dn > 0)
    radiance = gains[None, :] * dn + offsets[None, :]
    censor_level = np.where(censored, radiance, np.nan)
    # into the merged cube's (wavelength-sorted) band order
    order = cube_band_order(files['metadata'], nb)
    if order is not None:
        flag, dn, censored, invalid, valid = (x[:, order] for x in (flag, dn, censored, invalid, valid))
        radiance, censor_level = radiance[:, order], censor_level[:, order]
        ceiling = ceiling[order]
    soft = (dn >= 0.75 * 65535) & ~censored          # diagnostic only
    if verbose:
        n_any_c = int((censored.sum(axis=1) > 0).sum())
        n_any_i = int((invalid.sum(axis=1) > 0).sum())
        print(f'  [enmap_quality] {n_pts} pixels: {n_any_c} with >=1 saturated (censored) band, '
              f'{n_any_i} with >=1 flagged-invalid band; '
              f'median invalid bands/pixel {int(np.median(invalid.sum(axis=1)))}')
    return {'available': True, 'valid': valid, 'censored': censored, 'invalid': invalid,
            'censor_level': censor_level, 'dn': dn, 'ceiling': ceiling, 'flag': flag,
            'n_soft_saturated': soft.sum(axis=1)}

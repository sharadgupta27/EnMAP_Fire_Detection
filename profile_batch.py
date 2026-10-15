#!/usr/bin/env python
# coding: utf-8
"""
Multi-core execution of the profile-likelihood retrieval (profile_retrieval.py)
over many pixels.

WHY
---
profile_retrieval() evaluates the profile chi2_p(T) on a 200-point grid per
reweighting pass, then refines the minimum and both interval edges: some 430
non-negative least-squares solves per pixel and library draw. The retrieval
stage of run_pipeline() runs it about 26 times per fire pixel (transmittance
rounds, split-half validation, two calibrations, the diagonal pass, five
library draws, two transmittance draws). Every call within a step is
independent of the others, so each step is one parallel map over the pixels.

DEVICES
-------
'joblib' (default): the unchanged scalar retrieval on every CPU core, in
chunks of up to 8 pixels per task (7x on 16 logical cores, VERIFICATION.md
Section 19). 'serial': one core, the reference path; identical results.

The batched torch / CUDA engines that used to live here were removed on
14 Oct 2026 (VERIFICATION.md Section 28): through the real stage-4 call they
were ~3x slower than 'joblib' on this hardware and disagreed with the scalar
estimator by up to 59 K (Section 20.4), and they did not implement the
iterated reweighting of ProfileML-v2, the primary method, so every v2 call
already fell back to the CPU cores. They are in the git history (commit
abc269d) with their NVRTC kernel (nnls_cuda.py, nvrtc_kernel.py).

USE
---
    from profile_batch import profile_retrieval_many
    results = profile_retrieval_many(radiances, wavelength, libraries,
                                     per_pixel=[dict(valid_mask=..., ...), ...],
                                     device=None, **profile_retrieval_kwargs)

`device` is 'joblib', 'serial', or None -> the PROFILEML_DEVICE environment
variable, else 'joblib'.
"""

import os

import profile_retrieval as pr

__all__ = ['profile_retrieval_many', 'resolve_device']

DEVICES = ('joblib', 'serial')


def resolve_device(device=None):
    """'joblib' | 'serial' from the argument, else the PROFILEML_DEVICE
    environment variable, else 'joblib' (every CPU core)."""
    d = device if device is not None else os.environ.get('PROFILEML_DEVICE', 'joblib')
    d = str(d).strip().lower()
    if d in ('auto', 'parallel', ''):
        return 'joblib'
    if d not in DEVICES:
        raise ValueError(f"profile_batch: device {d!r} is not available; use 'joblib' (every CPU core) "
                         "or 'serial'. The torch/CUDA engines were removed (VERIFICATION.md Section 28).")
    return d


def _serial(radiances, wavelength, libraries, per_pixel, **kw):
    return [pr.profile_retrieval(radiances[i], wavelength, libraries[i], **per_pixel[i], **kw)
            for i in range(len(radiances))]


def _joblib(radiances, wavelength, libraries, per_pixel, n_jobs=-1, **kw):
    from joblib import Parallel, delayed
    n = len(radiances)
    if n <= 2:
        return _serial(radiances, wavelength, libraries, per_pixel, **kw)
    chunk = max(1, min(8, n // (4 * max(os.cpu_count() or 1, 1)) + 1))
    parts = Parallel(n_jobs=n_jobs)(
        delayed(_serial)(radiances[i:i + chunk], wavelength, libraries[i:i + chunk], per_pixel[i:i + chunk], **kw)
        for i in range(0, n, chunk))
    return [r for part in parts for r in part]


def profile_retrieval_many(radiances, wavelength, libraries, per_pixel=None, device=None,
                           n_jobs=-1, **kwargs):
    """profile_retrieval() for a list of pixels, returning the list of the
    same result dicts, in input order.

    radiances : list/array of (n_bands,) spectra over the fit window.
    libraries : list of (J_i, n_bands) background libraries, one per pixel
        (or one array used for all).
    per_pixel : optional list of dicts of per-pixel keyword arguments
        (valid_mask, censored_mask, censor_level, tau, rel_model_error).
    device : see resolve_device(). kwargs go to profile_retrieval()."""
    n = len(radiances)
    if n == 0:
        return []
    if not isinstance(libraries, (list, tuple)):
        libraries = [libraries] * n
    per_pixel = [dict(pp) for pp in (per_pixel or [{}] * n)]
    if resolve_device(device) == 'joblib':
        return _joblib(radiances, wavelength, libraries, per_pixel, n_jobs=n_jobs, **kwargs)
    return _serial(radiances, wavelength, libraries, per_pixel, **kwargs)

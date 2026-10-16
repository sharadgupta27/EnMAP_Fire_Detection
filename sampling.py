#!/usr/bin/env python
# coding: utf-8
"""Seed-free subset selection for this work's chain (detector, ProfileML).

Every subset the method takes - the detector's fire-free pool, the absorption
template's sample, the background libraries, the split of the fire pixels
into halves, the noise sample of the saturation bounds - used to be a draw
from a seeded random generator. A fixed seed makes a run repeatable, but it
leaves a free parameter that changes the result: on t014 the detection count
moved by 20-30 % between seeds (VERIFICATION.md S18). Here every subset is a
fixed function of the data's own order, so there is no seed to choose:

* systematic_indices(n, m, phase): m indices evenly spaced through range(n),
  the systematic sample of survey statistics (every (n/m)-th element in the
  given order, starting at `phase`). The spacing is exactly n/m, so a
  raster-ordered pool is covered uniformly in space, and the indices are
  computed in integer arithmetic, so they are identical on every platform.
* weyl_phase(k): the fractional part of k/phi (phi the golden ratio), the
  Kronecker sequence. Successive k give phases that are as evenly spread
  over [0, 1) as any sequence can be (the three-gap theorem), so the k-th
  library of a pixel is offset from the others without a random draw.

The pixel's own key for the phase is its raster index (row * n_cols + col),
so the library of a pixel does not depend on which other pixels were
detected, nor on their order.
"""
import numpy as np

# 1/phi = (sqrt 5 - 1)/2; the Kronecker sequence k/phi mod 1 has the lowest
# possible discrepancy among sequences of the form k*alpha mod 1.
INV_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0
# phases are resolved to 1/PHASE_RES of a spacing; an integer, so the index
# arithmetic below is exact
PHASE_RES = 1 << 20


def weyl_phase(k):
    """frac(k / phi) for integer k (scalar or array), in [0, 1).

    Computed exactly in integers: k * round(2^52 / phi) mod 2^52, scaled.
    Floating-point k * 0.618... would lose the low digits for large k."""
    k = np.asarray(k, dtype=np.uint64)
    # round(2^52 / phi), from 60-digit decimal arithmetic:
    # int(((Decimal(5).sqrt() - 1) / 2 * 2**52).to_integral_value())
    mult = np.uint64(2783377641436328)
    mask = np.uint64((1 << 52) - 1)
    with np.errstate(over='ignore'):
        frac = (k * mult) & mask              # wraps mod 2^64; the low 52 bits are k*mult mod 2^52
    out = frac.astype(np.float64) / float(1 << 52)
    return float(out) if out.ndim == 0 else out


def systematic_indices(n, m, phase=0.0):
    """`m` indices evenly spaced through range(`n`), starting at `phase`
    (a fraction of one spacing, in [0, 1)): floor((i + phase) * n / m) for
    i = 0..m-1. Strictly increasing and inside [0, n) whenever m <= n;
    m >= n returns every index. Integer arithmetic, so the result does not
    depend on the platform's floating point."""
    n = int(n); m = int(m)
    if n <= 0 or m <= 0:
        return np.zeros(0, dtype=np.int64)
    if m >= n:
        return np.arange(n, dtype=np.int64)
    p = int(np.floor(float(phase) % 1.0 * PHASE_RES))
    i = np.arange(m, dtype=np.int64)
    # floor((i + p/R) * n / m) = (i*n*R + p*n) // (m*R), exact in int64 for
    # n up to ~10^7 and m up to ~10^5 (i*n*R < 2^63)
    if m * n * PHASE_RES >= (1 << 62):
        # very large pools: phase resolved to 1/m of a spacing instead
        q = int(np.floor(float(phase) % 1.0 * m))
        return (i * n + (q * n) // m) // m
    return (i * n * PHASE_RES + p * n) // (m * PHASE_RES)


def interleaved_halves(n):
    """Split range(n) into two halves by alternation (0, 2, 4, ... and
    1, 3, 5, ...). In raster order this puts neighbouring fire pixels in
    different halves, so each half spans the whole fire front."""
    idx = np.arange(int(n))
    return idx[0::2], idx[1::2]

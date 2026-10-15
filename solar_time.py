#!/usr/bin/env python
# coding: utf-8
"""Mean local solar time - the kernel shared by the pipeline and the
thermal-partner search.

WHY THIS IS ITS OWN MODULE
--------------------------
Both Fire_analysis_with_EnMAP.py and thermal_partners.py need the same
conversion from a UTC timestamp to mean local solar time, and both wrap it
in guards of their own:

  Fire_analysis_with_EnMAP._local_solar_hour()  returns None when the
      time-of-day is not actually known - a Landsat filename carries a date
      but no time, and treating that as midnight would fabricate a "night
      overpass" warning for a scene acquired at ~10:30 local.

  thermal_partners.local_solar_hour()  first normalises whatever the
      catalogues returned (tz-aware ISO strings from CMR and CDSE, naive
      datetimes from EnMAP's own metadata) through _as_naive_utc().

Those two contracts are genuinely different and neither is redundant, so
they stay where they are. What was duplicated is only the arithmetic
underneath them, and neither module can import it from the other without
damaging the layering:

  * thermal_partners.py is usable on its own - it has a __main__ CLI and
    __all__, and today needs numpy, requests and pyproj. Importing
    Fire_analysis_with_EnMAP at module level would drag matplotlib, pandas,
    scipy and tqdm into that standalone path.
  * Fire_analysis_with_EnMAP.py is the base module, and thermal_partners is
    deliberately OPTIONAL to it (_load_thermal_partners() guards the import
    with try/except ImportError so a missing partner module can never turn a
    working run into a failed one). A module-level dependency the other way
    would make requests a hard requirement of the whole pipeline.

Hence this module: standard library only, no third-party imports at all, so
either side can depend on it freely.

MEAN, NOT APPARENT
------------------
No equation-of-time correction. That term reaches ~16 minutes, which matters
for sun-angle work but not for the day/night and hours-apart judgements these
callers make, and pulling in a solar ephemeris to shave 16 minutes off a
number that is only ever printed would be false precision.
"""

__all__ = ['hours_of_day', 'solar_hour_from_utc']


def hours_of_day(value):
    """Time of day of a datetime-like object, as fractional hours.

    Accepts anything exposing .hour/.minute/.second - datetime.datetime and
    pandas.Timestamp both do. Sub-second precision is deliberately dropped:
    it is far below the resolution of every judgement built on this.
    """
    return value.hour + value.minute / 60.0 + value.second / 3600.0


def solar_hour_from_utc(hours_utc, lon_deg):
    """Mean local solar time in hours, wrapped into [0, 24).

    The Earth turns 15 degrees of longitude per hour, so local solar time is
    the UTC time of day shifted by lon/15. The modulo is what keeps a
    pre-dawn eastern-hemisphere overpass from reporting a negative hour, or
    a late western one from reporting 25.
    """
    return (float(hours_utc) + float(lon_deg) / 15.0) % 24.0

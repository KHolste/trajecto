"""Zeit-Hilfsfunktionen: Modifiziertes Julianisches Datum (MJD) und Dauerformat.

Wiederverwendbar fuer Module mit Missionsuhr (Manoever-Sandbox, spaeter auch
Hohmann). MJD = JD - 2400000.5; die MJD eines Kalendertags um 00:00 ist ganzzahlig.
"""

from __future__ import annotations

import datetime as _dt


def datetime_to_mjd(when: _dt.datetime) -> float:
    """Modifiziertes Julianisches Datum aus einem ``datetime``."""
    a = (14 - when.month) // 12
    y = when.year + 4800 - a
    m = when.month + 12 * a - 3
    jdn = (
        when.day + (153 * m + 2) // 5 + 365 * y
        + y // 4 - y // 100 + y // 400 - 32045
    )
    frac = (
        (when.hour - 12) / 24.0
        + when.minute / 1440.0
        + when.second / 86400.0
        + when.microsecond / 86_400_000_000.0
    )
    return jdn + frac - 2400000.5


def today_mjd() -> float:
    """MJD des heutigen Tages um 00:00 UTC (Ganzzahl)."""
    now = _dt.datetime.now(_dt.timezone.utc)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime_to_mjd(midnight)


def format_duration(seconds: float) -> str:
    """Sekunden als ``"D d HH:MM:SS"`` formatieren (mit Vorzeichen)."""
    sign = "-" if seconds < 0 else ""
    total = abs(int(round(seconds)))
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{sign}{days} d {hours:02d}:{minutes:02d}:{secs:02d}"

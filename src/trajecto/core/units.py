"""Umrechnung zwischen SI-Einheiten (intern) und Anzeige-Einheiten (GUI).

Trajecto rechnet **intern konsequent in SI** (m, s, m/s, rad, ...). Dieses Modul
ist die einzige Stelle, an der Anzeige-Einheiten definiert und umgerechnet werden.

Jede Einheit ist über ihren Faktor nach SI beschrieben::

    wert_si = wert_anzeige * FAKTOR
    wert_anzeige = wert_si / FAKTOR
"""

from __future__ import annotations

import math

#: Anzeige-Einheit -> Faktor, mit dem multipliziert der SI-Wert entsteht.
UNIT_FACTORS: dict[str, float] = {
    # Laenge
    "m": 1.0,
    "km": 1_000.0,
    # Geschwindigkeit
    "m/s": 1.0,
    "km/s": 1_000.0,
    # Zeit
    "s": 1.0,
    "min": 60.0,
    "h": 3_600.0,
    "d": 86_400.0,
    # Winkel
    "rad": 1.0,
    "deg": math.pi / 180.0,
    # Spezifische Energie [m^2/s^2 = J/kg]
    "J/kg": 1.0,
    "MJ/kg": 1.0e6,
    # Spezifischer Drehimpuls [m^2/s]
    "m^2/s": 1.0,
    "km^2/s": 1.0e6,
    # Dimensionslos (z. B. Exzentrizitaet, Verhaeltnisse)
    "-": 1.0,
}


def to_si(value: float, unit: str) -> float:
    """Rechne einen Wert aus der Anzeige-Einheit ``unit`` nach SI um."""
    try:
        return value * UNIT_FACTORS[unit]
    except KeyError as exc:
        raise ValueError(f"Unbekannte Einheit: {unit!r}") from exc


def from_si(value_si: float, unit: str) -> float:
    """Rechne einen SI-Wert in die Anzeige-Einheit ``unit`` um."""
    try:
        return value_si / UNIT_FACTORS[unit]
    except KeyError as exc:
        raise ValueError(f"Unbekannte Einheit: {unit!r}") from exc

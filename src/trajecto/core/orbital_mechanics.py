"""Grundfunktionen der Bahnmechanik.

Reine, zustandslose Funktionen. Alle Ein- und Ausgaben in SI-Einheiten
(m, s, m/s, m^3/s^2). Ungueltige Eingaben loesen ``ValueError`` aus.
"""

from __future__ import annotations

import math


def _validate(mu: float, radius: float) -> None:
    if mu <= 0.0:
        raise ValueError("Gravitationsparameter mu muss positiv sein.")
    if radius <= 0.0:
        raise ValueError("Radius muss positiv sein.")


def circular_orbit_velocity(mu: float, radius: float) -> float:
    """Kreisbahngeschwindigkeit ``v = sqrt(mu / r)`` [m/s].

    Args:
        mu: Gravitationsparameter des Zentralkoerpers [m^3/s^2].
        radius: Bahnradius (vom Koerpermittelpunkt) [m].
    """
    _validate(mu, radius)
    return math.sqrt(mu / radius)


def circular_orbit_period(mu: float, radius: float) -> float:
    """Umlaufzeit einer Kreisbahn ``T = 2*pi*sqrt(r^3 / mu)`` [s]."""
    _validate(mu, radius)
    return 2.0 * math.pi * math.sqrt(radius**3 / mu)


def escape_velocity(mu: float, radius: float) -> float:
    """Fluchtgeschwindigkeit ``v_esc = sqrt(2*mu / r)`` [m/s].

    Es gilt ``v_esc = sqrt(2) * v_kreis`` am selben Radius.
    """
    _validate(mu, radius)
    return math.sqrt(2.0 * mu / radius)

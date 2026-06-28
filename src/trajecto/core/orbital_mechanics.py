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


# --- Elliptische Keplerbahnen -----------------------------------------------


def _validate_ellipse(mu: float, semi_major_axis: float, eccentricity: float) -> None:
    if mu <= 0.0:
        raise ValueError("Gravitationsparameter mu muss positiv sein.")
    if semi_major_axis <= 0.0:
        raise ValueError("Grosse Halbachse a muss positiv sein.")
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError("Exzentrizitaet e muss im Bereich 0 <= e < 1 liegen.")


def orbital_period(mu: float, semi_major_axis: float) -> float:
    """Umlaufzeit einer Ellipsenbahn ``T = 2*pi*sqrt(a^3 / mu)`` [s].

    Drittes Keplersches Gesetz. Fuer ``a = r`` identisch mit der Umlaufzeit
    der Kreisbahn.
    """
    if mu <= 0.0:
        raise ValueError("Gravitationsparameter mu muss positiv sein.")
    if semi_major_axis <= 0.0:
        raise ValueError("Grosse Halbachse a muss positiv sein.")
    return 2.0 * math.pi * math.sqrt(semi_major_axis**3 / mu)


def periapsis_radius(semi_major_axis: float, eccentricity: float) -> float:
    """Periapsisradius ``r_p = a*(1 - e)`` [m] (kleinster Bahnradius)."""
    _validate_ellipse(1.0, semi_major_axis, eccentricity)
    return semi_major_axis * (1.0 - eccentricity)


def apoapsis_radius(semi_major_axis: float, eccentricity: float) -> float:
    """Apoapsisradius ``r_a = a*(1 + e)`` [m] (groesster Bahnradius)."""
    _validate_ellipse(1.0, semi_major_axis, eccentricity)
    return semi_major_axis * (1.0 + eccentricity)


def vis_viva_speed(mu: float, radius: float, semi_major_axis: float) -> float:
    """Bahngeschwindigkeit nach Vis-Viva ``v = sqrt(mu*(2/r - 1/a))`` [m/s].

    Args:
        mu: Gravitationsparameter [m^3/s^2].
        radius: aktueller Bahnradius vom Fokus [m].
        semi_major_axis: grosse Halbachse a [m].
    """
    if mu <= 0.0:
        raise ValueError("Gravitationsparameter mu muss positiv sein.")
    if radius <= 0.0:
        raise ValueError("Radius muss positiv sein.")
    if semi_major_axis <= 0.0:
        raise ValueError("Grosse Halbachse a muss positiv sein.")
    return math.sqrt(mu * (2.0 / radius - 1.0 / semi_major_axis))


def solve_eccentric_anomaly(
    mean_anomaly: float, eccentricity: float, *, tol: float = 1e-12, max_iter: int = 60
) -> float:
    """Loese die Keplergleichung ``M = E - e*sin(E)`` nach E (Newton-Verfahren).

    Wird fuer die zeitgetreue Darstellung des 2. Keplerschen Gesetzes benoetigt
    (gleiche Flaechen in gleichen Zeiten ↔ gleiche Mittelpunktsanomalie-Intervalle).
    """
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError("Exzentrizitaet e muss im Bereich 0 <= e < 1 liegen.")
    e = eccentricity
    # Startwert: M (fuer kleine e gut), bei grossem e leicht angehoben.
    eccentric = mean_anomaly if e < 0.8 else math.pi
    for _ in range(max_iter):
        f = eccentric - e * math.sin(eccentric) - mean_anomaly
        f_prime = 1.0 - e * math.cos(eccentric)
        delta = f / f_prime
        eccentric -= delta
        if abs(delta) < tol:
            break
    return eccentric

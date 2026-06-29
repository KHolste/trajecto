"""Eingabemodus fuer orbitale Abstaende: Radius vom Mittelpunkt oder Hoehe.

Trajecto rechnet intern immer mit dem Bahnradius vom Mittelpunkt des
Zentralkoerpers (SI, m). Studierende denken jedoch oft in Hoehen ueber der
Oberflaeche (z. B. 500 km LEO, 35786 km GEO). Diese Hilfslogik kapselt die
Umrechnung zentral und wiederverwendbar – ohne Rechenlogik in der GUI.

Fuer den Hoehenmodus gilt::

    r = R_body + h
"""

from __future__ import annotations

from trajecto.core.bodies import CelestialBody

#: Eingabemodus: Wert ist der Bahnradius vom Mittelpunkt.
INPUT_MODE_RADIUS = "Radius vom Mittelpunkt"
#: Eingabemodus: Wert ist die Hoehe ueber der Oberflaeche.
INPUT_MODE_ALTITUDE = "Höhe über Oberfläche"
#: Alle verfuegbaren Eingabemodi (z. B. fuer eine Auswahl in der GUI).
INPUT_MODES: tuple[str, ...] = (INPUT_MODE_RADIUS, INPUT_MODE_ALTITUDE)


def radius_from_input(
    body: CelestialBody, value: float, mode: str = INPUT_MODE_RADIUS
) -> float:
    """Bestimme den Bahnradius [m] aus Eingabewert und Eingabemodus.

    Args:
        body: Zentralkoerper (liefert den mittleren Radius).
        value: Eingabewert in m – je nach ``mode`` Radius oder Hoehe.
        mode: ``INPUT_MODE_RADIUS`` oder ``INPUT_MODE_ALTITUDE``.

    Raises:
        ValueError: bei unbekanntem Modus oder wenn der resultierende
            Radius ``<= 0`` ist.
    """
    if mode == INPUT_MODE_RADIUS:
        radius = value
    elif mode == INPUT_MODE_ALTITUDE:
        radius = body.mean_radius + value
    else:
        raise ValueError(f"Unbekannter Eingabemodus: {mode!r}")

    if radius <= 0.0:
        raise ValueError(
            "Resultierender Bahnradius muss positiv sein (r <= 0). "
            "Pruefe Eingabewert und Eingabemodus."
        )
    return radius


def altitude_from_radius(body: CelestialBody, radius: float) -> float:
    """Hoehe ueber der Oberflaeche [m] aus dem Bahnradius (kann negativ sein)."""
    return radius - body.mean_radius

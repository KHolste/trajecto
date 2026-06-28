"""Datenmodell und Katalog fuer Himmelskoerper.

Alle Groessen in SI-Einheiten. Gespeichert wird bevorzugt der
Gravitationsparameter ``mu = G * M`` direkt, da er aus Bahnbeobachtungen
deutlich genauer bekannt ist als Masse und Gravitationskonstante einzeln.
"""

from __future__ import annotations

from dataclasses import dataclass

from trajecto.core.constants import GRAVITATIONAL_CONSTANT


@dataclass(frozen=True)
class CelestialBody:
    """Ein Zentralkoerper der Bahnmechanik.

    Attributes:
        name: Anzeigename (z. B. ``"Erde"``).
        mu: Gravitationsparameter ``G * M`` [m^3 / s^2].
        mean_radius: mittlerer Koerperradius [m].
    """

    name: str
    mu: float
    mean_radius: float

    def __post_init__(self) -> None:
        if self.mu <= 0.0:
            raise ValueError("mu muss positiv sein.")
        if self.mean_radius <= 0.0:
            raise ValueError("mean_radius muss positiv sein.")

    @property
    def mass(self) -> float:
        """Aus ``mu`` abgeleitete Masse ``M = mu / G`` [kg]."""
        return self.mu / GRAVITATIONAL_CONSTANT


# Vordefinierte Zentralkoerper (alle Werte in SI).
# mu = G*M [m^3/s^2] aus Standardquellen (IAU/JPL); mean_radius = mittlerer
# Koerperradius [m]. mu wird direkt gespeichert, da es genauer bekannt ist als
# Masse und G einzeln.

#: Sonne.
SUN = CelestialBody(name="Sonne", mu=1.327_124_400_18e20, mean_radius=6.957_00e8)

#: Erde.
EARTH = CelestialBody(name="Erde", mu=3.986_004_418e14, mean_radius=6.371_000e6)

#: Erdmond.
MOON = CelestialBody(name="Mond", mu=4.902_800e12, mean_radius=1.737_400e6)

#: Mars.
MARS = CelestialBody(name="Mars", mu=4.282_837e13, mean_radius=3.389_500e6)

#: Venus.
VENUS = CelestialBody(name="Venus", mu=3.248_590e14, mean_radius=6.051_800e6)

#: Jupiter.
JUPITER = CelestialBody(name="Jupiter", mu=1.266_865_34e17, mean_radius=6.991_100e7)

#: Katalog vordefinierter Koerper, adressierbar ueber den Namen.
BODIES: dict[str, CelestialBody] = {
    body.name: body
    for body in (SUN, EARTH, MOON, MARS, VENUS, JUPITER)
}


def get_body(name: str) -> CelestialBody:
    """Liefere den Katalogkoerper mit dem gegebenen Namen."""
    try:
        return BODIES[name]
    except KeyError as exc:
        raise ValueError(f"Unbekannter Himmelskoerper: {name!r}") from exc

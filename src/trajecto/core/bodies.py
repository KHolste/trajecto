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


#: Vordefinierter Zentralkoerper: Erde (mu nach EGM/GMAT, Radius = Aequatorradius).
EARTH = CelestialBody(
    name="Erde",
    mu=3.986_004_418e14,
    mean_radius=6.371_000e6,
)

#: Katalog vordefinierter Koerper, adressierbar ueber den Namen.
BODIES: dict[str, CelestialBody] = {
    EARTH.name: EARTH,
}


def get_body(name: str) -> CelestialBody:
    """Liefere den Katalogkoerper mit dem gegebenen Namen."""
    try:
        return BODIES[name]
    except KeyError as exc:
        raise ValueError(f"Unbekannter Himmelskoerper: {name!r}") from exc

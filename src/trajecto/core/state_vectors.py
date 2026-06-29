"""Zustandsvektoren und dynamische Invarianten in der Bahnebene (2D).

Trajecto stellt Bahnen in der Bahnebene dar. Der Zustand eines Koerpers wird
durch Positions- und Geschwindigkeitsvektor beschrieben (perifokales System:
die Periapsis liegt auf der +x-Achse, der Zentralkoerper im Ursprung/Fokus).

Alle Groessen in SI. Energien sind **spezifisch** (pro Masse); es wird keine
Satellitenmasse eingefuehrt:

    eps_kin = v^2 / 2
    eps_pot = -mu / r
    eps     = v^2 / 2 - mu / r
    h_vec   = r_vec x v_vec
    e_vec   = (v_vec x h_vec) / mu - r_vec / r

Vektoren sind 3D-Arrays mit z = 0, damit Kreuzprodukte sauber definiert sind.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OrbitState:
    """Momentaner Bahnzustand: Position und Geschwindigkeit im Fokus-System.

    Attributes:
        mu: Gravitationsparameter des Zentralkoerpers [m^3/s^2].
        position: Ortsvektor r_vec vom Fokus [m] (3D, z = 0).
        velocity: Geschwindigkeitsvektor v_vec [m/s] (3D, z = 0).
    """

    mu: float
    position: np.ndarray
    velocity: np.ndarray


def state_from_kepler(
    mu: float, semi_major_axis: float, eccentricity: float, true_anomaly: float
) -> OrbitState:
    """Bahnzustand aus a, e und wahrer Anomalie ``nu`` (perifokales System).

    Bahnradius ``r = a(1 - e^2) / (1 + e*cos(nu))``. Der Kreisfall ``e = 0``
    wird ohne Sonderbehandlung korrekt erfasst (p = a, r = a).

    Args:
        mu: Gravitationsparameter [m^3/s^2].
        semi_major_axis: grosse Halbachse a [m].
        eccentricity: Exzentrizitaet e (0 <= e < 1).
        true_anomaly: wahre Anomalie nu [rad].

    Raises:
        ValueError: bei mu <= 0, a <= 0 oder e ausserhalb [0, 1).
    """
    if mu <= 0.0:
        raise ValueError("Gravitationsparameter mu muss positiv sein.")
    if semi_major_axis <= 0.0:
        raise ValueError("Grosse Halbachse a muss positiv sein.")
    if not (0.0 <= eccentricity < 1.0):
        raise ValueError("Exzentrizitaet e muss im Bereich 0 <= e < 1 liegen.")

    p = semi_major_axis * (1.0 - eccentricity * eccentricity)  # Semi-latus rectum
    cos_nu = math.cos(true_anomaly)
    sin_nu = math.sin(true_anomaly)
    r = p / (1.0 + eccentricity * cos_nu)

    position = r * np.array([cos_nu, sin_nu, 0.0])
    # Geschwindigkeit im perifokalen System.
    factor = math.sqrt(mu / p)
    velocity = factor * np.array([-sin_nu, eccentricity + cos_nu, 0.0])
    return OrbitState(mu=mu, position=position, velocity=velocity)


def state_circular(mu: float, radius: float, true_anomaly: float) -> OrbitState:
    """Bahnzustand auf einer Kreisbahn (Sonderfall a = r, e = 0)."""
    if radius <= 0.0:
        raise ValueError("Radius muss positiv sein.")
    return state_from_kepler(mu, radius, 0.0, true_anomaly)


# --- Dynamische Groessen / Invarianten --------------------------------------


def radius(state: OrbitState) -> float:
    """Betrag des Ortsvektors |r_vec| [m]."""
    return float(np.linalg.norm(state.position))


def speed(state: OrbitState) -> float:
    """Betrag des Geschwindigkeitsvektors |v_vec| [m/s]."""
    return float(np.linalg.norm(state.velocity))


def specific_kinetic_energy(state: OrbitState) -> float:
    """Spezifische kinetische Energie ``v^2 / 2`` [J/kg]."""
    v = speed(state)
    return 0.5 * v * v


def specific_potential_energy(state: OrbitState) -> float:
    """Spezifische potentielle Energie ``-mu / r`` [J/kg]."""
    return -state.mu / radius(state)


def specific_total_energy(state: OrbitState) -> float:
    """Spezifische Gesamtenergie ``v^2/2 - mu/r`` [J/kg] (Bahninvariante)."""
    return specific_kinetic_energy(state) + specific_potential_energy(state)


def specific_angular_momentum(state: OrbitState) -> np.ndarray:
    """Spezifischer Drehimpulsvektor ``h_vec = r_vec x v_vec`` [m^2/s]."""
    return np.cross(state.position, state.velocity)


def eccentricity_vector(state: OrbitState) -> np.ndarray:
    """Exzentrizitaetsvektor ``e_vec = (v x h)/mu - r/|r|`` (dimensionslos).

    Im Zweikoerperproblem konstant, zeigt zur Periapsis; |e_vec| = e.
    """
    h_vec = specific_angular_momentum(state)
    r = radius(state)
    return np.cross(state.velocity, h_vec) / state.mu - state.position / r

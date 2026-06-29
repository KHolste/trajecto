"""Gemeinsame Darstellungshilfen fuer Bahnzustaende (Ergebnisse + Vektorplot).

Wiederverwendbar von ``circular_orbit`` und ``kepler_laws``. Die Physik liegt in
``core.state_vectors``; hier wird nur aus einem ``OrbitState`` der dynamische
Ergebnisblock gebaut und werden Radius-/Geschwindigkeitsvektoren gezeichnet.

Plot-Hinweis: Geschwindigkeitsvektoren werden rein **visuell** skaliert
(Richtung + qualitative Aenderung). Die echten Betraege stehen im Ergebnis.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from trajecto.core import state_vectors as sv
from trajecto.core.units import from_si
from trajecto.modules.base import ResultItem


def dynamic_result_items(state: sv.OrbitState) -> list[ResultItem]:
    """Dynamischer Ergebnisblock (Zustandsgroessen + Invarianten) in SI."""
    h_mag = float(np.linalg.norm(sv.specific_angular_momentum(state)))
    e_mag = float(np.linalg.norm(sv.eccentricity_vector(state)))
    return [
        ResultItem("Aktueller Radius", sv.radius(state), "km"),
        ResultItem("Aktuelle Geschwindigkeit", sv.speed(state), "km/s"),
        ResultItem("Spez. kinetische Energie", sv.specific_kinetic_energy(state), "MJ/kg"),
        ResultItem("Spez. potentielle Energie", sv.specific_potential_energy(state), "MJ/kg"),
        ResultItem("Spez. Gesamtenergie", sv.specific_total_energy(state), "MJ/kg"),
        ResultItem("Betrag spez. Drehimpuls", h_mag, "km^2/s"),
        ResultItem("Betrag Exzentrizitätsvektor", e_mag, "-"),
    ]


def draw_state_vectors(
    ax,
    position_m: np.ndarray,
    velocity_m: np.ndarray,
    *,
    velocity_length_km: float,
    show_radius: bool = True,
    show_velocity: bool = True,
    add_legend: bool = True,
    sat_color: str = "#222222",
    radius_color: str = "#5a189a",
    velocity_color: str = "#c1121f",
) -> None:
    """Zeichne Satellitenposition, Radiusvektor und (skalierten) v-Vektor in km.

    Args:
        ax: Matplotlib-Achse.
        position_m: Ortsvektor [m] (mind. 2 Komponenten).
        velocity_m: Geschwindigkeitsvektor [m/s] (mind. 2 Komponenten).
        velocity_length_km: gewuenschte visuelle Pfeillaenge des v-Vektors [km].
        show_radius: Radiusvektor zeichnen.
        show_velocity: Geschwindigkeitsvektor zeichnen.
    """
    pos = np.asarray(position_m)[:2] / 1000.0  # km
    vel = np.asarray(velocity_m)[:2]
    speed = float(np.linalg.norm(vel))

    # Radiusvektor vom Fokus (Ursprung) zum Satelliten – physisch massstaeblich.
    if show_radius:
        ax.annotate(
            "", xy=(pos[0], pos[1]), xytext=(0.0, 0.0),
            arrowprops=dict(arrowstyle="->", color=radius_color, lw=1.4), zorder=5,
        )
        if add_legend:
            ax.plot([], [], color=radius_color, lw=1.4, label="Radiusvektor r")
    # Geschwindigkeitsvektor – nur visuell skaliert (Richtung zaehlt).
    if show_velocity and speed > 0.0:
        vdir = vel / speed
        tip = (pos[0] + vdir[0] * velocity_length_km, pos[1] + vdir[1] * velocity_length_km)
        ax.annotate(
            "", xy=tip, xytext=(pos[0], pos[1]),
            arrowprops=dict(arrowstyle="->", color=velocity_color, lw=1.8), zorder=5,
        )
        if add_legend:
            ax.plot([], [], color=velocity_color, lw=1.8,
                    label="Geschwindigkeit v (skaliert)")
    ax.plot([pos[0]], [pos[1]], "o", color=sat_color, markersize=6, zorder=6)


def sample_energies(
    state_at: Callable[[float], sv.OrbitState], n: int = 181
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Spezifische Energien ueber tau in [0, 1] abtasten (alle in SI, J/kg).

    ``state_at(tau)`` liefert den Bahnzustand zur Zeitposition tau. Die
    eigentliche Physik kommt aus ``core.state_vectors``.
    """
    taus = np.linspace(0.0, 1.0, n)
    kin = np.empty(n)
    pot = np.empty(n)
    tot = np.empty(n)
    for i, tau in enumerate(taus):
        state = state_at(float(tau))
        kin[i] = sv.specific_kinetic_energy(state)
        pot[i] = sv.specific_potential_energy(state)
        tot[i] = sv.specific_total_energy(state)
    return taus, kin, pot, tot


def draw_energy_over_tau(
    ax,
    taus: np.ndarray,
    eps_kin: np.ndarray,
    eps_pot: np.ndarray,
    eps_tot: np.ndarray,
    *,
    current_tau: float | None = None,
) -> None:
    """Energieverlauf (kinetisch/potentiell/gesamt) ueber tau in MJ/kg.

    Keine festen Farben – Matplotlib-Defaults. Der aktuelle tau-Wert wird, wenn
    angegeben, als senkrechte Linie markiert.
    """
    def to_mj(values: np.ndarray) -> np.ndarray:
        return np.array([from_si(float(v), "MJ/kg") for v in values])

    ax.plot(taus, to_mj(eps_kin), label="kinetisch")
    ax.plot(taus, to_mj(eps_pot), label="potentiell")
    ax.plot(taus, to_mj(eps_tot), label="gesamt")
    if current_tau is not None:
        ax.axvline(float(current_tau), color="0.5", linestyle="--", linewidth=0.8)
    ax.set_xlabel("τ [-]")
    ax.set_ylabel("spez. Energie [MJ/kg]")
    ax.set_title("Energieverlauf über τ")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(loc="best", fontsize="small")

"""Modul ``hohmann_transfer`` – Hohmann-Transfer zwischen zwei Kreisbahnen.

Zwei Bedienmodi:

* **Idealer Hohmann-Transfer** – klassische Vorschau mit Transferellipse und
  idealen Delta-v-Werten.
* **Δv₁-Experiment** – manoevergetriebener Ablauf: Das Raumfahrzeug startet auf
  der Startkreisbahn; erst durch ``Δv₁ ausfuehren`` (und spaeter
  ``Δv₂ ausfuehren``) entstehen neue Bahnen. Vor dem ersten Impuls ist keine
  Transfer-/Experimentbahn sichtbar.

Die Fachlogik stammt aus ``core``; das Modul orchestriert Eingaben, einen
einfachen Simulationszustand (Phase), Ergebnisse, Visualisierung und Erklaerung.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from trajecto.core import applog
from trajecto.core import orbit_determination as od
from trajecto.core import orbital_mechanics as om
from trajecto.core import state_vectors as sv
from trajecto.core.bodies import BODIES, EARTH, get_body
from trajecto.core.orbit_input import (
    INPUT_MODE_ALTITUDE,
    INPUT_MODE_RADIUS,
    INPUT_MODES,
    altitude_from_radius,
    radius_from_input,
)
from trajecto.modules.base import (
    Action,
    ActionResult,
    ChoiceParameter,
    FloatParameter,
    Module,
    ModuleResult,
    Parameter,
    PlotOption,
    ResultItem,
    ResultSection,
)
from trajecto.modules.dynamics_view import (
    draw_energy_over_tau,
    draw_state_vectors,
    dynamic_result_items,
    sample_energies,
)

#: Default: Hoehenmodus, LEO (500 km) -> GEO (35786 km).
_DEFAULT_H1_SI = 500_000.0      # 500 km Starthoehe
_DEFAULT_H2_SI = 35_786_000.0   # geostationaere Hoehe
_DEFAULT_TAU = 0.0
#: Defaults fuer die experimentellen Delta-v: ideale Hohmann-Werte der
#: Standardbahnen (500 km -> 35786 km Hoehe).
_DEFAULT_TRANSFER = om.hohmann_transfer(
    EARTH.mu, EARTH.mean_radius + _DEFAULT_H1_SI, EARTH.mean_radius + _DEFAULT_H2_SI
)
_DEFAULT_DV1_SI = _DEFAULT_TRANSFER.dv1
_DEFAULT_DV2_SI = _DEFAULT_TRANSFER.dv2
#: Relative Toleranz fuer "Zielbahn erreicht".
_TARGET_REL_TOL = 1e-3
#: Didaktische Toleranz: Bahn gilt als (nahezu) kreisfoermig bzw. als Zielbahn
#: erreicht. Die physikalische Core-Klassifikation bleibt davon unberuehrt.
_CIRCLE_DIDACTIC_TOL = 5e-3

#: Bedienmodi.
MODE_IDEAL = "Idealer Hohmann-Transfer"
MODE_EXPERIMENT = "Δv₁-Experiment"
MODES = (MODE_IDEAL, MODE_EXPERIMENT)

#: Simulationsphasen im Experimentmodus.
PHASE_PARKING = "parking_orbit"
PHASE_AFTER_DV1 = "after_dv1"
PHASE_AFTER_DV2 = "after_dv2"
PHASE_LABELS = {
    PHASE_PARKING: "Startkreisbahn",
    PHASE_AFTER_DV1: "nach Δv₁",
    PHASE_AFTER_DV2: "nach Δv₂",
}


def _direction_word(dv_signed: float) -> str:
    return "prograd" if dv_signed >= 0.0 else "retrograd"


def _in_experiment(values: dict[str, Any]) -> bool:
    return values.get("mode") == MODE_EXPERIMENT


def transfer_state(
    mu: float, a_transfer: float, e_transfer: float, outward: bool, s: float
) -> sv.OrbitState:
    """Bahnzustand auf der Transferellipse an der Transferposition ``s`` (Idealmodus)."""
    theta_peri = 0.0 if outward else math.pi
    base = sv.state_from_kepler(mu, a_transfer, e_transfer, s - theta_peri)
    if outward:
        return base
    return sv.OrbitState(mu=mu, position=-base.position, velocity=-base.velocity)


def transfer_s_from_tau(tau: float, e_transfer: float, outward: bool) -> float:
    """Transferposition ``s`` aus der normierten Zeitposition ``tau`` (0..1)."""
    mean = math.pi * tau if outward else math.pi * (tau - 1.0)
    nu_transfer = om.true_anomaly_from_mean(mean, e_transfer)
    return nu_transfer if outward else nu_transfer + math.pi


def experiment_state(mu: float, r1: float, vy: float, tau: float) -> sv.OrbitState:
    """Zustand auf der nach Delta-v 1 entstehenden gebundenen Bahn (Impuls bei (r1,0)).

    Spezialfall des allgemeinen Propagators (Burn am Startpunkt +x); dient nur
    noch als Hilfs-/Testfunktion. Der eigentliche Ablauf nutzt den am tatsaech-
    lichen Klickpunkt erfassten Zustand.
    """
    return od.propagate(
        mu, np.array([r1, 0.0, 0.0]), np.array([0.0, vy, 0.0]), tau
    )


class HohmannTransferModule(Module):
    """Hohmann-Transfer zwischen zwei koplanaren Kreisbahnen."""

    id = "hohmann_transfer"
    title = "Hohmann-Transfer"
    subtitle = "Zwei-Impuls-Transfer zwischen zwei Kreisbahnen"

    def __init__(self) -> None:
        self._reset_state()

    def _reset_state(self) -> None:
        # Simulationszustand (nur im Experimentmodus genutzt).
        self._phase = PHASE_PARKING
        self._applied_dv1 = 0.0
        self._applied_dv2 = 0.0
        # Gespeicherte Burn-Zustaende (am tatsaechlichen Klickpunkt erfasst).
        self._burn1: dict | None = None  # position, velocity (nach Δv₁), impulse
        self._burn2: dict | None = None  # position, velocity (nach Δv₂), impulse

    def on_activated(self) -> None:
        # Variante A: Bei (erneuter) Auswahl des Moduls auf die Startphase
        # zuruecksetzen – didaktisch erwartbar fuer Lernsoftware.
        self._reset_state()

    # -- Parameter / Optionen / Aktionen --------------------------------------

    def parameters(self) -> list[Parameter]:
        return [
            ChoiceParameter(name="mode", label="Modus", choices=MODES,
                            default=MODE_IDEAL, group="Setup"),
            ChoiceParameter(name="body", label="Zentralkörper",
                            choices=tuple(BODIES.keys()), default=EARTH.name,
                            group="Setup"),
            ChoiceParameter(name="input_mode", label="Eingabemodus",
                            choices=INPUT_MODES, default=INPUT_MODE_ALTITUDE,
                            tooltip="Eingabe als Radius vom Mittelpunkt oder Höhe "
                                    "über der Oberfläche (gilt für r1 und r2)",
                            group="Setup"),
            FloatParameter(name="r1", label="Startwert", default_si=_DEFAULT_H1_SI,
                           display_unit="km", minimum_si=-1.0e7,
                           tooltip="Startbahn: r1 bzw. h1", group="Setup"),
            FloatParameter(name="r2", label="Zielwert", default_si=_DEFAULT_H2_SI,
                           display_unit="km", minimum_si=-1.0e7,
                           tooltip="Zielbahn: r2 bzw. h2", group="Setup"),
            FloatParameter(name="dv1_exp", label="Δv₁ exp.",
                           default_si=_DEFAULT_DV1_SI, display_unit="km/s",
                           minimum_si=-1.0e4, maximum_si=1.0e4,
                           visible_when=_in_experiment,
                           tooltip="Experimentelles Δv₁, positiv = prograd",
                           group="Manöver"),
            FloatParameter(name="dv2_exp", label="Δv₂ exp.",
                           default_si=_DEFAULT_DV2_SI, display_unit="km/s",
                           minimum_si=-1.0e4, maximum_si=1.0e4,
                           visible_when=_in_experiment,
                           tooltip="Experimentelles Δv₂, positiv = prograd",
                           group="Manöver"),
            FloatParameter(name="tau", label="τ", default_si=_DEFAULT_TAU,
                           display_unit="-", minimum_si=0.0, maximum_si=1.0,
                           animatable=True,
                           tooltip="Normierte Zeitposition entlang der aktiven Bahn",
                           group="Animation"),
        ]

    def plot_options(self) -> list[PlotOption]:
        return [
            # Default didaktisch ruhig: aktive Bahn + Geschwindigkeit; Radiusvektor,
            # vorherige Bahn und ideale Referenz sind standardmaessig aus.
            PlotOption("show_active_orbit", "Aktive Bahn anzeigen", True,
                       visible_when=_in_experiment),
            PlotOption("show_velocity", "Geschwindigkeitsvektor anzeigen", True),
            PlotOption("show_dv", "Delta-v-Pfeile anzeigen", True),
            PlotOption("show_radius", "Radiusvektor anzeigen", False),
            PlotOption("show_orbit_after_dv1", "Vorherige Bahn (nach Δv₁) anzeigen",
                       False, visible_when=_in_experiment),
            PlotOption("show_ideal_orbit", "Ideale Hohmann-Bahn anzeigen", False,
                       visible_when=_in_experiment),
            PlotOption("show_energy", "Energieplot anzeigen", False),
        ]

    def actions(self) -> list[Action]:
        return [
            Action("execute_dv1", "Δv₁ ausführen", visible_when=_in_experiment,
                   tooltip="Ersten tangentialen Impuls auf der Startkreisbahn ausführen"),
            Action("execute_dv2", "Δv₂ ausführen", visible_when=_in_experiment,
                   tooltip="Zweiten tangentialen Impuls an der aktuellen Position "
                           "ausführen (ideal am Apsidenpunkt, τ ≈ 0,5)"),
            Action("reset_sim", "Simulation zurücksetzen", visible_when=_in_experiment,
                   tooltip="Zurück zur Startkreisbahn"),
            Action("set_ideal", "Ideale Werte einsetzen", visible_when=_in_experiment,
                   tooltip="Ideale Delta-v-Werte für Δv₁ und Δv₂ einsetzen"),
        ]

    def is_action_enabled(self, name: str, values: dict[str, Any]) -> bool:
        if name == "execute_dv1":
            return self._phase == PHASE_PARKING
        if name == "execute_dv2":
            # Δv₂ ist nach Δv₁ jederzeit und an jeder Position moeglich – auch an
            # ungeeigneten Punkten wie der Periapsis (dann entsteht z. B. eine
            # Fluchtbahn). Die Zielkreisbahn entsteht ideal nur am Apsidenpunkt.
            return self._phase == PHASE_AFTER_DV1 and self._burn1 is not None
        if name == "reset_sim":
            return self._phase != PHASE_PARKING
        return True  # set_ideal

    def perform_action(self, name: str, values: dict[str, Any]) -> ActionResult:
        body, r1, r2 = self._read_bodies(values)
        mu = body.mu
        tau = float(values.get("tau", 0.0))

        if name == "execute_dv1" and self._phase == PHASE_PARKING:
            # Aktuellen Zustand auf der Startkreisbahn am Klickpunkt erfassen.
            v_circ_1 = om.circular_orbit_velocity(mu, r1)
            state = od.propagate(
                mu, np.array([r1, 0.0, 0.0]), np.array([0.0, v_circ_1, 0.0]), tau
            )
            dv1 = float(values.get("dv1_exp", _DEFAULT_DV1_SI))
            self._burn1 = self._apply_impulse(mu, state, dv1)
            self._applied_dv1 = dv1
            self._phase = PHASE_AFTER_DV1
            return ActionResult(reset_tau=True)

        if name == "execute_dv2" and self._phase == PHASE_AFTER_DV1 and self._burn1:
            if not self._is_bound(self._burn1):
                return ActionResult()
            # Aktuellen Zustand auf der Bahn nach Δv₁ am Klickpunkt erfassen.
            state = od.propagate(mu, self._burn1["position"], self._burn1["velocity"], tau)
            dv2 = float(values.get("dv2_exp", _DEFAULT_DV2_SI))
            self._burn2 = self._apply_impulse(mu, state, dv2)
            self._applied_dv2 = dv2
            self._phase = PHASE_AFTER_DV2
            return ActionResult(reset_tau=True)

        if name == "reset_sim":
            self._reset_state()
            return ActionResult(reset_tau=True, stop_animation=True)

        if name == "set_ideal":
            t = om.hohmann_transfer(mu, r1, r2)
            return ActionResult(set_params_si={"dv1_exp": t.dv1, "dv2_exp": t.dv2})
        return ActionResult()

    # -- Gemeinsame Hilfen ----------------------------------------------------

    @staticmethod
    def _read_bodies(values: dict[str, Any]):
        body = get_body(values["body"])
        mode = values.get("input_mode", INPUT_MODE_RADIUS)
        r1 = radius_from_input(body, float(values["r1"]), mode)
        r2 = radius_from_input(body, float(values["r2"]), mode)
        return body, r1, r2

    @staticmethod
    def _apply_impulse(mu: float, state: sv.OrbitState, dv: float) -> dict:
        """Tangentialen (prograden) Impuls ``dv`` auf den Geschwindigkeitsvektor.

        Die Position bleibt **exakt erhalten** (kein Sprung); nur die
        Geschwindigkeit aendert sich instantan. ``dv > 0`` ist prograd.
        """
        pos = np.asarray(state.position, dtype=float).copy()
        vel_before = np.asarray(state.velocity, dtype=float).copy()
        speed = float(np.linalg.norm(vel_before))
        tang = vel_before / speed if speed > 0.0 else np.zeros(3)
        impulse = dv * tang
        return {"mu": mu, "position": pos, "velocity": vel_before + impulse,
                "velocity_before": vel_before, "impulse": impulse}

    @staticmethod
    def _is_bound(burn: dict) -> bool:
        el = od.orbital_elements(burn["mu"], burn["position"], burn["velocity"])
        return el.is_bound

    def _active_reference(self, mu: float, r1: float, v_circ_1: float):
        """(position, velocity) als tau=0-Referenz der aktuell aktiven Bahn."""
        if self._phase == PHASE_PARKING:
            return np.array([r1, 0.0, 0.0]), np.array([0.0, v_circ_1, 0.0])
        if self._phase == PHASE_AFTER_DV1 and self._burn1:
            return self._burn1["position"], self._burn1["velocity"]
        if self._phase == PHASE_AFTER_DV2 and self._burn2:
            return self._burn2["position"], self._burn2["velocity"]
        # Fallback (sollte nicht auftreten).
        return np.array([r1, 0.0, 0.0]), np.array([0.0, v_circ_1, 0.0])

    @staticmethod
    def _spacecraft_state(mu, ref_pos, ref_vel, tau: float) -> sv.OrbitState:
        # propagate beherrscht gebundene und ungebundene Bahnen; ungebundene
        # werden entlang ihres Bogens animiert (kein Einfrieren mehr).
        try:
            return od.propagate(mu, ref_pos, ref_vel, tau)
        except Exception:  # defensiv: niemals die GUI haengen lassen
            applog.logger.exception("propagate fehlgeschlagen (Raumfahrzeug)")
            return sv.OrbitState(mu=mu, position=np.asarray(ref_pos, dtype=float),
                                 velocity=np.asarray(ref_vel, dtype=float))

    @staticmethod
    def _orbit_plot_data(mu: float, burn: dict | None) -> dict | None:
        """Plot-Daten einer Burn-Bahn (Burnpunkt, Geschwindigkeit, Impulspfeil)."""
        if burn is None:
            return None
        return {
            "position": burn["position"],
            "velocity": burn["velocity"],
            "impulse": burn["impulse"],
            "bound": HohmannTransferModule._is_bound(burn),
        }

    # -- Berechnung -----------------------------------------------------------

    def compute(self, values: dict[str, Any]) -> ModuleResult:
        body, r1, r2 = self._read_bodies(values)
        h1 = altitude_from_radius(body, r1)
        h2 = altitude_from_radius(body, r2)
        # core validiert mu>0, r1>0, r2>0, r1 != r2 (loest ValueError aus).
        t = om.hohmann_transfer(body.mu, r1, r2)

        if values.get("mode", MODE_IDEAL) == MODE_EXPERIMENT:
            return self._compute_experiment(values, body, r1, r2, h1, h2, t)
        return self._compute_ideal(values, body, r1, r2, h1, h2, t)

    def _compute_ideal(self, values, body, r1, r2, h1, h2, t) -> ModuleResult:
        e_transfer = abs(r2 - r1) / (r1 + r2)
        s = transfer_s_from_tau(float(values.get("tau", 0.0)), e_transfer, t.outward)
        nu_transfer = s if t.outward else s - math.pi
        state = transfer_state(body.mu, t.a_transfer, e_transfer, t.outward, s)

        energy_start = sv.specific_total_energy(sv.state_circular(body.mu, r1, 0.0))
        energy_transfer = sv.specific_total_energy(state)
        energy_target = sv.specific_total_energy(sv.state_circular(body.mu, r2, 0.0))

        items: list = [
            ResultSection("Idealtransfer"),
            ResultItem("Kreisbahngeschw. Startbahn", t.v_circ_1, "km/s"),
            ResultItem("Kreisbahngeschw. Zielbahn", t.v_circ_2, "km/s"),
            ResultItem("Transfergeschw. am Startpunkt", t.v_transfer_1, "km/s"),
            ResultItem("Transfergeschw. am Zielpunkt", t.v_transfer_2, "km/s"),
            ResultItem(f"Delta-v 1 ({_direction_word(t.dv1)})", abs(t.dv1), "km/s"),
            ResultItem(f"Delta-v 2 ({_direction_word(t.dv2)})", abs(t.dv2), "km/s"),
            ResultItem("Gesamt-Delta-v", t.dv_total, "km/s"),
            ResultItem("Transferzeit", t.transfer_time, "min"),
            ResultItem("Halbachse Transferellipse", t.a_transfer, "km"),
            ResultItem("Verwendeter Startradius", r1, "km"),
            ResultItem("Verwendeter Zielradius", r2, "km"),
            ResultItem("Verwendete Starthöhe", h1, "km"),
            ResultItem("Verwendete Zielhöhe", h2, "km"),
            ResultSection("Aktueller Zustand (Transfer)"),
            ResultItem("Aktuelle wahre Anomalie (Transfer)", nu_transfer, "deg"),
            *dynamic_result_items(state),
            ResultSection("Energie"),
            ResultItem("Gesamtenergie Startkreisbahn", energy_start, "MJ/kg"),
            ResultItem("Gesamtenergie Transferellipse", energy_transfer, "MJ/kg"),
            ResultItem("Gesamtenergie Zielkreisbahn", energy_target, "MJ/kg"),
        ]
        notes = ["Transferart: " + ("nach aussen (r2 > r1)" if t.outward
                                    else "nach innen (r2 < r1)")]
        self._add_altitude_warning(notes, body, h1, h2)

        data = {
            "experiment": False,
            "body_name": body.name, "body_radius": body.mean_radius,
            "r1": r1, "r2": r2, "mu": body.mu,
            "a_transfer": t.a_transfer, "e_transfer": e_transfer, "outward": t.outward,
            "dv1_prograde": t.dv1 >= 0.0, "dv2_prograde": t.dv2 >= 0.0,
            "position": state.position, "velocity": state.velocity,
        }
        return ModuleResult(items=items, notes=notes, data=data)

    def _compute_experiment(self, values, body, r1, r2, h1, h2, t) -> ModuleResult:
        mu = body.mu
        v_circ_1 = t.v_circ_1
        tau = float(values.get("tau", 0.0))
        applied_dv1 = self._applied_dv1 if self._phase in (PHASE_AFTER_DV1, PHASE_AFTER_DV2) else 0.0
        applied_dv2 = self._applied_dv2 if self._phase == PHASE_AFTER_DV2 else 0.0
        applied_total = abs(applied_dv1) + abs(applied_dv2)

        ref_pos, ref_vel = self._active_reference(mu, r1, v_circ_1)
        el = od.orbital_elements(mu, ref_pos, ref_vel)
        sc = self._spacecraft_state(mu, ref_pos, ref_vel, tau)

        budget_block = [
            ResultItem("Ideales Delta-v 1", t.dv1, "km/s"),
            ResultItem("Ideales Delta-v 2", t.dv2, "km/s"),
            ResultItem("Ideales Gesamt-Delta-v", t.dv_total, "km/s"),
            ResultItem("Angewendetes Delta-v 1", applied_dv1, "km/s"),
            ResultItem("Angewendetes Delta-v 2", applied_dv2, "km/s"),
            ResultItem("Angewendetes Gesamt-Delta-v", applied_total, "km/s"),
            ResultItem("Abweichung vom idealen Budget", applied_total - t.dv_total, "km/s"),
        ]
        orbit_block = [
            ResultItem("Exzentrizität", el.eccentricity, "-"),
            ResultItem("Spez. Gesamtenergie", el.specific_energy, "MJ/kg"),
            ResultItem("Periapsisradius", el.periapsis_radius, "km"),
            ResultItem("Periapsishöhe", el.periapsis_radius - body.mean_radius, "km"),
        ]
        if el.is_bound:
            orbit_block += [
                ResultItem("Apoapsisradius", el.apoapsis_radius, "km"),
                ResultItem("Apoapsishöhe", el.apoapsis_radius - body.mean_radius, "km"),
                ResultItem("Umlaufzeit", el.period, "min"),
            ]

        items: list = [
            ResultSection("Phase & Δv-Budget"),
            *budget_block,
            ResultSection(f"Aktuelle Bahn ({PHASE_LABELS[self._phase]})"),
            *orbit_block,
            ResultSection("Bahnen"),
            ResultItem("Verwendeter Startradius", r1, "km"),
            ResultItem("Verwendeter Zielradius", r2, "km"),
        ]
        # Didaktische (nutzerseitige) Bahnbezeichnung; die physikalische
        # Core-Klassifikation (el.classification) bleibt unveraendert.
        notes = [
            f"Aktuelle Bahn: {self._didactic_class(el, r2)}.",
            self._experiment_target_note(el, r1, r2),
        ]
        self._add_altitude_warning(notes, body, h1, h2)

        # Prominente Statusbox: Phase, Bewertung der aktiven Bahn, Δv-Budget,
        # naechster moeglicher Schritt (gibt dem Experiment Ablaufcharakter).
        status = (
            f"Phase: {PHASE_LABELS[self._phase]}  •  "
            f"Aktive Bahn: {self._status_short_eval(el, r1, r2)}\n"
            f"Budget: {applied_total / 1000.0:.3f} km/s angewendet "
            f"(ideal {t.dv_total / 1000.0:.3f} km/s)\n"
            f"Nächster Schritt: {self._next_step(el)}"
        )

        # Daten fuer den Plot: gespeicherte Burn-Bahnen (am tatsaechlichen
        # Klickpunkt erfasst, beliebig orientiert).
        dv1_orbit = self._orbit_plot_data(mu, self._burn1) \
            if self._phase in (PHASE_AFTER_DV1, PHASE_AFTER_DV2) else None
        dv2_orbit = self._orbit_plot_data(mu, self._burn2) \
            if self._phase == PHASE_AFTER_DV2 else None

        data = {
            "experiment": True,
            "body_name": body.name, "body_radius": body.mean_radius,
            "r1": r1, "r2": r2, "mu": body.mu,
            "a_transfer": t.a_transfer, "e_transfer": abs(r2 - r1) / (r1 + r2),
            "outward": t.outward,
            "exp_phase": self._phase,
            "exp_active_position": np.asarray(ref_pos, dtype=float),
            "exp_active_velocity": np.asarray(ref_vel, dtype=float),
            "exp_active_bound": el.is_bound,
            "exp_sc_position": sc.position, "exp_sc_velocity": sc.velocity,
            "exp_dv1_orbit": dv1_orbit, "exp_dv2_orbit": dv2_orbit,
        }
        return ModuleResult(items=items, notes=notes, data=data, status=status)

    def _status_short_eval(self, el: od.OrbitElements, r1: float, r2: float) -> str:
        """Knappe Bewertung der aktiven Bahn fuer die Statusbox."""
        if self._phase == PHASE_PARKING:
            return "Startkreisbahn"
        if not el.is_bound:
            return "Fluchtbahn (nicht gebunden)"
        if self._phase == PHASE_AFTER_DV2:
            return ("Zielkreisbahn erreicht" if self._near_target_circle(el, r2)
                    else "Ellipse (Ziel nicht erreicht)")
        opposite = el.apoapsis_radius if el.apoapsis_radius is not None else el.periapsis_radius
        if abs(opposite - r2) / r2 < _TARGET_REL_TOL:
            return "Apsis erreicht Zielbahn"
        return ("Apsis unter Zielbahn (Δv₁ zu klein)" if opposite < r2
                else "Apsis über Zielbahn (Δv₁ zu groß)")

    def _next_step(self, el: od.OrbitElements) -> str:
        """Hinweis auf den naechsten moeglichen Schritt im Ablauf."""
        if self._phase == PHASE_PARKING:
            return "Δv₁ ausführen"
        if self._phase == PHASE_AFTER_DV1:
            if not el.is_bound:
                return "Simulation zurücksetzen (Fluchtbahn)"
            return "zum Apsidenpunkt fliegen (τ ≈ 0,5), dann Δv₂ ausführen"
        return "fertig – Simulation zurücksetzen für neuen Versuch"

    def _didactic_class(self, el: od.OrbitElements, r2: float) -> str:
        """Nutzerseitige Bahnbezeichnung (glaettet die numerische Klassifikation)."""
        if not el.is_bound:
            return el.classification  # Fluchtbahn
        near_circular = el.eccentricity < _CIRCLE_DIDACTIC_TOL
        if near_circular and self._phase == PHASE_AFTER_DV2 and self._near_target_circle(el, r2):
            return "Zielkreisbahn erreicht (nahezu kreisförmig)"
        if near_circular:
            return "nahezu kreisförmige Bahn"
        return el.classification  # gebundene Ellipse

    @staticmethod
    def _near_target_circle(el: od.OrbitElements, r2: float) -> bool:
        if el.apoapsis_radius is None:
            return False
        return (abs(el.periapsis_radius - r2) / r2 < _CIRCLE_DIDACTIC_TOL
                and abs(el.apoapsis_radius - r2) / r2 < _CIRCLE_DIDACTIC_TOL)

    @staticmethod
    def _add_altitude_warning(notes, body, h1, h2) -> None:
        if h1 < 0.0 or h2 < 0.0:
            notes.append(
                f"Warnung: Start- oder Zielbahn liegt unterhalb der Oberfläche von "
                f"{body.name} (Radius {body.mean_radius / 1000.0:,.0f} km, Höhe < 0) – "
                f"physikalisch keine freie Bahn."
            )

    def _experiment_target_note(self, el: od.OrbitElements, r1: float, r2: float) -> str:
        """Zielbewertung je nach Phase und resultierender Bahn."""
        if self._phase == PHASE_PARKING:
            return "Raumfahrzeug kreist auf der Startkreisbahn (vor Δv₁)."
        if not el.is_bound:
            return ("Die resultierende Bahn ist nicht gebunden (Fluchtbahn); "
                    "keine geschlossene Endlosschleife.")
        if self._phase == PHASE_AFTER_DV2:
            # Zielkreisbahn erreicht, wenn Peri und Apo nahe r2 liegen.
            if self._near_target_circle(el, r2):
                return "Zielkreisbahn erreicht (Bahn ist nahezu kreisförmig bei r2)."
            return "Zielkreisbahn nicht erreicht – resultierende Bahn ist eine Ellipse."
        # PHASE_AFTER_DV1: gegenüberliegender Apsidenpunkt entscheidet.
        opposite = el.apoapsis_radius if el.apoapsis_radius is not None else el.periapsis_radius
        if abs(opposite - r2) / r2 < _TARGET_REL_TOL:
            return "Apsidenpunkt erreicht die Zielbahn (ideales Δv₁) – jetzt Δv₂ ausführen."
        if opposite < r2:
            return "Apsidenpunkt unter Zielbahn – Δv₁ zu klein."
        return "Apsidenpunkt über Zielbahn – Δv₁ zu groß."

    # -- Visualisierung -------------------------------------------------------

    def plot(self, figure, values, result, options=None) -> None:
        figure.clear()
        if self.is_option_enabled(options, "show_energy"):
            ax, ax_energy = figure.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        else:
            ax = figure.add_subplot(111)
            ax_energy = None

        data = result.data
        r1 = data["r1"] / 1000.0
        r2 = data["r2"] / 1000.0
        body_r = data["body_radius"] / 1000.0
        body_name = data["body_name"]
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        scale_km = max(r1, r2)
        show_radius = self.is_option_enabled(options, "show_radius")
        show_velocity = self.is_option_enabled(options, "show_velocity")
        experiment = data.get("experiment", False)

        # Start- und Zielkreisbahn (immer sichtbar).
        ax.plot(r1 * np.cos(theta), r1 * np.sin(theta), color="#3a7bd5",
                linewidth=1.4, zorder=2, label="Startbahn r1")
        ax.plot(r2 * np.cos(theta), r2 * np.sin(theta), color="#2a9d8f",
                linewidth=1.4, zorder=2, label="Zielbahn r2")
        # Zentralkoerper im Fokus.
        ax.fill(body_r * np.cos(theta), body_r * np.sin(theta), color="#264653",
                alpha=0.9, zorder=4, label=body_name)
        ax.plot([r1], [0.0], "o", color="#1d3557", zorder=5)
        ax.plot([-r2], [0.0], "o", color="#006d77", zorder=5)

        if experiment:
            self._plot_experiment(ax, data, values, scale_km, options,
                                  show_radius, show_velocity)
            phase = data["exp_phase"]
            subtitle = f"Experiment – {PHASE_LABELS[phase]}"
        else:
            self._plot_ideal(ax, data, scale_km, options, show_radius, show_velocity)
            subtitle = "Idealtransfer"

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"Hohmann-Transfer um {body_name} – {subtitle}")
        ax.grid(True, linestyle=":", alpha=0.35)
        # Schlanke Legende: nur Bahnen/Koerper, keine Vektor-/Pfeil-Eintraege.
        ax.legend(loc="upper right", fontsize="x-small", framealpha=0.7)

        if ax_energy is not None:
            self._plot_energy(ax_energy, data, values, experiment)

    def _plot_ideal(self, ax, data, scale_km, options, show_radius, show_velocity) -> None:
        r1 = data["r1"] / 1000.0
        r2 = data["r2"] / 1000.0
        a = 0.5 * (r1 + r2)
        cx0 = 0.5 * (r1 - r2)
        b = math.sqrt(max(a * a - cx0 * cx0, 0.0))
        t_half = np.linspace(0.0, np.pi, 181)
        ax.plot(cx0 + a * np.cos(t_half), b * np.sin(t_half), color="#e76f51",
                linewidth=2.0, linestyle="--", zorder=5, label="Transferellipse")
        if self.is_option_enabled(options, "show_dv"):
            arrow_len = 0.14 * scale_km
            dy1 = arrow_len if data["dv1_prograde"] else -arrow_len
            ax.annotate("", xy=(r1, dy1), xytext=(r1, 0.0),
                        arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.6))
            dy2 = -arrow_len if data["dv2_prograde"] else arrow_len
            ax.annotate("", xy=(-r2, dy2), xytext=(-r2, 0.0),
                        arrowprops=dict(arrowstyle="->", color="#c1121f", lw=1.6))
        draw_state_vectors(ax, data["position"], data["velocity"],
                           velocity_length_km=0.30 * scale_km,
                           show_radius=show_radius, show_velocity=show_velocity,
                           add_legend=False)

    def _plot_experiment(self, ax, data, values, scale_km, options,
                         show_radius, show_velocity) -> None:
        mu = data["mu"]
        phase = data["exp_phase"]
        show_dv = self.is_option_enabled(options, "show_dv")
        dv1_orbit = data["exp_dv1_orbit"]
        dv2_orbit = data["exp_dv2_orbit"]

        # 1) Referenz-/vergangene Bahnen zuerst, dezent im Hintergrund.
        if self.is_option_enabled(options, "show_ideal_orbit"):
            r1 = data["r1"] / 1000.0
            r2 = data["r2"] / 1000.0
            a = 0.5 * (r1 + r2)
            cx0 = 0.5 * (r1 - r2)
            b = math.sqrt(max(a * a - cx0 * cx0, 0.0))
            t_half = np.linspace(0.0, np.pi, 181)
            ax.plot(cx0 + a * np.cos(t_half), b * np.sin(t_half), color="#e76f51",
                    linewidth=0.9, linestyle=":", alpha=0.6, zorder=3,
                    label="ideale Hohmann-Bahn")
        if (phase == PHASE_AFTER_DV2 and dv1_orbit is not None
                and self.is_option_enabled(options, "show_orbit_after_dv1")):
            self._draw_orbit_curve(ax, mu, dv1_orbit, color="#9d7bbf",
                                   label="vorher: nach Δv₁", linewidth=1.0,
                                   linestyle="--", alpha=0.55, zorder=3)

        # 2) Aktive Bahn prominent im Vordergrund.
        if self.is_option_enabled(options, "show_active_orbit"):
            if phase == PHASE_AFTER_DV2 and dv2_orbit is not None:
                self._draw_orbit_curve(ax, mu, dv2_orbit, color="#e07a00",
                                       label="aktive Bahn (nach Δv₂)",
                                       linewidth=2.4, zorder=5)
            elif phase == PHASE_AFTER_DV1 and dv1_orbit is not None:
                self._draw_orbit_curve(ax, mu, dv1_orbit, color="#7b2cbf",
                                       label="aktive Bahn (nach Δv₁)",
                                       linewidth=2.4, zorder=5)

        # 3) Delta-v-Pfeile (ohne Legendeneintraege).
        if show_dv:
            if dv1_orbit is not None:
                self._draw_impulse_arrow(ax, dv1_orbit, scale_km, color="#7b2cbf")
            if dv2_orbit is not None:
                self._draw_impulse_arrow(ax, dv2_orbit, scale_km, color="#e07a00")

        # 4) Raumfahrzeug auf der aktiven Bahn (Vektoren ohne Legende).
        draw_state_vectors(ax, data["exp_sc_position"], data["exp_sc_velocity"],
                           velocity_length_km=0.30 * scale_km,
                           show_radius=show_radius, show_velocity=show_velocity,
                           add_legend=False)

    @staticmethod
    def _draw_orbit_curve(ax, mu, orbit, color, label, linewidth=1.6,
                          linestyle="-", alpha=1.0, zorder=3) -> None:
        """Bahnkurve durch den tatsaechlichen Burnpunkt (allgemein orientiert).

        Gebundene Bahn -> geschlossene Ellipse; ungebundene Bahn (Fluchtbahn)
        -> offener Bogen entlang der Hyperbel/Parabel.
        """
        try:
            taus = np.linspace(0.0, 1.0, 241)
            pts = np.array([
                od.propagate(mu, orbit["position"], orbit["velocity"], float(tt)).position
                for tt in taus
            ]) / 1000.0
        except Exception:
            applog.logger.exception("Bahnkurve konnte nicht gezeichnet werden")
            return
        ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=linewidth,
                linestyle=linestyle, alpha=alpha, zorder=zorder, label=label)

    @staticmethod
    def _draw_impulse_arrow(ax, orbit, scale_km, color) -> None:
        """Δv-Pfeil am Burnpunkt in Richtung des tatsaechlichen Impulsvektors."""
        impulse = np.asarray(orbit["impulse"], dtype=float)
        mag = float(np.linalg.norm(impulse[:2]))
        if mag <= 0.0:
            return
        base = np.asarray(orbit["position"], dtype=float)[:2] / 1000.0
        direction = impulse[:2] / mag
        tip = base + direction * (0.16 * scale_km)
        ax.annotate("", xy=(tip[0], tip[1]), xytext=(base[0], base[1]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.8), zorder=6)

    def _plot_energy(self, ax_energy, data, values, experiment) -> None:
        mu = data["mu"]
        current_tau = values.get("tau")
        if experiment:
            # propagate beherrscht auch ungebundene Bahnen (Fluchtbahn-Bogen).
            ref_pos = data["exp_active_position"]
            ref_vel = data["exp_active_velocity"]
            taus, kin, pot, tot = sample_energies(
                lambda tau: od.propagate(mu, ref_pos, ref_vel, tau))
        else:
            a_t, e_t, outward = data["a_transfer"], data["e_transfer"], data["outward"]
            taus, kin, pot, tot = sample_energies(
                lambda tau: transfer_state(
                    mu, a_t, e_t, outward, transfer_s_from_tau(tau, e_t, outward)))
        draw_energy_over_tau(ax_energy, taus, kin, pot, tot, current_tau=current_tau)

    # -- Erklaerung -----------------------------------------------------------

    def explanation(self) -> str:
        return (
            "Hohmann-Transfer\n"
            "================\n\n"
            "Der Hohmann-Transfer ist der treibstoffgünstigste Weg, um zwischen\n"
            "zwei koplanaren Kreisbahnen mit den Radien r1 und r2 zu wechseln.\n"
            "Er besteht aus zwei impulsiven Manövern: Delta-v 1 bringt das\n"
            "Raumfahrzeug von der Startkreisbahn auf eine Transferellipse, Delta-v 2\n"
            "macht am gegenüberliegenden Apsidenpunkt daraus die Zielkreisbahn.\n"
            "Die große Halbachse der Transferellipse ist a = (r1 + r2) / 2, die\n"
            "Transferzeit die halbe Umlaufzeit dieser Ellipse.\n\n"
            "Modus 'Idealer Hohmann-Transfer':\n"
            "  Zeigt die Transferellipse und die idealen Delta-v-Werte als Vorschau.\n\n"
            "Modus 'Δv₁-Experiment' (manövergetrieben):\n"
            "  Der Transfer wird als Folge impulsiver Geschwindigkeitsänderungen\n"
            "  erlebt. Vor dem ersten Impuls befindet sich das Raumfahrzeug auf der\n"
            "  Startkreisbahn; es ist noch keine Transfer-/Experimentbahn sichtbar.\n"
            "  - 'Δv₁ ausführen' ändert instantan den Geschwindigkeitsvektor und\n"
            "    erzeugt eine neue Keplerbahn. Ohne Δv₂ bleibt das Raumfahrzeug auf\n"
            "    dieser Bahn und kehrt bei gebundener Ellipse wieder zum Impulspunkt\n"
            "    zurück.\n"
            "  - 'Δv₂ ausführen' wirkt an der aktuellen Position auf der Bahn nach\n"
            "    Δv₁ und ist jederzeit möglich. Die Zielkreisbahn entsteht aber nur,\n"
            "    wenn Δv₂ am Apsidenpunkt (außen: Apoapsis, innen: Periapsis, τ ≈ 0,5)\n"
            "    mit passendem Betrag ausgeführt wird; an anderen Punkten entsteht\n"
            "    eine andere Ellipse.\n"
            "  - Nur bei passenden Δv₁ und Δv₂ entsteht die Zielkreisbahn. Zu kleine\n"
            "    oder zu große Impulse ergeben andere Ellipsen oder eine Fluchtbahn.\n"
            "  - Das Δv-Budget ist die Summe der angewendeten Impulse.\n"
            "  - 'Simulation zurücksetzen' bringt das Raumfahrzeug auf die\n"
            "    Startkreisbahn zurück; die Animation läuft je Phase zeitbasiert\n"
            "    über die aktuelle Bahn (τ = 0 am jeweiligen Impulspunkt).\n\n"
            "Eingabe: r1 und r2 können als Radius vom Mittelpunkt oder als Höhe\n"
            "über der Oberfläche angegeben werden. Idealisierungen: koplanare\n"
            "Kreisbahnen, impulsive Manöver, reines Zwei-Körper-Problem."
        )

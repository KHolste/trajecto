"""Modul ``kepler_laws`` – die drei Keplerschen Gesetze anschaulich.

Didaktisches Grundlagenmodul:

* 1. Gesetz – elliptische Bahn mit Zentralkoerper in einem Brennpunkt.
* 2. Gesetz – gleiche Flaechen in gleichen Zeiten (zeitgetreu markierte Segmente).
* 3. Gesetz – Zusammenhang von grosser Halbachse a und Umlaufzeit T.

Die gesamte Fachlogik stammt aus ``core``; das Modul orchestriert nur Eingaben,
Ergebnisse, Visualisierung und Erklaerung.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from matplotlib.figure import Figure

from trajecto.core import orbital_mechanics as om
from trajecto.core import state_vectors as sv
from trajecto.core.bodies import BODIES, EARTH, get_body
from trajecto.modules.base import (
    ChoiceParameter,
    FloatParameter,
    Module,
    ModuleResult,
    Parameter,
    PlotOption,
    ResultItem,
)
from trajecto.modules.dynamics_view import (
    draw_energy_over_tau,
    draw_state_vectors,
    dynamic_result_items,
    sample_energies,
)

#: Standardwerte: deutlich elliptische Erdbahn, kollisionsfrei.
_DEFAULT_A_SI = 24_000_000.0  # 24 000 km
_DEFAULT_E = 0.3
#: Default-Zeitposition tau (0..1).
_DEFAULT_TAU = 0.15


class KeplerLawsModule(Module):
    """Keplersche Gesetze an einer frei waehlbaren Ellipsenbahn."""

    id = "kepler_laws"
    title = "Keplersche Gesetze"
    subtitle = "Ellipsenbahn, Flaechensatz und 3. Gesetz"

    def parameters(self) -> list[Parameter]:
        return [
            ChoiceParameter(
                name="body",
                label="Zentralkörper",
                choices=tuple(BODIES.keys()),
                default=EARTH.name,
            ),
            FloatParameter(
                name="a",
                label="Große Halbachse a",
                default_si=_DEFAULT_A_SI,
                display_unit="km",
                minimum_si=1.0,
            ),
            FloatParameter(
                name="e",
                label="Exzentrizität e",
                default_si=_DEFAULT_E,
                display_unit="-",
                minimum_si=0.0,
                maximum_si=0.95,
            ),
            FloatParameter(
                name="tau",
                label="Zeitposition τ (0..1)",
                default_si=_DEFAULT_TAU,
                display_unit="-",
                minimum_si=0.0,
                maximum_si=1.0,
                animatable=True,
            ),
            ChoiceParameter(
                name="segments",
                label="Flächensegmente (2. Gesetz)",
                choices=("4", "6", "8", "12"),
                default="6",
            ),
        ]

    def plot_options(self) -> list[PlotOption]:
        return [
            PlotOption("show_radius", "Radiusvektor anzeigen", True),
            PlotOption("show_velocity", "Geschwindigkeitsvektor anzeigen", True),
            PlotOption("show_ecc", "Exzentrizitaetsvektor anzeigen", True),
            PlotOption("show_segments", "Flaechensegmente anzeigen", True),
            PlotOption("show_axes", "Halbachsen anzeigen", True),
            PlotOption("show_energy", "Energieplot anzeigen", False),
        ]

    def compute(self, values: dict[str, Any]) -> ModuleResult:
        body = get_body(values["body"])
        a = float(values["a"])
        e = float(values["e"])
        segments = int(values.get("segments", 6))
        # Zeitposition tau hat Vorrang: M = 2*pi*tau, dann zeitgetreu nu(M, e).
        # Direkter nu-Wert bleibt als Fallback moeglich.
        if "tau" in values:
            nu = om.true_anomaly_from_mean(2.0 * math.pi * float(values["tau"]), e)
        else:
            nu = float(values.get("nu", 0.0))  # wahre Anomalie [rad]

        # core validiert a > 0 und 0 <= e < 1 (loest ValueError aus).
        r_peri = om.periapsis_radius(a, e)
        r_apo = om.apoapsis_radius(a, e)
        period = om.orbital_period(body.mu, a)
        v_peri = om.vis_viva_speed(body.mu, r_peri, a)
        v_apo = om.vis_viva_speed(body.mu, r_apo, a)
        speed_ratio = v_peri / v_apo

        # Aktueller Bahnzustand am gewaehlten Bahnwinkel.
        state = sv.state_from_kepler(body.mu, a, e, nu)
        ecc_vec = sv.eccentricity_vector(state)

        items = [
            ResultItem("Periapsisradius", r_peri, "km"),
            ResultItem("Apoapsisradius", r_apo, "km"),
            ResultItem("Umlaufzeit", period, "min"),
            ResultItem("Geschwindigkeit im Periapsis", v_peri, "km/s"),
            ResultItem("Geschwindigkeit im Apoapsis", v_apo, "km/s"),
            ResultItem("Verhaeltnis v_peri / v_apo", speed_ratio, "-"),
            ResultItem("Aktuelle wahre Anomalie", nu, "deg"),
            *dynamic_result_items(state),
        ]
        data = {
            "body_name": body.name,
            "body_radius": body.mean_radius,
            "a": a,
            "e": e,
            "r_peri": r_peri,
            "r_apo": r_apo,
            "segments": segments,
            "mu": body.mu,
            "position": state.position,
            "velocity": state.velocity,
            "ecc_vector": ecc_vec,
        }
        return ModuleResult(items=items, data=data)

    # -- Visualisierung -------------------------------------------------------

    def plot(
        self,
        figure: Figure,
        values: dict[str, Any],
        result: ModuleResult,
        options: dict[str, bool] | None = None,
    ) -> None:
        figure.clear()
        if self.is_option_enabled(options, "show_energy"):
            ax, ax_energy = figure.subplots(1, 2, gridspec_kw={"width_ratios": [3, 2]})
        else:
            ax = figure.add_subplot(111)
            ax_energy = None

        a = result.data["a"] / 1000.0  # km
        e = result.data["e"]
        body_r = result.data["body_radius"] / 1000.0
        body_name = result.data["body_name"]
        segments = result.data["segments"]

        c = a * e  # lineare Exzentrizitaet (Fokus-Abstand vom Mittelpunkt)
        b = a * np.sqrt(1.0 - e * e)  # kleine Halbachse

        # Bahnpunkt relativ zum Fokus (Zentralkoerper im Ursprung), aus
        # exzentrischer Anomalie E. Mittelpunkt der Ellipse liegt bei (-c, 0).
        def point(eccentric: np.ndarray | float):
            x = a * np.cos(eccentric) - c
            y = b * np.sin(eccentric)
            return x, y

        # 2. Gesetz: gleiche Zeitintervalle ueberstreichen gleiche Flaechen.
        # Die Segmentgrenzen liegen bei aequidistanten Werten der mittleren
        # Anomalie M (M ist proportional zur Zeit). Pro Segment einen Keil vom
        # Fokus zeichnen; gleiche Flaechen, aber unterschiedliche Winkelbreiten.
        if self.is_option_enabled(options, "show_segments"):
            colors = ["#f4a259", "#5b8e7d"]
            for i in range(segments):
                m0 = 2.0 * np.pi * i / segments
                m1 = 2.0 * np.pi * (i + 1) / segments
                m_samples = np.linspace(m0, m1, 24)
                ecc = np.array(
                    [om.solve_eccentric_anomaly(float(m), e) for m in m_samples]
                )
                px, py = point(ecc)
                poly_x = np.concatenate(([0.0], px, [0.0]))
                poly_y = np.concatenate(([0.0], py, [0.0]))
                ax.fill(
                    poly_x, poly_y,
                    color=colors[i % 2], alpha=0.45, edgecolor="none", zorder=1,
                )

        # Vollstaendige Ellipsenbahn.
        ecc_full = np.linspace(0.0, 2.0 * np.pi, 361)
        ex, ey = point(ecc_full)
        ax.plot(ex, ey, color="#2b2d42", linewidth=1.6, zorder=3, label="Bahn")

        # Halbachsen dezent andeuten (a entlang Hauptachse, b durch Mittelpunkt).
        if self.is_option_enabled(options, "show_axes"):
            ax.plot(
                [-(a + c), a - c], [0.0, 0.0],
                color="#888888", linestyle="--", linewidth=0.8, zorder=2,
            )
            ax.plot(
                [-c, -c], [-b, b],
                color="#888888", linestyle=":", linewidth=0.8, zorder=2,
            )

        # Zentralkoerper im Brennpunkt (Ursprung).
        theta = np.linspace(0.0, 2.0 * np.pi, 181)
        ax.fill(
            body_r * np.cos(theta), body_r * np.sin(theta),
            color="#3a7bd5", alpha=0.9, zorder=4, label=body_name,
        )

        # Periapsis (+x) und Apoapsis (-x) markieren.
        r_peri = result.data["r_peri"] / 1000.0
        r_apo = result.data["r_apo"] / 1000.0
        ax.plot([r_peri], [0.0], "o", color="#c1121f", zorder=5)
        ax.annotate("Periapsis", (r_peri, 0.0), textcoords="offset points",
                    xytext=(4, 6), fontsize="small", color="#c1121f")
        ax.plot([-r_apo], [0.0], "o", color="#005f73", zorder=5)
        ax.annotate("Apoapsis", (-r_apo, 0.0), textcoords="offset points",
                    xytext=(4, 6), fontsize="small", color="#005f73")

        # Exzentrizitaetsvektor (zeigt zur Periapsis, +x). Visuell mit e skaliert,
        # sodass er fuer nahezu kreisfoermige Bahnen sichtbar schrumpft.
        if self.is_option_enabled(options, "show_ecc"):
            ecc_vec = np.asarray(result.data["ecc_vector"])
            ecc_mag = float(np.linalg.norm(ecc_vec[:2]))
            if ecc_mag > 1e-6:
                direction = ecc_vec[:2] / ecc_mag
                length = ecc_mag * a  # km
                ax.annotate(
                    "", xy=(direction[0] * length, direction[1] * length),
                    xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="->", color="#2a9d8f", lw=1.6), zorder=5,
                )
                ax.plot([], [], color="#2a9d8f", lw=1.6, label="Exzentrizitätsvektor e")

        # Aktueller Zustand: Satellit, Radius- und (skalierter) v-Vektor.
        draw_state_vectors(
            ax,
            result.data["position"],
            result.data["velocity"],
            velocity_length_km=0.35 * a,
            show_radius=self.is_option_enabled(options, "show_radius"),
            show_velocity=self.is_option_enabled(options, "show_velocity"),
        )

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("x [km]")
        ax.set_ylabel("y [km]")
        ax.set_title(f"Keplerbahn um {body_name}  (a={a:,.0f} km, e={e:.2f})")
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.legend(loc="upper right", fontsize="small")

        if ax_energy is not None:
            mu = result.data["mu"]
            a_si = result.data["a"]
            e_si = result.data["e"]
            taus, kin, pot, tot = sample_energies(
                lambda tau: sv.state_from_kepler(
                    mu, a_si, e_si,
                    om.true_anomaly_from_mean(2.0 * np.pi * tau, e_si),
                )
            )
            draw_energy_over_tau(ax_energy, taus, kin, pot, tot,
                                 current_tau=values.get("tau"))

    def explanation(self) -> str:
        return (
            "Keplersche Gesetze\n"
            "==================\n\n"
            "1. Gesetz (Bahnform):\n"
            "   Jeder Planet bewegt sich auf einer Ellipse, in deren einem\n"
            "   Brennpunkt der Zentralkörper steht – nicht im Mittelpunkt.\n\n"
            "2. Gesetz (Flächensatz):\n"
            "   Die Verbindungslinie Körper–Zentralkörper überstreicht in\n"
            "   gleichen Zeiten gleiche Flächen. Die farbigen Segmente im Plot\n"
            "   haben dieselbe Fläche und stehen für gleiche Zeitabschnitte:\n"
            "   nahe der Periapsis ist die Bahn schnell, nahe der Apoapsis langsam.\n\n"
            "3. Gesetz (Periodengesetz):\n"
            "   T = 2*pi*sqrt(a^3 / mu).  Die Umlaufzeit hängt nur von der\n"
            "   großen Halbachse a (und mu) ab, nicht von der Exzentrizität.\n\n"
            "Begriffe:\n"
            "  a  – große Halbachse (mittlerer Bahnradius)\n"
            "  e  – Exzentrizität (0 = Kreis, näher 1 = stark gestreckt)\n"
            "  Periapsis  r_p = a*(1 - e):  nächster Punkt, höchste Geschwindigkeit\n"
            "  Apoapsis   r_a = a*(1 + e):  fernster Punkt, kleinste Geschwindigkeit\n\n"
            "Daher ist die Geschwindigkeit im Periapsis stets größer als im\n"
            "Apoapsis: v_peri / v_apo = (1 + e) / (1 - e).\n\n"
            "Zustand entlang der Bahn:\n"
            "  Im Periapsis ist die Geschwindigkeit maximal, im Apoapsis minimal.\n"
            "  Kinetische und potentielle Energie ändern sich entlang der Bahn,\n"
            "  ihre Summe (Gesamtenergie) bleibt jedoch konstant – ebenso der\n"
            "  Drehimpuls. Der Exzentrizitätsvektor bleibt konstant und zeigt\n"
            "  stets zur Periapsis; sein Betrag entspricht e.\n\n"
            "Animation (zeitbasiert):\n"
            "  Der animierte Parameter ist die normierte Zeitposition τ (0..1),\n"
            "  also der Bruchteil einer Umlaufzeit. Aus τ wird zuerst die mittlere\n"
            "  Anomalie M = 2*pi*τ und daraus über die Keplergleichung die wahre\n"
            "  Anomalie nu berechnet (Ergebniszeile 'Aktuelle wahre Anomalie').\n"
            "  Deshalb bewegt sich der Körper im Periapsis schneller und im\n"
            "  Apoapsis langsamer – nicht gleichförmig im Winkel."
        )

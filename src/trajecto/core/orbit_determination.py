"""Bahnbestimmung aus einem Zustandsvektor (Orbit determination).

Aus Position und Geschwindigkeit werden die Bahngroessen und die Bahnklasse
bestimmt. Reine Physik in SI; keine GUI. Wiederverwendbar – wird vom
Impuls-Experiment im Hohmann-Modul genutzt.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trajecto.core import orbital_mechanics as om
from trajecto.core import state_vectors as sv

#: Bahnklassen (didaktische Bezeichnungen).
CLASS_CIRCLE = "Kreisbahn"
CLASS_ELLIPSE = "gebundene Ellipse"
CLASS_PARABOLA = "parabolischer Grenzfall"
CLASS_HYPERBOLA = "hyperbolische Fluchtbahn"

#: Numerische Toleranzen fuer die Klassifikation.
_CIRCLE_TOL = 1e-8
_ECC_TOL = 1e-6


@dataclass(frozen=True)
class OrbitElements:
    """Bahngroessen und Klassifikation aus einem Zustandsvektor (SI).

    Fuer nicht gebundene Bahnen (Parabel/Hyperbel) sind ``semi_major_axis``,
    ``apoapsis_radius`` und ``period`` ``None``.
    """

    mu: float
    specific_energy: float
    angular_momentum: np.ndarray
    angular_momentum_mag: float
    ecc_vector: np.ndarray
    eccentricity: float
    semi_latus_rectum: float
    periapsis_radius: float
    classification: str
    semi_major_axis: float | None
    apoapsis_radius: float | None
    period: float | None
    arg_periapsis: float  # Winkel des Exzentrizitaetsvektors in der Ebene [rad]

    @property
    def is_bound(self) -> bool:
        return self.classification in (CLASS_CIRCLE, CLASS_ELLIPSE)


def orbital_elements(
    mu: float, position: np.ndarray, velocity: np.ndarray
) -> OrbitElements:
    """Bestimme Bahngroessen und Klasse aus ``mu``, Position und Geschwindigkeit.

    Raises:
        ValueError: bei ``mu <= 0`` oder verschwindender Position.
    """
    if mu <= 0.0:
        raise ValueError("Gravitationsparameter mu muss positiv sein.")
    state = sv.OrbitState(
        mu=mu,
        position=np.asarray(position, dtype=float),
        velocity=np.asarray(velocity, dtype=float),
    )
    r = sv.radius(state)
    if r <= 0.0:
        raise ValueError("Position muss vom Ursprung verschieden sein.")

    eps = sv.specific_total_energy(state)
    h_vec = sv.specific_angular_momentum(state)
    h_mag = float(np.linalg.norm(h_vec))
    e_vec = sv.eccentricity_vector(state)
    e = float(np.linalg.norm(e_vec))
    p = h_mag * h_mag / mu  # Semi-latus rectum
    periapsis = p / (1.0 + e)

    if e < _CIRCLE_TOL:
        classification = CLASS_CIRCLE
    elif e < 1.0 - _ECC_TOL:
        classification = CLASS_ELLIPSE
    elif e <= 1.0 + _ECC_TOL:
        classification = CLASS_PARABOLA
    else:
        classification = CLASS_HYPERBOLA

    bound = classification in (CLASS_CIRCLE, CLASS_ELLIPSE)
    if bound:
        semi_major: float | None = -mu / (2.0 * eps)
        apoapsis: float | None = p / (1.0 - e)
        period: float | None = 2.0 * math.pi * math.sqrt(semi_major**3 / mu)
    else:
        semi_major = None
        apoapsis = None
        period = None

    arg_peri = math.atan2(e_vec[1], e_vec[0]) if e >= _CIRCLE_TOL else 0.0

    return OrbitElements(
        mu=mu,
        specific_energy=eps,
        angular_momentum=h_vec,
        angular_momentum_mag=h_mag,
        ecc_vector=e_vec,
        eccentricity=e,
        semi_latus_rectum=p,
        periapsis_radius=periapsis,
        classification=classification,
        semi_major_axis=semi_major,
        apoapsis_radius=apoapsis,
        period=period,
        arg_periapsis=arg_peri,
    )


def propagate(
    mu: float, position: np.ndarray, velocity: np.ndarray, tau: float
) -> sv.OrbitState:
    """Zustand nach der normierten Bahnposition ``tau`` aus einem Zustandsvektor.

    Die Bahn wird vollstaendig aus dem konkreten Zustandsvektor ``(position,
    velocity)`` bestimmt – **beliebig orientiert**, keine feste +x-Annahme.
    ``tau = 0`` liefert exakt den uebergebenen Zustand. Bewegungsrichtung
    (prograd/retrograd in der Ebene) wird aus dem Drehimpuls uebernommen.

    * Gebundene Bahn: ``tau`` laeuft zeitgetreu (mittlere Anomalie) ueber eine
      volle Umlaufperiode; ``tau = 1`` fuehrt zurueck zum Startpunkt.
    * Ungebundene Bahn (Parabel/Hyperbel, Fluchtbahn): ``tau`` laeuft entlang
      des offenen Bahnbogens vom Startpunkt in Bewegungsrichtung bis nahe an die
      Asymptote (keine Periode; die Animation wiederholt den Bogen).
    """
    el = orbital_elements(mu, position, velocity)
    e = el.eccentricity
    pos = np.asarray(position, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    r0 = float(math.hypot(pos[0], pos[1]))
    s = 1.0 if el.angular_momentum[2] >= 0.0 else -1.0  # Bewegungsrichtung

    if e < _CIRCLE_TOL:
        # Kreisbahn: keine Apsidenlinie -> Bezug auf die Startposition.
        omega = math.atan2(pos[1], pos[0])
        nu0 = 0.0
    else:
        ev = el.ecc_vector
        omega = math.atan2(ev[1], ev[0])  # Richtung der Periapsis
        cos_nu0 = (ev[0] * pos[0] + ev[1] * pos[1]) / (e * r0)
        nu0 = math.acos(max(-1.0, min(1.0, cos_nu0)))
        if (pos[0] * vel[0] + pos[1] * vel[1]) < 0.0:
            nu0 = -nu0  # vor der Periapsis (Annaeherung)

    if el.is_bound:
        # Mittlere Anomalie am Start; tau erhoeht die Zeit -> M waechst.
        e0 = 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(0.5 * nu0),
            math.sqrt(1.0 + e) * math.cos(0.5 * nu0),
        )
        m0 = e0 - e * math.sin(e0)
        nu = om.true_anomaly_from_mean(m0 + 2.0 * math.pi * tau, e)
    else:
        # Ungebunden: tau laeuft geometrisch entlang des Bogens bis nahe an die
        # Asymptote (radiale Begrenzung, damit der Plot endlich bleibt).
        nu_inf = math.acos(min(1.0, max(-1.0, -1.0 / e)))
        nu_max = nu0 + 0.7 * (nu_inf - nu0)
        nu = nu0 + tau * (nu_max - nu0)

    p = el.semi_latus_rectum  # = h^2/mu, fuer alle Kegelschnitte definiert
    r = p / (1.0 + e * math.cos(nu))
    alpha = omega + s * nu  # Lagewinkel im Inertialsystem
    h = el.angular_momentum_mag
    v_r = (mu / h) * e * math.sin(nu)    # radiale Komponente
    v_t = h / r                          # transversale Komponente
    ca, sa = math.cos(alpha), math.sin(alpha)
    r_hat = np.array([ca, sa, 0.0])
    t_hat = s * np.array([-sa, ca, 0.0])  # Bewegungsrichtung
    return sv.OrbitState(mu=mu, position=r * r_hat, velocity=v_r * r_hat + v_t * t_hat)


#: Begrenzung der (hyperbolischen) mittleren Anomalie, damit ungebundene
#: Propagation ueber sehr lange Zeit numerisch stabil bleibt.
_HYPERBOLIC_N_LIMIT = 1.0e6


def propagate_time(
    mu: float, position: np.ndarray, velocity: np.ndarray, dt: float
) -> sv.OrbitState:
    """Zustand nach der **Zeit** ``dt`` (Sekunden) aus einem Zustandsvektor.

    Zeitgetreue Kepler-Propagation – gebunden (elliptisch) **und** ungebunden
    (hyperbolisch). ``dt = 0`` liefert exakt den uebergebenen Zustand. Geeignet
    fuer eine durchlaufende Missionsuhr.
    """
    el = orbital_elements(mu, position, velocity)
    e = el.eccentricity
    pos = np.asarray(position, dtype=float)
    vel = np.asarray(velocity, dtype=float)
    r0 = float(math.hypot(pos[0], pos[1]))
    s = 1.0 if el.angular_momentum[2] >= 0.0 else -1.0
    p = el.semi_latus_rectum
    h = el.angular_momentum_mag

    if e < _CIRCLE_TOL:
        omega = math.atan2(pos[1], pos[0])
        nu0 = 0.0
    else:
        ev = el.ecc_vector
        omega = math.atan2(ev[1], ev[0])
        cos_nu0 = (ev[0] * pos[0] + ev[1] * pos[1]) / (e * r0)
        nu0 = math.acos(max(-1.0, min(1.0, cos_nu0)))
        if (pos[0] * vel[0] + pos[1] * vel[1]) < 0.0:
            nu0 = -nu0

    if el.is_bound:
        a = el.semi_major_axis
        n = math.sqrt(mu / a**3)  # mittlere Bewegung
        e0 = 2.0 * math.atan2(
            math.sqrt(1.0 - e) * math.sin(0.5 * nu0),
            math.sqrt(1.0 + e) * math.cos(0.5 * nu0),
        )
        m0 = e0 - e * math.sin(e0)
        nu = om.true_anomaly_from_mean(m0 + n * dt, e)
    else:
        # Hyperbolische Propagation ueber die hyperbolische mittlere Anomalie.
        a_abs = p / (e * e - 1.0)
        n = math.sqrt(mu / a_abs**3)
        h0 = 2.0 * math.atanh(
            math.sqrt((e - 1.0) / (e + 1.0)) * math.tan(0.5 * nu0)
        )
        n0 = e * math.sinh(h0) - h0
        mean_h = max(-_HYPERBOLIC_N_LIMIT, min(_HYPERBOLIC_N_LIMIT, n0 + n * dt))
        hh = om.solve_hyperbolic_anomaly(mean_h, e)
        nu = 2.0 * math.atan2(
            math.sqrt(e + 1.0) * math.sinh(0.5 * hh),
            math.sqrt(e - 1.0) * math.cosh(0.5 * hh),
        )

    r = p / (1.0 + e * math.cos(nu))
    alpha = omega + s * nu
    v_r = (mu / h) * e * math.sin(nu)
    v_t = h / r
    ca, sa = math.cos(alpha), math.sin(alpha)
    r_hat = np.array([ca, sa, 0.0])
    t_hat = s * np.array([-sa, ca, 0.0])
    return sv.OrbitState(mu=mu, position=r * r_hat, velocity=v_r * r_hat + v_t * t_hat)

"""Gemeinsames Interface und Datentypen fuer didaktische Module.

Ein Modul ist ein in sich geschlossenes Lernthema. Es beschreibt seine
**Eingabeparameter**, fuehrt eine **Berechnung** durch, liefert **Ergebnisse**,
zeichnet eine **Visualisierung** und liefert eine **Erklaerung**.

Die GUI ist generisch: Sie kennt nur diese Typen und baut sich aus der
Modulbeschreibung selbst auf. Neue Module erfordern keine GUI-Aenderung.

Konvention: Parameterwerte und Ergebnisse werden gegenueber Berechnungslogik
immer in **SI-Einheiten** gefuehrt. Anzeige-Einheiten dienen nur der GUI.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from matplotlib.figure import Figure

#: Optionale Sichtbarkeitsbedingung: erhaelt die aktuellen Parameterwerte
#: (``name -> Wert``) und liefert ``True``, wenn das Element sichtbar sein soll.
VisibleWhen = Callable[[dict[str, Any]], bool]


@dataclass(frozen=True)
class FloatParameter:
    """Numerischer Eingabeparameter.

    ``default_si``, ``minimum_si`` und ``maximum_si`` sind in SI. ``display_unit``
    bestimmt nur, in welcher Einheit die GUI den Wert anzeigt/entgegennimmt.
    """

    name: str
    label: str
    default_si: float
    display_unit: str
    minimum_si: float = 0.0
    maximum_si: float | None = None
    #: Wenn True, kann die GUI diesen Parameter ueber min..max animieren
    #: (z. B. die normierte Zeitposition tau). Erfordert ein endliches maximum_si.
    animatable: bool = False
    #: Optionale Sichtbarkeitsbedingung (siehe :data:`VisibleWhen`).
    visible_when: VisibleWhen | None = None
    #: Optionaler Tooltip mit der ausfuehrlichen Bedeutung.
    tooltip: str = ""
    #: Optionale Gruppenueberschrift fuer die GUI (z. B. "Setup").
    group: str = ""
    kind: str = "float"


@dataclass(frozen=True)
class ChoiceParameter:
    """Auswahlparameter aus festen Optionen (z. B. Zentralkoerper)."""

    name: str
    label: str
    choices: tuple[str, ...]
    default: str
    #: Optionale Sichtbarkeitsbedingung (siehe :data:`VisibleWhen`).
    visible_when: VisibleWhen | None = None
    #: Optionaler Tooltip mit der ausfuehrlichen Bedeutung.
    tooltip: str = ""
    #: Optionale Gruppenueberschrift fuer die GUI (z. B. "Setup").
    group: str = ""
    kind: str = "choice"


Parameter = FloatParameter | ChoiceParameter


@dataclass(frozen=True)
class ResultItem:
    """Ein einzelnes Ergebnis: SI-Wert plus gewuenschte Anzeige-Einheit."""

    label: str
    value_si: float
    display_unit: str


@dataclass(frozen=True)
class ResultSection:
    """Gruppenueberschrift in der Ergebnisliste (eine nicht-numerische Zeile)."""

    title: str


#: Eintrag in ``ModuleResult.items``: Kennzahl oder Gruppenueberschrift.
ResultEntry = ResultItem | ResultSection


@dataclass
class ModuleResult:
    """Ergebnis einer Modulberechnung.

    ``items`` sind die anzuzeigenden Kennzahlen (optional gegliedert durch
    ``ResultSection``-Ueberschriften), ``notes`` enthaelt textuelle Hinweise/
    Warnungen und ``data`` beliebige Zusatzdaten fuer die Visualisierung.
    """

    items: list[ResultEntry] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    #: Optionale, prominent angezeigte Statuszeile (z. B. aktuelle Phase).
    status: str | None = None


@dataclass(frozen=True)
class PlotOption:
    """Ein- und ausblendbares Plot-Element, das ein Modul deklariert.

    Die GUI erzeugt daraus generisch eine Checkbox; die ``plot``-Funktion des
    Moduls wertet den Zustand ueber :meth:`Module.is_option_enabled` aus.
    """

    name: str
    label: str
    default: bool = True
    #: Optionale Sichtbarkeitsbedingung anhand der Parameterwerte
    #: (siehe :data:`VisibleWhen`); die Checkbox wird sonst ausgeblendet.
    visible_when: VisibleWhen | None = None


@dataclass(frozen=True)
class Action:
    """Ein vom Modul angebotener Knopf (z. B. ``Δv₁ ausfuehren``).

    Die GUI erzeugt je Action einen Button. Aktiviert-Zustand und Wirkung
    bestimmt das Modul ueber :meth:`Module.is_action_enabled` /
    :meth:`Module.perform_action` – die GUI bleibt generisch.
    """

    name: str
    label: str
    #: Optionale Sichtbarkeitsbedingung anhand der Parameterwerte.
    visible_when: VisibleWhen | None = None
    #: Optionaler Tooltip-Text fuer den Button.
    tooltip: str = ""


@dataclass
class ActionResult:
    """Rueckgabe von :meth:`Module.perform_action` an die GUI.

    ``reset_tau`` bittet die GUI, die Zeitposition auf 0 zu setzen;
    ``set_params_si`` setzt einzelne Parameter (Werte in SI).
    """

    reset_tau: bool = False
    set_params_si: dict[str, float] = field(default_factory=dict)
    #: Wenn True, soll die GUI eine laufende Animation stoppen (z. B. bei Reset).
    #: Standard False: eine laufende Animation laeuft nach der Aktion weiter.
    stop_animation: bool = False


class Module(ABC):
    """Basisklasse fuer alle didaktischen Module."""

    #: stabile, technische ID (z. B. fuer Persistenz/Tests)
    id: str = ""
    #: Anzeigetitel in der Navigation
    title: str = ""
    #: einzeiliger Untertitel
    subtitle: str = ""

    @abstractmethod
    def parameters(self) -> list[Parameter]:
        """Liste der Eingabeparameter dieses Moduls."""

    @abstractmethod
    def compute(self, values: dict[str, Any]) -> ModuleResult:
        """Berechne Ergebnisse aus ``values`` (Floats in SI, Choices als str)."""

    @abstractmethod
    def plot(
        self,
        figure: Figure,
        values: dict[str, Any],
        result: ModuleResult,
        options: dict[str, bool] | None = None,
    ) -> None:
        """Zeichne die Visualisierung; ``options`` schaltet Plot-Elemente.

        Ist ``options`` ``None`` oder fehlt ein Schluessel, gilt der jeweilige
        Default aus :meth:`plot_options`.
        """

    @abstractmethod
    def explanation(self) -> str:
        """Didaktischer Erklaertext (einfaches Markdown/Plaintext)."""

    def plot_options(self) -> list[PlotOption]:
        """Optionale, ein-/ausblendbare Plot-Elemente (Standard: keine)."""
        return []

    def is_option_enabled(
        self, options: dict[str, bool] | None, name: str
    ) -> bool:
        """Zustand einer Plot-Option; faellt auf den deklarierten Default zurueck."""
        if options is not None and name in options:
            return bool(options[name])
        for option in self.plot_options():
            if option.name == name:
                return option.default
        return False

    # -- Optionale Aktionen (manoevergetriebene Module) -----------------------

    def actions(self) -> list["Action"]:
        """Vom Modul angebotene Aktions-Buttons (Standard: keine)."""
        return []

    def is_action_enabled(self, name: str, values: dict[str, Any]) -> bool:
        """Ob die Aktion aktuell ausgefuehrt werden kann (Standard: ja)."""
        return True

    def perform_action(self, name: str, values: dict[str, Any]) -> "ActionResult":
        """Fuehre die Aktion aus und aendere ggf. den internen Modulzustand."""
        return ActionResult()

    def on_activated(self) -> None:
        """Wird aufgerufen, wenn das Modul (erneut) ausgewaehlt wird.

        Module mit internem Zustand koennen sich hier zuruecksetzen (Standard:
        nichts tun).
        """

    # -- Optionale durchlaufende Missionsuhr ----------------------------------

    def is_clock_driven(self) -> bool:
        """True, wenn die Animation eine kontinuierliche Missionsuhr treibt."""
        return False

    def advance_clock(self, orbit_fraction: float) -> None:
        """Missionsuhr um den Bruchteil ``orbit_fraction`` einer Bahn fortschreiten.

        ``orbit_fraction`` ist der Anteil einer (charakteristischen) Umlaufzeit pro
        Animations-Tick; das Modul rechnet daraus die verstrichene Zeit (Standard:
        nichts tun).
        """

    def reset_clock(self) -> None:
        """Missionsuhr (und ggf. Bahn/Log) zuruecksetzen (Standard: nichts tun)."""

    # -- Optionales Log (z. B. Manoever-Log) ----------------------------------

    def has_log(self) -> bool:
        """True, wenn das Modul ein eigenes Log-Fenster (Tab) anbietet."""
        return False

    def log_entries(self) -> list[str]:
        """Zeilen fuer das Log-Fenster (neueste ggf. zuletzt; Standard: leer)."""
        return []

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
from dataclasses import dataclass, field
from typing import Any

from matplotlib.figure import Figure


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
    kind: str = "float"


@dataclass(frozen=True)
class ChoiceParameter:
    """Auswahlparameter aus festen Optionen (z. B. Zentralkoerper)."""

    name: str
    label: str
    choices: tuple[str, ...]
    default: str
    kind: str = "choice"


Parameter = FloatParameter | ChoiceParameter


@dataclass(frozen=True)
class ResultItem:
    """Ein einzelnes Ergebnis: SI-Wert plus gewuenschte Anzeige-Einheit."""

    label: str
    value_si: float
    display_unit: str


@dataclass
class ModuleResult:
    """Ergebnis einer Modulberechnung.

    ``items`` sind die anzuzeigenden Kennzahlen, ``notes`` enthaelt textuelle
    Hinweise/Warnungen (z. B. Transferart oder Plausibilitaetswarnungen) und
    ``data`` beliebige Zusatzdaten (z. B. Bahnpunkte) fuer die Visualisierung.
    """

    items: list[ResultItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)


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
    def plot(self, figure: Figure, values: dict[str, Any], result: ModuleResult) -> None:
        """Zeichne die Visualisierung in die gegebene Matplotlib-Figure."""

    @abstractmethod
    def explanation(self) -> str:
        """Didaktischer Erklaertext (einfaches Markdown/Plaintext)."""

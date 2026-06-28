# Trajecto

**Trajecto** ist eine interaktive Desktop-Lernsoftware, die Studierenden die
Grundlagen und Anwendungen der **Orbitalmechanik** anschaulich vermittelt.

Trajecto ist bewusst als klassische Desktop-Anwendung (PySide6) konzipiert –
keine Web-, Browser- oder Server-Lösung – und kann später als eigenständige
Windows-`.exe` (PyInstaller / Nuitka) verteilt werden.

## Ziel

Orbitalmechanik soll nicht nur als Formelsammlung, sondern interaktiv erfahrbar
werden: Parameter verändern, Ergebnis sehen, Bahn visualisieren, Erklärung lesen.
Das Projekt ist von Anfang an **modular** aufgebaut, sodass didaktische
Themenmodule schrittweise ergänzt werden können (Keplersche Gesetze,
Hohmann-Transfer, Gravity Assist, numerische Propagation …).

## Architektur (Kurzüberblick)

```
src/trajecto/
├── core/       physikalische & mathematische Grundlagen (SI-intern)
├── modules/    didaktische Themenmodule (Parameter, Rechnung, Plot, Erklärung)
├── ui/         PySide6-Desktopoberfläche (Navigation, Parameter, Plot, Ergebnis, Erklärung)
└── resources/  Icons, Bilder, Stylesheets (paketierbar)
```

Eine ausführliche Beschreibung steht in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

**Wichtige Konvention:** Intern wird **konsequent in SI-Einheiten** gerechnet
(m, s, m/s, rad, m³/s²). Erst in der GUI werden nutzerfreundliche Einheiten
(km, km/s, min, h, d, °) angezeigt.

## Installation (Entwicklung)

Voraussetzung: Python ≥ 3.10.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -e ".[dev]"
```

## Start der Anwendung

```bash
trajecto
```

oder direkt über das Modul:

```bash
python -m trajecto
```

## Tests

```bash
pytest
```

## Status

Erstes Fundament: Rechenkern (Kreisbahn, Umlaufzeit, Fluchtgeschwindigkeit),
Referenzmodul `circular_orbit` und eine minimale, erweiterbare GUI.
Fachmodule wie Hohmann-Transfer folgen in späteren Schritten.

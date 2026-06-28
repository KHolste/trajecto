# Trajecto – Architektur

Dieses Dokument beschreibt die geplante Modulstruktur von Trajecto sowie die
Konventionen, die die Erweiterbarkeit sicherstellen.

## Leitprinzipien

1. **Trennung der Verantwortlichkeiten.**
   Physik/Mathematik (`core`), didaktische Module (`modules`) und Oberfläche
   (`ui`) sind strikt getrennt. Die GUI enthält **keine** Physikformeln, der
   Rechenkern kennt **kein** Qt.
2. **SI-intern, nutzerfreundlich extern.**
   Alle Berechnungen und Datenmodelle arbeiten in SI-Einheiten. Einheiten-
   Umrechnung (km, km/s, min, h, d, °) passiert ausschließlich an der
   GUI-Grenze über `core/units.py`.
3. **Module sind Plugins.**
   Jedes Themenmodul implementiert dasselbe Interface (`modules/base.py`) und
   wird in der `registry` registriert. Die GUI ist generisch und baut sich aus
   der Modulbeschreibung selbst auf – neue Module erfordern **keine**
   GUI-Änderungen.
4. **Paketierbarkeit.**
   Ressourcen werden über `core/paths.py` adressiert, damit PyInstaller/Nuitka
   (`sys._MEIPASS`) später transparent funktionieren.

## Verzeichnisstruktur

```
Trajecto/
├── pyproject.toml          Projekt-, Build- und Test-Konfiguration (src-Layout)
├── README.md
├── docs/
│   └── ARCHITECTURE.md
├── src/
│   └── trajecto/
│       ├── __init__.py
│       ├── __main__.py     ermöglicht `python -m trajecto`
│       ├── app.py          Einstiegspunkt (QApplication, Hauptfenster)
│       ├── core/
│       │   ├── constants.py        Naturkonstanten (G, ...)
│       │   ├── units.py            SI <-> Anzeige-Einheiten
│       │   ├── paths.py            paketierungssichere Ressourcenpfade
│       │   ├── bodies.py           Himmelskörper-Datenmodell + Katalog (Erde …)
│       │   └── orbital_mechanics.py  Grundfunktionen der Bahnmechanik
│       ├── modules/
│       │   ├── base.py             Modul-Interface + Parameter-/Ergebnis-Typen
│       │   ├── registry.py         verfügbare Module
│       │   └── circular_orbit.py   Referenzmodul „Kreisbahn“
│       ├── ui/
│       │   ├── main_window.py      Fenster-Layout & Verdrahtung
│       │   └── widgets/
│       │       ├── parameter_panel.py    baut Eingaben aus Modulparametern
│       │       ├── plot_panel.py         Matplotlib-Canvas
│       │       ├── result_panel.py       Ergebnistabelle
│       │       └── explanation_panel.py  didaktische Erklärung
│       └── resources/      Icons, Stylesheets, Erklärgrafiken (paketiert)
└── tests/                  pytest-Tests für Rechenkern und Module
```

## core

- **constants.py** – Physikalische Konstanten in SI (z. B. Gravitationskonstante `G`).
- **units.py** – Tabelle von Anzeige-Einheiten mit Umrechnungsfaktor nach SI sowie
  `to_si()` / `from_si()`. Single Source of Truth für jede Einheitenumrechnung.
- **paths.py** – `resource_path()` für paketierungssichere Pfade.
- **bodies.py** – `CelestialBody` (Name, μ = G·M, mittlerer Radius) und ein
  Katalog vordefinierter Körper (aktuell: Erde). μ wird bevorzugt direkt
  gespeichert, da es genauer bekannt ist als G·M einzeln.
- **orbital_mechanics.py** – reine Funktionen ohne Zustand:
  `circular_orbit_velocity`, `circular_orbit_period`, `escape_velocity`.
  Eingaben/Ausgaben in SI; ungültige Eingaben lösen `ValueError` aus.

## modules

Jedes Modul kapselt ein didaktisches Thema und liefert fünf Dinge:

| Aspekt          | Mechanismus                                   |
|-----------------|-----------------------------------------------|
| Eingabeparameter| `parameters()` → Liste von Parameter-Specs    |
| Berechnung      | `compute(values)` → `ModuleResult`            |
| Ergebnisse      | `ModuleResult.items` (Label, SI-Wert, Einheit)|
| Visualisierung  | `plot(figure, values, result)`                |
| Erklärung       | `explanation()` → Text                        |

`base.py` definiert dazu:

- `FloatParameter` – numerischer Eingabewert mit Default und Anzeige-Einheit (SI-intern).
- `ChoiceParameter` – Auswahl aus festen Optionen (z. B. Zentralkörper).
- `ResultItem` / `ModuleResult` – Ergebnisdaten in SI plus freie `data` für den Plot.
- `Module` (ABC) – das gemeinsame Interface.

Neue Module werden lediglich in `registry.available_modules()` ergänzt.

## ui

Die Oberfläche ist **datengetrieben**: Sie liest das aktive Modul, baut den
Parameterbereich aus dessen `parameters()` auf, ruft bei Änderung `compute()`
und `plot()` auf und stellt `items` sowie `explanation()` dar. Layout:

```
┌────────────┬───────────────────────────┬──────────────────┐
│ Module     │                           │ Parameter        │
│ (Liste)    │      Plot (Matplotlib)    │ Ergebnisse       │
│            │                           │ Erklärung        │
└────────────┴───────────────────────────┴──────────────────┘
```

## Geplante Themenmodule (Roadmap)

Keplersche Gesetze · Kreis-/Ellipsenbahnen · Hohmann-Transfer ·
bielliptischer Transfer · Ebenenwechsel · Fluchtbahnen · Rendezvous/Phasing ·
Gravity Assist · spezielle Bahnprobleme · einfache numerische Propagation.

Diese werden inkrementell als eigenständige Module ergänzt, ohne den Rechenkern
oder die GUI-Struktur umbauen zu müssen.

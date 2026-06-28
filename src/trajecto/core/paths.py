"""Paketierungssichere Ressourcenpfade.

Damit Ressourcen sowohl im Entwicklungsbetrieb als auch in einer mit
PyInstaller/Nuitka erzeugten ``.exe`` gefunden werden, darf nie ein fixer
relativer Pfad verwendet werden. PyInstaller entpackt Daten zur Laufzeit in
ein temporaeres Verzeichnis und legt dessen Pfad in ``sys._MEIPASS`` ab.
"""

from __future__ import annotations

import sys
from pathlib import Path

#: Wurzel des installierten Pakets ``trajecto`` (Verzeichnis dieser Datei).
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _base_dir() -> Path:
    """Basisverzeichnis fuer Ressourcen – beruecksichtigt PyInstaller."""
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        return Path(meipass) / "trajecto"
    return _PACKAGE_ROOT


def resource_path(*parts: str) -> Path:
    """Pfad zu einer Ressource unterhalb von ``trajecto/resources``.

    Beispiel::

        resource_path("icons", "app.png")
    """
    return _base_dir() / "resources" / Path(*parts)

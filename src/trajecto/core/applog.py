"""Einfaches Datei-Logging fuer Trajecto (Diagnose / Crash-Log).

Schreibt Ereignisse und unbehandelte Ausnahmen in eine Logdatei. Standardort:
``<Repo>/logs/trajecto.log`` (Entwicklung) bzw. ``~/.trajecto/trajecto.log``,
falls das Repo-Verzeichnis nicht beschreibbar ist (z. B. paketiert).

Verwendung::

    from trajecto.core import applog
    applog.logger.info("…")
    applog.logger.exception("Fehler bei X")  # mit Traceback
"""

from __future__ import annotations

import logging
from pathlib import Path


def log_file() -> Path:
    """Pfad der Logdatei (Verzeichnis wird bei Bedarf angelegt)."""
    # src/trajecto/core/applog.py -> parents[3] ist die Repo-Wurzel.
    repo_root = Path(__file__).resolve().parents[3]
    candidates = (repo_root / "logs", Path.home() / ".trajecto")
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            return directory / "trajecto.log"
        except OSError:
            continue
    # Letzter Ausweg: aktuelles Arbeitsverzeichnis.
    return Path("trajecto.log")


def _build_logger() -> logging.Logger:
    log = logging.getLogger("trajecto")
    if log.handlers:  # nur einmal konfigurieren
        return log
    log.setLevel(logging.DEBUG)
    log.propagate = False
    try:
        handler: logging.Handler = logging.FileHandler(log_file(), encoding="utf-8")
    except OSError:
        handler = logging.NullHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    log.addHandler(handler)
    return log


#: Der gemeinsame Anwendungs-Logger.
logger = _build_logger()

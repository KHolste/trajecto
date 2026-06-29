"""Registrierung der verfuegbaren Module.

Neue Themenmodule werden ausschliesslich hier ergaenzt – die GUI baut ihre
Navigation aus dieser Liste auf.
"""

from __future__ import annotations

from trajecto.modules.base import Module
from trajecto.modules.circular_orbit import CircularOrbitModule
from trajecto.modules.hohmann_transfer import HohmannTransferModule
from trajecto.modules.kepler_laws import KeplerLawsModule
from trajecto.modules.maneuver_sandbox import ManeuverSandboxModule


def available_modules() -> list[Module]:
    """Liefere frische Instanzen aller verfuegbaren Module."""
    return [
        CircularOrbitModule(),
        KeplerLawsModule(),
        HohmannTransferModule(),
        ManeuverSandboxModule(),
    ]

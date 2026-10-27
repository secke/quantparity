"""quantparity: an independent reference engine to validate pricers and risk models.

All functions take and return plain Python objects. Structured inputs (trades, markets,
scenarios) follow the JSON schema documented in the README.
"""

from ._core import (  # noqa: F401
    Market,
    __version__,
    black76,
    bond,
    bsm,
    calibration,
    christoffersen,
    compare,
    crr,
    discrimination,
    implied_vol,
    kupiec,
    pla_test,
    pnl_attribution,
    position_values,
    price,
    psi,
    revalue,
    sensitivities,
    stress,
    traffic_light,
    var_backtest,
    var_es,
)

__all__ = [n for n in dir() if not n.startswith("_")] + ["__version__"]

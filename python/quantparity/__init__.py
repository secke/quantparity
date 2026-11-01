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
    es_backtest,
    implied_vol,
    kupiec,
    norm_inv,
    pla_test,
    pnl_attribution,
    position_values,
    price,
    psi,
    revalue,
    sensitivities,
    stress,
    t_cdf,
    t_inv,
    traffic_light,
    var_backtest,
    var_es,
    var_forecasts,
)

__all__ = [n for n in dir() if not n.startswith("_")] + ["__version__"]

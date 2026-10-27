"""Deterministic synthetic 300-day history of risk-factor moves.
Correlations: equities fall as volatilities rise, €STR and Euribor curves move together, OAT tracks rates.
A crisis week (days 200 to 206) fattens the tail of the distribution.
Units: SPOT relative (0.01 = +1%), VOL absolute (0.01 = +1 vol point), CURVE and YIELD in basis points."""
import math, random

rng = random.Random(2026)
cols = ["day", "SPOT:SX5E", "VOL:SX5E", "SPOT:CAC40", "VOL:CAC40", "SPOT:BRENT", "VOL:BRENT",
        "CURVE:EUR-ESTR", "CURVE:EUR-6M", "YIELD:OAT"]
rows = []
for d in range(1, 301):
    crisis = 200 <= d <= 206
    k = 3.0 if crisis else 1.0
    t = rng.gauss(0, 1) / math.sqrt(rng.gammavariate(2.5, 1 / 2.5))  # fat tails
    eq = (-0.012 if crisis else 0.0) + 0.012 * k * t
    cac = 0.9 * eq + 0.006 * k * rng.gauss(0, 1)
    rates = 4.5 * k * rng.gauss(0, 1) - (8 if crisis else 0)
    rows.append([
        d,
        eq,
        -0.6 * eq + 0.004 * k * rng.gauss(0, 1),
        cac,
        -0.55 * cac + 0.004 * k * rng.gauss(0, 1),
        0.02 * k * rng.gauss(0, 1),
        0.006 * k * rng.gauss(0, 1),
        rates,
        0.95 * rates + 1.2 * rng.gauss(0, 1),
        0.8 * rates + 2.5 * k * rng.gauss(0, 1) + (6 if crisis else 0),
    ])
with open("data/factor_history.csv", "w") as f:
    f.write(",".join(cols) + "\n")
    for r in rows:
        f.write(",".join([str(r[0])] + [f"{x:.8f}" for x in r[1:]]) + "\n")

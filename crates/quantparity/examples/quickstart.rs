//! Pricing, a multi-curve swap and historical VaR on the bundled EUR example data.

use quantparity::{bsm, crr, curves::Market, risk, value_in, OptionType, Trade};
use serde_json::{json, Value};

fn main() -> Result<(), String> {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/../../examples/data/");
    let read = |f: &str| -> Result<Value, String> {
        let s = std::fs::read_to_string(format!("{data}{f}")).map_err(|e| e.to_string())?;
        serde_json::from_str(&s).map_err(|e| e.to_string())
    };

    let g = bsm(OptionType::Call, 42.0, 40.0, 0.10, 0.0, 0.20, 0.5)?;
    println!(
        "BSM call: price {:.4}, delta {:.4}, vega {:.4}",
        g.price, g.delta, g.vega
    );
    let am = crr(OptionType::Put, 50.0, 50.0, 0.10, 0.0, 0.40, 5.0 / 12.0, 500, true)?;
    println!("American put (CRR, 500 steps): {am:.4}");

    let market_json = read("eur_market.json")?;
    let market = Market::from_value(&market_json)?;
    let swap: Trade = serde_json::from_value(json!({
        "id": "IRS-10Y", "product": "vanilla_swap", "notional": 10_000_000.0,
        "fixed_rate": 0.0274, "tenor": "10Y", "pay_fixed": true, "forward_curve": "EUR-6M"
    }))
    .map_err(|e| e.to_string())?;
    let v = value_in(&swap, &market)?;
    println!(
        "10Y payer swap: NPV {:.2} EUR, par rate {:.5}, DV01 {:.2}",
        v.price, v.measures["par_rate"], v.measures["dv01"]
    );

    let trades: Vec<Trade> =
        serde_json::from_value(read("portfolio.json")?["trades"].clone()).map_err(|e| e.to_string())?;
    let csv = std::fs::read_to_string(format!("{data}factor_history.csv")).map_err(|e| e.to_string())?;
    let mut lines = csv.lines();
    let header: Vec<&str> = lines.next().ok_or("empty history")?.split(',').skip(1).collect();
    let mut table = serde_json::Map::new();
    for l in lines {
        for (h, x) in header.iter().zip(l.split(',').skip(1)) {
            let col = table.entry(h.to_string()).or_insert_with(|| json!([]));
            col.as_array_mut()
                .unwrap()
                .push(json!(x.parse::<f64>().map_err(|e| e.to_string())?));
        }
    }
    let scenarios = risk::scenarios_from_table(&Value::Object(table), None)?;
    let (_, pnl) = risk::revalue(&trades, &market_json, &scenarios)?;
    let pnl: Vec<f64> = pnl.iter().map(|p| p.pnl).collect();
    let r = risk::var_es(&pnl, 0.99, 0.975).ok_or("no P&L")?;
    println!(
        "Portfolio, {} scenarios: VaR 99% {:.0}, ES 97.5% {:.0}",
        pnl.len(),
        r.var,
        r.es
    );
    Ok(())
}

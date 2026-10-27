//! Yield curves: single-curve (time in years) and dated multi-curve (OIS discounting, Ibor projection).

pub mod multicurve;
pub mod simple;

pub use multicurve::{build_all, ibor_swap_legs, DatedCurve, DatedCurveSpec};
pub use simple::{swap_npv, Curve, CurveSpec, Market};

//! Dates, holiday calendars, business-day conventions, day counts and schedules.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Calendar date (days since 1970-01-01).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Date(i64);

fn days_from_civil(y: i64, m: u32, d: u32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = (m as i64 + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

pub fn is_leap(y: i64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

pub fn days_in_month(y: i64, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        _ if is_leap(y) => 29,
        _ => 28,
    }
}

impl Date {
    pub fn from_ymd(y: i64, m: u32, d: u32) -> Result<Self, String> {
        if !(1..=12).contains(&m) || d == 0 || d > days_in_month(y, m) {
            return Err(format!("invalid date {y:04}-{m:02}-{d:02}"));
        }
        Ok(Date(days_from_civil(y, m, d)))
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let p: Vec<&str> = s.trim().split('-').collect();
        let num = |x: &str| x.parse::<i64>().map_err(|_| format!("invalid date `{s}` (YYYY-MM-DD)"));
        if p.len() != 3 {
            return Err(format!("invalid date `{s}` (YYYY-MM-DD)"));
        }
        Self::from_ymd(num(p[0])?, num(p[1])? as u32, num(p[2])? as u32)
    }

    pub fn ymd(self) -> (i64, u32, u32) {
        let z = self.0 + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z - era * 146_097;
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        (yoe + era * 400 + i64::from(m <= 2), m, d)
    }

    /// 0 = Monday … 6 = Sunday.
    pub fn weekday(self) -> u32 {
        (self.0 + 3).rem_euclid(7) as u32
    }

    pub fn add_days(self, n: i64) -> Self {
        Date(self.0 + n)
    }

    /// Adds months; the day is clamped to the end of the month if needed (Jan 31 + 1M = Feb 28).
    pub fn add_months(self, n: i64) -> Self {
        let (y, m, d) = self.ymd();
        let total = y * 12 + (m as i64 - 1) + n;
        let (ny, nm) = (total.div_euclid(12), (total.rem_euclid(12) + 1) as u32);
        Date(days_from_civil(ny, nm, d.min(days_in_month(ny, nm))))
    }

    pub fn days_until(self, other: Date) -> i64 {
        other.0 - self.0
    }

    pub fn is_end_of_month(self) -> bool {
        let (y, m, d) = self.ymd();
        d == days_in_month(y, m)
    }
}

impl fmt::Display for Date {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let (y, m, d) = self.ymd();
        write!(f, "{y:04}-{m:02}-{d:02}")
    }
}

/// Easter Sunday (anonymous Gregorian algorithm).
pub fn easter_sunday(y: i64) -> Date {
    let a = y % 19;
    let b = y / 100;
    let c = y % 100;
    let d = b / 4;
    let e = b % 4;
    let f = (b + 8) / 25;
    let g = (b - f + 1) / 3;
    let h = (19 * a + b - d - g + 15) % 30;
    let i = c / 4;
    let k = c % 4;
    let l = (32 + 2 * e + 2 * i - h - k) % 7;
    let m = (a + 11 * h + 22 * l) / 451;
    let month = (h + l - 7 * m + 114) / 31;
    let day = (h + l - 7 * m + 114) % 31 + 1;
    Date(days_from_civil(y, month as u32, day as u32))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Calendar {
    /// Saturdays and Sundays only.
    Weekends,
    /// TARGET2 (euro area): Jan 1, Good Friday, Easter Monday, May 1, Dec 25 and 26.
    Target,
}

impl Calendar {
    pub fn is_business_day(self, d: Date) -> bool {
        if d.weekday() >= 5 {
            return false;
        }
        match self {
            Calendar::Weekends => true,
            Calendar::Target => {
                let (y, m, dd) = d.ymd();
                let easter = easter_sunday(y);
                let fixed = matches!((m, dd), (1, 1) | (5, 1) | (12, 25) | (12, 26))
                    || (m == 12 && dd == 31 && matches!(y, 1998 | 1999 | 2001));
                let mobile = y >= 2000 && (d == easter.add_days(-2) || d == easter.add_days(1));
                !(fixed || mobile)
            }
        }
    }

    pub fn adjust(self, d: Date, c: Convention) -> Date {
        let next = |mut x: Date, step: i64| {
            while !self.is_business_day(x) {
                x = x.add_days(step);
            }
            x
        };
        match c {
            Convention::Unadjusted => d,
            Convention::Following => next(d, 1),
            Convention::Preceding => next(d, -1),
            Convention::ModifiedFollowing => {
                let f = next(d, 1);
                if f.ymd().1 != d.ymd().1 {
                    next(d, -1)
                } else {
                    f
                }
            }
        }
    }

    /// Advances by `n` business days.
    pub fn advance_business_days(self, mut d: Date, n: i64) -> Date {
        let step = n.signum();
        let mut left = n.abs();
        while left > 0 {
            d = d.add_days(step);
            if self.is_business_day(d) {
                left -= 1;
            }
        }
        d
    }

    /// Last business day of the month of `d`.
    pub fn end_of_month(self, d: Date) -> Date {
        let (y, m, _) = d.ymd();
        self.adjust(Date(days_from_civil(y, m, days_in_month(y, m))), Convention::Preceding)
    }

    pub fn is_end_of_month(self, d: Date) -> bool {
        d == self.end_of_month(d)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Convention {
    Unadjusted,
    Following,
    ModifiedFollowing,
    Preceding,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DayCount {
    #[serde(rename = "act/360")]
    Act360,
    #[serde(rename = "act/365f")]
    Act365F,
    /// 30/360 bond basis (ISDA).
    #[serde(rename = "30/360")]
    Thirty360,
}

impl DayCount {
    pub fn year_fraction(self, d1: Date, d2: Date) -> f64 {
        match self {
            DayCount::Act360 => d1.days_until(d2) as f64 / 360.0,
            DayCount::Act365F => d1.days_until(d2) as f64 / 365.0,
            DayCount::Thirty360 => {
                let ((y1, m1, mut a), (y2, m2, mut b)) = (d1.ymd(), d2.ymd());
                if a == 31 {
                    a = 30;
                }
                if b == 31 && a >= 30 {
                    b = 30;
                }
                (360 * (y2 - y1) + 30 * (m2 as i64 - m1 as i64) + (b as i64 - a as i64)) as f64 / 360.0
            }
        }
    }
}

/// Tenor: `3M`, `6M`, `1Y`, `10Y`, `2W`, `7D`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tenor {
    Days(i64),
    Months(i64),
}

impl Tenor {
    pub fn parse(s: &str) -> Result<Self, String> {
        let s = s.trim().to_ascii_uppercase();
        let (n, unit) = s.split_at(s.len().saturating_sub(1));
        let n: i64 = n.parse().map_err(|_| format!("invalid tenor `{s}`"))?;
        if n <= 0 {
            return Err(format!("invalid tenor `{s}`"));
        }
        match unit {
            "D" => Ok(Tenor::Days(n)),
            "W" => Ok(Tenor::Days(7 * n)),
            "M" => Ok(Tenor::Months(n)),
            "Y" => Ok(Tenor::Months(12 * n)),
            _ => Err(format!("invalid tenor `{s}` (units D, W, M, Y)")),
        }
    }

    pub fn months(self) -> Result<i64, String> {
        match self {
            Tenor::Months(m) => Ok(m),
            Tenor::Days(_) => Err("day tenors are not supported here".into()),
        }
    }

    pub fn add_to(self, d: Date) -> Date {
        match self {
            Tenor::Days(n) => d.add_days(n),
            Tenor::Months(m) => d.add_months(m),
        }
    }
}

/// Schedule generated backward from the unadjusted end date (market rule for swaps),
/// dates adjusted with `convention`, the last one with `termination`.
pub fn backward_schedule(
    start: Date,
    end_unadjusted: Date,
    period_months: i64,
    cal: Calendar,
    convention: Convention,
    termination: Convention,
) -> Result<Vec<Date>, String> {
    if period_months <= 0 || end_unadjusted <= start {
        return Err("invalid schedule".into());
    }
    let mut raw = vec![end_unadjusted];
    let mut i = 1;
    loop {
        let d = end_unadjusted.add_months(-i * period_months);
        if d <= start {
            break;
        }
        raw.push(d);
        i += 1;
    }
    raw.push(start);
    raw.reverse();
    let n = raw.len();
    let mut out: Vec<Date> = raw
        .iter()
        .enumerate()
        .map(|(k, d)| cal.adjust(*d, if k == n - 1 { termination } else { convention }))
        .collect();
    out.dedup();
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(s: &str) -> Date {
        Date::parse(s).unwrap()
    }

    #[test]
    fn civil_roundtrip_and_weekday() {
        for s in ["1970-01-01", "2000-02-29", "2026-01-13", "2099-12-31"] {
            assert_eq!(d(s).to_string(), s);
        }
        assert_eq!(d("2026-01-13").weekday(), 1); // Tuesday
        assert!(Date::parse("2026-02-29").is_err());
    }

    #[test]
    fn month_arithmetic_clamps() {
        assert_eq!(d("2026-01-31").add_months(1), d("2026-02-28"));
        assert_eq!(d("2024-01-31").add_months(1), d("2024-02-29"));
        assert_eq!(d("2026-03-31").add_months(-1), d("2026-02-28"));
        assert_eq!(d("2026-01-15").add_months(120), d("2036-01-15"));
    }

    #[test]
    fn easter_dates() {
        assert_eq!(easter_sunday(2026), d("2026-04-05"));
        assert_eq!(easter_sunday(2033), d("2033-04-17"));
        assert_eq!(easter_sunday(2024), d("2024-03-31"));
    }

    #[test]
    fn target_calendar() {
        let t = Calendar::Target;
        assert!(!t.is_business_day(d("2026-04-03"))); // Good Friday
        assert!(!t.is_business_day(d("2026-04-06"))); // Easter Monday
        assert!(!t.is_business_day(d("2026-05-01")));
        assert!(!t.is_business_day(d("2026-12-25")));
        assert!(t.is_business_day(d("2026-04-07")));
        assert_eq!(t.advance_business_days(d("2026-01-13"), 2), d("2026-01-15"));
        assert_eq!(t.advance_business_days(d("2026-04-02"), 1), d("2026-04-07"));
    }

    #[test]
    fn business_day_conventions() {
        let t = Calendar::Target;
        // 2026-01-31 is a Saturday: Following -> Feb 2, Modified Following -> Jan 30.
        assert_eq!(t.adjust(d("2026-01-31"), Convention::Following), d("2026-02-02"));
        assert_eq!(
            t.adjust(d("2026-01-31"), Convention::ModifiedFollowing),
            d("2026-01-30")
        );
        assert_eq!(
            t.adjust(d("2033-04-15"), Convention::ModifiedFollowing),
            d("2033-04-19")
        ); // Good Friday
    }

    #[test]
    fn day_counts() {
        assert_eq!(
            DayCount::Act360.year_fraction(d("2026-01-15"), d("2026-07-15")),
            181.0 / 360.0
        );
        assert_eq!(
            DayCount::Thirty360.year_fraction(d("2026-01-31"), d("2026-03-31")),
            60.0 / 360.0
        );
        assert_eq!(
            DayCount::Thirty360.year_fraction(d("2026-02-28"), d("2026-03-31")),
            33.0 / 360.0
        );
    }

    #[test]
    fn schedules() {
        let s = backward_schedule(
            d("2026-01-15"),
            d("2028-01-15"),
            6,
            Calendar::Target,
            Convention::ModifiedFollowing,
            Convention::ModifiedFollowing,
        )
        .unwrap();
        let txt: Vec<String> = s.iter().map(|x| x.to_string()).collect();
        assert_eq!(
            txt,
            ["2026-01-15", "2026-07-15", "2027-01-15", "2027-07-15", "2028-01-17"]
        );
        assert_eq!(Tenor::parse("10Y").unwrap(), Tenor::Months(120));
        assert!(Tenor::parse("3Q").is_err());
    }
}

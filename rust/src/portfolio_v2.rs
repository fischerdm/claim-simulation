use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

/// A single insurance policy for the v2 multi-year simulation.
///
/// v2 differences vs Policy (v1):
///   - No `bonus_malus` field (dropped — too hard to project forward)
///   - `claims_hist: [u32; 3]` — synthetic claim history seed for the rolling
///     3-year window: [t=-3 (oldest), t=-2, t=-1 (most recent)]
///
/// Feature order in `to_feature_row()` must match `ALL_FEATURES` in train_v2.py:
///   [VehPower, VehAge, DrivAge, Density, Area, VehBrand, VehGas, Region, PriorClaims3Y]
///
/// VehAge and DrivAge are passed as parameters (not stored on the struct) because
/// they change every projection year; the struct holds the t=0 baseline values.
pub struct PolicyV2 {
    // Stable features — do not change across simulation years
    pub veh_power: f32, // engine power (clipped to 15)
    pub density:   f32, // population density of driver's municipality
    pub area:      f32, // area category: A=0 … F=5
    pub veh_brand: f32, // vehicle brand code
    pub veh_gas:   f32, // fuel type: Diesel=0, Regular=1
    pub region:    f32, // French administrative region code

    // Dynamic features — baseline at t=0, incremented each projection year
    pub veh_age:  f32, // vehicle age in years at t=0
    pub driv_age: f32, // driver age in years at t=0

    // Exposure at t=0; simulation years t=1..4 always use exposure=1.0
    pub exposure: f32,

    // Synthetic claim history seed: [oldest=t-3, t-2, newest=t-1].
    // The simulator maintains a rolling window, shifting one position each year.
    pub claims_hist: [u32; 3],
}

impl PolicyV2 {
    /// Returns the 9-element feature vector in the exact order the v2 ONNX model expects.
    ///
    /// `veh_age` and `driv_age` are passed in (not read from `self`) because they
    /// are advanced by 1 each projection year.
    /// `prior_claims_3y` is the sum of the current 3-year rolling window.
    pub fn to_feature_row(&self, veh_age: f32, driv_age: f32, prior_claims_3y: f32) -> [f32; 9] {
        [
            self.veh_power,
            veh_age,
            driv_age,
            self.density,
            self.area,
            self.veh_brand,
            self.veh_gas,
            self.region,
            prior_claims_3y,
        ]
    }
}

/// CSV row written by python/export_portfolio_v2.py.
/// Column names match the CSV headers exactly — serde maps them by field name.
#[derive(Deserialize)]
struct PortfolioRowV2 {
    veh_power:     f32,
    veh_age:       f32,
    driv_age:      f32,
    density:       f32,
    area:          f32,
    veh_brand:     f32,
    veh_gas:       f32,
    region:        f32,
    exposure:      f32,
    claims_hist_1: u32, // t = -3 (oldest)
    claims_hist_2: u32, // t = -2
    claims_hist_3: u32, // t = -1 (most recent)
}

/// Load a v2 portfolio from the CSV produced by python/export_portfolio_v2.py.
pub fn load_v2_from_csv(path: &Path) -> anyhow::Result<Vec<PolicyV2>> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open v2 portfolio CSV at {}", path.display()))?;

    let policies = rdr
        .deserialize()
        .enumerate()
        .map(|(i, result)| {
            let row: PortfolioRowV2 = result
                .with_context(|| format!("failed to parse row {} in v2 portfolio CSV", i + 1))?;
            Ok(PolicyV2 {
                veh_power:   row.veh_power,
                veh_age:     row.veh_age,
                driv_age:    row.driv_age,
                density:     row.density,
                area:        row.area,
                veh_brand:   row.veh_brand,
                veh_gas:     row.veh_gas,
                region:      row.region,
                exposure:    row.exposure,
                claims_hist: [row.claims_hist_1, row.claims_hist_2, row.claims_hist_3],
            })
        })
        .collect::<anyhow::Result<Vec<PolicyV2>>>()?;

    Ok(policies)
}

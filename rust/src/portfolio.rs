use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

/// A single insurance policy for the multi-year simulation.
///
/// BonusMalus is excluded — it cannot be projected forward reliably.
/// Instead, PriorClaims3Y (a rolling 3-year claim window) is used as the
/// experience feature, which the simulation updates each projection year.
///
/// Feature order in `to_feature_row()` must match `ALL_FEATURES` in train.py (V2):
///   [VehPower, VehAge, DrivAge, Density, PriorClaims3Y, Area, VehBrand, VehGas, Region]
///
/// Categorical encodings (from models/feature_metadata_v2.json):
///   Area:     A=0, B=1, C=2, D=3, E=4, F=5
///   VehBrand: B1=0, B10=1, B11=2, B12=3, B13=4, B14=5, B2=6, B3=7, B4=8, B5=9, B6=10
///   VehGas:   Diesel=0, Regular=1
///   Region:   R11=0, R21=1, R22=2, ..., R94=21  (see feature_metadata_v2.json)
pub struct Policy {
    // Stable features — do not change across simulation years
    pub veh_power: f32,
    pub density:   f32,
    pub area:      f32,
    pub veh_brand: f32,
    pub veh_gas:   f32,
    pub region:    f32,

    // Dynamic features — baseline at t=0, incremented each projection year
    pub veh_age:  f32,
    pub driv_age: f32,

    // Exposure at t=0; simulation years t=1..N_YEARS-1 always use exposure=1.0
    pub exposure: f32,

    // Synthetic claim history seed: [oldest=t-3, t-2, newest=t-1].
    // The simulator maintains a rolling window, shifting one position each year.
    pub claims_hist: [u32; 3],
}

impl Policy {
    /// Returns the 9-element feature vector the v2 ONNX model expects.
    ///
    /// `veh_age`, `driv_age`, and `prior_claims_3y` are passed in (not read
    /// from `self`) because they change every projection year.
    pub fn to_feature_row(&self, veh_age: f32, driv_age: f32, prior_claims_3y: f32) -> [f32; 9] {
        [
            self.veh_power,
            veh_age,
            driv_age,
            self.density,
            prior_claims_3y,
            self.area,
            self.veh_brand,
            self.veh_gas,
            self.region,
        ]
    }
}

/// CSV row written by python/export_portfolio.py (v2 section).
/// Column names match the CSV headers exactly — serde maps them by field name.
/// All categoricals are already label-encoded (encoding applied in Python).
#[derive(Deserialize)]
struct PortfolioRow {
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

/// Load a portfolio from `data/portfolio_v2.csv`.
pub fn load_from_csv(path: &Path) -> anyhow::Result<Vec<Policy>> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open portfolio CSV at {}", path.display()))?;

    rdr.deserialize()
        .enumerate()
        .map(|(i, result)| {
            let row: PortfolioRow = result
                .with_context(|| format!("failed to parse row {} in portfolio CSV", i + 1))?;
            Ok(Policy {
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
        .collect()
}

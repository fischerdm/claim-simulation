use std::path::Path;

use anyhow::Context;
use serde::Deserialize;

/// A single insurance policy with all features needed for frequency prediction.
///
/// Feature order in `to_feature_row()` must match `ALL_FEATURES` in train.py:
///   [VehPower, VehAge, DrivAge, BonusMalus, Density, Area, VehBrand, VehGas, Region]
///
/// Categorical features are stored as their label-encoded integer values cast to f32,
/// which is what the ONNX model expects (LightGBM/onnxmltools convention).
/// The encoding comes from models/feature_metadata.json:
///   Area:     A=0, B=1, C=2, D=3, E=4, F=5
///   VehBrand: B1=0, B10=1, B11=2, B12=3, B13=4, B14=5, B2=6, B3=7, B4=8, B5=9, B6=10
///   VehGas:   Diesel=0, Regular=1
///   Region:   R11=0, R21=1, R22=2, ..., R94=21  (see feature_metadata.json for full list)
pub struct Policy {
    // Numeric features
    pub veh_power:   f32,  // engine power (clipped to 15)
    pub veh_age:     f32,  // vehicle age in years
    pub driv_age:    f32,  // driver age in years
    pub bonus_malus: f32,  // bonus-malus score (50 = best, 200 = worst, clipped)
    pub density:     f32,  // population density of the driver's municipality

    // Categorical features (label-encoded as integers stored as f32)
    pub area:      f32,    // area category: A=0 ... F=5
    pub veh_brand: f32,    // vehicle brand code
    pub veh_gas:   f32,    // fuel type: Diesel=0, Regular=1
    pub region:    f32,    // French administrative region code

    // Exposure: fraction of the year the policy was active.
    // NOT a model input — used as the offset: μ = λ × exposure
    pub exposure: f32,
}

impl Policy {
    /// Returns the 9-element feature vector in the exact order the ONNX model expects.
    /// In Python terms: [VehPower, VehAge, DrivAge, BonusMalus, Density, Area, VehBrand, VehGas, Region]
    pub fn to_feature_row(&self) -> [f32; 9] {
        [
            self.veh_power,
            self.veh_age,
            self.driv_age,
            self.bonus_malus,
            self.density,
            self.area,
            self.veh_brand,
            self.veh_gas,
            self.region,
        ]
    }
}

/// CSV row written by python/export_portfolio.py.
/// Column names match the CSV headers exactly — serde maps them by field name.
/// All categoricals are already label-encoded (encoding applied in Python).
#[derive(Deserialize)]
struct PortfolioRow {
    veh_power:   f32,
    veh_age:     f32,
    driv_age:    f32,
    bonus_malus: f32,
    density:     f32,
    area:        f32,
    veh_brand:   f32,
    veh_gas:     f32,
    region:      f32,
    exposure:    f32,
}

/// Load a portfolio from the CSV file produced by python/export_portfolio.py.
///
/// The CSV contains numeric columns only — categoricals are already label-encoded
/// with the same orderings as during model training. Column names must exactly
/// match the `PortfolioRow` fields above.
///
/// Equivalent Python:
///   df = pd.read_csv(path)
pub fn load_from_csv(path: &Path) -> anyhow::Result<Vec<Policy>> {
    let mut rdr = csv::Reader::from_path(path)
        .with_context(|| format!("failed to open portfolio CSV at {}", path.display()))?;

    let policies = rdr
        .deserialize()
        .enumerate()
        .map(|(i, result)| {
            let row: PortfolioRow = result
                .with_context(|| format!("failed to parse row {} in portfolio CSV", i + 1))?;
            Ok(Policy {
                veh_power:   row.veh_power,
                veh_age:     row.veh_age,
                driv_age:    row.driv_age,
                bonus_malus: row.bonus_malus,
                density:     row.density,
                area:        row.area,
                veh_brand:   row.veh_brand,
                veh_gas:     row.veh_gas,
                region:      row.region,
                exposure:    row.exposure,
            })
        })
        .collect::<anyhow::Result<Vec<Policy>>>()?;

    Ok(policies)
}

#[cfg(test)]
pub fn test_portfolio() -> Vec<Policy> {
    vec![
        // 1. Low risk: experienced driver, best bonus-malus, rural area
        Policy {
            veh_power: 6.0, veh_age: 5.0, driv_age: 45.0, bonus_malus: 50.0, density: 200.0,
            area: 1.0,  // B
            veh_brand: 0.0,  // B1
            veh_gas: 0.0,    // Diesel
            region: 2.0,     // R22
            exposure: 1.0,
        },
        // 2. High risk: young driver, high bonus-malus, dense urban area
        Policy {
            veh_power: 8.0, veh_age: 2.0, driv_age: 22.0, bonus_malus: 150.0, density: 8000.0,
            area: 4.0,   // E
            veh_brand: 3.0,  // B12
            veh_gas: 1.0,    // Regular
            region: 0.0,     // R11
            exposure: 1.0,
        },
        // 3. Medium risk: middle-aged driver, partial exposure (joined mid-year)
        Policy {
            veh_power: 7.0, veh_age: 8.0, driv_age: 35.0, bonus_malus: 70.0, density: 800.0,
            area: 3.0,   // D
            veh_brand: 6.0,  // B2
            veh_gas: 0.0,    // Diesel
            region: 11.0,    // R52
            exposure: 0.75,
        },
        // 4. Senior driver, low bonus-malus, mid-size city
        Policy {
            veh_power: 5.0, veh_age: 3.0, driv_age: 60.0, bonus_malus: 50.0, density: 500.0,
            area: 2.0,   // C
            veh_brand: 10.0, // B6
            veh_gas: 0.0,    // Diesel
            region: 4.0,     // R24
            exposure: 1.0,
        },
        // 5. High-power vehicle, experienced driver
        Policy {
            veh_power: 12.0, veh_age: 1.0, driv_age: 40.0, bonus_malus: 60.0, density: 3000.0,
            area: 5.0,   // F
            veh_brand: 7.0,  // B3
            veh_gas: 1.0,    // Regular
            region: 19.0,    // R91
            exposure: 0.5,
        },
        // 6. Rural policy, very low density, oldest vehicle
        Policy {
            veh_power: 5.0, veh_age: 12.0, driv_age: 50.0, bonus_malus: 50.0, density: 50.0,
            area: 0.0,   // A
            veh_brand: 9.0,  // B5
            veh_gas: 0.0,    // Diesel
            region: 6.0,     // R26
            exposure: 1.0,
        },
        // 7. Short exposure — new policy, started late in the year
        Policy {
            veh_power: 7.0, veh_age: 4.0, driv_age: 30.0, bonus_malus: 85.0, density: 1200.0,
            area: 2.0,   // C
            veh_brand: 3.0,  // B12
            veh_gas: 1.0,    // Regular
            region: 17.0,    // R82
            exposure: 0.25,
        },
        // 8. Worst risk: high bonus-malus, young driver, dense city
        Policy {
            veh_power: 9.0, veh_age: 3.0, driv_age: 28.0, bonus_malus: 200.0, density: 10000.0,
            area: 4.0,   // E
            veh_brand: 8.0,  // B4
            veh_gas: 1.0,    // Regular
            region: 0.0,     // R11
            exposure: 1.0,
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Smoke test: the test portfolio builds without panicking and has the expected size.
    #[test]
    fn test_portfolio_has_eight_policies() {
        let portfolio = test_portfolio();
        assert_eq!(portfolio.len(), 8);
    }

    /// Every policy must produce a 9-element feature row.
    #[test]
    fn feature_rows_have_correct_length() {
        for policy in test_portfolio() {
            assert_eq!(policy.to_feature_row().len(), 9);
        }
    }
}

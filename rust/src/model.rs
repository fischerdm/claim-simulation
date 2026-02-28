use std::path::Path;

use anyhow::Context;
use ort::{inputs, session::Session, value::Tensor};

use crate::portfolio::Policy;

/// Wraps the ONNX Runtime session for the LightGBM frequency model.
pub struct FrequencyModel {
    session: Session,
    /// Output tensor name, read from the model at load time.
    output_name: String,
}

impl FrequencyModel {
    /// Load the ONNX model from disk.
    ///
    /// Equivalent to Python's:
    ///   sess = onnxruntime.InferenceSession("model.onnx")
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        let session = Session::builder()
            .context("failed to create ONNX session builder")?
            .commit_from_file(path)
            .with_context(|| format!("failed to load ONNX model from {}", path.display()))?;

        let output_name = session
            .outputs()
            .first()
            .context("ONNX model has no outputs")?
            .name()
            .to_string();

        Ok(Self { session, output_name })
    }

    /// Compute the Poisson parameter μ per policy: μ = λ × exposure.
    ///
    /// The ONNX model outputs λ (annual claim frequency) directly.
    /// Multiplying by exposure gives the expected claim count for the period.
    /// The model is deterministic, so we call it ONCE here before the parallel
    /// simulation loop. Each simulation then just draws Poisson(μ) per policy.
    ///
    /// Note: `session.run` requires `&mut self` in ort 2.x, which is why this
    /// method takes `&mut self`. That also means it cannot be called from
    /// multiple threads at once, which is fine — we call it once upfront.
    pub fn compute_lambdas(&mut self, policies: &[Policy]) -> anyhow::Result<Vec<f64>> {
        let n = policies.len();

        // Build a flat Vec<f32> row-by-row, then hand it to ort as a [n, 9] tensor.
        // ort accepts a (shape, Vec<T>) tuple directly — no ndarray needed here.
        let flat: Vec<f32> = policies
            .iter()
            .flat_map(|p| p.to_feature_row())
            .collect();

        // Create the input tensor: shape [n_policies, 9], dtype f32.
        let tensor = Tensor::<f32>::from_array(([n, 9_usize], flat))
            .context("failed to create input tensor")?;

        // Run the model. `inputs!["name" => value]` builds the named input map.
        // "float_input" is the input name set in export_onnx.py's initial_types.
        // `run` takes `&mut self` — one call, sequential, before parallelism.
        let mut outputs = self
            .session
            .run(inputs!["float_input" => tensor])
            .context("ONNX inference failed")?;

        // `remove` gives us an owned DynValue, freeing the borrow on `session`.
        let output = outputs
            .remove(self.output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("output '{}' not found in model", self.output_name))?;

        // Extract the raw data slice: (shape, &[f32]).
        // The ONNX model outputs λ (annual claim frequency) directly.
        // booster.predict() for LightGBM Poisson returns exp(f(X)) = λ,
        // and onnxmltools preserves this — the output is already in original space.
        let (_, lambda_slice) = output
            .try_extract_tensor::<f32>()
            .context("failed to extract output tensor as f32")?;

        // Expected claims per policy = λ × exposure.
        // λ is the annual frequency; exposure is the fraction of year active.
        let lambdas = lambda_slice
            .iter()
            .zip(policies.iter())
            .map(|(&lambda, policy)| {
                let mu = lambda as f64 * policy.exposure as f64;
                mu.max(1e-9) // guard against zero/negative due to rounding
            })
            .collect();

        Ok(lambdas)
    }
}

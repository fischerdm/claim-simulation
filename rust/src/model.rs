use std::path::Path;

use anyhow::Context;
use ort::{inputs, session::Session, value::Tensor};

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

    /// Generic inference: takes a pre-assembled row-major feature matrix and
    /// returns λ (annual frequency) per policy.
    ///
    /// `flat_features` is a Vec<f32> of length `n_policies × n_features`.
    /// This is the workhorse for the v2 multi-year simulation, where features
    /// change each year (VehAge, DrivAge, PriorClaims3Y) and must be rebuilt
    /// before every ONNX call.
    pub fn run_inference(
        &mut self,
        flat_features: Vec<f32>,
        n_policies: usize,
        n_features: usize,
    ) -> anyhow::Result<Vec<f64>> {
        let tensor = Tensor::<f32>::from_array(([n_policies, n_features], flat_features))
            .context("failed to create input tensor")?;

        let mut outputs = self
            .session
            .run(inputs!["float_input" => tensor])
            .context("ONNX inference failed")?;

        let output = outputs
            .remove(self.output_name.as_str())
            .ok_or_else(|| anyhow::anyhow!("output '{}' not found", self.output_name))?;

        let (_, lambda_slice) = output
            .try_extract_tensor::<f32>()
            .context("failed to extract output tensor as f32")?;

        Ok(lambda_slice
            .iter()
            .map(|&l| (l as f64).max(1e-9))
            .collect())
    }
}

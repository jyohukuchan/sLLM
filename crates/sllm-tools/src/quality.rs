//! Deterministic, bounded quality evaluators.
//!
//! The quality tools intentionally consume JSON fixtures rather than reaching
//! out to a model or a leaderboard.  This keeps the evaluator reproducible and
//! lets the runtime (or a future common manifest implementation) attach its
//! identity without coupling this crate to a particular execution backend.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::cmp::Ordering;
use std::fmt;
use std::fs;
use std::path::PathBuf;

pub const QUALITY_INPUT_SCHEMA_VERSION: &str = "sllm-phase46-quality-input-v1";
pub const QUALITY_RESULT_SCHEMA_VERSION: &str = "sllm-phase46-quality-result-v1";
pub const QUALITY_RESULT_STRUCT_SIZE_V1: u32 = 7;

/// Bounds are deliberately conservative.  Callers may lower them for a
/// particular run, but may not raise the hard process limits.
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct QualityLimits {
    #[serde(default = "default_max_samples")]
    pub max_samples: usize,
    #[serde(default = "default_max_logit_width")]
    pub max_logit_width: usize,
    #[serde(default = "default_max_task_choices")]
    pub max_task_choices: usize,
    #[serde(default = "default_max_context_tokens")]
    pub max_context_tokens: usize,
    #[serde(default = "default_max_input_bytes")]
    pub max_input_bytes: usize,
}

const fn default_max_samples() -> usize {
    100_000
}
const fn default_max_logit_width() -> usize {
    1_000_000
}
const fn default_max_task_choices() -> usize {
    100_000
}
const fn default_max_context_tokens() -> usize {
    10_000_000
}
const fn default_max_input_bytes() -> usize {
    64 * 1024 * 1024
}

impl Default for QualityLimits {
    fn default() -> Self {
        Self {
            max_samples: default_max_samples(),
            max_logit_width: default_max_logit_width(),
            max_task_choices: default_max_task_choices(),
            max_context_tokens: default_max_context_tokens(),
            max_input_bytes: default_max_input_bytes(),
        }
    }
}

impl QualityLimits {
    fn validate(self) -> Result<Self, QualityError> {
        let hard = Self::default();
        if self.max_samples == 0
            || self.max_samples > hard.max_samples
            || self.max_logit_width == 0
            || self.max_logit_width > hard.max_logit_width
            || self.max_task_choices == 0
            || self.max_task_choices > hard.max_task_choices
            || self.max_context_tokens == 0
            || self.max_context_tokens > hard.max_context_tokens
            || self.max_input_bytes == 0
            || self.max_input_bytes > hard.max_input_bytes
        {
            return Err(QualityError::OverLimit(
                "invalid evaluator limits".to_owned(),
            ));
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QualityError {
    Invalid(String),
    Empty(String),
    NonFinite(String),
    OverLimit(String),
    Unsupported(String),
}

impl fmt::Display for QualityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(message) => write!(f, "invalid quality input: {message}"),
            Self::Empty(message) => write!(f, "empty quality input: {message}"),
            Self::NonFinite(message) => write!(f, "non-finite quality value: {message}"),
            Self::OverLimit(message) => write!(f, "quality input exceeds bound: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported quality input: {message}"),
        }
    }
}

impl std::error::Error for QualityError {}

#[derive(Debug, Deserialize)]
struct InputEnvelope {
    schema_version: String,
    #[serde(default)]
    #[serde(alias = "type")]
    kind: Option<String>,
    #[serde(default)]
    limits: Option<QualityLimits>,
    #[serde(default)]
    manifest: Option<crate::tool_manifest::ToolRunManifestV1>,
    #[serde(flatten)]
    body: Map<String, Value>,
}

/// Evaluate one versioned JSON input and return a versioned machine-readable
/// result.  Exactly one of `perplexity`, `logit_comparison`/`logits`, `task`,
/// or `long_context` must be present.
pub fn evaluate_quality(input: &Value) -> Result<Value, QualityError> {
    let encoded = serde_json::to_vec(input)
        .map_err(|error| QualityError::Invalid(format!("cannot encode input: {error}")))?;
    let envelope: InputEnvelope = serde_json::from_value(input.clone())
        .map_err(|error| QualityError::Invalid(error.to_string()))?;
    if envelope.schema_version != QUALITY_INPUT_SCHEMA_VERSION {
        return Err(QualityError::Unsupported(format!(
            "schema_version must be {QUALITY_INPUT_SCHEMA_VERSION}"
        )));
    }
    let limits = envelope.limits.unwrap_or_default().validate()?;
    if encoded.len() > limits.max_input_bytes {
        return Err(QualityError::OverLimit(format!(
            "input bytes {} > {}",
            encoded.len(),
            limits.max_input_bytes
        )));
    }

    let mut selected = Vec::new();
    for key in [
        "perplexity",
        "logit_comparison",
        "logits",
        "kld",
        "task",
        "long_context",
    ] {
        if envelope.body.contains_key(key) {
            selected.push(key);
        }
    }
    if selected.len() != 1 {
        return Err(QualityError::Invalid(
            "exactly one evaluator section is required".to_owned(),
        ));
    }
    if envelope.body.keys().any(|key| {
        !matches!(
            key.as_str(),
            "perplexity" | "logit_comparison" | "logits" | "kld" | "task" | "long_context"
        )
    }) {
        return Err(QualityError::Unsupported(
            "quality input contains an unknown top-level field".to_owned(),
        ));
    }
    let selected_kind = envelope.kind.as_deref();
    let section = envelope.body.get(selected[0]).expect("selected section");
    let kind = selected_kind.unwrap_or(selected[0]);
    let (metric, result) = match (kind, selected[0]) {
        ("perplexity", "perplexity") => ("perplexity", evaluate_perplexity(section, limits)?),
        ("kld", "logit_comparison")
        | ("logits", "logit_comparison")
        | ("logit_comparison", "logit_comparison")
        | ("kld", "logits")
        | ("logits", "logits")
        | ("logit_comparison", "logits") => ("logit_comparison", evaluate_logits(section, limits)?),
        ("kld", "kld") | ("logits", "kld") | ("logit_comparison", "kld") => {
            ("logit_comparison", evaluate_logits(section, limits)?)
        }
        ("task", "task") => ("task", evaluate_task(section, limits)?),
        ("long_context", "long_context") => {
            ("long_context", evaluate_long_context(section, limits)?)
        }
        _ => {
            return Err(QualityError::Invalid(format!(
                "kind {kind:?} does not match evaluator section {}",
                selected[0]
            )));
        }
    };
    let mut output = Map::new();
    output.insert(
        "$schema".to_owned(),
        Value::String("https://sllm.dev/schema/phase46-quality-result-v1.schema.json".to_owned()),
    );
    output.insert(
        "schema_version".to_owned(),
        Value::String(QUALITY_RESULT_SCHEMA_VERSION.to_owned()),
    );
    output.insert("state".to_owned(), Value::String("PASS".to_owned()));
    output.insert(
        "struct_size".to_owned(),
        Value::from(QUALITY_RESULT_STRUCT_SIZE_V1),
    );
    output.insert("metric".to_owned(), Value::String(metric.to_owned()));
    let manifest = envelope.manifest.ok_or_else(|| {
        QualityError::Invalid("an identity-bound tool manifest is required".to_owned())
    })?;
    manifest
        .validate()
        .map_err(|error| QualityError::Invalid(format!("invalid tool manifest: {error}")))?;
    output.insert(
        "manifest".to_owned(),
        serde_json::to_value(manifest)
            .map_err(|error| QualityError::Invalid(format!("serialize tool manifest: {error}")))?,
    );
    output.insert("result".to_owned(), result);
    output.insert("extensions".to_owned(), Value::Object(Map::new()));
    Ok(Value::Object(output))
}

/// Attach the common typed run manifest without making the evaluator own the
/// manifest construction.  This is the integration point used by callers
/// which already have a verified `ToolRunManifestV1`.
pub fn evaluate_quality_with_manifest(
    input: &Value,
    manifest: &crate::tool_manifest::ToolRunManifestV1,
) -> Result<Value, QualityError> {
    manifest
        .validate()
        .map_err(|error| QualityError::Invalid(format!("invalid tool manifest: {error}")))?;
    let mut bound_input = input.clone();
    let input_object = bound_input
        .as_object_mut()
        .ok_or_else(|| QualityError::Invalid("quality input must be an object".to_owned()))?;
    if input_object.contains_key("manifest") {
        return Err(QualityError::Invalid(
            "embedded manifest conflicts with the explicitly supplied manifest".to_owned(),
        ));
    }
    input_object.insert(
        "manifest".to_owned(),
        serde_json::to_value(manifest)
            .map_err(|error| QualityError::Invalid(format!("serialize tool manifest: {error}")))?,
    );
    let mut result = evaluate_quality(&bound_input)?;
    let object = result
        .as_object_mut()
        .ok_or_else(|| QualityError::Invalid("quality result is not an object".to_owned()))?;
    object.insert(
        "manifest".to_owned(),
        serde_json::to_value(manifest)
            .map_err(|error| QualityError::Invalid(format!("serialize tool manifest: {error}")))?,
    );
    let extensions = object
        .get_mut("extensions")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| QualityError::Invalid("quality extensions are not an object".to_owned()))?;
    extensions.insert(
        "manifest_sha256".to_owned(),
        Value::String(
            manifest
                .sha256()
                .map_err(|error| QualityError::Invalid(format!("hash tool manifest: {error}")))?,
        ),
    );
    Ok(result)
}

/// Entry point for the `sllm-eval` binary.  Input and output are both bounded;
/// output publication is atomic when `--output` is supplied.
pub fn run_eval_cli(mut arguments: impl Iterator<Item = String>) -> Result<(), String> {
    let mut input_path = None;
    let mut output_path = None;
    let mut manifest_path = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--help" | "-h" => {
                print_eval_help();
                return Ok(());
            }
            "--input" => input_path = Some(next_cli_path(&mut arguments, "--input")?),
            "--output" => output_path = Some(next_cli_path(&mut arguments, "--output")?),
            "--manifest" => manifest_path = Some(next_cli_path(&mut arguments, "--manifest")?),
            value if value.starts_with('-') => {
                return Err(format!("unknown sllm-eval option {value}"));
            }
            value => {
                if input_path.is_some() {
                    return Err(format!("unexpected positional argument {value}"));
                }
                input_path = Some(PathBuf::from(value));
            }
        }
    }
    let input_path = input_path.ok_or_else(|| "--input PATH is required".to_owned())?;
    let bytes = fs::read(&input_path).map_err(|error| format!("read input: {error}"))?;
    if bytes.len() > QualityLimits::default().max_input_bytes {
        return Err("input exceeds the 64 MiB evaluator bound".to_owned());
    }
    let input: Value =
        serde_json::from_slice(&bytes).map_err(|error| format!("parse input: {error}"))?;
    let manifest_path = manifest_path.ok_or_else(|| {
        "--manifest RUN.json is required so quality output is identity-bound".to_owned()
    })?;
    let manifest_bytes =
        fs::read(manifest_path).map_err(|error| format!("read manifest: {error}"))?;
    let manifest = serde_json::from_slice(&manifest_bytes)
        .map_err(|error| format!("parse manifest: {error}"))?;
    let result =
        evaluate_quality_with_manifest(&input, &manifest).map_err(|error| error.to_string())?;
    if let Some(path) = output_path {
        crate::tool_manifest::atomic_write_json(&path, &result)
            .map_err(|error| format!("publish output: {error}"))?;
    } else {
        let encoded = serde_json::to_string_pretty(&result)
            .map_err(|error| format!("serialize result: {error}"))?;
        println!("{encoded}");
    }
    Ok(())
}

fn next_cli_path(
    arguments: &mut impl Iterator<Item = String>,
    option: &str,
) -> Result<PathBuf, String> {
    arguments
        .next()
        .map(PathBuf::from)
        .ok_or_else(|| format!("{option} requires a path"))
}

fn print_eval_help() {
    println!("sllm-eval --input INPUT.json --manifest RUN.json [--output RESULT.json]");
    println!("  INPUT schema: {QUALITY_INPUT_SCHEMA_VERSION}");
    println!("  sections: perplexity, logit_comparison, task, long_context");
    println!(
        "  output is {QUALITY_RESULT_SCHEMA_VERSION}; all empty/non-finite/over-limit input fails"
    );
}

fn evaluate_perplexity(section: &Value, limits: QualityLimits) -> Result<Value, QualityError> {
    let object = if let Some(object) = section.as_object() {
        object
    } else if let Some(losses) = section.as_array() {
        return evaluate_perplexity(&json!({ "losses": losses }), limits);
    } else {
        return Err(QualityError::Invalid(
            "perplexity must be an object or loss array".to_owned(),
        ));
    };
    reject_unknown_keys(
        object,
        &[
            "loss_sum",
            "loss_total",
            "token_count",
            "target_token_count",
            "losses",
            "token_losses",
        ],
        "perplexity",
    )?;
    let (loss_sum, token_count) = if let (Some(loss), Some(tokens)) = (
        object
            .get("loss_sum")
            .or_else(|| object.get("loss_total"))
            .and_then(Value::as_f64),
        object
            .get("token_count")
            .or_else(|| object.get("target_token_count"))
            .and_then(Value::as_u64),
    ) {
        (
            loss,
            usize::try_from(tokens)
                .map_err(|_| QualityError::OverLimit("token_count".to_owned()))?,
        )
    } else if let Some(values) = object.get("losses").or_else(|| object.get("token_losses")) {
        let values = values
            .as_array()
            .ok_or_else(|| QualityError::Invalid("losses must be an array".to_owned()))?;
        if values.is_empty() {
            return Err(QualityError::Empty("perplexity losses".to_owned()));
        }
        if values.len() > limits.max_context_tokens {
            return Err(QualityError::OverLimit("perplexity token count".to_owned()));
        }
        let mut sum = 0.0_f64;
        for (index, value) in values.iter().enumerate() {
            let loss = value
                .as_f64()
                .ok_or_else(|| QualityError::Invalid(format!("losses[{index}] is not a number")))?;
            ensure_finite(loss, &format!("losses[{index}]"))?;
            if loss < 0.0 {
                return Err(QualityError::Invalid(format!(
                    "losses[{index}] is negative"
                )));
            }
            sum += loss;
            ensure_finite(sum, "loss_sum")?;
        }
        (sum, values.len())
    } else {
        return Err(QualityError::Invalid(
            "perplexity requires loss_sum+token_count or losses".to_owned(),
        ));
    };
    ensure_finite(loss_sum, "loss_sum")?;
    if loss_sum < 0.0 {
        return Err(QualityError::Invalid(
            "perplexity loss_sum is negative".to_owned(),
        ));
    }
    if token_count == 0 {
        return Err(QualityError::Empty("perplexity token_count".to_owned()));
    }
    if token_count > limits.max_context_tokens {
        return Err(QualityError::OverLimit("perplexity token_count".to_owned()));
    }
    let mean_nll = loss_sum / token_count as f64;
    ensure_finite(mean_nll, "mean_nll")?;
    let perplexity = mean_nll.exp();
    ensure_finite(perplexity, "perplexity")?;
    Ok(json!({
        "loss_sum": loss_sum,
        "token_count": token_count,
        "mean_nll": mean_nll,
        "perplexity": perplexity,
    }))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LogitSample {
    #[serde(default)]
    position: Option<u64>,
    #[serde(alias = "reference", alias = "baseline_logits", alias = "fp16")]
    baseline: Vec<f64>,
    #[serde(alias = "candidate_logits", alias = "quantized")]
    candidate: Vec<f64>,
}

fn evaluate_logits(section: &Value, limits: QualityLimits) -> Result<Value, QualityError> {
    let object = if let Some(object) = section.as_object() {
        object
    } else if let Some(samples) = section.as_array() {
        return evaluate_logits(&json!({ "samples": samples }), limits);
    } else {
        return Err(QualityError::Invalid(
            "logit_comparison must be an object or sample array".to_owned(),
        ));
    };
    reject_unknown_keys(object, &["samples", "positions"], "logit comparison")?;
    let raw_samples = object
        .get("samples")
        .or_else(|| object.get("positions"))
        .ok_or_else(|| QualityError::Invalid("logit comparison requires samples".to_owned()))?;
    let samples: Vec<LogitSample> = serde_json::from_value(raw_samples.clone())
        .map_err(|error| QualityError::Invalid(format!("invalid logit sample: {error}")))?;
    if samples.is_empty() {
        return Err(QualityError::Empty("logit samples".to_owned()));
    }
    if samples.len() > limits.max_samples {
        return Err(QualityError::OverLimit("logit sample count".to_owned()));
    }
    let mut top1_matches = 0usize;
    let mut first_divergence = None;
    let mut max_kld = 0.0_f64;
    let mut klds = Vec::with_capacity(samples.len());
    let mut differences = Vec::new();
    let mut positions = Vec::with_capacity(samples.len());
    let mut seen_positions = std::collections::BTreeSet::new();
    for (index, sample) in samples.iter().enumerate() {
        let position = sample.position.unwrap_or(index as u64);
        if position > limits.max_context_tokens as u64 {
            return Err(QualityError::OverLimit(format!(
                "logit position {position}"
            )));
        }
        if !seen_positions.insert(position) {
            return Err(QualityError::Invalid(format!(
                "duplicate logit position {position}"
            )));
        }
        validate_logits(&sample.baseline, limits.max_logit_width, "baseline")?;
        validate_logits(&sample.candidate, limits.max_logit_width, "candidate")?;
        if sample.baseline.len() != sample.candidate.len() {
            return Err(QualityError::Invalid(format!(
                "logit width mismatch at position {position}"
            )));
        }
        let baseline_top1 = argmax(&sample.baseline);
        let candidate_top1 = argmax(&sample.candidate);
        let top1_match = baseline_top1 == candidate_top1;
        if top1_match {
            top1_matches += 1;
        } else if first_divergence
            .map(|previous| position < previous)
            .unwrap_or(true)
        {
            first_divergence = Some(position);
        }
        let kld = kl_divergence(&sample.baseline, &sample.candidate)?;
        max_kld = max_kld.max(kld);
        klds.push(kld);
        for (reference, candidate) in sample.baseline.iter().zip(&sample.candidate) {
            let difference = (reference - candidate).abs();
            ensure_finite(difference, "logit difference")?;
            differences.push(difference);
        }
        positions.push(json!({
            "position": position,
            "baseline_top1": baseline_top1,
            "candidate_top1": candidate_top1,
            "top1_match": top1_match,
            "kld": kld,
        }));
    }
    if differences.is_empty() {
        return Err(QualityError::Empty("logit values".to_owned()));
    }
    if differences.len() > limits.max_samples.saturating_mul(limits.max_logit_width) {
        return Err(QualityError::OverLimit("logit value count".to_owned()));
    }
    differences.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    klds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    let quantiles = json!({
        "p50": percentile(&differences, 0.50),
        "p90": percentile(&differences, 0.90),
        "p99": percentile(&differences, 0.99),
    });
    Ok(json!({
        "sample_count": samples.len(),
        "top1_matches": top1_matches,
        "top1_agreement": top1_matches as f64 / samples.len() as f64,
        "kld_max": max_kld,
        "kld_p50": percentile(&klds, 0.50),
        "kld_p90": percentile(&klds, 0.90),
        "kld_p99": percentile(&klds, 0.99),
        "logit_abs_diff_max": *differences.last().expect("nonempty"),
        "logit_abs_diff": quantiles,
        "first_divergence_position": first_divergence,
        "positions": positions,
    }))
}

fn validate_logits(values: &[f64], max_width: usize, label: &str) -> Result<(), QualityError> {
    if values.is_empty() {
        return Err(QualityError::Empty(format!("{label} logits")));
    }
    if values.len() > max_width {
        return Err(QualityError::OverLimit(format!("{label} logit width")));
    }
    for (index, value) in values.iter().enumerate() {
        ensure_finite(*value, &format!("{label}[{index}]"))?;
    }
    Ok(())
}

fn argmax(values: &[f64]) -> usize {
    let mut best = 0usize;
    for index in 1..values.len() {
        // Strict comparison intentionally keeps the lowest index on ties.
        if values[index] > values[best] {
            best = index;
        }
    }
    best
}

fn kl_divergence(reference: &[f64], candidate: &[f64]) -> Result<f64, QualityError> {
    let reference_probs = softmax(reference)?;
    let candidate_probs = softmax(candidate)?;
    let mut result = 0.0_f64;
    for (reference, candidate) in reference_probs.iter().zip(candidate_probs) {
        // Both softmax outputs are strictly positive for finite logits.  The
        // guard keeps this fail-closed if the implementation changes later.
        if *reference <= 0.0 || candidate <= 0.0 {
            return Err(QualityError::NonFinite(
                "invalid softmax probability".to_owned(),
            ));
        }
        result += reference * (reference / candidate).ln();
    }
    ensure_finite(result, "KLD")?;
    Ok(result.max(0.0))
}

fn softmax(values: &[f64]) -> Result<Vec<f64>, QualityError> {
    let max = values
        .iter()
        .copied()
        .reduce(f64::max)
        .ok_or_else(|| QualityError::Empty("softmax values".to_owned()))?;
    let mut exponents = Vec::with_capacity(values.len());
    let mut sum = 0.0_f64;
    for value in values {
        let exponent = (*value - max).exp();
        ensure_finite(exponent, "softmax exponent")?;
        exponents.push(exponent);
        sum += exponent;
    }
    ensure_finite(sum, "softmax sum")?;
    if sum <= 0.0 {
        return Err(QualityError::NonFinite("softmax sum is zero".to_owned()));
    }
    Ok(exponents.into_iter().map(|value| value / sum).collect())
}

fn percentile(values: &[f64], quantile: f64) -> f64 {
    debug_assert!(!values.is_empty());
    let index = ((values.len() - 1) as f64 * quantile).ceil() as usize;
    values[index.min(values.len() - 1)]
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskSection {
    #[serde(default)]
    task_version: Option<String>,
    #[serde(default)]
    renderer: Option<String>,
    #[serde(default)]
    few_shot: Option<u32>,
    #[serde(default)]
    samples: Vec<TaskSample>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct TaskSample {
    #[serde(default, alias = "prediction_text", alias = "predicted")]
    prediction: Option<String>,
    #[serde(default, alias = "expected", alias = "target", alias = "gold")]
    reference: Option<String>,
    #[serde(default)]
    answer: Option<Value>,
    #[serde(default, alias = "logits")]
    choice_logits: Option<Vec<f64>>,
    #[serde(default)]
    choices: Option<Vec<Value>>,
    #[serde(default, alias = "correct", alias = "correct_index")]
    answer_index: Option<usize>,
}

fn evaluate_task(section: &Value, limits: QualityLimits) -> Result<Value, QualityError> {
    let task: TaskSection = serde_json::from_value(section.clone())
        .map_err(|error| QualityError::Invalid(format!("invalid task section: {error}")))?;
    if task.samples.is_empty() {
        return Err(QualityError::Empty("task samples".to_owned()));
    }
    if task.samples.len() > limits.max_samples {
        return Err(QualityError::OverLimit("task sample count".to_owned()));
    }
    for (label, value) in [
        ("task_version", task.task_version.as_deref()),
        ("renderer", task.renderer.as_deref()),
    ] {
        if value.is_some_and(str::is_empty) {
            return Err(QualityError::Invalid(format!(
                "task {label} must not be empty"
            )));
        }
    }
    let mut exact_matches = 0usize;
    let mut multiple_choice = 0usize;
    let mut multiple_choice_matches = 0usize;
    for (index, sample) in task.samples.iter().enumerate() {
        let choice_logits = if let Some(logits) = &sample.choice_logits {
            Some(logits.clone())
        } else if let Some(choices) = &sample.choices {
            if choices.len() > limits.max_task_choices {
                return Err(QualityError::OverLimit("task choice count".to_owned()));
            }
            let mut logits = Vec::with_capacity(choices.len());
            let mut numeric = true;
            for choice in choices {
                if let Some(logit) = choice.as_f64() {
                    logits.push(logit);
                } else if let Some(object) = choice.as_object() {
                    if let Some(logit) = object
                        .get("logit")
                        .or_else(|| object.get("score"))
                        .and_then(Value::as_f64)
                    {
                        logits.push(logit);
                    } else {
                        numeric = false;
                        break;
                    }
                } else {
                    numeric = false;
                    break;
                }
            }
            numeric.then_some(logits)
        } else {
            None
        };
        if let Some(logits) = choice_logits.as_deref() {
            multiple_choice += 1;
            validate_logits(logits, limits.max_task_choices, "choice_logits")?;
            let answer = sample
                .answer_index
                .or_else(|| {
                    sample
                        .answer
                        .as_ref()
                        .and_then(Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok())
                })
                .ok_or_else(|| {
                    QualityError::Invalid(format!("task sample {index} lacks answer_index"))
                })?;
            if answer >= logits.len() {
                return Err(QualityError::Invalid(format!(
                    "task sample {index} answer_index out of range"
                )));
            }
            if argmax(logits) == answer {
                multiple_choice_matches += 1;
            }
        } else if let Some(choices) = sample
            .choices
            .as_ref()
            .filter(|choices| !choices.is_empty() && choices.iter().all(Value::is_string))
        {
            multiple_choice += 1;
            let answer = sample
                .answer_index
                .or_else(|| {
                    sample
                        .answer
                        .as_ref()
                        .and_then(Value::as_u64)
                        .and_then(|value| usize::try_from(value).ok())
                })
                .ok_or_else(|| {
                    QualityError::Invalid(format!("task sample {index} lacks answer_index"))
                })?;
            if answer >= choices.len() {
                return Err(QualityError::Invalid(format!(
                    "task sample {index} answer_index out of range"
                )));
            }
            let predicted = sample.prediction.as_deref().and_then(|prediction| {
                choices
                    .iter()
                    .position(|choice| choice.as_str() == Some(prediction))
            });
            if predicted == Some(answer) {
                multiple_choice_matches += 1;
            }
        } else {
            let prediction = sample.prediction.as_deref().ok_or_else(|| {
                QualityError::Invalid(format!("task sample {index} lacks prediction"))
            })?;
            let reference = sample
                .reference
                .as_deref()
                .or_else(|| sample.answer.as_ref().and_then(Value::as_str))
                .ok_or_else(|| {
                    QualityError::Invalid(format!("task sample {index} lacks reference"))
                })?;
            if prediction == reference {
                exact_matches += 1;
            }
        }
    }
    let exact_count = task.samples.len().saturating_sub(multiple_choice);
    Ok(json!({
        "task_version": task.task_version,
        "renderer": task.renderer,
        "few_shot": task.few_shot,
        "sample_count": task.samples.len(),
        "exact_match": if exact_count == 0 { Value::Null } else { json!(exact_matches as f64 / exact_count as f64) },
        "exact_match_count": exact_matches,
        "multiple_choice_count": multiple_choice,
        "multiple_choice_accuracy": if multiple_choice == 0 { Value::Null } else { json!(multiple_choice_matches as f64 / multiple_choice as f64) },
        "multiple_choice_correct": multiple_choice_matches,
    }))
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct LongContextSample {
    position: u64,
    #[serde(default)]
    max_position: Option<u64>,
    #[serde(default)]
    band: Option<String>,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    key: Option<bool>,
    #[serde(default)]
    value: Option<bool>,
    #[serde(default)]
    #[serde(alias = "plane")]
    kv_plane: Option<String>,
    #[serde(default)]
    layer: Option<u32>,
    #[serde(default)]
    kv_head: Option<u32>,
    #[serde(default)]
    block_tail: Option<bool>,
}

fn evaluate_long_context(section: &Value, limits: QualityLimits) -> Result<Value, QualityError> {
    let object = if let Some(object) = section.as_object() {
        object
    } else if let Some(samples) = section.as_array() {
        return evaluate_long_context(&json!({ "samples": samples }), limits);
    } else {
        return Err(QualityError::Invalid(
            "long_context must be an object or sample array".to_owned(),
        ));
    };
    reject_unknown_keys(
        object,
        &["capacity", "samples", "positions"],
        "long_context",
    )?;
    let raw_samples = object
        .get("samples")
        .or_else(|| object.get("positions"))
        .ok_or_else(|| QualityError::Invalid("long_context requires samples".to_owned()))?;
    let samples: Vec<LongContextSample> = serde_json::from_value(raw_samples.clone())
        .map_err(|error| QualityError::Invalid(format!("invalid long-context sample: {error}")))?;
    if samples.is_empty() {
        return Err(QualityError::Empty("long-context samples".to_owned()));
    }
    if samples.len() > limits.max_samples {
        return Err(QualityError::OverLimit(
            "long-context sample count".to_owned(),
        ));
    }
    let declared_capacity = object.get("capacity").and_then(Value::as_u64);
    let max_position = samples
        .iter()
        .map(|sample| sample.position)
        .max()
        .unwrap_or(0);
    if max_position > limits.max_context_tokens as u64 {
        return Err(QualityError::OverLimit("long-context position".to_owned()));
    }
    let capacity = declared_capacity.unwrap_or_else(|| {
        samples
            .iter()
            .filter_map(|sample| sample.max_position)
            .max()
            .unwrap_or(max_position)
            .max(max_position)
            .saturating_add(1)
    });
    if capacity == 0 || capacity > limits.max_context_tokens as u64 {
        return Err(QualityError::OverLimit("long-context capacity".to_owned()));
    }
    let mut early = 0usize;
    let mut middle = 0usize;
    let mut tail = 0usize;
    let mut key = 0usize;
    let mut value = 0usize;
    let mut layers = std::collections::BTreeSet::new();
    let mut heads = std::collections::BTreeSet::new();
    let mut block_tail = 0usize;
    let mut kinds = std::collections::BTreeSet::new();
    let mut positions = std::collections::BTreeSet::new();
    for sample in &samples {
        if sample.position >= capacity
            || sample
                .max_position
                .is_some_and(|position| position < sample.position || position >= capacity)
        {
            return Err(QualityError::Invalid(
                "long-context position is outside declared capacity".to_owned(),
            ));
        }
        positions.insert(sample.position);
        let band = sample.band.as_deref().unwrap_or_else(|| {
            if sample.position.saturating_mul(3) < capacity {
                "early"
            } else if sample.position.saturating_mul(3) < capacity.saturating_mul(2) {
                "middle"
            } else {
                "tail"
            }
        });
        match band {
            "early" => early += 1,
            "middle" => middle += 1,
            "tail" => tail += 1,
            _ => {
                return Err(QualityError::Invalid(format!(
                    "unknown long-context band {band}"
                )));
            }
        }
        if sample.key.unwrap_or(false)
            || sample
                .kv_plane
                .as_deref()
                .is_some_and(|plane| plane.eq_ignore_ascii_case("k"))
        {
            key += 1;
        }
        if sample.kv_plane.as_deref().is_some_and(|plane| {
            !plane.eq_ignore_ascii_case("k") && !plane.eq_ignore_ascii_case("v")
        }) {
            return Err(QualityError::Invalid(
                "long-context kv_plane must be K or V".to_owned(),
            ));
        }
        if sample.value.unwrap_or(false)
            || sample
                .kv_plane
                .as_deref()
                .is_some_and(|plane| plane.eq_ignore_ascii_case("v"))
        {
            value += 1;
        }
        if let Some(layer) = sample.layer {
            layers.insert(layer);
        }
        if let Some(head) = sample.kv_head {
            heads.insert(head);
        }
        if sample.block_tail.unwrap_or(false) {
            block_tail += 1;
        }
        if let Some(kind) = sample.kind.as_deref() {
            if kind.is_empty() {
                return Err(QualityError::Invalid(
                    "long-context kind must not be empty".to_owned(),
                ));
            }
            kinds.insert(kind.to_owned());
        }
    }
    if early == 0 || middle == 0 || tail == 0 {
        return Err(QualityError::Invalid(
            "long-context coverage requires early, middle, and tail samples".to_owned(),
        ));
    }
    if key == 0 || value == 0 {
        return Err(QualityError::Invalid(
            "long-context coverage requires both K and V samples".to_owned(),
        ));
    }
    if layers.is_empty() || heads.is_empty() || block_tail == 0 {
        return Err(QualityError::Invalid(
            "long-context coverage requires layer, KV-head, and block-tail samples".to_owned(),
        ));
    }
    Ok(json!({
        "sample_count": samples.len(),
        "capacity": capacity,
        "position_min": samples.iter().map(|sample| sample.position).min().unwrap_or(0),
        "position_max": max_position,
        "early": early,
        "middle": middle,
        "tail": tail,
        "coverage_ratio": positions.len() as f64 / capacity as f64,
        "key_samples": key,
        "value_samples": value,
        "layers": layers.into_iter().collect::<Vec<_>>(),
        "kv_heads": heads.into_iter().collect::<Vec<_>>(),
        "block_tail_samples": block_tail,
        "kinds": kinds.into_iter().collect::<Vec<_>>(),
    }))
}

fn reject_unknown_keys(
    object: &Map<String, Value>,
    allowed: &[&str],
    label: &str,
) -> Result<(), QualityError> {
    if let Some(key) = object.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(QualityError::Unsupported(format!(
            "{label} contains unknown field {key}"
        )));
    }
    Ok(())
}

fn ensure_finite(value: f64, label: &str) -> Result<(), QualityError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(QualityError::NonFinite(label.to_owned()))
    }
}

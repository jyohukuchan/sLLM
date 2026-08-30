//! Deterministic aggregation of identity-bound benchmark samples.

use crate::tool_manifest::{
    AtomicBundleV1, TOOL_JSON_CANONICALIZATION_V1, TOOL_RUN_SCHEMA_VERSION_V1,
    TOOL_RUN_STRUCT_SIZE_V1, ToolError, ToolFileIdentityV1, ToolIdentityV1, ToolRecipeIdentityV1,
    ToolRunManifestV1, ToolRunStateV1, canonical_json_bytes, sha256_bytes,
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};

pub const BENCHMARK_INPUT_SCHEMA_V1: &str = "sllm-phase46-benchmark-input-v1";
pub const BENCHMARK_RESULT_SCHEMA_V1: &str = "sllm-phase46-benchmark-result-v1";
pub const BENCHMARK_RESULT_STRUCT_SIZE_V1: u32 = 7;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SampleStateV1 {
    Pass,
    Timeout,
    Crash,
    Oom,
    BackendFallback,
    CleanupFailure,
    Rejected,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSampleV1 {
    pub iteration: u32,
    pub state: SampleStateV1,
    pub reason: Option<String>,
    pub wall_ns: Option<u64>,
    pub gpu_ns: Option<u64>,
    pub model_load_ns: Option<u64>,
    pub e2e_ns: Option<u64>,
    pub ttft_ns: Option<u64>,
    pub tpot_ns: Option<u64>,
    pub prefill_ns: Option<u64>,
    pub decode_ns: Option<u64>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "status", rename_all = "kebab-case", deny_unknown_fields)]
pub enum ResourceMeasurementV1 {
    Measured { bytes: u64 },
    Unsupported { reason: String },
    Missing { reason: String },
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkResourcesV1 {
    pub hbm_before: ResourceMeasurementV1,
    pub hbm_peak: ResourceMeasurementV1,
    pub hbm_settled: ResourceMeasurementV1,
    pub gtt_before: ResourceMeasurementV1,
    pub gtt_peak: ResourceMeasurementV1,
    pub gtt_settled: ResourceMeasurementV1,
    pub model_resident: ResourceMeasurementV1,
    pub kv_logical: ResourceMeasurementV1,
    pub kv_physical: ResourceMeasurementV1,
    pub workspace: ResourceMeasurementV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkConfigurationV1 {
    pub request_count: u32,
    pub parallelism: u32,
    pub context_tokens: u64,
    pub sampling: String,
    pub kv_encoding: String,
    pub gpu_identity: String,
    pub provider: String,
    pub fallback: bool,
    pub cleanup: bool,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkInputV1 {
    pub schema_version: String,
    pub model_lock: String,
    pub tokenizer_files: Vec<String>,
    pub dataset_files: Vec<String>,
    pub configuration: BenchmarkConfigurationV1,
    pub warmups: Vec<BenchmarkSampleV1>,
    pub measured: Vec<BenchmarkSampleV1>,
    pub resources: BenchmarkResourcesV1,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DistributionV1 {
    pub count: u64,
    pub min: u64,
    pub p10: u64,
    pub median: u64,
    pub p90: u64,
    pub max: u64,
    pub mad: u64,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct TimingSummaryV1 {
    pub wall_ns: DistributionV1,
    pub gpu_ns: Option<DistributionV1>,
    pub model_load_ns: DistributionV1,
    pub e2e_ns: DistributionV1,
    pub ttft_ns: DistributionV1,
    pub tpot_ns: DistributionV1,
    pub prefill_ns: DistributionV1,
    pub decode_ns: DistributionV1,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum DecisionV1 {
    Pass,
    Fail,
    Unsupported,
    Missing,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkDecisionsV1 {
    pub correctness: DecisionV1,
    pub quality: DecisionV1,
    pub performance: DecisionV1,
    pub memory: DecisionV1,
    pub fallback: DecisionV1,
    pub cleanup: DecisionV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkPayloadV1 {
    pub configuration: BenchmarkConfigurationV1,
    pub warmups: Vec<BenchmarkSampleV1>,
    pub measured: Vec<BenchmarkSampleV1>,
    pub rejected: Vec<BenchmarkSampleV1>,
    pub timing: Option<TimingSummaryV1>,
    pub resources: BenchmarkResourcesV1,
    pub decisions: BenchmarkDecisionsV1,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkResultV1 {
    #[serde(rename = "$schema")]
    pub schema_uri: String,
    pub schema_version: String,
    pub struct_size: u32,
    pub state: ToolRunStateV1,
    pub manifest: ToolRunManifestV1,
    pub payload: BenchmarkPayloadV1,
    pub extensions: BTreeMap<String, Value>,
}

pub fn aggregate_benchmark(
    input: BenchmarkInputV1,
) -> Result<(ToolRunStateV1, BenchmarkPayloadV1), ToolError> {
    validate_input(&input)?;
    let (mut accepted, mut rejected) = (Vec::new(), Vec::new());
    for sample in &input.measured {
        if sample.state == SampleStateV1::Pass && complete_timings(sample) {
            accepted.push(sample.clone());
        } else {
            let mut rejected_sample = sample.clone();
            if rejected_sample
                .reason
                .as_deref()
                .unwrap_or_default()
                .is_empty()
            {
                rejected_sample.reason = Some("missing-or-zero-timing".to_owned());
            }
            rejected.push(rejected_sample);
        }
    }
    let run_ok = input.warmups.iter().all(|s| s.state == SampleStateV1::Pass)
        && !accepted.is_empty()
        && rejected.is_empty()
        && !input.configuration.fallback
        && input.configuration.cleanup;
    let state = if run_ok {
        ToolRunStateV1::Pass
    } else {
        ToolRunStateV1::Fail
    };
    let timing = if accepted.is_empty() {
        None
    } else {
        Some(summarize(&accepted)?)
    };
    let decisions = BenchmarkDecisionsV1 {
        correctness: decision(run_ok),
        quality: DecisionV1::Missing,
        performance: decision(run_ok),
        memory: resource_decision(&input.resources),
        fallback: decision(!input.configuration.fallback),
        cleanup: decision(input.configuration.cleanup),
    };
    Ok((
        state,
        BenchmarkPayloadV1 {
            configuration: input.configuration,
            warmups: input.warmups,
            measured: accepted,
            rejected,
            timing,
            resources: input.resources,
            decisions,
        },
    ))
}

fn validate_input(input: &BenchmarkInputV1) -> Result<(), ToolError> {
    if input.schema_version != BENCHMARK_INPUT_SCHEMA_V1 {
        return Err(ToolError::invalid("unknown benchmark input schema"));
    }
    if input.warmups.is_empty() || input.measured.is_empty() {
        return Err(ToolError::invalid(
            "benchmark has zero selected warmup or measured samples",
        ));
    }
    if input.model_lock.is_empty()
        || input.tokenizer_files.is_empty()
        || input.dataset_files.is_empty()
    {
        return Err(ToolError::invalid("benchmark identities are incomplete"));
    }
    let c = &input.configuration;
    if c.request_count == 0
        || c.parallelism == 0
        || c.context_tokens == 0
        || c.sampling.is_empty()
        || c.kv_encoding.is_empty()
        || c.gpu_identity.is_empty()
        || c.provider.is_empty()
    {
        return Err(ToolError::invalid("benchmark configuration is incomplete"));
    }
    for samples in [&input.warmups, &input.measured] {
        let mut seen = BTreeSet::new();
        for sample in samples {
            if sample.iteration == 0 || !seen.insert(sample.iteration) {
                return Err(ToolError::invalid(
                    "benchmark iteration is zero or duplicated",
                ));
            }
            if sample.state != SampleStateV1::Pass
                && sample.reason.as_deref().unwrap_or_default().is_empty()
            {
                return Err(ToolError::invalid("failed benchmark sample has no reason"));
            }
        }
    }
    Ok(())
}

fn complete_timings(s: &BenchmarkSampleV1) -> bool {
    [
        s.wall_ns,
        s.model_load_ns,
        s.e2e_ns,
        s.ttft_ns,
        s.tpot_ns,
        s.prefill_ns,
        s.decode_ns,
    ]
    .iter()
    .all(|v| v.is_some_and(|v| v > 0))
}

fn summarize(samples: &[BenchmarkSampleV1]) -> Result<TimingSummaryV1, ToolError> {
    fn req(
        samples: &[BenchmarkSampleV1],
        pick: impl Fn(&BenchmarkSampleV1) -> Option<u64>,
    ) -> Result<DistributionV1, ToolError> {
        distribution(samples.iter().filter_map(pick).collect())
    }
    let gpu: Vec<u64> = samples.iter().filter_map(|s| s.gpu_ns).collect();
    Ok(TimingSummaryV1 {
        wall_ns: req(samples, |s| s.wall_ns)?,
        gpu_ns: if gpu.len() == samples.len() {
            Some(distribution(gpu)?)
        } else {
            None
        },
        model_load_ns: req(samples, |s| s.model_load_ns)?,
        e2e_ns: req(samples, |s| s.e2e_ns)?,
        ttft_ns: req(samples, |s| s.ttft_ns)?,
        tpot_ns: req(samples, |s| s.tpot_ns)?,
        prefill_ns: req(samples, |s| s.prefill_ns)?,
        decode_ns: req(samples, |s| s.decode_ns)?,
    })
}

fn distribution(mut values: Vec<u64>) -> Result<DistributionV1, ToolError> {
    if values.is_empty() {
        return Err(ToolError::invalid("cannot summarize zero samples"));
    }
    values.sort_unstable();
    let median = percentile(&values, 50);
    let mut deviations: Vec<u64> = values.iter().map(|v| v.abs_diff(median)).collect();
    deviations.sort_unstable();
    Ok(DistributionV1 {
        count: values.len() as u64,
        min: values[0],
        p10: percentile(&values, 10),
        median,
        p90: percentile(&values, 90),
        max: *values.last().expect("nonempty"),
        mad: percentile(&deviations, 50),
    })
}

fn percentile(values: &[u64], percent: usize) -> u64 {
    values[percent
        .saturating_mul(values.len().saturating_sub(1))
        .div_ceil(100)]
}

fn decision(pass: bool) -> DecisionV1 {
    if pass {
        DecisionV1::Pass
    } else {
        DecisionV1::Fail
    }
}

fn resource_decision(r: &BenchmarkResourcesV1) -> DecisionV1 {
    let all = [
        &r.hbm_before,
        &r.hbm_peak,
        &r.hbm_settled,
        &r.gtt_before,
        &r.gtt_peak,
        &r.gtt_settled,
        &r.model_resident,
        &r.kv_logical,
        &r.kv_physical,
        &r.workspace,
    ];
    if all
        .iter()
        .any(|v| matches!(v, ResourceMeasurementV1::Missing { .. }))
    {
        DecisionV1::Missing
    } else if all
        .iter()
        .any(|v| matches!(v, ResourceMeasurementV1::Unsupported { .. }))
    {
        DecisionV1::Unsupported
    } else {
        DecisionV1::Pass
    }
}

pub fn publish_benchmark(
    input_path: &Path,
    output_bundle: &Path,
    tool_commit: &str,
) -> Result<PathBuf, ToolError> {
    let bytes = fs::read(input_path)
        .map_err(|e| ToolError::invalid(format!("read benchmark input: {e}")))?;
    let input: BenchmarkInputV1 = serde_json::from_slice(&bytes)
        .map_err(|e| ToolError::invalid(format!("parse benchmark input: {e}")))?;
    let model_lock = PathBuf::from(&input.model_lock);
    let tokenizers: Vec<PathBuf> = input.tokenizer_files.iter().map(PathBuf::from).collect();
    let datasets: Vec<PathBuf> = input.dataset_files.iter().map(PathBuf::from).collect();
    let (state, payload) = aggregate_benchmark(input)?;
    let bundle = AtomicBundleV1::create(output_bundle)?;
    let payload_path = bundle.write_json("benchmark-payload.json", &payload)?;
    let mut sources = vec![
        ToolFileIdentityV1::from_path("benchmark-input", "benchmark-input.json", input_path)?,
        ToolFileIdentityV1::from_path("model-lock", "model-lock.json", &model_lock)?,
    ];
    for (i, path) in tokenizers.iter().enumerate() {
        sources.push(ToolFileIdentityV1::from_path(
            "tokenizer",
            format!("tokenizer-{i}"),
            path,
        )?);
    }
    for (i, path) in datasets.iter().enumerate() {
        sources.push(ToolFileIdentityV1::from_path(
            "dataset",
            format!("dataset-{i}"),
            path,
        )?);
    }
    let recipe = serde_json::to_value(&payload.configuration)
        .map_err(|e| ToolError::invalid(format!("serialize recipe: {e}")))?;
    let executable = std::env::current_exe()
        .map_err(|e| ToolError::invalid(format!("resolve benchmark executable: {e}")))?;
    let executable_sha256 =
        ToolFileIdentityV1::from_path("tool-binary", "sllm-bench", executable)?.sha256;
    let manifest = ToolRunManifestV1 {
        schema_version: TOOL_RUN_SCHEMA_VERSION_V1.to_owned(),
        struct_size: TOOL_RUN_STRUCT_SIZE_V1,
        canonicalization: TOOL_JSON_CANONICALIZATION_V1.to_owned(),
        operation: "benchmark".to_owned(),
        state,
        selected_count: payload.measured.len() as u64,
        tool: ToolIdentityV1 {
            repository: "https://github.com/89chin/sLLM".to_owned(),
            commit: tool_commit.to_owned(),
            package: "sllm-tools".to_owned(),
            version: env!("CARGO_PKG_VERSION").to_owned(),
            executable_sha256,
            arguments: std::env::args().collect(),
            environment: crate::tool_manifest::rust_toolchain_environment(),
        },
        recipe: ToolRecipeIdentityV1 {
            id: "benchmark".to_owned(),
            version: "v1".to_owned(),
            config_sha256: sha256_bytes(&canonical_json_bytes(&recipe)?),
        },
        sources,
        outputs: vec![ToolFileIdentityV1::from_path(
            "benchmark-payload",
            "benchmark-payload.json",
            &payload_path,
        )?],
        raw_evidence: vec![ToolFileIdentityV1::for_bytes(
            "benchmark-input-copy",
            "benchmark-input.json",
            &bytes,
        )?],
        identities: BTreeMap::from([
            ("gpu".to_owned(), payload.configuration.gpu_identity.clone()),
            (
                "provider".to_owned(),
                payload.configuration.provider.clone(),
            ),
            (
                "kv-encoding".to_owned(),
                payload.configuration.kv_encoding.clone(),
            ),
        ]),
        metrics: BTreeMap::from([(
            "timing".to_owned(),
            serde_json::to_value(&payload.timing)
                .map_err(|e| ToolError::invalid(format!("serialize timing: {e}")))?,
        )]),
        extensions: BTreeMap::new(),
    };
    manifest.validate()?;
    let result = BenchmarkResultV1 {
        schema_uri: "https://sllm.dev/schema/phase46-benchmark-result-v1.schema.json".to_owned(),
        schema_version: BENCHMARK_RESULT_SCHEMA_V1.to_owned(),
        struct_size: BENCHMARK_RESULT_STRUCT_SIZE_V1,
        state,
        manifest,
        payload,
        extensions: BTreeMap::new(),
    };
    bundle.write_json("benchmark.json", &result)?;
    bundle.write_bytes("raw/benchmark-input.json", &bytes)?;
    bundle.commit()
}

pub fn run_bench_cli<I, S>(args: I) -> Result<(), ToolError>
where
    I: IntoIterator<Item = S>,
    S: Into<OsString>,
{
    let args: Vec<OsString> = args.into_iter().map(Into::into).collect();
    if args.len() == 1 && matches!(args[0].to_str(), Some("-h" | "--help")) {
        println!(
            "sllm-bench aggregate --input INPUT.json --output-bundle DIR --tool-commit COMMIT"
        );
        return Ok(());
    }
    if args.first().and_then(|v| v.to_str()) != Some("aggregate") {
        return Err(ToolError::invalid("expected `aggregate` (use --help)"));
    }
    let value = |flag: &str| -> Result<PathBuf, ToolError> {
        let i = args
            .iter()
            .position(|arg| arg == flag)
            .ok_or_else(|| ToolError::invalid(format!("missing {flag}")))?;
        args.get(i + 1)
            .map(PathBuf::from)
            .ok_or_else(|| ToolError::invalid(format!("missing value for {flag}")))
    };
    let input = value("--input")?;
    let output = value("--output-bundle")?;
    let commit = value("--tool-commit")?;
    let commit = commit
        .to_str()
        .ok_or_else(|| ToolError::invalid("tool commit is not UTF-8"))?;
    println!("{}", publish_benchmark(&input, &output, commit)?.display());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn sample(iteration: u32, value: u64) -> BenchmarkSampleV1 {
        BenchmarkSampleV1 {
            iteration,
            state: SampleStateV1::Pass,
            reason: None,
            wall_ns: Some(value),
            gpu_ns: Some(value - 1),
            model_load_ns: Some(value),
            e2e_ns: Some(value),
            ttft_ns: Some(value),
            tpot_ns: Some(value),
            prefill_ns: Some(value),
            decode_ns: Some(value),
        }
    }
    fn resources(value: ResourceMeasurementV1) -> BenchmarkResourcesV1 {
        BenchmarkResourcesV1 {
            hbm_before: value.clone(),
            hbm_peak: value.clone(),
            hbm_settled: value.clone(),
            gtt_before: value.clone(),
            gtt_peak: value.clone(),
            gtt_settled: value.clone(),
            model_resident: value.clone(),
            kv_logical: value.clone(),
            kv_physical: value.clone(),
            workspace: value,
        }
    }
    fn input() -> BenchmarkInputV1 {
        BenchmarkInputV1 {
            schema_version: BENCHMARK_INPUT_SCHEMA_V1.to_owned(),
            model_lock: "model.lock.json".to_owned(),
            tokenizer_files: vec!["tokenizer.json".to_owned()],
            dataset_files: vec!["dataset.json".to_owned()],
            configuration: BenchmarkConfigurationV1 {
                request_count: 1,
                parallelism: 1,
                context_tokens: 17,
                sampling: "greedy".to_owned(),
                kv_encoding: "fp16".to_owned(),
                gpu_identity: "gfx1030".to_owned(),
                provider: "hip".to_owned(),
                fallback: false,
                cleanup: true,
            },
            warmups: vec![sample(1, 10)],
            measured: vec![sample(1, 10), sample(2, 20), sample(3, 30)],
            resources: resources(ResourceMeasurementV1::Measured { bytes: 0 }),
        }
    }
    #[test]
    fn wall_and_gpu_are_distinct() {
        let (state, out) = aggregate_benchmark(input()).unwrap();
        assert_eq!(state, ToolRunStateV1::Pass);
        let timing = out.timing.unwrap();
        assert_eq!(timing.wall_ns.median, 20);
        assert_eq!(timing.gpu_ns.unwrap().median, 19);
    }
    #[test]
    fn zero_and_oom_fail_closed() {
        let mut zero = input();
        zero.measured.clear();
        assert!(aggregate_benchmark(zero).is_err());
        let mut oom = input();
        oom.measured[1].state = SampleStateV1::Oom;
        oom.measured[1].reason = Some("oom".to_owned());
        let (state, out) = aggregate_benchmark(oom).unwrap();
        assert_eq!(state, ToolRunStateV1::Fail);
        assert_eq!(out.rejected.len(), 1);
    }
    #[test]
    fn missing_differs_from_unsupported() {
        let mut missing = input();
        missing.resources.hbm_peak = ResourceMeasurementV1::Missing {
            reason: "sampler".to_owned(),
        };
        assert_eq!(
            aggregate_benchmark(missing).unwrap().1.decisions.memory,
            DecisionV1::Missing
        );
        let mut unsupported = input();
        unsupported.resources.gtt_peak = ResourceMeasurementV1::Unsupported {
            reason: "platform".to_owned(),
        };
        assert_eq!(
            aggregate_benchmark(unsupported).unwrap().1.decisions.memory,
            DecisionV1::Unsupported
        );
    }
}

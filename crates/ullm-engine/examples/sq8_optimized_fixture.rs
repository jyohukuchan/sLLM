// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use serde::Serialize;
use std::ffi::{CString, OsString};
use std::fs::{self, File, OpenOptions};
use std::io::Write;
use std::os::raw::{c_char, c_int, c_uint};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use ullm_engine::host_bytes::encode_f32_to_bytes;
use ullm_engine::sq_canonical::read_sq8_canonical_artifact;
use ullm_engine::sq_optimized_reference::{
    SQ8_OPTIMIZED_COSINE_THRESHOLD, SQ8_OPTIMIZED_RELATIVE_L2_THRESHOLD,
    quantize_sq8_dynamic_activation, run_sq8_optimized_reference_projection,
};
use ullm_engine::sq_reference::sq8_reference_activation;

const SCHEMA_VERSION: &str = "sq8-optimized-fixture-v0.2";
const MAX_FIXTURE_WORKING_BYTES: usize = 2 * 1024 * 1024 * 1024;
const AT_FDCWD: c_int = -100;
const RENAME_NOREPLACE: c_uint = 1;

unsafe extern "C" {
    fn renameat2(
        old_dir_fd: c_int,
        old_path: *const c_char,
        new_dir_fd: c_int,
        new_path: *const c_char,
        flags: c_uint,
    ) -> c_int;
}

#[derive(Debug)]
struct Args {
    artifact_dir: PathBuf,
    tensor: String,
    m: usize,
    output_dir: PathBuf,
}

#[derive(Serialize)]
struct TensorReport {
    name: String,
    weight_shape: [u64; 2],
    weight_sha256: String,
    weight_scale_shape: [u64; 2],
    weight_scale_sha256: String,
    block_shape: [u64; 2],
}

#[derive(Serialize)]
struct RawFileReport {
    file: &'static str,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: usize,
    sha256: String,
}

#[derive(Serialize)]
struct ThresholdReport {
    relative_l2_max: f64,
    cosine_min: f64,
    finite_output_required: bool,
}

#[derive(Serialize)]
struct FixtureReport {
    schema_version: &'static str,
    artifact_dir: String,
    artifact_schema_version: String,
    artifact_content_sha256: String,
    tensor: TensorReport,
    m: usize,
    n: usize,
    k: usize,
    activation_block_cols: usize,
    cpu_worker_threads: usize,
    estimated_working_bytes: usize,
    activation: RawFileReport,
    quantized_activation: RawFileReport,
    activation_scales: RawFileReport,
    oracle_output: RawFileReport,
    thresholds: ThresholdReport,
}

fn take_value(values: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    values
        .next()
        .ok_or_else(|| format!("{flag} requires a value"))
}

fn parse_args() -> Result<Option<Args>, String> {
    let mut artifact_dir = None;
    let mut tensor = None;
    let mut m = None;
    let mut output_dir = None;
    let mut values = std::env::args().skip(1);
    while let Some(flag) = values.next() {
        match flag.as_str() {
            "--artifact-dir" => {
                artifact_dir = Some(PathBuf::from(take_value(&mut values, &flag)?));
            }
            "--tensor" => tensor = Some(take_value(&mut values, &flag)?),
            "--m" => {
                let value = take_value(&mut values, &flag)?;
                let parsed = value
                    .parse::<usize>()
                    .map_err(|err| format!("invalid --m {value:?}: {err}"))?;
                if parsed == 0 {
                    return Err("--m must be greater than zero".to_string());
                }
                m = Some(parsed);
            }
            "--output-dir" => {
                output_dir = Some(PathBuf::from(take_value(&mut values, &flag)?));
            }
            "--help" | "-h" => {
                println!(
                    "usage: sq8_optimized_fixture --artifact-dir DIR --tensor NAME --m M --output-dir DIR"
                );
                return Ok(None);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Some(Args {
        artifact_dir: artifact_dir.ok_or_else(|| "--artifact-dir is required".to_string())?,
        tensor: tensor.ok_or_else(|| "--tensor is required".to_string())?,
        m: m.ok_or_else(|| "--m is required".to_string())?,
        output_dir: output_dir.ok_or_else(|| "--output-dir is required".to_string())?,
    }))
}

fn staging_dir_for(output_dir: &Path) -> Result<PathBuf, String> {
    let file_name = output_dir.file_name().ok_or_else(|| {
        format!(
            "output directory {} has no final component",
            output_dir.display()
        )
    })?;
    let parent = output_dir
        .parent()
        .filter(|path| !path.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let mut staging_name = OsString::from(".");
    staging_name.push(file_name);
    staging_name.push(format!(".tmp-{}", std::process::id()));
    Ok(parent.join(staging_name))
}

fn path_exists(path: &Path) -> Result<bool, String> {
    match fs::symlink_metadata(path) {
        Ok(_) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(format!("failed to inspect {}: {err}", path.display())),
    }
}

fn rename_noreplace(source: &Path, destination: &Path) -> Result<(), String> {
    let source_c = CString::new(source.as_os_str().as_bytes())
        .map_err(|_| format!("source path {} contains NUL", source.display()))?;
    let destination_c = CString::new(destination.as_os_str().as_bytes())
        .map_err(|_| format!("destination path {} contains NUL", destination.display()))?;
    // Linux renameat2 makes the no-clobber check and directory promotion one operation.
    let result = unsafe {
        renameat2(
            AT_FDCWD,
            source_c.as_ptr(),
            AT_FDCWD,
            destination_c.as_ptr(),
            RENAME_NOREPLACE,
        )
    };
    if result == 0 {
        return Ok(());
    }
    Err(format!(
        "failed to publish fixture {} from {} without replacement: {}",
        destination.display(),
        source.display(),
        std::io::Error::last_os_error()
    ))
}

fn checked_fixture_working_bytes(m: usize, n: usize, k: usize) -> Result<usize, String> {
    let activation_elements = m
        .checked_mul(k)
        .ok_or_else(|| format!("activation shape [{m},{k}] overflows usize"))?;
    let output_elements = m
        .checked_mul(n)
        .ok_or_else(|| format!("output shape [{m},{n}] overflows usize"))?;
    let scale_elements = m
        .checked_mul(k.div_ceil(128))
        .ok_or_else(|| "activation scale shape overflows usize".to_string())?;
    let activation_bytes = activation_elements
        .checked_mul(17)
        .ok_or_else(|| "activation working bytes overflow usize".to_string())?;
    let output_bytes = output_elements
        .checked_mul(16)
        .ok_or_else(|| "output working bytes overflow usize".to_string())?;
    let scale_bytes = scale_elements
        .checked_mul(8)
        .ok_or_else(|| "activation scale working bytes overflow usize".to_string())?;
    activation_bytes
        .checked_add(output_bytes)
        .and_then(|bytes| bytes.checked_add(scale_bytes))
        .and_then(|bytes| bytes.checked_add(16 * 1024 * 1024))
        .ok_or_else(|| "fixture working bytes overflow usize".to_string())
}

fn write_synced(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|err| format!("failed to create {}: {err}", path.display()))?;
    file.write_all(bytes)
        .map_err(|err| format!("failed to write {}: {err}", path.display()))?;
    file.sync_all()
        .map_err(|err| format!("failed to sync {}: {err}", path.display()))
}

fn publish_fixture(
    output_dir: &Path,
    activation_f32: &[u8],
    activation_f8: &[u8],
    activation_scales_f32: &[u8],
    oracle_output_f32: &[u8],
    report_json: &[u8],
) -> Result<(), String> {
    if path_exists(output_dir)? {
        return Err(format!(
            "refusing to replace existing fixture directory {}",
            output_dir.display()
        ));
    }
    let staging_dir = staging_dir_for(output_dir)?;
    if let Some(parent) = output_dir.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent).map_err(|err| {
            format!(
                "failed to create fixture parent directory {}: {err}",
                parent.display()
            )
        })?;
    }
    fs::create_dir(&staging_dir).map_err(|err| {
        format!(
            "failed to create staging directory {}: {err}",
            staging_dir.display()
        )
    })?;
    let write_result = (|| {
        write_synced(&staging_dir.join("activation.f32le"), activation_f32)?;
        write_synced(&staging_dir.join("activation.f8"), activation_f8)?;
        write_synced(
            &staging_dir.join("activation_scales.f32le"),
            activation_scales_f32,
        )?;
        write_synced(&staging_dir.join("oracle_output.f32le"), oracle_output_f32)?;
        write_synced(&staging_dir.join("fixture.json"), report_json)?;
        File::open(&staging_dir)
            .and_then(|directory| directory.sync_all())
            .map_err(|err| {
                format!(
                    "failed to sync staging directory {}: {err}",
                    staging_dir.display()
                )
            })?;
        rename_noreplace(&staging_dir, output_dir)?;
        let parent = output_dir
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .unwrap_or_else(|| Path::new("."));
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|err| {
                format!(
                    "failed to sync published fixture parent {}: {err}",
                    parent.display()
                )
            })
    })();
    if write_result.is_err() {
        let _ = fs::remove_dir_all(&staging_dir);
    }
    write_result
}

fn main() -> Result<(), String> {
    let Some(args) = parse_args()? else {
        return Ok(());
    };
    if path_exists(&args.output_dir)? {
        return Err(format!(
            "refusing to replace existing fixture directory {}",
            args.output_dir.display()
        ));
    }

    let artifact = read_sq8_canonical_artifact(&args.artifact_dir)?;
    let pair = artifact.tensor_pair(&args.tensor)?;
    let n = usize::try_from(pair.shape[0])
        .map_err(|_| format!("SQ8 tensor {} N does not fit usize", pair.name))?;
    let k = usize::try_from(pair.shape[1])
        .map_err(|_| format!("SQ8 tensor {} K does not fit usize", pair.name))?;
    let estimated_working_bytes = checked_fixture_working_bytes(args.m, n, k)?;
    if estimated_working_bytes > MAX_FIXTURE_WORKING_BYTES {
        return Err(format!(
            "fixture estimated working set {estimated_working_bytes} exceeds {} bytes",
            MAX_FIXTURE_WORKING_BYTES
        ));
    }
    let input_elements = args
        .m
        .checked_mul(k)
        .ok_or_else(|| format!("activation shape [{},{}] overflows usize", args.m, k))?;
    let activation = sq8_reference_activation(input_elements);
    let quantized = quantize_sq8_dynamic_activation(&activation, args.m, k)?;
    let projection = run_sq8_optimized_reference_projection(&artifact, &args.tensor, &quantized)?;
    if projection.output_rows != args.m || projection.output_cols != n {
        return Err(format!(
            "optimized oracle returned unexpected shape [{},{}], expected [{},{}]",
            projection.output_rows, projection.output_cols, args.m, n
        ));
    }

    let activation_bytes = encode_f32_to_bytes(&activation);
    let activation_scale_bytes = encode_f32_to_bytes(quantized.scales());
    let oracle_output_bytes = encode_f32_to_bytes(&projection.output);
    let hashes = quantized.hashes()?;
    let report = FixtureReport {
        schema_version: SCHEMA_VERSION,
        artifact_dir: artifact.artifact_dir().display().to_string(),
        artifact_schema_version: artifact.manifest().schema_version.clone(),
        artifact_content_sha256: artifact.manifest().integrity.content_sha256.clone(),
        tensor: TensorReport {
            name: pair.name.clone(),
            weight_shape: pair.shape,
            weight_sha256: pair.weight.sha256.clone(),
            weight_scale_shape: pair.scale.shape,
            weight_scale_sha256: pair.scale.sha256.clone(),
            block_shape: pair.scale.block_shape,
        },
        m: args.m,
        n,
        k,
        activation_block_cols: quantized.block_cols(),
        cpu_worker_threads: projection.cpu_worker_threads,
        estimated_working_bytes,
        activation: RawFileReport {
            file: "activation.f32le",
            dtype: "f32-le",
            shape: vec![args.m, k],
            bytes: activation_bytes.len(),
            sha256: hashes.input_f32_le_sha256,
        },
        quantized_activation: RawFileReport {
            file: "activation.f8",
            dtype: "ocp-fp8-e4m3",
            shape: vec![args.m, k],
            bytes: quantized.values().len(),
            sha256: hashes.encoded_bytes_sha256,
        },
        activation_scales: RawFileReport {
            file: "activation_scales.f32le",
            dtype: "f32-le",
            shape: vec![args.m, quantized.blocks_per_row()],
            bytes: activation_scale_bytes.len(),
            sha256: hashes.scales_f32_le_sha256,
        },
        oracle_output: RawFileReport {
            file: "oracle_output.f32le",
            dtype: "f32-le",
            shape: vec![args.m, n],
            bytes: oracle_output_bytes.len(),
            sha256: projection.output_f32_le_sha256()?,
        },
        thresholds: ThresholdReport {
            relative_l2_max: SQ8_OPTIMIZED_RELATIVE_L2_THRESHOLD,
            cosine_min: SQ8_OPTIMIZED_COSINE_THRESHOLD,
            finite_output_required: true,
        },
    };
    let mut report_json = serde_json::to_vec_pretty(&report)
        .map_err(|err| format!("failed to serialize SQ8 optimized fixture report: {err}"))?;
    report_json.push(b'\n');

    publish_fixture(
        &args.output_dir,
        &activation_bytes,
        quantized.values(),
        &activation_scale_bytes,
        &oracle_output_bytes,
        &report_json,
    )?;
    println!(
        "{}",
        String::from_utf8(report_json)
            .map_err(|err| format!("fixture report was not UTF-8: {err}"))?
    );
    Ok(())
}

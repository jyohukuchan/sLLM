//! Deterministic streaming GGUF writing and derived-artifact lock verification.

use crate::gguf::{
    GGUF_ALIGNMENT, GGUF_VERSION, GgufArray, GgufError, GgufTensorType, GgufValue, VerifiedGguf,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufWriter, Read, Seek, Write};
use std::path::{Path, PathBuf};

#[cfg(unix)]
use std::os::unix::fs::{FileExt, OpenOptionsExt};

const GGUF_MAGIC: &[u8; 4] = b"GGUF";
const WRITE_CHUNK_BYTES: usize = 16 * 1024 * 1024;
const MAX_WRITE_METADATA: usize = 16_384;
const MAX_WRITE_TENSORS: usize = 65_536;

#[cfg(unix)]
const O_CLOEXEC: i32 = 0o2000000;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufWriteTensor {
    pub name: String,
    pub source_name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GgufTensorType,
}

impl GgufWriteTensor {
    pub fn byte_length(&self) -> Result<u64, GgufError> {
        if self.name.is_empty() || self.source_name.is_empty() {
            return Err(invalid("write tensor name is empty"));
        }
        if self.dimensions.is_empty() || self.dimensions.len() > 8 {
            return Err(invalid("write tensor dimension count is invalid"));
        }
        if self.dimensions.iter().any(|dimension| *dimension == 0) {
            return Err(invalid("write tensor has a zero dimension"));
        }
        let block_size = self.tensor_type.block_size();
        if self.dimensions[0] % block_size != 0 {
            return Err(invalid(format!(
                "write tensor {} first dimension is not divisible by {block_size}",
                self.name
            )));
        }
        let elements = self
            .dimensions
            .iter()
            .try_fold(1_u64, |product, dimension| {
                product
                    .checked_mul(*dimension)
                    .ok_or_else(|| invalid("write tensor element count overflows"))
            })?;
        elements
            .checked_div(block_size)
            .and_then(|blocks| blocks.checked_mul(self.tensor_type.type_size()))
            .ok_or_else(|| invalid("write tensor byte length overflows"))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct GgufWritePlan {
    pub metadata: BTreeMap<String, GgufValue>,
    pub tensors: Vec<GgufWriteTensor>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GgufWriteReport {
    pub output_path: PathBuf,
    pub size_bytes: u64,
    pub sha256: String,
    pub metadata_sha256: String,
    pub tensor_catalog_sha256: String,
    pub tensor_count: usize,
}

/// Write one deterministic GGUF without retaining a full tensor payload in host memory.
///
/// `read_source` receives the source name, tensor-relative byte offset, and a
/// bounded length. It must return exactly that many verified or converted bytes.
pub fn write_gguf<F>(
    output_path: impl AsRef<Path>,
    plan: &GgufWritePlan,
    mut read_source: F,
) -> Result<GgufWriteReport, GgufError>
where
    F: FnMut(&str, u64, usize) -> Result<Vec<u8>, GgufError>,
{
    validate_write_plan(plan)?;
    let output_path = output_path.as_ref();
    if output_path.exists() {
        return Err(invalid("GGUF output path already exists"));
    }
    let partial_path = partial_path(output_path)?;
    let mut options = OpenOptions::new();
    options.read(true).write(true).create_new(true);
    #[cfg(unix)]
    options.custom_flags(O_CLOEXEC);
    let file = options
        .open(&partial_path)
        .map_err(|error| io_error(&partial_path, error))?;
    let result = write_open_file(&partial_path, file, plan, &mut read_source);
    let report = match result {
        Ok(report) => report,
        Err(error) => {
            let _ = fs::remove_file(&partial_path);
            return Err(error);
        }
    };
    fs::rename(&partial_path, output_path).map_err(|error| io_error(output_path, error))?;
    Ok(GgufWriteReport {
        output_path: output_path.to_path_buf(),
        ..report
    })
}

fn write_open_file<F>(
    path: &Path,
    file: File,
    plan: &GgufWritePlan,
    read_source: &mut F,
) -> Result<GgufWriteReport, GgufError>
where
    F: FnMut(&str, u64, usize) -> Result<Vec<u8>, GgufError>,
{
    let mut tensors = plan.tensors.clone();
    tensors.sort_by(|left, right| left.name.cmp(&right.name));
    let mut offsets = Vec::with_capacity(tensors.len());
    let mut next_offset = 0_u64;
    for tensor in &tensors {
        next_offset = align_up(next_offset, GGUF_ALIGNMENT)?;
        offsets.push(next_offset);
        next_offset = next_offset
            .checked_add(tensor.byte_length()?)
            .ok_or_else(|| invalid("GGUF tensor data size overflows"))?;
    }

    let mut writer = BufWriter::new(file);
    writer
        .write_all(GGUF_MAGIC)
        .map_err(|error| io_error(path, error))?;
    write_u32(&mut writer, GGUF_VERSION, path)?;
    write_u64(&mut writer, tensors.len() as u64, path)?;
    write_u64(&mut writer, plan.metadata.len() as u64, path)?;
    let metadata_start = writer
        .stream_position()
        .map_err(|error| io_error(path, error))?;
    for (key, value) in &plan.metadata {
        write_string(&mut writer, key, path)?;
        write_value(&mut writer, value, path)?;
    }
    let metadata_end = writer
        .stream_position()
        .map_err(|error| io_error(path, error))?;
    let catalog_start = metadata_end;
    for (tensor, relative_offset) in tensors.iter().zip(&offsets) {
        write_string(&mut writer, &tensor.name, path)?;
        write_u32(&mut writer, tensor.dimensions.len() as u32, path)?;
        for dimension in &tensor.dimensions {
            write_u64(&mut writer, *dimension, path)?;
        }
        write_u32(&mut writer, tensor.tensor_type.raw(), path)?;
        write_u64(&mut writer, *relative_offset, path)?;
    }
    let catalog_end = writer
        .stream_position()
        .map_err(|error| io_error(path, error))?;
    let data_offset = if tensors.is_empty() {
        catalog_end
    } else {
        align_up(catalog_end, GGUF_ALIGNMENT)?
    };
    write_padding(&mut writer, data_offset - catalog_end, path)?;

    for ((tensor, relative_offset), next) in tensors.iter().zip(&offsets).zip(
        offsets
            .iter()
            .skip(1)
            .copied()
            .chain(std::iter::once(next_offset)),
    ) {
        let expected = data_offset
            .checked_add(*relative_offset)
            .ok_or_else(|| invalid("GGUF write position overflows"))?;
        let current = writer
            .stream_position()
            .map_err(|error| io_error(path, error))?;
        if current > expected {
            return Err(invalid(
                "GGUF tensor write position exceeded planned offset",
            ));
        }
        write_padding(&mut writer, expected - current, path)?;
        let byte_length = tensor.byte_length()?;
        let mut offset = 0_u64;
        while offset < byte_length {
            let length = usize::try_from((byte_length - offset).min(WRITE_CHUNK_BYTES as u64))
                .map_err(|_| invalid("GGUF write chunk does not fit usize"))?;
            let bytes = read_source(&tensor.source_name, offset, length)?;
            if bytes.len() != length {
                return Err(invalid(format!(
                    "source {} returned a short tensor chunk",
                    tensor.source_name
                )));
            }
            writer
                .write_all(&bytes)
                .map_err(|error| io_error(path, error))?;
            offset += length as u64;
        }
        let padded_end = data_offset
            .checked_add(next)
            .ok_or_else(|| invalid("GGUF padded tensor end overflows"))?;
        let current = writer
            .stream_position()
            .map_err(|error| io_error(path, error))?;
        if current < padded_end {
            write_padding(&mut writer, padded_end - current, path)?;
        }
    }
    writer.flush().map_err(|error| io_error(path, error))?;
    let file = writer
        .into_inner()
        .map_err(|error| io_error(path, error.into_error()))?;
    file.sync_all().map_err(|error| io_error(path, error))?;
    let size_bytes = file
        .metadata()
        .map_err(|error| io_error(path, error))?
        .len();
    let sha256 = sha256_file(path, &file, size_bytes)?;
    let metadata_sha256 = sha256_range(path, &file, metadata_start, metadata_end - metadata_start)?;
    let tensor_catalog_sha256 =
        sha256_range(path, &file, catalog_start, catalog_end - catalog_start)?;
    Ok(GgufWriteReport {
        output_path: path.to_path_buf(),
        size_bytes,
        sha256,
        metadata_sha256,
        tensor_catalog_sha256,
        tensor_count: tensors.len(),
    })
}

fn validate_write_plan(plan: &GgufWritePlan) -> Result<(), GgufError> {
    if plan.metadata.len() > MAX_WRITE_METADATA {
        return Err(invalid("GGUF write metadata count exceeds bound"));
    }
    if plan.tensors.len() > MAX_WRITE_TENSORS {
        return Err(invalid("GGUF write tensor count exceeds bound"));
    }
    let architecture = match plan.metadata.get("general.architecture") {
        Some(GgufValue::String(value)) => value.as_str(),
        _ => return Err(invalid("GGUF write architecture is missing")),
    };
    if !matches!(architecture, "qwen35" | "qwen35moe" | "gemma4") {
        return Err(invalid("GGUF write architecture is unsupported"));
    }
    match plan.metadata.get("general.alignment") {
        None | Some(GgufValue::U32(32)) => {}
        _ => return Err(invalid("GGUF write alignment must be u32 32")),
    }
    let mut names = BTreeSet::new();
    for tensor in &plan.tensors {
        if !names.insert(tensor.name.as_str()) {
            return Err(invalid(format!(
                "duplicate GGUF write tensor {}",
                tensor.name
            )));
        }
        tensor.byte_length()?;
    }
    Ok(())
}

fn write_value(
    writer: &mut BufWriter<File>,
    value: &GgufValue,
    path: &Path,
) -> Result<(), GgufError> {
    macro_rules! scalar {
        ($type_id:expr, $bytes:expr) => {{
            write_u32(writer, $type_id, path)?;
            writer
                .write_all(&$bytes)
                .map_err(|error| io_error(path, error))?;
        }};
    }
    match value {
        GgufValue::U8(value) => scalar!(0, value.to_le_bytes()),
        GgufValue::I8(value) => scalar!(1, value.to_le_bytes()),
        GgufValue::U16(value) => scalar!(2, value.to_le_bytes()),
        GgufValue::I16(value) => scalar!(3, value.to_le_bytes()),
        GgufValue::U32(value) => scalar!(4, value.to_le_bytes()),
        GgufValue::I32(value) => scalar!(5, value.to_le_bytes()),
        GgufValue::F32(value) => {
            if !value.is_finite() {
                return Err(invalid("cannot write non-finite f32 metadata"));
            }
            scalar!(6, value.to_le_bytes());
        }
        GgufValue::Bool(value) => scalar!(7, [u8::from(*value)]),
        GgufValue::String(value) => {
            write_u32(writer, 8, path)?;
            write_string(writer, value, path)?;
        }
        GgufValue::Array(array) => write_array(writer, array, path)?,
        GgufValue::U64(value) => scalar!(10, value.to_le_bytes()),
        GgufValue::I64(value) => scalar!(11, value.to_le_bytes()),
        GgufValue::F64(value) => {
            if !value.is_finite() {
                return Err(invalid("cannot write non-finite f64 metadata"));
            }
            scalar!(12, value.to_le_bytes());
        }
    }
    Ok(())
}

fn write_array(
    writer: &mut BufWriter<File>,
    array: &GgufArray,
    path: &Path,
) -> Result<(), GgufError> {
    write_u32(writer, 9, path)?;
    macro_rules! numeric {
        ($type_id:expr, $values:expr) => {{
            write_u32(writer, $type_id, path)?;
            write_u64(writer, $values.len() as u64, path)?;
            for value in $values {
                writer
                    .write_all(&value.to_le_bytes())
                    .map_err(|error| io_error(path, error))?;
            }
        }};
    }
    match array {
        GgufArray::U8(values) => numeric!(0, values),
        GgufArray::I8(values) => numeric!(1, values),
        GgufArray::U16(values) => numeric!(2, values),
        GgufArray::I16(values) => numeric!(3, values),
        GgufArray::U32(values) => numeric!(4, values),
        GgufArray::I32(values) => numeric!(5, values),
        GgufArray::F32(values) => {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(invalid("cannot write non-finite f32 array metadata"));
            }
            numeric!(6, values);
        }
        GgufArray::Bool(values) => {
            write_u32(writer, 7, path)?;
            write_u64(writer, values.len() as u64, path)?;
            for value in values {
                writer
                    .write_all(&[u8::from(*value)])
                    .map_err(|error| io_error(path, error))?;
            }
        }
        GgufArray::String(values) => {
            write_u32(writer, 8, path)?;
            write_u64(writer, values.len() as u64, path)?;
            for value in values {
                write_string(writer, value, path)?;
            }
        }
        GgufArray::U64(values) => numeric!(10, values),
        GgufArray::I64(values) => numeric!(11, values),
        GgufArray::F64(values) => {
            if values.iter().any(|value| !value.is_finite()) {
                return Err(invalid("cannot write non-finite f64 array metadata"));
            }
            numeric!(12, values);
        }
    }
    Ok(())
}

fn write_string(writer: &mut BufWriter<File>, value: &str, path: &Path) -> Result<(), GgufError> {
    write_u64(writer, value.len() as u64, path)?;
    writer
        .write_all(value.as_bytes())
        .map_err(|error| io_error(path, error))
}

fn write_u32(writer: &mut BufWriter<File>, value: u32, path: &Path) -> Result<(), GgufError> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|error| io_error(path, error))
}

fn write_u64(writer: &mut BufWriter<File>, value: u64, path: &Path) -> Result<(), GgufError> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(|error| io_error(path, error))
}

fn write_padding(writer: &mut BufWriter<File>, length: u64, path: &Path) -> Result<(), GgufError> {
    const ZEROS: [u8; 32] = [0; 32];
    let mut remaining = length;
    while remaining > 0 {
        let chunk = usize::try_from(remaining.min(ZEROS.len() as u64))
            .map_err(|_| invalid("padding length does not fit usize"))?;
        writer
            .write_all(&ZEROS[..chunk])
            .map_err(|error| io_error(path, error))?;
        remaining -= chunk as u64;
    }
    Ok(())
}

fn partial_path(output_path: &Path) -> Result<PathBuf, GgufError> {
    let name = output_path
        .file_name()
        .and_then(|value| value.to_str())
        .ok_or_else(|| invalid("GGUF output file name is not UTF-8"))?;
    Ok(output_path.with_file_name(format!(".{name}.sllm-partial-{}", std::process::id())))
}

fn align_up(value: u64, alignment: u64) -> Result<u64, GgufError> {
    let remainder = value % alignment;
    if remainder == 0 {
        Ok(value)
    } else {
        value
            .checked_add(alignment - remainder)
            .ok_or_else(|| invalid("GGUF alignment overflows"))
    }
}

fn invalid(message: impl Into<String>) -> GgufError {
    GgufError::Invalid(message.into())
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 71
        && value.starts_with("sha256:")
        && value[7..]
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
}

fn io_error(path: &Path, error: impl std::fmt::Display) -> GgufError {
    GgufError::Io {
        path: path.to_path_buf(),
        message: error.to_string(),
    }
}

#[cfg(unix)]
fn read_exact_at(
    path: &Path,
    file: &File,
    mut offset: u64,
    mut output: &mut [u8],
) -> Result<(), GgufError> {
    while !output.is_empty() {
        let read = file
            .read_at(output, offset)
            .map_err(|error| io_error(path, error))?;
        if read == 0 {
            return Err(invalid("truncated GGUF while hashing"));
        }
        offset = offset
            .checked_add(read as u64)
            .ok_or_else(|| invalid("hash offset overflows"))?;
        output = &mut output[read..];
    }
    Ok(())
}

#[cfg(not(unix))]
fn read_exact_at(
    path: &Path,
    file: &File,
    offset: u64,
    output: &mut [u8],
) -> Result<(), GgufError> {
    use std::io::{Read, SeekFrom};
    let mut file = file.try_clone().map_err(|error| io_error(path, error))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| io_error(path, error))?;
    file.read_exact(output)
        .map_err(|error| io_error(path, error))
}

fn sha256_range(path: &Path, file: &File, start: u64, length: u64) -> Result<String, GgufError> {
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1024 * 1024];
    let mut offset = 0_u64;
    while offset < length {
        let chunk = usize::try_from((length - offset).min(buffer.len() as u64))
            .map_err(|_| invalid("hash chunk does not fit usize"))?;
        read_exact_at(path, file, start + offset, &mut buffer[..chunk])?;
        hasher.update(&buffer[..chunk]);
        offset += chunk as u64;
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn sha256_file(path: &Path, file: &File, size: u64) -> Result<String, GgufError> {
    sha256_range(path, file, 0, size)
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedGgufConverter {
    pub repository: String,
    pub commit: String,
    pub arguments: Vec<String>,
    pub effective_config: BTreeMap<String, String>,
    pub environment: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedGgufOutput {
    pub path: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub metadata_sha256: String,
    pub tensor_catalog_sha256: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct DerivedGgufLock {
    pub schema_version: String,
    pub fingerprint: String,
    pub semantic_model_id: String,
    pub source_lock_fingerprints: Vec<String>,
    pub converter: DerivedGgufConverter,
    pub output: DerivedGgufOutput,
}

#[derive(Serialize)]
struct DerivedGgufPayload<'a> {
    schema_version: &'a str,
    semantic_model_id: &'a str,
    source_lock_fingerprints: &'a [String],
    converter: &'a DerivedGgufConverter,
    output: &'a DerivedGgufOutput,
}

impl DerivedGgufLock {
    pub fn new(
        semantic_model_id: String,
        source_lock_fingerprints: Vec<String>,
        converter: DerivedGgufConverter,
        report: &GgufWriteReport,
    ) -> Result<Self, GgufError> {
        let output = DerivedGgufOutput {
            path: report.output_path.display().to_string(),
            size_bytes: report.size_bytes,
            sha256: report.sha256.clone(),
            metadata_sha256: report.metadata_sha256.clone(),
            tensor_catalog_sha256: report.tensor_catalog_sha256.clone(),
        };
        let mut lock = Self {
            schema_version: "derived-gguf-lock-v1".to_owned(),
            fingerprint: String::new(),
            semantic_model_id,
            source_lock_fingerprints,
            converter,
            output,
        };
        lock.fingerprint = lock.compute_fingerprint()?;
        lock.validate()?;
        Ok(lock)
    }

    pub fn parse(bytes: &[u8]) -> Result<Self, GgufError> {
        let lock: Self = serde_json::from_slice(bytes)
            .map_err(|error| invalid(format!("derived GGUF lock JSON: {error}")))?;
        lock.validate()?;
        if serde_json::to_vec(&lock).map_err(|error| invalid(error.to_string()))? != bytes {
            return Err(invalid("derived GGUF lock JSON is not canonical"));
        }
        Ok(lock)
    }

    pub fn canonical_json(&self) -> Result<Vec<u8>, GgufError> {
        self.validate()?;
        serde_json::to_vec(self)
            .map_err(|error| invalid(format!("serialize derived lock: {error}")))
    }

    fn payload(&self) -> DerivedGgufPayload<'_> {
        DerivedGgufPayload {
            schema_version: &self.schema_version,
            semantic_model_id: &self.semantic_model_id,
            source_lock_fingerprints: &self.source_lock_fingerprints,
            converter: &self.converter,
            output: &self.output,
        }
    }

    fn compute_fingerprint(&self) -> Result<String, GgufError> {
        let bytes = serde_json::to_vec(&self.payload())
            .map_err(|error| invalid(format!("serialize derived lock payload: {error}")))?;
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        Ok(format!("sha256:{:x}", hasher.finalize()))
    }

    fn validate(&self) -> Result<(), GgufError> {
        if self.schema_version != "derived-gguf-lock-v1"
            || self.semantic_model_id.is_empty()
            || self.source_lock_fingerprints.is_empty()
            || self.converter.repository.is_empty()
            || self.converter.arguments.is_empty()
            || self.output.path.is_empty()
            || self.output.size_bytes == 0
        {
            return Err(invalid("derived GGUF lock required field is invalid"));
        }
        if !valid_sha256(&self.fingerprint)
            || !self
                .source_lock_fingerprints
                .iter()
                .all(|value| valid_sha256(value))
            || !valid_sha256(&self.output.sha256)
            || !valid_sha256(&self.output.metadata_sha256)
            || !valid_sha256(&self.output.tensor_catalog_sha256)
        {
            return Err(invalid("derived GGUF lock digest is invalid"));
        }
        if self.converter.commit.len() != 40
            || !self
                .converter
                .commit
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        {
            return Err(invalid("derived GGUF converter commit is invalid"));
        }
        let mut unique = BTreeSet::new();
        if !self
            .source_lock_fingerprints
            .iter()
            .all(|value| unique.insert(value))
        {
            return Err(invalid("derived GGUF source fingerprints are duplicated"));
        }
        let computed = self.compute_fingerprint()?;
        if self.fingerprint != computed {
            return Err(invalid(format!(
                "derived GGUF lock fingerprint differs: expected {}, computed {computed}",
                self.fingerprint
            )));
        }
        Ok(())
    }
}

pub struct VerifiedDerivedGguf {
    pub lock: DerivedGgufLock,
    pub gguf: VerifiedGguf,
}

pub fn verify_derived_gguf(
    lock: DerivedGgufLock,
    path: impl AsRef<Path>,
) -> Result<VerifiedDerivedGguf, GgufError> {
    lock.validate()?;
    let gguf = VerifiedGguf::open(path.as_ref())?;
    if gguf.file_size() != lock.output.size_bytes
        || gguf.metadata_sha256() != lock.output.metadata_sha256
        || gguf.tensor_catalog_sha256() != lock.output.tensor_catalog_sha256
        || gguf.file_sha256()? != lock.output.sha256
    {
        return Err(invalid("derived GGUF output identity differs"));
    }
    Ok(VerifiedDerivedGguf { lock, gguf })
}

pub fn read_derived_gguf_lock(path: impl AsRef<Path>) -> Result<DerivedGgufLock, GgufError> {
    const MAX_DERIVED_LOCK_BYTES: u64 = 16 * 1024 * 1024;
    let path = path.as_ref();
    let path_metadata = fs::symlink_metadata(path).map_err(|error| io_error(path, error))?;
    if path_metadata.file_type().is_symlink()
        || !path_metadata.is_file()
        || path_metadata.len() == 0
        || path_metadata.len() > MAX_DERIVED_LOCK_BYTES
    {
        return Err(invalid("derived GGUF lock file contract differs"));
    }
    let mut file = File::open(path).map_err(|error| io_error(path, error))?;
    let opened = file.metadata().map_err(|error| io_error(path, error))?;
    if opened.len() != path_metadata.len() {
        return Err(invalid("derived GGUF lock changed while opening"));
    }
    let mut bytes = Vec::with_capacity(
        usize::try_from(opened.len()).map_err(|_| invalid("derived lock length exceeds usize"))?,
    );
    file.read_to_end(&mut bytes)
        .map_err(|error| io_error(path, error))?;
    if bytes.len() as u64 != opened.len() {
        return Err(invalid("derived GGUF lock returned a short read"));
    }
    DerivedGgufLock::parse(&bytes)
}

impl VerifiedGguf {
    pub fn file_sha256(&self) -> Result<String, GgufError> {
        sha256_file(self.path(), &self.file_for_hash(), self.file_size())
    }

    fn file_for_hash(&self) -> std::sync::Arc<File> {
        self.owned_file()
    }
}

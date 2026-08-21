use std::fs::{self, OpenOptions};
use std::io::Read;
use std::path::Path;

use serde::de::{DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use sllm_frontend::{
    GENERIC_TEMPLATE_MAX_KWARGS_BYTES_V1, GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1,
    GenericTemplateProviderV1,
};

/// Reads one caller-selected template source without exposing its local path
/// in any error.  The file is checked as a regular, non-symlink file, read
/// once through an opened descriptor, and compared against the descriptor's
/// post-read size before MiniJinja parses it.
pub(crate) fn read_verified_template(
    path: &Path,
    digest: &str,
) -> Result<GenericTemplateProviderV1, String> {
    let path_type = fs::symlink_metadata(path)
        .map_err(|_| "custom template file could not be inspected".to_owned())?;
    if !path_type.file_type().is_file() || path_type.file_type().is_symlink() {
        return Err("custom template file must be a regular non-symlink file".to_owned());
    }
    let mut open_options = OpenOptions::new();
    open_options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        open_options.custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    }
    let mut file = open_options
        .open(path)
        .map_err(|_| "custom template file could not be opened".to_owned())?;
    let descriptor_type = file
        .metadata()
        .map_err(|_| "custom template file metadata is unavailable".to_owned())?;
    if !descriptor_type.file_type().is_file() {
        return Err("custom template file must be a regular file".to_owned());
    }
    let expected_size = usize::try_from(descriptor_type.len())
        .map_err(|_| "custom template file size is invalid".to_owned())?;
    if expected_size == 0 || expected_size > GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1 {
        return Err("custom template source exceeds the bounded size".to_owned());
    }
    let mut bytes = Vec::with_capacity(expected_size);
    let max_read = u64::try_from(GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1)
        .ok()
        .and_then(|value| value.checked_add(1))
        .ok_or_else(|| "custom template source size limit is invalid".to_owned())?;
    (&mut file)
        .take(max_read)
        .read_to_end(&mut bytes)
        .map_err(|_| "custom template file could not be read".to_owned())?;
    let actual_size = file
        .metadata()
        .map_err(|_| "custom template file metadata is unavailable".to_owned())?
        .len();
    if bytes.len() > GENERIC_TEMPLATE_MAX_SOURCE_BYTES_V1
        || bytes.len() != expected_size
        || actual_size != descriptor_type.len()
    {
        return Err("custom template file changed while it was being read".to_owned());
    }
    if bytes.contains(&0) {
        return Err("custom template source contains NUL bytes".to_owned());
    }
    GenericTemplateProviderV1::from_bytes(&bytes, digest)
        .map_err(|_| "custom template source failed verification".to_owned())
}

/// Parses the kwargs object with duplicate-key rejection. `serde_json::Value`
/// normally keeps the last duplicate key; that behavior is not acceptable for
/// a digest-bound prompt context.
pub(crate) fn parse_kwargs_json(input: &str) -> Result<Map<String, Value>, String> {
    if input.len() > GENERIC_TEMPLATE_MAX_KWARGS_BYTES_V1 {
        return Err("template kwargs exceed the bounded size".to_owned());
    }
    let mut deserializer = serde_json::Deserializer::from_str(input);
    let value = StrictValue.deserialize(&mut deserializer).map_err(|_| {
        "template kwargs must be a finite JSON object without duplicate keys".to_owned()
    })?;
    deserializer
        .end()
        .map_err(|_| "template kwargs contain trailing data".to_owned())?;
    match value {
        Value::Object(value) => Ok(value),
        _ => Err("template kwargs must be a JSON object".to_owned()),
    }
}

struct StrictValue;

impl<'de> DeserializeSeed<'de> for StrictValue {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictVisitor)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str("a finite JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Value::from(value))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Value::from(value))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        if value.is_finite() {
            Ok(Value::from(value))
        } else {
            Err(E::custom("non-finite number"))
        }
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Value::String(value))
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: serde::de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        StrictValue.deserialize(deserializer)
    }

    fn visit_seq<A>(self, mut access: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = access.next_element_seed(StrictValue)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut access: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut values = Map::new();
        while let Some(key) = access.next_key::<String>()? {
            if values.contains_key(&key) {
                return Err(serde::de::Error::custom("duplicate object key"));
            }
            let value = access.next_value_seed(StrictValue)?;
            values.insert(key, value);
        }
        Ok(Value::Object(values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;
    use std::io::Write;
    use std::os::unix::fs::symlink;
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEQUENCE: AtomicU64 = AtomicU64::new(0);

    #[test]
    fn duplicate_nonfinite_and_nonobject_kwargs_fail_closed() {
        assert!(parse_kwargs_json(r#"{"x":1,"x":2}"#).is_err());
        assert!(parse_kwargs_json(r#"{"x":NaN}"#).is_err());
        assert!(parse_kwargs_json(r#"[1]"#).is_err());
        assert_eq!(
            parse_kwargs_json(r#"{"x":{"y":true}}"#).unwrap()["x"]["y"],
            true
        );
    }

    #[test]
    fn file_reader_rejects_symlink_and_accepts_verified_regular_source() {
        let directory = std::env::temp_dir().join(format!(
            "sllm-cli-template-file-{}-{}",
            std::process::id(),
            SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&directory).unwrap();
        let source = "{{ messages[0].content }}";
        let path = directory.join("template.jinja");
        std::fs::File::create(&path)
            .unwrap()
            .write_all(source.as_bytes())
            .unwrap();
        let digest = format!("sha256:{:x}", sha2::Sha256::digest(source.as_bytes()));
        assert!(read_verified_template(&path, &digest).is_ok());
        let link = directory.join("link.jinja");
        symlink(&path, &link).unwrap();
        assert!(read_verified_template(&link, &digest).is_err());
        let _ = std::fs::remove_dir_all(directory);
    }

    #[test]
    fn file_reader_errors_do_not_disclose_local_paths() {
        let path = std::env::temp_dir().join("sllm-private-template-name.jinja");
        let error = read_verified_template(&path, "sha256:invalid").unwrap_err();
        assert!(!error.contains(path.to_string_lossy().as_ref()));
    }
}

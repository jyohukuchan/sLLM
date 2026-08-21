use serde_json::{Value, json};
use sllm_server::{
    MAX_MODEL_MANIFEST_ARTIFACTS_V1, MAX_MODEL_MANIFEST_BYTES_V1, MAX_MODEL_MANIFEST_MODELS_V1,
    MODEL_MANIFEST_SCHEMA_VERSION_V1, ModelManifestErrorV1, parse_model_manifest_v1,
};
use std::fs::{self, File};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

static NEXT_DIRECTORY: AtomicU64 = AtomicU64::new(0);

struct Fixture {
    root: PathBuf,
    manifest: PathBuf,
    gguf: PathBuf,
    derived_lock: PathBuf,
    adapter_lock: PathBuf,
    adapter_payload: PathBuf,
    control_lock: PathBuf,
    control_payload: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let id = NEXT_DIRECTORY.fetch_add(1, Ordering::Relaxed);
        let root =
            std::env::temp_dir().join(format!("sllm-phase45-manifest-{}-{id}", std::process::id()));
        fs::create_dir(&root).unwrap();
        let fixture = Self {
            manifest: root.join("models.json"),
            gguf: root.join("model.gguf"),
            derived_lock: root.join("derived.lock"),
            adapter_lock: root.join("adapter.lock"),
            adapter_payload: root.join("adapter.payload"),
            control_lock: root.join("control.lock"),
            control_payload: root.join("control.payload"),
            root,
        };
        for path in [
            &fixture.gguf,
            &fixture.derived_lock,
            &fixture.adapter_lock,
            &fixture.adapter_payload,
            &fixture.control_lock,
            &fixture.control_payload,
        ] {
            fs::write(path, b"offline-artifact").unwrap();
        }
        fixture
    }

    fn valid_manifest(&self) -> Value {
        json!({
            "schema_version": MODEL_MANIFEST_SCHEMA_VERSION_V1,
            "models": [{
                "alias": "qwen-a",
                "gguf": self.gguf,
                "derived_lock": self.derived_lock,
                "device_index": 0,
                "target": "gfx1030",
                "declared_resident_bytes": 1,
                "preload": true,
                "adapters": [{
                    "alias": "style-a",
                    "lock": self.adapter_lock,
                    "payload": self.adapter_payload,
                }],
                "control_vectors": [{
                    "alias": "control-a",
                    "lock": self.control_lock,
                    "payload": self.control_payload,
                }]
            }]
        })
    }

    fn write(&self, value: &Value) {
        fs::write(&self.manifest, serde_json::to_vec(value).unwrap()).unwrap();
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.root);
    }
}

#[test]
fn accepts_v1_manifest_and_redacts_paths_from_debug() {
    let fixture = Fixture::new();
    fixture.write(&fixture.valid_manifest());
    let manifest = parse_model_manifest_v1(&fixture.manifest).unwrap();
    assert_eq!(manifest.schema_version(), MODEL_MANIFEST_SCHEMA_VERSION_V1);
    assert_eq!(manifest.models().len(), 1);
    let model = &manifest.models()[0];
    assert_eq!(model.alias(), "qwen-a");
    assert_eq!(model.gguf(), fixture.gguf.as_path());
    assert_eq!(model.derived_lock(), fixture.derived_lock.as_path());
    assert_eq!(model.device_index(), 0);
    assert_eq!(model.target(), "gfx1030");
    assert!(model.preload());
    assert_eq!(model.adapters().len(), 1);
    assert_eq!(model.control_vectors().len(), 1);
    let debug = format!("{manifest:?}");
    assert!(!debug.contains(fixture.adapter_payload.to_string_lossy().as_ref()));
    assert!(!debug.contains("offline-artifact"));
}

#[test]
fn model_alias_limit_and_duplicate_aliases_are_fail_closed() {
    let fixture = Fixture::new();
    let mut value = fixture.valid_manifest();
    value["models"] = Value::Array(
        (0..=MAX_MODEL_MANIFEST_MODELS_V1)
            .map(|index| {
                json!({
                    "alias": format!("m{index}"),
                    "gguf": fixture.gguf,
                    "derived_lock": fixture.derived_lock,
                    "device_index": 0,
                    "target": "gfx1030",
                    "declared_resident_bytes": 1,
                    "preload": false,
                })
            })
            .collect(),
    );
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::AliasLimit)
    );

    value["models"] = json!([
        {
            "alias": "same",
            "gguf": fixture.gguf,
            "derived_lock": fixture.derived_lock,
            "device_index": 0,
            "target": "gfx1030",
            "declared_resident_bytes": 1,
            "preload": false,
        },
        {
            "alias": "same",
            "gguf": fixture.gguf,
            "derived_lock": fixture.derived_lock,
            "device_index": 0,
            "target": "gfx1030",
            "declared_resident_bytes": 1,
            "preload": false,
        }
    ]);
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::DuplicateAlias)
    );
}

#[test]
fn artifact_lists_require_eight_or_fewer_sorted_unique_aliases() {
    let fixture = Fixture::new();
    let mut value = fixture.valid_manifest();
    value["models"][0]["adapters"] = Value::Array(
        (0..=MAX_MODEL_MANIFEST_ARTIFACTS_V1)
            .map(|index| {
                json!({
                    "alias": format!("a{index}"),
                    "lock": fixture.adapter_lock,
                    "payload": fixture.adapter_payload,
                })
            })
            .collect(),
    );
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::ArtifactLimit)
    );

    for aliases in [["b", "a"], ["a", "a"]] {
        value["models"][0]["adapters"] = Value::Array(
            aliases
                .into_iter()
                .map(|alias| {
                    json!({
                        "alias": alias,
                        "lock": fixture.adapter_lock,
                        "payload": fixture.adapter_payload,
                    })
                })
                .collect(),
        );
        fixture.write(&value);
        assert_eq!(
            parse_model_manifest_v1(&fixture.manifest),
            Err(ModelManifestErrorV1::ArtifactOrder)
        );
    }
}

#[test]
fn rejects_unknown_duplicate_and_invalid_schema_values() {
    let fixture = Fixture::new();
    let mut value = fixture.valid_manifest();
    value["models"][0]["alias"] = json!("a".repeat(128));
    value["models"][0]["target"] = json!("gfx1201");
    value["models"][0]["device_index"] = json!(u32::MAX);
    fixture.write(&value);
    let boundary = parse_model_manifest_v1(&fixture.manifest).unwrap();
    assert_eq!(boundary.models()[0].alias().len(), 128);
    assert_eq!(boundary.models()[0].target(), "gfx1201");
    assert_eq!(boundary.models()[0].device_index(), u32::MAX);

    value["models"][0]["alias"] = json!("a".repeat(129));
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    );

    value = fixture.valid_manifest();
    value["credential"] = json!("do-not-store");
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::UnknownField)
    );

    let raw = r#"{"schema_version":"sllm-model-manifest-v1","schema_version":"sllm-model-manifest-v1","models":[]}"#
        .to_owned();
    fs::write(&fixture.manifest, raw).unwrap();
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::DuplicateField)
    );

    let mut value = fixture.valid_manifest();
    value["models"][0]["alias"] = json!("bad/name");
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    );
    value["models"][0]["target"] = json!("https://gfx1030");
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    );
    value["models"][0]["target"] = json!("gfx9999");
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    );
    value["models"][0]["declared_resident_bytes"] = json!(0);
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    );

    value["models"] = json!([]);
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    );
}

#[test]
fn adapter_and_control_aliases_share_a_global_namespace() {
    let fixture = Fixture::new();
    let mut value = fixture.valid_manifest();
    value["models"][0]["control_vectors"][0]["alias"] = json!("style-a");
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::DuplicateAlias)
    );
}

#[test]
fn rejects_relative_traversal_network_nonregular_and_symlink_paths() {
    let fixture = Fixture::new();
    let mut value = fixture.valid_manifest();
    value["models"][0]["gguf"] = json!("relative/model.gguf");
    fixture.write(&value);
    assert!(matches!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    ));

    value["models"][0]["gguf"] = json!(format!("{}/../model.gguf", fixture.root.display()));
    fixture.write(&value);
    assert!(matches!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    ));

    value["models"][0]["gguf"] = json!("https://example.invalid/model.gguf");
    fixture.write(&value);
    assert!(matches!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::InvalidValue)
    ));

    let directory = fixture.root.join("not-a-file");
    fs::create_dir(&directory).unwrap();
    value["models"][0]["gguf"] = json!(directory);
    fixture.write(&value);
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::NotRegularFile)
    );

    #[cfg(unix)]
    {
        let link = fixture.root.join("payload-link");
        std::os::unix::fs::symlink(&fixture.adapter_payload, &link).unwrap();
        value["models"][0]["gguf"] = json!(fixture.gguf);
        value["models"][0]["adapters"][0]["payload"] = json!(link);
        fixture.write(&value);
        assert_eq!(
            parse_model_manifest_v1(&fixture.manifest),
            Err(ModelManifestErrorV1::Symlink)
        );

        let nested = fixture.root.join("nested");
        fs::create_dir(&nested).unwrap();
        let nested_link = fixture.root.join("nested-link");
        std::os::unix::fs::symlink(&nested, &nested_link).unwrap();
        let nested_manifest = nested_link.join("models.json");
        fs::write(
            nested.join("models.json"),
            serde_json::to_vec(&fixture.valid_manifest()).unwrap(),
        )
        .unwrap();
        assert_eq!(
            parse_model_manifest_v1(&nested_manifest),
            Err(ModelManifestErrorV1::Symlink)
        );

        let manifest_link = fixture.root.join("manifest-link.json");
        std::os::unix::fs::symlink(&fixture.manifest, &manifest_link).unwrap();
        assert_eq!(
            parse_model_manifest_v1(&manifest_link),
            Err(ModelManifestErrorV1::Symlink)
        );
    }
}

#[test]
fn manifest_size_limit_is_checked_before_json_parse() {
    let fixture = Fixture::new();
    let mut file = File::create(&fixture.manifest).unwrap();
    let bytes = vec![b' '; MAX_MODEL_MANIFEST_BYTES_V1 + 1];
    std::io::Write::write_all(&mut file, &bytes).unwrap();
    assert_eq!(
        parse_model_manifest_v1(&fixture.manifest),
        Err(ModelManifestErrorV1::TooLarge)
    );
}

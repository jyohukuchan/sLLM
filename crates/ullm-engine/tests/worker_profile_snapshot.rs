// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

use ullm_engine::sq8_worker_protocol::{
    Sq8OrderedJsonlWriter, Sq8WorkerEvent, Sq8WorkerProfile, inspect_sq8_worker_command,
};

fn profile() -> Sq8WorkerProfile {
    Sq8WorkerProfile {
        model: "snapshot-model".into(),
        model_revision: "snapshot-revision".into(),
        artifact_content_sha256: "a".repeat(64),
        package_manifest_sha256: "b".repeat(64),
        device: "snapshot-device".into(),
        execution_profile: "snapshot-execution".into(),
        context_length: 8,
        max_new_tokens: 4,
        vocab_size: 32,
        eos_token_ids: vec![2, 3],
        top_k: 1,
    }
}

#[test]
fn ready_json_is_derived_from_the_explicit_profile_snapshot() {
    let profile = profile();
    let mut writer = Sq8OrderedJsonlWriter::with_profile(Vec::new(), profile.clone());
    writer
        .write_ready_event(&Sq8WorkerEvent::ready_with_profile(&profile))
        .unwrap();
    let json = String::from_utf8(writer.into_inner()).unwrap();
    assert_eq!(
        json,
        format!(
            "{{\"type\":\"ready\",\"schema_version\":\"ullm.worker.v1\",\"model\":\"snapshot-model\",\"model_revision\":\"snapshot-revision\",\"artifact_content_sha256\":\"{}\",\"package_manifest_sha256\":\"{}\",\"device\":\"snapshot-device\",\"execution_profile\":\"snapshot-execution\",\"context_length\":8,\"max_new_tokens\":4}}\n",
            "a".repeat(64),
            "b".repeat(64),
        )
    );
}

#[test]
fn command_decode_and_request_validation_share_the_explicit_snapshot() {
    let payload = br#"{"schema_version":"ullm.worker.v1","type":"generate","request_id":"req-1","prompt_token_ids":[1,4],"max_new_tokens":2,"sampling":{"temperature":0.0,"top_p":1.0,"top_k":1,"seed":7},"eos_token_ids":[2,3]}"#;
    let profile = profile();
    let command = inspect_sq8_worker_command(payload)
        .unwrap()
        .decode_with_profile(&profile)
        .unwrap();
    let ullm_engine::sq8_worker_protocol::Sq8WorkerCommand::Generate(generate) = command else {
        panic!("expected generate command");
    };
    let request = generate
        .into_serving_request_with_profile(&profile)
        .unwrap();
    assert_eq!(request.prompt_token_ids, vec![1, 4]);
    assert_eq!(request.eos_token_ids, vec![2, 3]);
}

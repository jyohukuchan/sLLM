use ullm_engine::qwen35_aq4_direct_trace::{
    Qwen35Aq4DirectTraceBinding, Qwen35Aq4DirectTraceCollector, Qwen35Aq4DirectTraceRoute,
};

fn main() -> Result<(), String> {
    let mut collector = Qwen35Aq4DirectTraceCollector::default();
    collector.begin_request(Qwen35Aq4DirectTraceBinding {
        side: "candidate".into(),
        binding_kind: "run".into(),
        binding_id: "run-1".into(),
        request_id: "request-1".into(),
        implementation_id: "qwen35-aq4-direct-v1".into(),
        source_id: "runtime-route-apply".into(),
        source_sha256: "a".repeat(64),
        case_id: "case-1".into(),
        case_sha256: "b".repeat(64),
        identity_sha256: "c".repeat(64),
        direct_sequence_output_enabled: true,
    })?;
    collector.record_invocation(Qwen35Aq4DirectTraceRoute::Direct, 512, 2, 4096)?;
    let observation = collector.finish_request("completed")?;
    let raw = observation.to_json_bytes()?;
    print!(
        "{}",
        String::from_utf8(raw).map_err(|error| error.to_string())?
    );
    Ok(())
}

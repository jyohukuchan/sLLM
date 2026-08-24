use std::time::Instant;

use serde_json::{Value, json};
use sllm_core::AllocationSnapshot;

pub(crate) const RENDER_TOKENIZE_BENCHMARK_SCHEMA_VERSION: &str = "engine-performance-render-v1";

pub(crate) const DIRECT_BENCHMARK_SCHEMA_VERSION: &str = "engine-performance-direct-v2";

// Keep the historical default for code paths and host fixtures that model the
// pre-tokenized lane. The model layer emits the lane-specific schema version.
#[allow(dead_code)]
pub(crate) const BENCHMARK_SCHEMA_VERSION: &str = DIRECT_BENCHMARK_SCHEMA_VERSION;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BenchmarkEvent {
    PrefillSubmit,
    PrefillComplete,
    FirstToken,
    LaterToken,
    Stop,
    Cleanup,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct MonotonicClock {
    origin: Instant,
}

impl MonotonicClock {
    pub(crate) fn start() -> Self {
        Self {
            origin: Instant::now(),
        }
    }

    pub(crate) fn now_ns(self) -> u64 {
        u64::try_from(self.origin.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct BenchmarkTiming {
    clock: MonotonicClock,
}

impl BenchmarkTiming {
    pub(crate) fn start() -> Self {
        Self {
            clock: MonotonicClock::start(),
        }
    }

    pub(crate) fn model_load_start_ns(self) -> u64 {
        0
    }

    pub(crate) fn now_ns(self) -> u64 {
        self.clock.now_ns()
    }

    pub(crate) fn request_clock(self) -> MonotonicClock {
        self.clock
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct FakeClock {
    now_ns: u64,
}

#[cfg(test)]
impl FakeClock {
    pub(crate) const fn new() -> Self {
        Self { now_ns: 0 }
    }

    #[cfg(test)]
    pub(crate) const fn at(self) -> u64 {
        self.now_ns
    }

    #[cfg(test)]
    pub(crate) fn advance(&mut self, delta_ns: u64) -> Result<u64, String> {
        self.now_ns = self
            .now_ns
            .checked_add(delta_ns)
            .ok_or_else(|| "fake clock overflowed".to_owned())?;
        Ok(self.now_ns)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BenchmarkTimeline {
    request_start_ns: u64,
    prefill_submit_ns: Option<u64>,
    prefill_complete_ns: Option<u64>,
    first_token_ns: Option<u64>,
    later_token_publications_ns: Vec<u64>,
    stop_ns: Option<u64>,
    cleanup_ns: Option<u64>,
    last_event_ns: u64,
}

pub(crate) struct BenchmarkSampleInput<'a> {
    pub(crate) input_token_ids: &'a [u32],
    pub(crate) generated_token_ids: &'a [u32],
    pub(crate) visible_token_ids: &'a [u32],
    pub(crate) decode_input_token_ids: &'a [u32],
    pub(crate) stop: Value,
    pub(crate) audit: Value,
    pub(crate) memory: Value,
    pub(crate) cleanup: Value,
}

impl BenchmarkTimeline {
    pub(crate) fn new(request_start_ns: u64) -> Self {
        Self {
            request_start_ns,
            prefill_submit_ns: None,
            prefill_complete_ns: None,
            first_token_ns: None,
            later_token_publications_ns: Vec::new(),
            stop_ns: None,
            cleanup_ns: None,
            last_event_ns: request_start_ns,
        }
    }

    pub(crate) fn record(
        &mut self,
        event: BenchmarkEvent,
        timestamp_ns: u64,
    ) -> Result<(), String> {
        if timestamp_ns < self.request_start_ns || timestamp_ns < self.last_event_ns {
            return Err(
                "benchmark event timestamps must be monotonic and relative to request start"
                    .to_owned(),
            );
        }
        match event {
            BenchmarkEvent::PrefillSubmit if self.prefill_submit_ns.is_none() => {
                self.prefill_submit_ns = Some(timestamp_ns);
            }
            BenchmarkEvent::PrefillComplete
                if self.prefill_submit_ns.is_some() && self.prefill_complete_ns.is_none() =>
            {
                self.prefill_complete_ns = Some(timestamp_ns);
            }
            BenchmarkEvent::FirstToken
                if self.prefill_complete_ns.is_some() && self.first_token_ns.is_none() =>
            {
                self.first_token_ns = Some(timestamp_ns);
            }
            BenchmarkEvent::LaterToken
                if self.first_token_ns.is_some() && self.stop_ns.is_none() =>
            {
                self.later_token_publications_ns.push(timestamp_ns);
            }
            BenchmarkEvent::Stop
                if self.first_token_ns.is_some()
                    && self.stop_ns.is_none()
                    && timestamp_ns
                        >= self
                            .later_token_publications_ns
                            .last()
                            .copied()
                            .unwrap_or(self.first_token_ns.unwrap_or(timestamp_ns)) =>
            {
                self.stop_ns = Some(timestamp_ns);
            }
            BenchmarkEvent::Cleanup if self.stop_ns.is_some() && self.cleanup_ns.is_none() => {
                self.cleanup_ns = Some(timestamp_ns);
            }
            _ => {
                return Err(format!(
                    "benchmark event {event:?} is missing, duplicated, or out of order"
                ));
            }
        }
        self.last_event_ns = timestamp_ns;
        Ok(())
    }

    pub(crate) fn finish(self, sample: BenchmarkSampleInput<'_>) -> Result<Value, String> {
        let BenchmarkSampleInput {
            input_token_ids,
            generated_token_ids,
            visible_token_ids,
            decode_input_token_ids,
            stop,
            audit,
            memory,
            cleanup,
        } = sample;
        let prefill_submit_ns = self
            .prefill_submit_ns
            .ok_or_else(|| "benchmark sample is missing prefill_submit".to_owned())?;
        let prefill_complete_ns = self
            .prefill_complete_ns
            .ok_or_else(|| "benchmark sample is missing prefill_complete".to_owned())?;
        let first_token_ns = self
            .first_token_ns
            .ok_or_else(|| "benchmark sample is missing first_token".to_owned())?;
        let stop_ns = self
            .stop_ns
            .ok_or_else(|| "benchmark sample is missing stop".to_owned())?;
        let cleanup_ns = self
            .cleanup_ns
            .ok_or_else(|| "benchmark sample is missing cleanup".to_owned())?;
        if generated_token_ids.is_empty()
            || generated_token_ids.len() != self.later_token_publications_ns.len() + 1
        {
            return Err(
                "benchmark token publication count does not match generated tokens".to_owned(),
            );
        }

        let ttft_ns = checked_sub(first_token_ns, self.request_start_ns, "TTFT")?;
        let prefill_ns = checked_sub(prefill_complete_ns, prefill_submit_ns, "prefill")?;
        let e2e_ns = checked_sub(cleanup_ns, self.request_start_ns, "E2E")?;
        let prefill_tokens = u64::try_from(input_token_ids.len())
            .map_err(|_| "prefill token count overflowed".to_owned())?;
        let prefill_tokens_per_second = if prefill_ns != 0 {
            Some(
                prefill_tokens
                    .checked_mul(1_000_000_000)
                    .ok_or_else(|| "prefill token/s arithmetic overflowed".to_owned())?
                    as f64
                    / prefill_ns as f64,
            )
        } else {
            None
        };
        let mut tpot_ns = Vec::with_capacity(self.later_token_publications_ns.len());
        let mut previous = first_token_ns;
        for publication in &self.later_token_publications_ns {
            tpot_ns.push(checked_sub(*publication, previous, "TPOT")?);
            previous = *publication;
        }
        let decode_ns = self
            .later_token_publications_ns
            .last()
            .copied()
            .map(|last| checked_sub(last, first_token_ns, "decode"))
            .transpose()?;
        let decode_tokens = u64::try_from(generated_token_ids.len() - 1)
            .map_err(|_| "decode token count overflowed".to_owned())?;
        let decode_tokens_per_second = match decode_ns {
            Some(duration) if duration != 0 => Some(
                decode_tokens
                    .checked_mul(1_000_000_000)
                    .ok_or_else(|| "decode token/s arithmetic overflowed".to_owned())?
                    as f64
                    / duration as f64,
            ),
            Some(_) | None => None,
        };

        Ok(json!({
            "execution_path": "timed-production",
            "timing_instrumentation": "on",
            "events": {
                "request_start_ns": self.request_start_ns,
                "prefill_submit_ns": prefill_submit_ns,
                "prefill_complete_ns": prefill_complete_ns,
                "first_token_ns": first_token_ns,
                "later_token_publications_ns": self.later_token_publications_ns,
                "stop_ns": stop_ns,
                "cleanup_ns": cleanup_ns,
            },
            "derived": {
                "ttft_ns": ttft_ns,
                "prefill_ns": prefill_ns,
                "prefill_tokens_per_second": prefill_tokens_per_second,
                "e2e_ns": e2e_ns,
                "tpot_ns": tpot_ns,
                "decode_tokens": decode_tokens,
                "decode_tokens_per_second": decode_tokens_per_second,
            },
            "tokens": {
                "input_token_ids": input_token_ids,
                "generated_token_ids": generated_token_ids,
                "visible_token_ids": visible_token_ids,
                "decode_input_token_ids": decode_input_token_ids,
            },
            "stop": stop,
            "audit": audit,
            "memory": memory,
            "cleanup": cleanup,
        }))
    }
}

pub(crate) fn checked_sub(end: u64, start: u64, label: &str) -> Result<u64, String> {
    end.checked_sub(start)
        .ok_or_else(|| format!("benchmark {label} duration underflowed"))
}

pub(crate) fn validate_sample_count(warmups: u32, measured: u32) -> Result<(), String> {
    if measured == 0 {
        return Err("benchmark measured sample count must be non-zero".to_owned());
    }
    if warmups > 100 || measured > 100 {
        return Err("benchmark warmups and measured counts must be at most 100".to_owned());
    }
    Ok(())
}

pub(crate) fn validate_fixed_input_token_ids(
    seed_input_token_ids: &[u32],
    input_token_ids: &[u32],
) -> Result<(), String> {
    if input_token_ids.is_empty() || input_token_ids != seed_input_token_ids {
        return Err("benchmark request tokenization changed the fixed input token IDs".to_owned());
    }
    Ok(())
}

pub(crate) fn allocation_snapshot_value(snapshot: AllocationSnapshot) -> Value {
    fn category(value: sllm_core::AllocationCategorySnapshot) -> Value {
        json!({
            "current_bytes": value.current_bytes(),
            "high_water_bytes": value.high_water_bytes(),
        })
    }
    json!({
        "model_resident": category(snapshot.model_resident()),
        "request_state": category(snapshot.request_state()),
        "workspace": category(snapshot.workspace()),
        "current_bytes": snapshot.current_bytes(),
        "high_water_bytes": snapshot.high_water_bytes(),
        "poisoned": snapshot.poisoned(),
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct SnapshotAccountingValues {
    model_current_bytes: u64,
    model_high_water_bytes: u64,
    request_current_bytes: u64,
    workspace_current_bytes: u64,
    total_current_bytes: u64,
    total_high_water_bytes: u64,
}

fn snapshot_accounting_values(
    snapshot: &Value,
    label: &str,
) -> Result<SnapshotAccountingValues, String> {
    let object = snapshot
        .as_object()
        .ok_or_else(|| format!("{label} allocation snapshot is not an object"))?;
    if object.get("poisoned").and_then(Value::as_bool) != Some(false) {
        return Err(format!(
            "{label} allocation accounting is poisoned or missing"
        ));
    }
    let category = |name: &str| -> Result<(u64, u64), String> {
        let category = object
            .get(name)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{label}.{name} allocation category is not an object"))?;
        let current = category
            .get("current_bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{label}.{name}.current_bytes is not an unsigned integer"))?;
        let high_water = category
            .get("high_water_bytes")
            .and_then(Value::as_u64)
            .ok_or_else(|| format!("{label}.{name}.high_water_bytes is not an unsigned integer"))?;
        if high_water < current {
            return Err(format!(
                "{label}.{name} high-water bytes are below current bytes"
            ));
        }
        Ok((current, high_water))
    };
    let (model_current_bytes, model_high_water_bytes) = category("model_resident")?;
    let (request_current_bytes, _) = category("request_state")?;
    let (workspace_current_bytes, _) = category("workspace")?;
    let total_current_bytes = object
        .get("current_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.current_bytes is not an unsigned integer"))?;
    let total_high_water_bytes = object
        .get("high_water_bytes")
        .and_then(Value::as_u64)
        .ok_or_else(|| format!("{label}.high_water_bytes is not an unsigned integer"))?;
    if total_high_water_bytes < total_current_bytes {
        return Err(format!(
            "{label} total high-water bytes are below total current bytes"
        ));
    }
    let category_sum = model_current_bytes
        .checked_add(request_current_bytes)
        .and_then(|sum| sum.checked_add(workspace_current_bytes))
        .ok_or_else(|| format!("{label} allocation category sum overflowed"))?;
    if category_sum != total_current_bytes {
        return Err(format!(
            "{label} total current bytes do not equal the category current sum"
        ));
    }
    Ok(SnapshotAccountingValues {
        model_current_bytes,
        model_high_water_bytes,
        request_current_bytes,
        workspace_current_bytes,
        total_current_bytes,
        total_high_water_bytes,
    })
}

pub(crate) fn validate_snapshot_accounting(snapshot: &Value, label: &str) -> Result<(), String> {
    snapshot_accounting_values(snapshot, label).map(|_| ())
}

pub(crate) fn validate_model_ready_snapshot(snapshot: &Value) -> Result<u64, String> {
    let values = snapshot_accounting_values(snapshot, "model-ready memory")?;
    if values.request_current_bytes != 0 || values.workspace_current_bytes != 0 {
        return Err(
            "model-ready memory has non-zero request-state or workspace current bytes".to_owned(),
        );
    }
    if values.model_current_bytes == 0 || values.model_high_water_bytes == 0 {
        return Err("model-ready memory has no resident model allocation".to_owned());
    }
    if values.total_high_water_bytes < values.model_high_water_bytes {
        return Err(
            "model-ready total high-water bytes are below model-resident high-water bytes"
                .to_owned(),
        );
    }
    Ok(values.model_high_water_bytes)
}

pub(crate) fn validate_request_cleanup_snapshot(
    snapshot: &Value,
    ready_model_current_bytes: u64,
) -> Result<(), String> {
    let values = snapshot_accounting_values(snapshot, "request cleanup memory")?;
    if values.request_current_bytes != 0 || values.workspace_current_bytes != 0 {
        return Err(
            "request cleanup left non-zero request-state or workspace current bytes".to_owned(),
        );
    }
    if values.model_current_bytes != ready_model_current_bytes {
        return Err("request cleanup changed the model-resident current allocation".to_owned());
    }
    Ok(())
}

pub(crate) fn validate_resident_drop_snapshot(snapshot: &Value) -> Result<(), String> {
    let values = snapshot_accounting_values(snapshot, "resident cleanup memory")?;
    if values.model_current_bytes != 0
        || values.request_current_bytes != 0
        || values.workspace_current_bytes != 0
        || values.total_current_bytes != 0
    {
        return Err("resident cleanup left non-zero current allocation bytes".to_owned());
    }
    Ok(())
}

pub(crate) fn validate_peak_vram_snapshot(
    snapshot: &Value,
    model_resident_high_water_bytes: u64,
) -> Result<(), String> {
    let values = snapshot_accounting_values(snapshot, "resident cleanup memory")?;
    if values.total_high_water_bytes < model_resident_high_water_bytes {
        return Err(
            "runtime allocator total high-water bytes are below model-resident high-water bytes"
                .to_owned(),
        );
    }
    Ok(())
}

const CONTROL_TOKEN_FIELDS: [&str; 4] = [
    "input_token_ids",
    "generated_token_ids",
    "visible_token_ids",
    "decode_input_token_ids",
];
const CONTROL_STOP_FIELDS: [&str; 4] = ["version", "reason_version", "kind", "token_id"];
const CONTROL_DISPATCH_FIELDS: [&str; 11] = [
    "selected_backend",
    "target",
    "device_index",
    "model_fingerprint",
    "plan_digest",
    "fallback_used",
    "all_dispatches_hip",
    "submission_count",
    "kernel_dispatch_count",
    "segment_count",
    "boundary_count",
];

pub(crate) fn control_comparison_contract() -> Value {
    json!({
        "mode": "exact",
        "scope": "every_warmup_and_measured_sample",
        "token_fields": CONTROL_TOKEN_FIELDS,
        "stop_fields": CONTROL_STOP_FIELDS,
        "dispatch_fields": CONTROL_DISPATCH_FIELDS,
        "dispatch_count_rule": "exact_when_token_and_stop_fields_match",
    })
}

pub(crate) fn compare_control_sample(control: &Value, sample: &Value) -> Result<(), String> {
    let compare_fields = |section: &str, fields: &[&str]| -> Result<(), String> {
        let control_section = control
            .get(section)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("correctness control {section} is not an object"))?;
        let sample_section = sample
            .get(section)
            .and_then(Value::as_object)
            .ok_or_else(|| format!("timed sample {section} is not an object"))?;
        for field in fields {
            if control_section.get(*field) != sample_section.get(*field) {
                return Err(format!(
                    "timed sample {section}.{field} differs from correctness control"
                ));
            }
        }
        Ok(())
    };
    compare_fields("tokens", &CONTROL_TOKEN_FIELDS)?;
    compare_fields("stop", &CONTROL_STOP_FIELDS)?;
    compare_fields("audit", &CONTROL_DISPATCH_FIELDS)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(timeline: BenchmarkTimeline) -> Value {
        timeline
            .finish(BenchmarkSampleInput {
                input_token_ids: &[1, 3, 17],
                generated_token_ids: &[7, 8, 9],
                visible_token_ids: &[7, 8, 9],
                decode_input_token_ids: &[7, 8],
                stop: json!({"kind": "max_new_tokens"}),
                audit: json!({"all_dispatches_hip": true}),
                memory: json!({"current_bytes": 1}),
                cleanup: json!({"retryable_cleanup": 0, "durable_quarantine": 0}),
            })
            .expect("synthetic benchmark sample")
    }

    #[test]
    fn fake_clock_and_event_arithmetic_are_checked() {
        let mut clock = FakeClock::new();
        assert_eq!(clock.advance(10).unwrap(), 10);
        assert_eq!(clock.at(), 10);
        let mut timeline = BenchmarkTimeline::new(0);
        for (event, delta) in [
            (BenchmarkEvent::PrefillSubmit, 10),
            (BenchmarkEvent::PrefillComplete, 20),
            (BenchmarkEvent::FirstToken, 30),
            (BenchmarkEvent::LaterToken, 50),
            (BenchmarkEvent::LaterToken, 70),
            (BenchmarkEvent::Stop, 80),
            (BenchmarkEvent::Cleanup, 100),
        ] {
            timeline.record(event, delta).unwrap();
        }
        let value = sample(timeline);
        assert_eq!(value["derived"]["ttft_ns"], 30);
        assert_eq!(value["derived"]["prefill_ns"], 10);
        assert_eq!(value["derived"]["e2e_ns"], 100);
        assert_eq!(value["derived"]["tpot_ns"], json!([20, 20]));
        assert!(clock.advance(u64::MAX).is_err());
    }

    #[test]
    fn ordering_and_missing_events_fail_closed() {
        let mut timeline = BenchmarkTimeline::new(10);
        assert!(timeline.record(BenchmarkEvent::FirstToken, 11).is_err());
        assert!(timeline.record(BenchmarkEvent::PrefillSubmit, 12).is_ok());
        assert!(
            timeline
                .record(BenchmarkEvent::PrefillComplete, 11)
                .is_err()
        );
        assert!(
            timeline
                .finish(BenchmarkSampleInput {
                    input_token_ids: &[1],
                    generated_token_ids: &[7],
                    visible_token_ids: &[7],
                    decode_input_token_ids: &[],
                    stop: json!({}),
                    audit: json!({}),
                    memory: json!({}),
                    cleanup: json!({}),
                })
                .is_err()
        );
        assert!(checked_sub(1, 2, "test").is_err());
    }

    #[test]
    fn sample_zero_is_rejected() {
        assert!(validate_sample_count(3, 0).is_err());
        assert!(validate_sample_count(3, 10).is_ok());
        assert!(validate_sample_count(101, 1).is_err());
    }

    #[test]
    fn fixed_input_validation_requires_exact_token_ids() {
        assert!(validate_fixed_input_token_ids(&[1, 3, 17], &[1, 3, 17]).is_ok());
        assert!(validate_fixed_input_token_ids(&[1, 3, 17], &[1, 3, 18]).is_err());
        assert!(validate_fixed_input_token_ids(&[1, 3], &[1, 3, 17]).is_err());
        assert!(validate_fixed_input_token_ids(&[1, 3, 17], &[]).is_err());
    }

    #[test]
    fn cli_lane_timing_starts_before_render_and_tokenize() {
        let mut timeline = BenchmarkTimeline::new(100);
        timeline.record(BenchmarkEvent::PrefillSubmit, 500).unwrap();
        timeline
            .record(BenchmarkEvent::PrefillComplete, 700)
            .unwrap();
        timeline.record(BenchmarkEvent::FirstToken, 800).unwrap();
        timeline.record(BenchmarkEvent::LaterToken, 850).unwrap();
        timeline.record(BenchmarkEvent::LaterToken, 875).unwrap();
        timeline.record(BenchmarkEvent::Stop, 900).unwrap();
        timeline.record(BenchmarkEvent::Cleanup, 1_000).unwrap();
        let value = sample(timeline);
        assert_eq!(value["derived"]["ttft_ns"], 700);
        assert_eq!(value["derived"]["e2e_ns"], 900);
    }

    fn snapshot(
        model_current: u64,
        model_high_water: u64,
        request_current: u64,
        workspace_current: u64,
        total_current: u64,
        total_high_water: u64,
        poisoned: bool,
    ) -> Value {
        json!({
            "model_resident": {
                "current_bytes": model_current,
                "high_water_bytes": model_high_water,
            },
            "request_state": {
                "current_bytes": request_current,
                "high_water_bytes": request_current,
            },
            "workspace": {
                "current_bytes": workspace_current,
                "high_water_bytes": workspace_current,
            },
            "current_bytes": total_current,
            "high_water_bytes": total_high_water,
            "poisoned": poisoned,
        })
    }

    #[test]
    fn snapshot_validation_enforces_ready_request_and_resident_lifecycles() {
        let ready = snapshot(100, 120, 0, 0, 100, 120, false);
        assert_eq!(validate_model_ready_snapshot(&ready), Ok(120));
        assert!(validate_snapshot_accounting(&ready, "test").is_ok());

        let request_cleanup = snapshot(100, 120, 0, 0, 100, 140, false);
        assert!(validate_request_cleanup_snapshot(&request_cleanup, 100).is_ok());

        let resident_drop = snapshot(0, 120, 0, 0, 0, 140, false);
        assert!(validate_resident_drop_snapshot(&resident_drop).is_ok());
        assert!(validate_peak_vram_snapshot(&resident_drop, 120).is_ok());

        for invalid in [
            snapshot(100, 120, 1, 0, 101, 120, false),
            snapshot(100, 120, 0, 0, 99, 120, false),
            snapshot(100, 120, 0, 0, 100, 120, true),
            snapshot(100, 120, 0, 0, 100, 119, false),
        ] {
            assert!(validate_model_ready_snapshot(&invalid).is_err());
        }
        assert!(
            validate_request_cleanup_snapshot(&snapshot(101, 120, 0, 0, 101, 140, false), 100)
                .is_err()
        );
        assert!(validate_resident_drop_snapshot(&snapshot(1, 120, 0, 0, 1, 140, false)).is_err());
        assert!(validate_peak_vram_snapshot(&snapshot(0, 120, 0, 0, 0, 119, false), 120).is_err());
    }

    #[test]
    fn control_comparison_rejects_all_semantic_and_dispatch_differences() {
        let control = json!({
            "tokens": {
                "input_token_ids": [1, 3, 17],
                "generated_token_ids": [7, 8, 248044],
                "visible_token_ids": [7, 8],
                "decode_input_token_ids": [7, 8],
            },
            "stop": {"version": 1, "reason_version": 1, "kind": "stop_token", "token_id": 248044},
            "audit": {
                "selected_backend": "hip",
                "target": "gfx1030",
                "device_index": 0,
                "model_fingerprint": "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "plan_digest": "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                "fallback_used": false,
                "all_dispatches_hip": true,
                "submission_count": 12,
                "kernel_dispatch_count": 12,
                "segment_count": 3,
                "boundary_count": 4,
            },
        });
        assert!(compare_control_sample(&control, &control).is_ok());
        for (section, field, replacement) in [
            ("tokens", "input_token_ids", json!([1, 3, 18])),
            ("tokens", "visible_token_ids", json!([7])),
            ("stop", "token_id", json!(248046)),
            ("audit", "target", json!("gfx1201")),
            ("audit", "kernel_dispatch_count", json!(13)),
            ("audit", "boundary_count", json!(5)),
        ] {
            let mut sample = control.clone();
            sample[section][field] = replacement;
            assert!(compare_control_sample(&control, &sample).is_err());
        }
    }
}

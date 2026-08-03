use std::time::Duration;

use ullm_hip::{EVIDENCE_CASE_SIZES, Status, expected_evidence_output, run_evidence};

fn main() {
    let mut timeout_ms = 1_000_u32;
    let mut args = std::env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--timeout-ms" {
            let Some(value) = args.next() else {
                eprintln!("--timeout-ms requires an integer");
                std::process::exit(2);
            };
            timeout_ms = match value.parse::<u32>() {
                Ok(value) => value,
                Err(_) => {
                    eprintln!("--timeout-ms must be an integer");
                    std::process::exit(2);
                }
            };
        } else {
            eprintln!("unknown argument: {argument}");
            std::process::exit(2);
        }
    }

    let mut rows = Vec::with_capacity(EVIDENCE_CASE_SIZES.len());
    let mut total_allocation_count = 0_u64;
    let mut total_copy_count = 0_u64;
    let mut total_dispatch_count = 0_u64;
    for size in EVIDENCE_CASE_SIZES {
        let input: Vec<u8> = (0..size)
            .map(|index| ((size.wrapping_add(index * 31)) & 0xff) as u8)
            .collect();
        match run_evidence(&input, Duration::from_millis(u64::from(timeout_ms))) {
            Ok((output, report)) => {
                if output != expected_evidence_output(&input) {
                    emit_failure("FAIL", "byte-exact contract failed", &rows);
                }
                total_allocation_count =
                    match total_allocation_count.checked_add(report.allocation_count) {
                        Some(total) => total,
                        None => emit_failure("FAIL", "allocation count overflow", &rows),
                    };
                total_copy_count = match total_copy_count.checked_add(report.copy_count) {
                    Some(total) => total,
                    None => emit_failure("FAIL", "copy count overflow", &rows),
                };
                total_dispatch_count = match total_dispatch_count.checked_add(report.dispatch_count)
                {
                    Some(total) => total,
                    None => emit_failure("FAIL", "dispatch count overflow", &rows),
                };
                rows.push(format!(
                    "{{\"size\":{size},\"state\":\"PASS\",\"byte_exact\":true,\"dispatch_count\":{},\"allocation_count\":{},\"copy_count\":{},\"timed_out\":false,\"fallback_used\":false}}",
                    report.dispatch_count,
                    report.allocation_count,
                    report.copy_count
                ));
            }
            Err(error) => {
                let state = match error.status() {
                    Status::HipUnavailable => "UNAVAILABLE",
                    Status::Timeout => "TIMEOUT",
                    _ => "FAIL",
                };
                emit_failure(state, error.message(), &rows);
            }
        }
    }
    if rows.len() != EVIDENCE_CASE_SIZES.len() {
        emit_failure("FAIL", "evidence case count is not six", &rows);
    }
    println!(
        "{{\"schema_version\":\"g1-report-v1\",\"state\":\"PASS\",\"selected_backend\":\"hip\",\"fallback_used\":false,\"case_count\":{},\"allocation_count\":{},\"copy_count\":{},\"kernel_dispatch_count\":{},\"dispatch_count\":{},\"cases\":[{}]}}",
        rows.len(),
        total_allocation_count,
        total_copy_count,
        total_dispatch_count,
        total_dispatch_count,
        rows.join(",")
    );
}

fn emit_failure(state: &str, reason: &str, cases: &[String]) -> ! {
    println!(
        "{{\"schema_version\":\"g1-report-v1\",\"state\":\"{}\",\"case_count\":{},\"cases\":[{}],\"error\":\"{}\"}}",
        json_escape(state),
        cases.len(),
        cases.join(","),
        json_escape(reason)
    );
    std::process::exit(1);
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '\\' => escaped.push_str("\\\\"),
            '"' => escaped.push_str("\\\""),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            character if character.is_control() => {
                escaped.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => escaped.push(character),
        }
    }
    escaped
}

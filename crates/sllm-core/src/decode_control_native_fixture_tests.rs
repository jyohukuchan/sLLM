//! Actual ResultV1 bytes emitted by decode_control_gpu_test on gfx1030 and
//! gfx1201; both captures were byte-identical. This tests the host/native wire
//! boundary independently of the Rust synthetic record builder.
use crate::decode_control::*;

fn native_bytes(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len(), DECODE_RESULT_BYTES_V1 * 2);
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

#[test]
fn gpu_produced_result_fixtures_match_rust_semantics() {
    // decode-control-fixtures-gfx1030.txt: sha256 5bb1ce1530c6b469f15678d6ed3568183988d2022b3c5b15fbbb3b1f2bf6db0e
    // decode-control-fixtures-gfx1201.txt: sha256 b9b13c4c68aa122ff16609d3d639d42fb9af266136cbec716bc4ef0af6bc9f33
    let fixtures = [
        (
            "ordinary-selector",
            "01000000000000000000000001000000010000000000000001000000000000000000000000000000640000000000000065000000000000000700000000000000080000000000000016000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000e0bf000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "all-accept",
            "010000000000000000000000030000000300000002000000020000001000000000000000000000006400000000000000670000000000000007000000000000000a000000000000000a0000000b0000000c00000000000000000000000000000000000000000000000000000000000000000000000000d0bf000000000000f4bf00000000000002c00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "partial",
            "0100000000000000000000000200000002000000010000000200000000000000000000000000000064000000000000006600000000000000070000000000000009000000000000000a0000000b0000000000000000000000000000000000000000000000000000000000000000000000000000000000d0bf000000000000f4bf00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "valid-eos",
            "0100000000000000000000000200000002000000020000000200000011000000010000000000000064000000000000006600000000000000070000000000000009000000000000000a0000001f0000000000000000000000000000000000000000000000000000000000000000000000000000000000d0bf000000000000f4bf00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "budget-noop",
            "0100000000000000000000000100000001000000020000000200000012000000000000000000000064000000000000006500000000000000070000000000000008000000000000000a000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000d0bf000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "error",
            "00000000000000000a000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            "halted-noop",
            "000000000000000001000000000000000000000000000000020000000800000000000000000000006400000000000000640000000000000007000000000000000700000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        ),
    ];
    for (name, hex) in fixtures {
        let bytes = native_bytes(hex);
        let is_target = name == "ordinary-selector";
        let is_noop = name == "halted-noop";
        let generation = if is_noop || name == "error" { 0 } else { 1 };
        let mode = if is_target {
            DecodeControlModeV1::TargetOnly
        } else {
            DecodeControlModeV1::Mtp
        };
        let decoded = decode_result_le(
            &bytes,
            generation as usize % 2,
            generation,
            mode,
            if is_target { 1 } else { 2 },
            100,
            7,
            1,
            if name == "budget-noop" { 2 } else { 128 },
            64,
        );
        if name == "error" {
            assert_eq!(u32::from_le_bytes(bytes[8..12].try_into().unwrap()), 10);
            assert!(decoded.is_err(), "native sticky error must not publish");
            continue;
        }
        let result = decoded.unwrap_or_else(|error| panic!("{name}: {error}"));
        let (ids, accepted, halted, stopped): (&[u32], u32, bool, bool) = match name {
            "ordinary-selector" => (&[22], 0, false, false),
            "all-accept" => (&[10, 11, 12], 2, false, false),
            "partial" => (&[10, 11], 1, false, false),
            "valid-eos" => (&[10, 31], 2, true, true),
            "budget-noop" => (&[10], 2, true, false),
            "halted-noop" => (&[], 0, true, false),
            _ => unreachable!(),
        };
        assert_eq!(result.selected_token_ids(), ids, "{name}");
        assert_eq!(result.count as usize, ids.len(), "{name}");
        assert_eq!(result.commit_rows as usize, ids.len(), "{name}");
        assert_eq!(result.accepted, accepted, "{name}");
        assert_eq!(
            result.model_position_after,
            100 + ids.len() as u64,
            "{name}"
        );
        assert_eq!(result.counter_after, 7 + ids.len() as u64, "{name}");
        assert_eq!(result.halted(), halted, "{name}");
        assert_eq!(result.stopped(), stopped, "{name}");
        assert_eq!(result.reserved_padding, 0, "{name}");
        assert_eq!(result.reserved_tail, 0, "{name}");
        assert_eq!(
            result.status,
            if is_noop {
                DecodeControlStatusV1::Noop
            } else {
                DecodeControlStatusV1::Ok
            }
        );
        if stopped {
            assert_eq!(result.stop_row, 1);
        }
        for (row, selection) in result.selections.iter().enumerate() {
            assert_eq!(
                selection.logprob,
                if is_target { -0.5 } else { -0.25 - row as f64 }
            );
        }
    }
}

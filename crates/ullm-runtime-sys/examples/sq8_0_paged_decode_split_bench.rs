// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! Explicit split-API probe for SQ8_0 R9700 paged decode attention.
//!
//! This is intentionally an example, not a decoder selector. The normal
//! `paged_decode_attn_f32` dispatch remains untouched; this program calls the
//! public split API explicitly so the two paths can be compared on identical
//! buffers.

use std::env;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use ullm_runtime_sys::{
    RuntimeBuffer, RuntimeContext, RuntimeStream, device_info, paged_decode_attn_f32,
    paged_decode_attn_split_f32, paged_decode_attn_split_workspace_bytes,
};

#[derive(Debug)]
struct Options {
    output: PathBuf,
    runtime_device: u32,
    cache_len: usize,
    warmups: usize,
    repeats: usize,
}

#[derive(Debug, Clone, Copy)]
struct Diff {
    max_abs: f64,
    max_rel: f64,
    max_abs_index: usize,
    non_finite: usize,
}

#[derive(Debug, Clone)]
struct Timing {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn usage() -> ! {
    panic!(
        "usage: sq8_0_paged_decode_split_bench --output PATH \
         [--runtime-device N] [--cache-len N] [--warmups N] [--repeats N]"
    );
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must be a non-negative integer"))
}

fn parse_options() -> Options {
    let mut output = None;
    let mut runtime_device = 1_u32;
    let mut cache_len = 1036_usize;
    let mut warmups = 3_usize;
    let mut repeats = 20_usize;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match argument.as_str() {
            "--output" => output = Some(PathBuf::from(value())),
            "--runtime-device" => {
                runtime_device = parse_usize(&value(), "--runtime-device")
                    .try_into()
                    .unwrap_or_else(|_| panic!("--runtime-device is too large"));
            }
            "--cache-len" => cache_len = parse_usize(&value(), "--cache-len"),
            "--warmups" => warmups = parse_usize(&value(), "--warmups"),
            "--repeats" => repeats = parse_usize(&value(), "--repeats"),
            "--help" | "-h" => usage(),
            _ => panic!("unknown argument: {argument}"),
        }
    }
    let output = output.unwrap_or_else(|| usage());
    assert!(cache_len > 0, "--cache-len must be positive");
    assert!(repeats > 0, "--repeats must be positive");
    Options {
        output,
        runtime_device,
        cache_len,
        warmups,
        repeats,
    }
}

fn xorshift_values(count: usize, salt: u32) -> Vec<f32> {
    let mut state = 0x9e37_79b9_u32 ^ salt;
    (0..count)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xffff) as f32 / 32767.5 - 1.0
        })
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn u32_bytes(values: &[u32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect()
}

fn bytes_f32(values: &[u8]) -> Vec<f32> {
    values
        .chunks_exact(std::mem::size_of::<f32>())
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

fn copy_f32(
    buffer: &RuntimeBuffer,
    elements: usize,
    stream: &mut RuntimeStream,
) -> Result<Vec<f32>, String> {
    let mut bytes = vec![0_u8; elements * std::mem::size_of::<f32>()];
    buffer.copy_to_host(0, &mut bytes, Some(stream))?;
    stream.synchronize()?;
    Ok(bytes_f32(&bytes))
}

fn compare(actual: &[f32], expected: &[f32]) -> Diff {
    assert_eq!(actual.len(), expected.len());
    let mut result = Diff {
        max_abs: 0.0,
        max_rel: 0.0,
        max_abs_index: 0,
        non_finite: 0,
    };
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        if !observed.is_finite() || !reference.is_finite() {
            result.non_finite += 1;
            continue;
        }
        let absolute = (observed as f64 - reference as f64).abs();
        let relative = absolute / (reference as f64).abs().max(1.0e-30);
        if absolute > result.max_abs {
            result.max_abs = absolute;
            result.max_abs_index = index;
        }
        result.max_rel = result.max_rel.max(relative);
    }
    result
}

fn direct_call(
    q: &RuntimeBuffer,
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    table: &RuntimeBuffer,
    cache_len: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    output: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
) -> Result<(), String> {
    paged_decode_attn_f32(
        q,
        k,
        v,
        table,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        output,
        Some(stream),
    )
}

#[allow(clippy::too_many_arguments)]
fn split_call(
    q: &RuntimeBuffer,
    k: &RuntimeBuffer,
    v: &RuntimeBuffer,
    table: &RuntimeBuffer,
    cache_len: usize,
    block_size: usize,
    cache_blocks: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
    source_tile: usize,
    workspace: &mut RuntimeBuffer,
    output: &mut RuntimeBuffer,
    stream: &mut RuntimeStream,
) -> Result<(), String> {
    paged_decode_attn_split_f32(
        q,
        k,
        v,
        table,
        cache_len,
        block_size,
        cache_blocks,
        q_heads,
        kv_heads,
        head_dim,
        value_dim,
        softmax_scale,
        source_tile,
        workspace,
        output,
        Some(stream),
    )
}

fn timing(
    mut call: impl FnMut(&mut RuntimeStream) -> Result<(), String>,
    stream: &mut RuntimeStream,
    warmups: usize,
    repeats: usize,
) -> Result<Timing, String> {
    for _ in 0..warmups {
        call(stream)?;
        stream.synchronize()?;
    }
    let mut samples = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        call(stream)?;
        stream.synchronize()?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
    }
    let mean_ms = samples.iter().sum::<f64>() / samples.len() as f64;
    let min_ms = samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_ms = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    Ok(Timing {
        mean_ms,
        min_ms,
        max_ms,
    })
}

fn diff_json(value: Diff) -> String {
    format!(
        "{{\"max_abs\":{:.12},\"max_rel\":{:.12},\"max_abs_index\":{},\"non_finite\":{}}}",
        value.max_abs, value.max_rel, value.max_abs_index, value.non_finite
    )
}

fn timing_json(value: &Timing) -> String {
    format!(
        "{{\"mean_ms\":{:.12},\"min_ms\":{:.12},\"max_ms\":{:.12}}}",
        value.mean_ms, value.min_ms, value.max_ms
    )
}

fn run(options: Options) -> Result<(), String> {
    let info = device_info(options.runtime_device)?;
    if info.backend != "hip" || !info.gcn_arch_name.contains("gfx1201") {
        return Err(format!(
            "refusing non-R9700 target: runtime_device={} backend={} gcn_arch_name={}",
            options.runtime_device, info.backend, info.gcn_arch_name
        ));
    }

    const Q_HEADS: usize = 40;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const VALUE_DIM: usize = 128;
    const BLOCK_SIZE: usize = 16;
    const CACHE_BLOCKS: usize = 256;
    let softmax_scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let physical_tokens = CACHE_BLOCKS * BLOCK_SIZE;
    let q_elements = Q_HEADS * HEAD_DIM;
    let kv_cache_elements = physical_tokens * KV_HEADS;
    let k_elements = kv_cache_elements * HEAD_DIM;
    let v_elements = kv_cache_elements * VALUE_DIM;
    let output_elements = Q_HEADS * VALUE_DIM;
    let table_entries = options.cache_len.div_ceil(BLOCK_SIZE);

    let q_values = xorshift_values(q_elements, 0x1001);
    let k_values = xorshift_values(k_elements, 0x1002);
    let v_values = xorshift_values(v_elements, 0x1003);
    let table = (0..table_entries as u32).collect::<Vec<_>>();

    let mut context = RuntimeContext::create(options.runtime_device)?;
    let mut stream = context.create_stream()?;
    let mut q = context.alloc_buffer(q_elements * 4)?;
    let mut k = context.alloc_buffer(k_elements * 4)?;
    let mut v = context.alloc_buffer(v_elements * 4)?;
    let mut table_buffer = context.alloc_buffer(table_entries * 4)?;
    let mut direct_output = context.alloc_buffer(output_elements * 4)?;
    let mut split_output = context.alloc_buffer(output_elements * 4)?;
    q.copy_from_host(0, &f32_bytes(&q_values), Some(&mut stream))?;
    k.copy_from_host(0, &f32_bytes(&k_values), Some(&mut stream))?;
    v.copy_from_host(0, &f32_bytes(&v_values), Some(&mut stream))?;
    table_buffer.copy_from_host(0, &u32_bytes(&table), Some(&mut stream))?;
    stream.synchronize()?;

    direct_call(
        &q,
        &k,
        &v,
        &table_buffer,
        options.cache_len,
        BLOCK_SIZE,
        CACHE_BLOCKS,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        VALUE_DIM,
        softmax_scale,
        &mut direct_output,
        &mut stream,
    )?;
    stream.synchronize()?;
    let direct_values = copy_f32(&direct_output, output_elements, &mut stream)?;

    let mut rows = Vec::new();
    for source_tile in [128_usize, 256, 512] {
        let workspace_bytes = paged_decode_attn_split_workspace_bytes(
            Q_HEADS,
            VALUE_DIM,
            options.cache_len,
            source_tile,
        )?;
        let mut workspace = context.alloc_buffer(workspace_bytes)?;
        split_call(
            &q,
            &k,
            &v,
            &table_buffer,
            options.cache_len,
            BLOCK_SIZE,
            CACHE_BLOCKS,
            Q_HEADS,
            KV_HEADS,
            HEAD_DIM,
            VALUE_DIM,
            softmax_scale,
            source_tile,
            &mut workspace,
            &mut split_output,
            &mut stream,
        )?;
        stream.synchronize()?;
        let split_values = copy_f32(&split_output, output_elements, &mut stream)?;
        let differential = compare(&split_values, &direct_values);
        let split_timing = timing(
            |timing_stream| {
                split_call(
                    &q,
                    &k,
                    &v,
                    &table_buffer,
                    options.cache_len,
                    BLOCK_SIZE,
                    CACHE_BLOCKS,
                    Q_HEADS,
                    KV_HEADS,
                    HEAD_DIM,
                    VALUE_DIM,
                    softmax_scale,
                    source_tile,
                    &mut workspace,
                    &mut split_output,
                    timing_stream,
                )
            },
            &mut stream,
            options.warmups,
            options.repeats,
        )?;
        rows.push((source_tile, workspace_bytes, differential, split_timing));
    }
    let direct_timing = timing(
        |timing_stream| {
            direct_call(
                &q,
                &k,
                &v,
                &table_buffer,
                options.cache_len,
                BLOCK_SIZE,
                CACHE_BLOCKS,
                Q_HEADS,
                KV_HEADS,
                HEAD_DIM,
                VALUE_DIM,
                softmax_scale,
                &mut direct_output,
                timing_stream,
            )
        },
        &mut stream,
        options.warmups,
        options.repeats,
    )?;

    let mut summary = String::from("{\n");
    summary.push_str("  \"schema_version\": \"ullm.sq8_0.r9700.paged_decode_split.v0.1\",\n");
    summary.push_str(
        "  \"scope\": \"explicit existing split API comparison; direct legacy dispatch unchanged\",\n",
    );
    summary.push_str(&format!(
        "  \"device\": {{\"runtime_device\":{},\"backend\":\"{}\",\"gcn_arch_name\":\"{}\"}},\n",
        options.runtime_device, info.backend, info.gcn_arch_name,
    ));
    summary.push_str(&format!(
        "  \"shape\": {{\"cache_len\":{},\"block_size\":{},\"cache_blocks\":{},\"q_heads\":{},\"kv_heads\":{},\"head_dim\":{},\"value_dim\":{}}},\n",
        options.cache_len, BLOCK_SIZE, CACHE_BLOCKS, Q_HEADS, KV_HEADS, HEAD_DIM, VALUE_DIM,
    ));
    summary.push_str(
        "  \"timing_scope\": \"host API call plus stream synchronize per M=1 attention invocation; excludes model layers and profiler\",\n",
    );
    summary.push_str(&format!(
        "  \"direct_legacy\": {},\n",
        timing_json(&direct_timing)
    ));
    summary.push_str("  \"tiles\": [\n");
    for (index, (tile, workspace_bytes, differential, tile_timing)) in rows.iter().enumerate() {
        summary.push_str(&format!(
            "    {{\"source_tile\":{},\"split_count\":{},\"workspace_bytes\":{},\"differential_vs_direct\":{},\"timing\":{}}}{}\n",
            tile,
            options.cache_len.div_ceil(*tile),
            workspace_bytes,
            diff_json(*differential),
            timing_json(tile_timing),
            if index + 1 == rows.len() { "" } else { "," },
        ));
    }
    summary.push_str("  ]\n}\n");
    if let Some(parent) = options.output.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(&options.output, &summary).map_err(|error| error.to_string())?;
    print!("{summary}");
    Ok(())
}

fn main() {
    if let Err(error) = run(parse_options()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

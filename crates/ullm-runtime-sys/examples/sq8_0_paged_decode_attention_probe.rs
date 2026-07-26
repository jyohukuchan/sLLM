// Copyright 2026 uLLM contributors
// SPDX-License-Identifier: Apache-2.0

//! R9700-only minimal correctness and timing probe for paged decode attention.
//!
//! The probe deliberately calls the public direct and split APIs on identical
//! deterministic paged KV buffers.  It records the direct/split bit comparison,
//! a CPU reference comparison, and optionally writes the direct F32 output bytes
//! so separately compiled HIPRTC variants can be compared byte for byte.

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
    direct_output: Option<PathBuf>,
    runtime_device: u32,
    cache_len: usize,
    source_tile: usize,
    warmups: usize,
    repeats: usize,
}

#[derive(Clone, Copy, Debug)]
struct Diff {
    max_abs: f64,
    max_rel: f64,
    max_abs_index: usize,
    bit_mismatches: usize,
    non_finite: usize,
}

#[derive(Clone, Debug)]
struct Timing {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
}

fn usage() -> ! {
    panic!(
        "usage: sq8_0_paged_decode_attention_probe --output PATH \
         [--direct-output PATH] [--runtime-device N] [--cache-len N] \
         [--source-tile N] [--warmups N] [--repeats N]"
    );
}

fn parse_usize(value: &str, flag: &str) -> usize {
    value
        .parse()
        .unwrap_or_else(|_| panic!("{flag} must be a non-negative integer"))
}

fn parse_options() -> Options {
    let mut output = None;
    let mut direct_output = None;
    let mut runtime_device = 1_u32;
    let mut cache_len = 1036_usize;
    let mut source_tile = 128_usize;
    let mut warmups = 3_usize;
    let mut repeats = 20_usize;
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        let mut value = || args.next().unwrap_or_else(|| usage());
        match argument.as_str() {
            "--output" => output = Some(PathBuf::from(value())),
            "--direct-output" => direct_output = Some(PathBuf::from(value())),
            "--runtime-device" => {
                runtime_device = parse_usize(&value(), "--runtime-device")
                    .try_into()
                    .unwrap_or_else(|_| panic!("--runtime-device is too large"));
            }
            "--cache-len" => cache_len = parse_usize(&value(), "--cache-len"),
            "--source-tile" => source_tile = parse_usize(&value(), "--source-tile"),
            "--warmups" => warmups = parse_usize(&value(), "--warmups"),
            "--repeats" => repeats = parse_usize(&value(), "--repeats"),
            "--help" | "-h" => usage(),
            _ => panic!("unknown argument: {argument}"),
        }
    }
    let output = output.unwrap_or_else(|| usage());
    assert!(cache_len > 0, "--cache-len must be positive");
    assert!(source_tile > 0, "--source-tile must be positive");
    assert!(repeats > 0, "--repeats must be positive");
    Options {
        output,
        direct_output,
        runtime_device,
        cache_len,
        source_tile,
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

fn copy_bytes(
    buffer: &RuntimeBuffer,
    elements: usize,
    stream: &mut RuntimeStream,
) -> Result<Vec<u8>, String> {
    let mut bytes = vec![0_u8; elements * std::mem::size_of::<f32>()];
    buffer.copy_to_host(0, &mut bytes, Some(stream))?;
    stream.synchronize()?;
    Ok(bytes)
}

fn compare(actual: &[f32], expected: &[f32]) -> Diff {
    assert_eq!(actual.len(), expected.len());
    let mut result = Diff {
        max_abs: 0.0,
        max_rel: 0.0,
        max_abs_index: 0,
        bit_mismatches: 0,
        non_finite: 0,
    };
    for (index, (&observed, &reference)) in actual.iter().zip(expected).enumerate() {
        if observed.to_bits() != reference.to_bits() {
            result.bit_mismatches += 1;
        }
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

#[allow(clippy::too_many_arguments)]
fn cpu_reference(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_table: &[u32],
    cache_len: usize,
    block_size: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    value_dim: usize,
    softmax_scale: f32,
) -> Vec<f32> {
    let mut output = vec![0.0_f32; q_heads * value_dim];
    let q_per_kv = q_heads / kv_heads;
    for q_head in 0..q_heads {
        let kv_head = q_head / q_per_kv;
        let q_base = q_head * head_dim;
        let mut scores = Vec::with_capacity(cache_len);
        for source_timestep in 0..cache_len {
            let block = block_table[source_timestep / block_size] as usize;
            let physical_timestep = block * block_size + source_timestep % block_size;
            let k_base = (physical_timestep * kv_heads + kv_head) * head_dim;
            let score = (0..head_dim)
                .map(|dim| q[q_base + dim] * k_cache[k_base + dim])
                .sum::<f32>()
                * softmax_scale;
            scores.push(score);
        }
        let max_score = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores
            .iter()
            .map(|score| (*score - max_score).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f32>();
        for value in 0..value_dim {
            let mut weighted = 0.0_f32;
            for (source_timestep, weight) in weights.iter().enumerate() {
                let block = block_table[source_timestep / block_size] as usize;
                let physical_timestep = block * block_size + source_timestep % block_size;
                let v_index = (physical_timestep * kv_heads + kv_head) * value_dim + value;
                weighted += *weight * v_cache[v_index];
            }
            output[q_head * value_dim + value] = weighted / denominator;
        }
    }
    output
}

#[allow(clippy::too_many_arguments)]
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
    Ok(Timing {
        mean_ms: samples.iter().sum::<f64>() / samples.len() as f64,
        min_ms: samples.iter().copied().fold(f64::INFINITY, f64::min),
        max_ms: samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    })
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut state = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        state ^= u64::from(*byte);
        state = state.wrapping_mul(0x0000_0100_0000_01b3);
    }
    state
}

fn experimental_flag_is_one(name: &str) -> bool {
    matches!(env::var(name).as_deref(), Ok("1"))
}

fn selected_split_mode() -> &'static str {
    if experimental_flag_is_one("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_PIPELINED_SPLIT") {
        "gqa-grouped-pipelined"
    } else if experimental_flag_is_one("ULLM_EXPERIMENTAL_PAGED_DECODE_GQA_GROUPED_SPLIT") {
        "gqa-grouped"
    } else {
        "generic-per-q-head"
    }
}

fn diff_json(value: Diff) -> String {
    format!(
        "{{\"max_abs\":{:.12},\"max_rel\":{:.12},\"max_abs_index\":{},\"bit_mismatches\":{},\"non_finite\":{}}}",
        value.max_abs, value.max_rel, value.max_abs_index, value.bit_mismatches, value.non_finite,
    )
}

fn timing_json(value: &Timing) -> String {
    format!(
        "{{\"mean_ms\":{:.12},\"min_ms\":{:.12},\"max_ms\":{:.12}}}",
        value.mean_ms, value.min_ms, value.max_ms,
    )
}

fn write_file(path: &PathBuf, bytes: &[u8]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    }
    fs::write(path, bytes).map_err(|error| error.to_string())
}

fn run(options: Options) -> Result<(), String> {
    let info = device_info(options.runtime_device)?;
    if info.backend != "hip" || !info.gcn_arch_name.contains("gfx1201") {
        return Err(format!(
            "refusing non-R9700 target: runtime_device={} backend={} gcn_arch_name={}",
            options.runtime_device, info.backend, info.gcn_arch_name,
        ));
    }

    const Q_HEADS: usize = 40;
    const KV_HEADS: usize = 8;
    const HEAD_DIM: usize = 128;
    const VALUE_DIM: usize = 128;
    const BLOCK_SIZE: usize = 16;
    const CACHE_BLOCKS: usize = 256;
    let physical_tokens = CACHE_BLOCKS * BLOCK_SIZE;
    if options.cache_len > physical_tokens {
        return Err(format!(
            "cache_len {} exceeds the fixed probe capacity {physical_tokens}",
            options.cache_len,
        ));
    }

    let softmax_scale = 1.0_f32 / (HEAD_DIM as f32).sqrt();
    let table_entries = options.cache_len.div_ceil(BLOCK_SIZE);
    let q_values = xorshift_values(Q_HEADS * HEAD_DIM, 0x1001);
    let k_values = xorshift_values(physical_tokens * KV_HEADS * HEAD_DIM, 0x1002);
    let v_values = xorshift_values(physical_tokens * KV_HEADS * VALUE_DIM, 0x1003);
    let table = (0..table_entries)
        .map(|index| ((index * 13 + 7) % CACHE_BLOCKS) as u32)
        .collect::<Vec<_>>();
    let expected = cpu_reference(
        &q_values,
        &k_values,
        &v_values,
        &table,
        options.cache_len,
        BLOCK_SIZE,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        VALUE_DIM,
        softmax_scale,
    );

    let mut context = RuntimeContext::create(options.runtime_device)?;
    let mut stream = context.create_stream()?;
    let mut q = context.alloc_buffer(q_values.len() * std::mem::size_of::<f32>())?;
    let mut k = context.alloc_buffer(k_values.len() * std::mem::size_of::<f32>())?;
    let mut v = context.alloc_buffer(v_values.len() * std::mem::size_of::<f32>())?;
    let mut table_buffer = context.alloc_buffer(table.len() * std::mem::size_of::<u32>())?;
    let mut direct_output =
        context.alloc_buffer(Q_HEADS * VALUE_DIM * std::mem::size_of::<f32>())?;
    let mut split_output =
        context.alloc_buffer(Q_HEADS * VALUE_DIM * std::mem::size_of::<f32>())?;
    let workspace_bytes = paged_decode_attn_split_workspace_bytes(
        Q_HEADS,
        VALUE_DIM,
        options.cache_len,
        options.source_tile,
    )?;
    let mut workspace = context.alloc_buffer(workspace_bytes)?;
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
    let direct_bytes = copy_bytes(&direct_output, Q_HEADS * VALUE_DIM, &mut stream)?;
    let direct_values = bytes_f32(&direct_bytes);

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
        options.source_tile,
        &mut workspace,
        &mut split_output,
        &mut stream,
    )?;
    stream.synchronize()?;
    let split_bytes = copy_bytes(&split_output, Q_HEADS * VALUE_DIM, &mut stream)?;
    let split_values = bytes_f32(&split_bytes);

    let direct_cpu = compare(&direct_values, &expected);
    let split_cpu = compare(&split_values, &expected);
    let split_direct = compare(&split_values, &direct_values);
    if direct_cpu.non_finite != 0 || split_cpu.non_finite != 0 {
        return Err("direct or split output contained a non-finite value".to_string());
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
                options.source_tile,
                &mut workspace,
                &mut split_output,
                timing_stream,
            )
        },
        &mut stream,
        options.warmups,
        options.repeats,
    )?;

    if let Some(path) = &options.direct_output {
        write_file(path, &direct_bytes)?;
    }
    let split_mode = selected_split_mode();
    let split_count = options.cache_len.div_ceil(options.source_tile);
    let grouped = split_mode != "generic-per-q-head";
    let partial_workgroups = if grouped {
        KV_HEADS * split_count
    } else {
        Q_HEADS * split_count
    };
    let semantic_kv_load_bytes = if grouped {
        KV_HEADS * options.cache_len * (HEAD_DIM + VALUE_DIM) * std::mem::size_of::<f32>()
    } else {
        Q_HEADS * options.cache_len * (HEAD_DIM + VALUE_DIM) * std::mem::size_of::<f32>()
    };
    let per_source_cta_barriers = if split_mode == "gqa-grouped-pipelined" {
        1
    } else {
        2
    };
    let initial_cta_barriers_per_partial = usize::from(split_mode == "gqa-grouped-pipelined");
    let experimental_wave_scalar =
        env::var_os("ULLM_EXPERIMENTAL_PAGED_DECODE_WAVE_SCALAR_SOFTMAX").is_some();
    let summary = format!(
        concat!(
            "{{\n",
            "  \"schema_version\": \"ullm.sq8_0.r9700.paged_decode_attention_probe.v0.1\",\n",
            "  \"mode\": \"{}\",\n",
            "  \"split_mode\": \"{}\",\n",
            "  \"device\": {{\"runtime_device\":{},\"backend\":\"{}\",\"gcn_arch_name\":\"{}\"}},\n",
            "  \"shape\": {{\"cache_len\":{},\"block_size\":{},\"cache_blocks\":{},\"q_heads\":{},\"kv_heads\":{},\"head_dim\":{},\"value_dim\":{},\"source_tile\":{},\"split_count\":{}}},\n",
            "  \"selected_split_geometry\": {{\"partial_workgroups\":{},\"merge_workgroups\":{},\"workgroup_threads\":256,\"semantic_kv_load_bytes\":{},\"semantic_kv_load_scope\":\"algorithmic K+V accesses; not a physical HBM measurement\",\"cta_barriers_per_source_row\":{},\"initial_cta_barriers_per_partial\":{}}},\n",
            "  \"page_table\": \"logical block i maps to (13*i+7) mod 256; unique for this probe capacity\",\n",
            "  \"cpu_reference\": \"F32 two-pass softmax reference; compares are diagnostic rather than bitwise GPU requirements\",\n",
            "  \"direct_vs_cpu\": {},\n",
            "  \"split_vs_cpu\": {},\n",
            "  \"split_vs_direct\": {},\n",
            "  \"direct_output\": {{\"bytes\":{},\"fnv1a64\":\"{:016x}\"}},\n",
            "  \"timing_scope\": \"host API call plus stream synchronize per M=1 attention invocation; excludes model layers and profiler\",\n",
            "  \"direct_timing\": {},\n",
            "  \"split_timing\": {}\n",
            "}}\n"
        ),
        if experimental_wave_scalar {
            "experimental-wave-scalar-softmax"
        } else {
            "legacy-direct-softmax"
        },
        split_mode,
        options.runtime_device,
        info.backend,
        info.gcn_arch_name,
        options.cache_len,
        BLOCK_SIZE,
        CACHE_BLOCKS,
        Q_HEADS,
        KV_HEADS,
        HEAD_DIM,
        VALUE_DIM,
        options.source_tile,
        split_count,
        partial_workgroups,
        Q_HEADS,
        semantic_kv_load_bytes,
        per_source_cta_barriers,
        initial_cta_barriers_per_partial,
        diff_json(direct_cpu),
        diff_json(split_cpu),
        diff_json(split_direct),
        direct_bytes.len(),
        fnv1a64(&direct_bytes),
        timing_json(&direct_timing),
        timing_json(&split_timing),
    );
    write_file(&options.output, summary.as_bytes())?;
    print!("{summary}");
    Ok(())
}

fn main() {
    if let Err(error) = run(parse_options()) {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}

// Backend-neutral wire contracts for whole-decode control and results.
//
// The native HIP implementation uses the same field offsets, but this module
// deliberately serializes every value with explicit little-endian accesses.
// No native padding, pointer, or transmuted representation crosses this
// boundary.

use crate::{ExecutionError, SamplingLogprobV1, SamplingSelectionV1};

pub const DECODE_CONTROL_VERSION_V1: u32 = 1;
pub const DECODE_CONTROL_BYTES_V1: usize = 144;
pub const DECODE_RESULT_BYTES_V1: usize = 192;
pub const DECODE_RESULT_RING_SLOTS_V1: usize = 2;
pub const DECODE_MAX_WIDTH_V1: u32 = 8;
pub const DECODE_MAX_EMITTED_V1: usize = 9;
pub const DECODE_NO_STOP_V1: u32 = u32::MAX;
pub const DECODE_Q_DRAFT_SEED_DOMAIN_V1: u64 = 0x5144_5241_4654_0001;

const RESULT_HALT_STOP: u32 = 1 << 0;
const RESULT_HALT_BUDGET: u32 = 1 << 1;
const RESULT_HALT_INVALID: u32 = 1 << 2;
const RESULT_HALT_NOOP: u32 = 1 << 3;
const RESULT_HALT_ALL_ACCEPT: u32 = 1 << 4;
const RESULT_HALT_KNOWN: u32 = RESULT_HALT_STOP
    | RESULT_HALT_BUDGET
    | RESULT_HALT_INVALID
    | RESULT_HALT_NOOP
    | RESULT_HALT_ALL_ACCEPT;

fn invalid(reason: impl Into<String>) -> ExecutionError {
    ExecutionError::InvalidRequest {
        reason: reason.into(),
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeControlModeV1 {
    TargetOnly = 0,
    Mtp = 1,
}

impl DecodeControlModeV1 {
    fn from_raw(value: u32) -> Result<Self, ExecutionError> {
        match value {
            0 => Ok(Self::TargetOnly),
            1 => Ok(Self::Mtp),
            _ => Err(invalid("decode control mode is unsupported")),
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodePhaseKindV1 {
    Target = 0,
    Draft = 1,
    MtpAlign = 2,
}

impl DecodePhaseKindV1 {
    fn from_raw(value: u32) -> Result<Self, ExecutionError> {
        match value {
            0 => Ok(Self::Target),
            1 => Ok(Self::Draft),
            2 => Ok(Self::MtpAlign),
            _ => Err(invalid("decode phase kind is unsupported")),
        }
    }
}

#[repr(u32)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DecodeControlStatusV1 {
    Ok = 0,
    Noop = 1,
    InvalidVersion = 2,
    InvalidMode = 3,
    InvalidPhase = 4,
    InvalidWidth = 5,
    InvalidGeneration = 6,
    InvalidCounter = 7,
    InvalidPosition = 8,
    InvalidCapacity = 9,
    InvalidDecision = 10,
    InvalidSelector = 11,
    InvalidToken = 12,
    InvalidStopIds = 13,
    InvalidBudget = 14,
    NullPointer = 15,
}

impl DecodeControlStatusV1 {
    fn from_raw(value: u32) -> Result<Self, ExecutionError> {
        match value {
            0 => Ok(Self::Ok),
            1 => Ok(Self::Noop),
            2 => Ok(Self::InvalidVersion),
            3 => Ok(Self::InvalidMode),
            4 => Ok(Self::InvalidPhase),
            5 => Ok(Self::InvalidWidth),
            6 => Ok(Self::InvalidGeneration),
            7 => Ok(Self::InvalidCounter),
            8 => Ok(Self::InvalidPosition),
            9 => Ok(Self::InvalidCapacity),
            10 => Ok(Self::InvalidDecision),
            11 => Ok(Self::InvalidSelector),
            12 => Ok(Self::InvalidToken),
            13 => Ok(Self::InvalidStopIds),
            14 => Ok(Self::InvalidBudget),
            15 => Ok(Self::NullPointer),
            _ => Err(invalid("decode control status is unknown")),
        }
    }

    pub const fn is_error(self) -> bool {
        !matches!(self, Self::Ok | Self::Noop)
    }
}

/// Device control state using the exact native field order as semantic fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeControlV1 {
    pub version: u32,
    pub status: DecodeControlStatusV1,
    pub mode: DecodeControlModeV1,
    pub width: u32,
    pub model_position: u64,
    pub sampler_counter: u64,
    pub seed: u64,
    pub generation: u64,
    pub capacity: u64,
    pub output_limit: u64,
    pub output_count: u64,
    pub pending_token: u32,
    pub halted: bool,
    pub commit_rows: u32,
    pub publish_count: u32,
    pub accepted_count: u32,
    pub stop_row: u32,
    pub hidden_row: u32,
    pub active_width: u32,
    pub phase_position: u64,
    pub phase_counter: u64,
    pub phase_seed: u64,
    pub phase_rows: u32,
    pub phase_index: u32,
    pub phase_kind: DecodePhaseKindV1,
    pub phase_active: bool,
}

impl DecodeControlV1 {
    /// Creates the post-prefill state. `output_count=1` accounts for the
    /// already-returned prefill selection; `output_limit` is the total limit.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        mode: DecodeControlModeV1,
        width: u32,
        capacity: u64,
        model_position: u64,
        sampler_counter: u64,
        seed: u64,
        output_limit: u64,
        pending_token: u32,
        vocabulary_size: u32,
    ) -> Result<Self, ExecutionError> {
        if !(1..=DECODE_MAX_WIDTH_V1).contains(&width) {
            return Err(invalid("decode control width must be in 1..=8"));
        }
        if mode == DecodeControlModeV1::TargetOnly && width != 1 {
            return Err(invalid("target-only decode control requires width 1"));
        }
        if capacity == 0 || model_position > capacity {
            return Err(invalid(
                "decode control capacity or model position is invalid",
            ));
        }
        if output_limit < 1 {
            return Err(invalid("decode control output limit must include prefill"));
        }
        if vocabulary_size == 0 || pending_token >= vocabulary_size {
            return Err(invalid(
                "decode control pending token is outside vocabulary",
            ));
        }
        Ok(Self {
            version: DECODE_CONTROL_VERSION_V1,
            status: DecodeControlStatusV1::Ok,
            mode,
            width,
            model_position,
            sampler_counter,
            seed,
            generation: 0,
            capacity,
            output_limit,
            output_count: 1,
            pending_token,
            halted: false,
            commit_rows: 0,
            publish_count: 0,
            accepted_count: 0,
            stop_row: DECODE_NO_STOP_V1,
            hidden_row: DECODE_NO_STOP_V1,
            active_width: width,
            phase_position: model_position,
            phase_counter: sampler_counter,
            phase_seed: seed,
            phase_rows: 0,
            phase_index: 0,
            phase_kind: DecodePhaseKindV1::Target,
            phase_active: false,
        })
    }

    pub fn validate(&self, vocabulary_size: u32) -> Result<(), ExecutionError> {
        if self.version != DECODE_CONTROL_VERSION_V1 {
            return Err(invalid("decode control version is unsupported"));
        }
        if !(1..=DECODE_MAX_WIDTH_V1).contains(&self.width)
            || (self.mode == DecodeControlModeV1::TargetOnly && self.width != 1)
            || self.active_width > self.width
        {
            return Err(invalid("decode control mode or width is invalid"));
        }
        if self.capacity == 0
            || self.model_position > self.capacity
            || self.output_count > self.output_limit
            || self.output_limit - self.output_count > self.capacity - self.model_position
            || vocabulary_size == 0
            || self.pending_token >= vocabulary_size
        {
            return Err(invalid("decode control bounds are invalid"));
        }
        if self.commit_rows > self.width + 1 || self.publish_count > self.width + 1 {
            return Err(invalid("decode control publication rows are invalid"));
        }
        if self.phase_active {
            self.validate_phase(vocabulary_size)?;
        }
        Ok(())
    }

    fn validate_phase(&self, _vocabulary_size: u32) -> Result<(), ExecutionError> {
        match self.phase_kind {
            DecodePhaseKindV1::Target => {
                if self.phase_rows == 0
                    || self.phase_rows
                        > if self.mode == DecodeControlModeV1::Mtp {
                            self.active_width + 1
                        } else {
                            1
                        }
                    || self.phase_index > self.width
                    || (self.mode == DecodeControlModeV1::TargetOnly
                        && (self.phase_index != 0 || self.phase_rows != 1))
                {
                    return Err(invalid("decode target phase bounds are invalid"));
                }
            }
            DecodePhaseKindV1::Draft => {
                if self.phase_rows != 1 || self.phase_index >= self.active_width {
                    return Err(invalid("decode draft phase bounds are invalid"));
                }
            }
            DecodePhaseKindV1::MtpAlign => {
                if self.phase_rows != 1 || self.phase_index != self.active_width {
                    return Err(invalid("decode MTP-align phase bounds are invalid"));
                }
            }
        }
        Ok(())
    }

    pub fn begin_phase(
        &mut self,
        kind: DecodePhaseKindV1,
        index: u32,
        rows: u32,
    ) -> Result<(), ExecutionError> {
        if self.status != DecodeControlStatusV1::Ok || self.halted {
            return Err(invalid("decode control is halted or sticky-error"));
        }
        let command_valid = match kind {
            DecodePhaseKindV1::Target => {
                rows != 0
                    && rows <= self.width + 1
                    && index <= self.width
                    && (self.mode == DecodeControlModeV1::Mtp || (index == 0 && rows == 1))
            }
            DecodePhaseKindV1::Draft => rows == 1 && index < self.width,
            DecodePhaseKindV1::MtpAlign => rows == 1 && index == self.width,
        };
        if !command_valid {
            return Err(invalid("decode phase command bounds are invalid"));
        }
        if self.mode == DecodeControlModeV1::Mtp && kind == DecodePhaseKindV1::Draft && index == 0 {
            let budget_rows = self
                .output_limit
                .checked_sub(self.output_count)
                .ok_or_else(|| invalid("decode output count exceeds its limit"))?;
            let capacity_rows = self
                .capacity
                .checked_sub(self.model_position)
                .and_then(|rows| rows.checked_sub(1))
                .ok_or_else(|| invalid("decode state has no target row capacity"))?;
            self.active_width = u64::from(self.width).min(budget_rows).min(capacity_rows) as u32;
        }
        if self.mode == DecodeControlModeV1::Mtp
            && kind == DecodePhaseKindV1::Draft
            && index >= self.active_width
        {
            self.phase_kind = kind;
            self.phase_index = index;
            self.phase_rows = 0;
            self.phase_active = false;
            return Ok(());
        }
        if self.mode == DecodeControlModeV1::Mtp
            && kind == DecodePhaseKindV1::Target
            && rows == 1
            && index > self.active_width
        {
            self.phase_kind = kind;
            self.phase_index = index;
            self.phase_rows = 0;
            self.phase_active = false;
            return Ok(());
        }
        let effective_index =
            if self.mode == DecodeControlModeV1::Mtp && kind == DecodePhaseKindV1::MtpAlign {
                self.active_width
            } else {
                index
            };
        let effective_rows =
            if self.mode == DecodeControlModeV1::Mtp && kind == DecodePhaseKindV1::Target {
                self.active_width + 1
            } else {
                rows
            };
        self.phase_kind = kind;
        self.phase_index = effective_index;
        self.phase_rows = effective_rows;
        self.validate_phase(1)?;
        match kind {
            DecodePhaseKindV1::Target => {
                self.phase_position = self.model_position;
                self.phase_counter = self
                    .sampler_counter
                    .checked_add(u64::from(effective_index))
                    .ok_or_else(|| invalid("target phase sampler counter overflowed"))?;
                self.phase_seed = self.seed;
            }
            DecodePhaseKindV1::Draft | DecodePhaseKindV1::MtpAlign => {
                self.phase_position = self
                    .model_position
                    .checked_add(u64::from(effective_index))
                    .ok_or_else(|| invalid("draft phase position overflowed"))?;
                self.phase_counter = self
                    .sampler_counter
                    .checked_mul(9)
                    .and_then(|value| value.checked_add(u64::from(effective_index)))
                    .ok_or_else(|| invalid("draft phase sampler counter overflowed"))?;
                self.phase_seed = self.seed ^ DECODE_Q_DRAFT_SEED_DOMAIN_V1;
            }
        }
        self.phase_active = true;
        Ok(())
    }

    pub fn encode_le(&self) -> Result<[u8; DECODE_CONTROL_BYTES_V1], ExecutionError> {
        self.validate(u32::MAX)?;
        let mut bytes = [0_u8; DECODE_CONTROL_BYTES_V1];
        put_u32(&mut bytes, 0, self.version);
        put_u32(&mut bytes, 4, self.status as u32);
        put_u32(&mut bytes, 8, self.mode as u32);
        put_u32(&mut bytes, 12, self.width);
        put_u64(&mut bytes, 16, self.model_position);
        put_u64(&mut bytes, 24, self.sampler_counter);
        put_u64(&mut bytes, 32, self.seed);
        put_u64(&mut bytes, 40, self.generation);
        put_u64(&mut bytes, 48, self.capacity);
        put_u64(&mut bytes, 56, self.output_limit);
        put_u64(&mut bytes, 64, self.output_count);
        put_u32(&mut bytes, 72, self.pending_token);
        put_u32(&mut bytes, 76, u32::from(self.halted));
        put_u32(&mut bytes, 80, self.commit_rows);
        put_u32(&mut bytes, 84, self.publish_count);
        put_u32(&mut bytes, 88, self.accepted_count);
        put_u32(&mut bytes, 92, self.stop_row);
        put_u32(&mut bytes, 96, self.hidden_row);
        put_u32(&mut bytes, 100, self.active_width);
        put_u64(&mut bytes, 104, self.phase_position);
        put_u64(&mut bytes, 112, self.phase_counter);
        put_u64(&mut bytes, 120, self.phase_seed);
        put_u32(&mut bytes, 128, self.phase_rows);
        put_u32(&mut bytes, 132, self.phase_index);
        put_u32(&mut bytes, 136, self.phase_kind as u32);
        put_u32(&mut bytes, 140, u32::from(self.phase_active));
        Ok(bytes)
    }

    pub fn decode_le(bytes: &[u8]) -> Result<Self, ExecutionError> {
        if bytes.len() != DECODE_CONTROL_BYTES_V1 {
            return Err(invalid("decode control wire length differs"));
        }
        let status = DecodeControlStatusV1::from_raw(get_u32(bytes, 4)?)?;
        let mode = DecodeControlModeV1::from_raw(get_u32(bytes, 8)?)?;
        let phase_kind = DecodePhaseKindV1::from_raw(get_u32(bytes, 136)?)?;
        if get_u32(bytes, 0)? != DECODE_CONTROL_VERSION_V1 || get_u32(bytes, 140)? > 1 {
            return Err(invalid("decode control wire header is invalid"));
        }
        Ok(Self {
            version: get_u32(bytes, 0)?,
            status,
            mode,
            width: get_u32(bytes, 12)?,
            model_position: get_u64(bytes, 16)?,
            sampler_counter: get_u64(bytes, 24)?,
            seed: get_u64(bytes, 32)?,
            generation: get_u64(bytes, 40)?,
            capacity: get_u64(bytes, 48)?,
            output_limit: get_u64(bytes, 56)?,
            output_count: get_u64(bytes, 64)?,
            pending_token: get_u32(bytes, 72)?,
            halted: get_u32(bytes, 76)? != 0,
            commit_rows: get_u32(bytes, 80)?,
            publish_count: get_u32(bytes, 84)?,
            accepted_count: get_u32(bytes, 88)?,
            stop_row: get_u32(bytes, 92)?,
            hidden_row: get_u32(bytes, 96)?,
            active_width: get_u32(bytes, 100)?,
            phase_position: get_u64(bytes, 104)?,
            phase_counter: get_u64(bytes, 112)?,
            phase_seed: get_u64(bytes, 120)?,
            phase_rows: get_u32(bytes, 128)?,
            phase_index: get_u32(bytes, 132)?,
            phase_kind,
            phase_active: get_u32(bytes, 140)? != 0,
        })
    }
}

/// Host representation of a validated native result ring record.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodeResultV1 {
    pub generation: u64,
    pub status: DecodeControlStatusV1,
    pub count: u32,
    pub commit_rows: u32,
    pub accepted: u32,
    pub width: u32,
    pub halt_flags: u32,
    pub stop_row: u32,
    pub reserved_padding: u32,
    pub reserved_tail: u64,
    pub model_position_before: u64,
    pub model_position_after: u64,
    pub counter_before: u64,
    pub counter_after: u64,
    pub selections: Vec<SamplingSelectionV1>,
}

impl DecodeResultV1 {
    pub const fn halted(&self) -> bool {
        self.halt_flags & (RESULT_HALT_STOP | RESULT_HALT_BUDGET | RESULT_HALT_NOOP) != 0
    }

    pub const fn stopped(&self) -> bool {
        self.halt_flags & RESULT_HALT_STOP != 0
    }

    pub fn selected_token_ids(&self) -> Vec<u32> {
        self.selections
            .iter()
            .map(|selection| selection.token_id)
            .collect()
    }
}

#[allow(clippy::too_many_arguments)]
pub fn decode_result_le(
    bytes: &[u8],
    ring_slot: usize,
    expected_generation: u64,
    expected_mode: DecodeControlModeV1,
    expected_width: u32,
    expected_model_position: u64,
    expected_sampler_counter: u64,
    expected_output_count: u64,
    output_limit: u64,
    vocabulary_size: u32,
) -> Result<DecodeResultV1, ExecutionError> {
    if bytes.len() != DECODE_RESULT_BYTES_V1 {
        return Err(invalid("decode result wire length differs"));
    }
    if ring_slot != (expected_generation as usize % DECODE_RESULT_RING_SLOTS_V1) {
        return Err(invalid("decode result ring slot is stale"));
    }
    let status = DecodeControlStatusV1::from_raw(get_u32(bytes, 8)?)?;
    let generation = get_u64(bytes, 0)?;
    let count = get_u32(bytes, 12)?;
    let commit_rows = get_u32(bytes, 16)?;
    let accepted = get_u32(bytes, 20)?;
    let width = get_u32(bytes, 24)?;
    let halt_flags = get_u32(bytes, 28)?;
    let stop_row = get_u32(bytes, 32)?;
    let reserved_padding = get_u32(bytes, 36)?;
    let reserved_tail = get_u64(bytes, 184)?;
    if generation != expected_generation
        || reserved_padding != 0
        || get_u32(bytes, 108)? != 0
        || reserved_tail != 0
        || halt_flags & !RESULT_HALT_KNOWN != 0
        || width > expected_width
        || width > DECODE_MAX_WIDTH_V1
        || (expected_mode == DecodeControlModeV1::TargetOnly && width != 1)
        || accepted > width
        || output_limit < expected_output_count
        || vocabulary_size == 0
    {
        return Err(invalid(
            "decode result header, generation, or mode is invalid",
        ));
    }
    let model_before = get_u64(bytes, 40)?;
    let model_after = get_u64(bytes, 48)?;
    let counter_before = get_u64(bytes, 56)?;
    let counter_after = get_u64(bytes, 64)?;
    if model_before != expected_model_position || counter_before != expected_sampler_counter {
        return Err(invalid(
            "decode result starts from a stale position or counter",
        ));
    }
    if status == DecodeControlStatusV1::Noop {
        if count != 0
            || commit_rows != 0
            || model_after != model_before
            || counter_after != counter_before
            || halt_flags & RESULT_HALT_NOOP == 0
        {
            return Err(invalid("decode noop result advances state"));
        }
        if bytes[72..108].iter().any(|value| *value != 0)
            || bytes[112..184].iter().any(|value| *value != 0)
        {
            return Err(invalid("decode noop result contains selected payload"));
        }
        return Ok(DecodeResultV1::from_parts(
            generation,
            status,
            count,
            commit_rows,
            accepted,
            width,
            halt_flags,
            stop_row,
            reserved_padding,
            reserved_tail,
            model_before,
            model_after,
            counter_before,
            counter_after,
            Vec::new(),
        ));
    }
    if status.is_error() {
        return Err(invalid("decode result reports a sticky GPU error"));
    }
    if status != DecodeControlStatusV1::Ok
        || count == 0
        || commit_rows != count
        || count > accepted.saturating_add(1)
        || u64::from(count) > output_limit - expected_output_count
        || halt_flags & RESULT_HALT_INVALID != 0
        || model_after
            != model_before
                .checked_add(u64::from(commit_rows))
                .ok_or_else(|| invalid("decode result model position overflowed"))?
        || counter_after
            != counter_before
                .checked_add(u64::from(count))
                .ok_or_else(|| invalid("decode result counter overflowed"))?
        || (halt_flags & RESULT_HALT_STOP != 0 && stop_row >= count)
        || (halt_flags & RESULT_HALT_STOP == 0 && stop_row != 0)
        || (halt_flags & RESULT_HALT_ALL_ACCEPT != 0 && accepted != width)
        || (expected_mode == DecodeControlModeV1::TargetOnly && accepted != 0)
    {
        return Err(invalid(
            "decode result prefix, state delta, or acceptance is invalid",
        ));
    }
    let mut selections = Vec::with_capacity(count as usize);
    for index in 0..DECODE_MAX_EMITTED_V1 {
        let token = get_u32(bytes, 72 + index * 4)?;
        let logprob = get_f64(bytes, 112 + index * 8)?;
        if index < count as usize {
            if token >= vocabulary_size || !logprob.is_finite() || logprob > 0.0 {
                return Err(invalid(
                    "decode result selected token or logprob is invalid",
                ));
            }
            selections.push(SamplingSelectionV1 {
                token_id: token,
                logprob,
                top_logprobs: Vec::<SamplingLogprobV1>::new(),
            });
        } else if token != 0 || logprob != 0.0 {
            return Err(invalid("decode result has a nonzero unused entry"));
        }
    }
    Ok(DecodeResultV1::from_parts(
        generation,
        status,
        count,
        commit_rows,
        accepted,
        width,
        halt_flags,
        stop_row,
        reserved_padding,
        reserved_tail,
        model_before,
        model_after,
        counter_before,
        counter_after,
        selections,
    ))
}

impl DecodeResultV1 {
    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        generation: u64,
        status: DecodeControlStatusV1,
        count: u32,
        commit_rows: u32,
        accepted: u32,
        width: u32,
        halt_flags: u32,
        stop_row: u32,
        reserved_padding: u32,
        reserved_tail: u64,
        model_position_before: u64,
        model_position_after: u64,
        counter_before: u64,
        counter_after: u64,
        selections: Vec<SamplingSelectionV1>,
    ) -> Self {
        Self {
            generation,
            status,
            count,
            commit_rows,
            accepted,
            width,
            halt_flags,
            stop_row,
            reserved_padding,
            reserved_tail,
            model_position_before,
            model_position_after,
            counter_before,
            counter_after,
            selections,
        }
    }
}

fn put_u32(bytes: &mut [u8], offset: usize, value: u32) {
    bytes[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(bytes: &mut [u8], offset: usize, value: u64) {
    bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn get_u32(bytes: &[u8], offset: usize) -> Result<u32, ExecutionError> {
    let chunk = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| invalid("decode wire u32 is truncated"))?;
    Ok(u32::from_le_bytes(
        chunk.try_into().expect("checked u32 length"),
    ))
}

fn get_u64(bytes: &[u8], offset: usize) -> Result<u64, ExecutionError> {
    let chunk = bytes
        .get(offset..offset + 8)
        .ok_or_else(|| invalid("decode wire u64 is truncated"))?;
    Ok(u64::from_le_bytes(
        chunk.try_into().expect("checked u64 length"),
    ))
}

fn get_f64(bytes: &[u8], offset: usize) -> Result<f64, ExecutionError> {
    Ok(f64::from_bits(get_u64(bytes, offset)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn control() -> DecodeControlV1 {
        DecodeControlV1::new(
            DecodeControlModeV1::Mtp,
            2,
            1000,
            100,
            7,
            0x1234,
            128,
            3,
            64,
        )
        .unwrap()
    }

    #[allow(clippy::too_many_arguments)]
    fn result_bytes(
        generation: u64,
        status: u32,
        count: u32,
        commit_rows: u32,
        accepted: u32,
        width: u32,
        halt_flags: u32,
        stop_row: u32,
        before: u64,
        after: u64,
        counter_before: u64,
        counter_after: u64,
    ) -> [u8; DECODE_RESULT_BYTES_V1] {
        let mut bytes = [0_u8; DECODE_RESULT_BYTES_V1];
        put_u64(&mut bytes, 0, generation);
        put_u32(&mut bytes, 8, status);
        put_u32(&mut bytes, 12, count);
        put_u32(&mut bytes, 16, commit_rows);
        put_u32(&mut bytes, 20, accepted);
        put_u32(&mut bytes, 24, width);
        put_u32(&mut bytes, 28, halt_flags);
        put_u32(&mut bytes, 32, stop_row);
        put_u64(&mut bytes, 40, before);
        put_u64(&mut bytes, 48, after);
        put_u64(&mut bytes, 56, counter_before);
        put_u64(&mut bytes, 64, counter_after);
        if count != 0 {
            put_u32(&mut bytes, 72, 10);
            put_u64(&mut bytes, 112, (-0.5_f64).to_bits());
        }
        bytes
    }

    #[test]
    fn control_wire_roundtrip_uses_native_offsets_and_active_width() {
        let mut value = control();
        value.begin_phase(DecodePhaseKindV1::Draft, 1, 1).unwrap();
        let encoded = value.encode_le().unwrap();
        assert_eq!(encoded.len(), DECODE_CONTROL_BYTES_V1);
        assert_eq!(&encoded[100..104], &2_u32.to_le_bytes());
        assert_eq!(DecodeControlV1::decode_le(&encoded).unwrap(), value);
    }

    #[test]
    fn mtp_phase_width_clamps_to_budget_and_capacity_boundaries() {
        for width in 1..=DECODE_MAX_WIDTH_V1 {
            for remaining in 1..=width + 1 {
                let mut value = DecodeControlV1::new(
                    DecodeControlModeV1::Mtp,
                    width,
                    100 + u64::from(remaining),
                    100,
                    7,
                    0x1234,
                    1 + u64::from(remaining),
                    3,
                    64,
                )
                .unwrap();
                value.begin_phase(DecodePhaseKindV1::Draft, 0, 1).unwrap();
                assert_eq!(value.active_width, width.min(remaining - 1));
                value
                    .begin_phase(DecodePhaseKindV1::Target, 0, width + 1)
                    .unwrap();
                assert_eq!(value.phase_rows, value.active_width + 1);
                value
                    .begin_phase(DecodePhaseKindV1::MtpAlign, width, 1)
                    .unwrap();
                assert_eq!(value.phase_index, value.active_width);
                assert_eq!(value.phase_position, 100 + u64::from(value.active_width));
            }
            for budget_rows in 1..=width {
                let mut value = DecodeControlV1::new(
                    DecodeControlModeV1::Mtp,
                    width,
                    1_000,
                    100,
                    7,
                    0x1234,
                    1 + u64::from(budget_rows),
                    3,
                    64,
                )
                .unwrap();
                value.begin_phase(DecodePhaseKindV1::Draft, 0, 1).unwrap();
                assert_eq!(value.active_width, budget_rows);
                value
                    .begin_phase(DecodePhaseKindV1::Target, 0, width + 1)
                    .unwrap();
                assert_eq!(value.phase_rows, budget_rows + 1);
            }
        }
    }

    #[test]
    fn result_accepts_zero_and_narrow_active_widths() {
        for width in 0..=2 {
            let bytes = result_bytes(
                1,
                DecodeControlStatusV1::Ok as u32,
                1,
                1,
                0,
                width,
                if width == 0 {
                    RESULT_HALT_ALL_ACCEPT
                } else {
                    0
                },
                0,
                100,
                101,
                7,
                8,
            );
            assert!(
                decode_result_le(
                    &bytes,
                    1,
                    1,
                    DecodeControlModeV1::Mtp,
                    2,
                    100,
                    7,
                    1,
                    128,
                    64,
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn control_constructor_rejects_target_width_and_pending_token() {
        assert!(
            DecodeControlV1::new(DecodeControlModeV1::TargetOnly, 2, 100, 0, 0, 0, 4, 1, 4,)
                .is_err()
        );
        assert!(
            DecodeControlV1::new(DecodeControlModeV1::TargetOnly, 1, 100, 0, 0, 0, 4, 4, 4,)
                .is_err()
        );
    }

    #[test]
    fn control_rejects_output_budget_beyond_physical_capacity() {
        let mut value =
            DecodeControlV1::new(DecodeControlModeV1::Mtp, 2, 102, 100, 7, 0x1234, 3, 3, 64)
                .unwrap();
        value.output_count = 1;
        value.output_limit = 4;
        assert!(value.validate(64).is_err());
    }

    #[test]
    fn result_accepts_eos_with_positive_commit_rows() {
        let bytes = result_bytes(
            1,
            DecodeControlStatusV1::Ok as u32,
            2,
            2,
            2,
            2,
            RESULT_HALT_STOP,
            1,
            100,
            102,
            7,
            9,
        );
        let result = decode_result_le(
            &bytes,
            1,
            1,
            DecodeControlModeV1::Mtp,
            2,
            100,
            7,
            1,
            128,
            64,
        )
        .unwrap();
        assert!(result.stopped());
        assert_eq!(result.commit_rows, 2);
        assert_eq!(result.selections.len(), 2);
    }

    #[test]
    fn result_noop_does_not_advance_and_stale_generation_fails() {
        let bytes = result_bytes(
            4,
            DecodeControlStatusV1::Noop as u32,
            0,
            0,
            0,
            2,
            RESULT_HALT_NOOP,
            0,
            100,
            100,
            7,
            7,
        );
        assert!(
            decode_result_le(
                &bytes,
                0,
                4,
                DecodeControlModeV1::Mtp,
                2,
                100,
                7,
                128,
                128,
                64,
            )
            .is_ok()
        );
        assert!(
            decode_result_le(
                &bytes,
                0,
                3,
                DecodeControlModeV1::Mtp,
                2,
                100,
                7,
                128,
                128,
                64,
            )
            .is_err()
        );
    }

    #[test]
    fn result_rejects_errors_and_malformed_prefix() {
        let error = result_bytes(
            1,
            DecodeControlStatusV1::InvalidDecision as u32,
            0,
            0,
            0,
            2,
            RESULT_HALT_INVALID,
            0,
            100,
            100,
            7,
            7,
        );
        assert!(
            decode_result_le(
                &error,
                1,
                1,
                DecodeControlModeV1::Mtp,
                2,
                100,
                7,
                1,
                128,
                64,
            )
            .is_err()
        );
        let malformed = result_bytes(
            1,
            DecodeControlStatusV1::Ok as u32,
            2,
            1,
            2,
            2,
            0,
            0,
            100,
            102,
            7,
            9,
        );
        assert!(
            decode_result_le(
                &malformed,
                1,
                1,
                DecodeControlModeV1::Mtp,
                2,
                100,
                7,
                1,
                128,
                64,
            )
            .is_err()
        );
    }
}

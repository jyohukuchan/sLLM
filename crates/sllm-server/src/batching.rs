//! Host-side continuous request-batching policy and checked row mapping.
//!
//! This module intentionally owns no model or device resource.  A GPU backend
//! may consume a [`BatchRowMapV1`] only after it can bind every row to an
//! independent request-local KV/GDN owner.

#![cfg_attr(not(test), allow(dead_code))]

use std::collections::{BTreeMap, BTreeSet, VecDeque};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct BatchRequestIdV1(u64);

impl BatchRequestIdV1 {
    pub(crate) fn new(value: u64) -> Result<Self, BatchPlannerErrorV1> {
        (value != 0)
            .then_some(Self(value))
            .ok_or(BatchPlannerErrorV1::ZeroRequestId)
    }
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) enum BatchCompatibilityLaneV1 {
    DenseBf16Greedy,
    DenseBf16Sampled,
    Singleton,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub(crate) struct BatchCompatibilityClassV1 {
    model_fingerprint: String,
    lane: BatchCompatibilityLaneV1,
}

impl BatchCompatibilityClassV1 {
    pub(crate) fn new(
        model_fingerprint: impl Into<String>,
        lane: BatchCompatibilityLaneV1,
    ) -> Result<Self, BatchPlannerErrorV1> {
        let model_fingerprint = model_fingerprint.into();
        if model_fingerprint.is_empty() {
            return Err(BatchPlannerErrorV1::EmptyCompatibilityIdentity);
        }
        Ok(Self {
            model_fingerprint,
            lane,
        })
    }

    fn batch_limit(&self, configured: usize) -> usize {
        match self.lane {
            BatchCompatibilityLaneV1::Singleton => 1,
            BatchCompatibilityLaneV1::DenseBf16Greedy
            | BatchCompatibilityLaneV1::DenseBf16Sampled => configured,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BatchRequestPhaseV1 {
    Queued,
    PrefillInFlight,
    DecodeReady,
    DecodeInFlight,
    Backpressured,
    Finished,
    Cancelled,
    Failed,
}

impl BatchRequestPhaseV1 {
    const fn terminal(self) -> bool {
        matches!(self, Self::Finished | Self::Cancelled | Self::Failed)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BatchRowV1 {
    pub(crate) row: usize,
    pub(crate) request_id: BatchRequestIdV1,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct BatchRowMapV1 {
    rows: Vec<BatchRowV1>,
}

impl BatchRowMapV1 {
    fn new(requests: Vec<BatchRequestIdV1>) -> Result<Self, BatchPlannerErrorV1> {
        if requests.is_empty() {
            return Err(BatchPlannerErrorV1::EmptyRowMap);
        }
        let unique = requests.iter().copied().collect::<BTreeSet<_>>();
        if unique.len() != requests.len() {
            return Err(BatchPlannerErrorV1::DuplicateRowRequest);
        }
        Ok(Self {
            rows: requests
                .into_iter()
                .enumerate()
                .map(|(row, request_id)| BatchRowV1 { row, request_id })
                .collect(),
        })
    }

    pub(crate) fn rows(&self) -> &[BatchRowV1] {
        &self.rows
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BatchRoundV1 {
    Prefill(BatchRequestIdV1),
    Decode(BatchRowMapV1),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BatchDecodeDispositionV1 {
    Ready,
    Backpressured,
    Finished,
    Cancelled,
    Failed,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ContinuousBatchConfigV1 {
    max_active: usize,
    max_batch: usize,
    max_decode_rounds_before_prefill: usize,
}

impl ContinuousBatchConfigV1 {
    pub(crate) fn new(
        max_active: usize,
        max_batch: usize,
        max_decode_rounds_before_prefill: usize,
    ) -> Result<Self, BatchPlannerErrorV1> {
        if max_active == 0 || max_batch == 0 || max_decode_rounds_before_prefill == 0 {
            return Err(BatchPlannerErrorV1::ZeroBound);
        }
        if max_batch > max_active {
            return Err(BatchPlannerErrorV1::BatchExceedsActive);
        }
        Ok(Self {
            max_active,
            max_batch,
            max_decode_rounds_before_prefill,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum BatchPlannerErrorV1 {
    ZeroRequestId,
    ZeroBound,
    BatchExceedsActive,
    EmptyCompatibilityIdentity,
    DuplicateRequest,
    UnknownRequest,
    ActiveCapacity,
    InvalidTransition,
    EmptyRowMap,
    DuplicateRowRequest,
    StaleRowMap,
}

#[derive(Clone, Debug)]
struct RequestRecordV1 {
    class: BatchCompatibilityClassV1,
    phase: BatchRequestPhaseV1,
    service_rounds: u64,
    cancel_pending: bool,
}

/// Deterministic waiting/running policy used to freeze the Phase 26 host
/// lifecycle independently from any unsafe GPU state aggregation.
pub(crate) struct ContinuousBatchPlannerV1 {
    config: ContinuousBatchConfigV1,
    requests: BTreeMap<BatchRequestIdV1, RequestRecordV1>,
    waiting: VecDeque<BatchRequestIdV1>,
    decode_ready: VecDeque<BatchRequestIdV1>,
    active_count: usize,
    decode_rounds_since_prefill: usize,
}

impl ContinuousBatchPlannerV1 {
    pub(crate) fn new(config: ContinuousBatchConfigV1) -> Self {
        Self {
            config,
            requests: BTreeMap::new(),
            waiting: VecDeque::new(),
            decode_ready: VecDeque::new(),
            active_count: 0,
            decode_rounds_since_prefill: 0,
        }
    }

    pub(crate) fn admit(
        &mut self,
        request_id: BatchRequestIdV1,
        class: BatchCompatibilityClassV1,
    ) -> Result<(), BatchPlannerErrorV1> {
        if self.requests.contains_key(&request_id) {
            return Err(BatchPlannerErrorV1::DuplicateRequest);
        }
        if self.active_count == self.config.max_active {
            return Err(BatchPlannerErrorV1::ActiveCapacity);
        }
        self.requests.insert(
            request_id,
            RequestRecordV1 {
                class,
                phase: BatchRequestPhaseV1::Queued,
                service_rounds: 0,
                cancel_pending: false,
            },
        );
        self.waiting.push_back(request_id);
        self.active_count += 1;
        Ok(())
    }

    pub(crate) fn next_round(&mut self) -> Result<Option<BatchRoundV1>, BatchPlannerErrorV1> {
        self.compact_queues();
        let prefill_due = !self.waiting.is_empty()
            && (self.decode_ready.is_empty()
                || self.decode_rounds_since_prefill
                    >= self.config.max_decode_rounds_before_prefill);
        if prefill_due {
            let request_id = self
                .waiting
                .pop_front()
                .ok_or(BatchPlannerErrorV1::UnknownRequest)?;
            self.transition(
                request_id,
                BatchRequestPhaseV1::Queued,
                BatchRequestPhaseV1::PrefillInFlight,
            )?;
            self.decode_rounds_since_prefill = 0;
            return Ok(Some(BatchRoundV1::Prefill(request_id)));
        }
        let Some(seed) = self.decode_ready.pop_front() else {
            return Ok(None);
        };
        let class = self.record(seed)?.class.clone();
        let limit = class.batch_limit(self.config.max_batch);
        let mut selected = vec![seed];
        let scan = self.decode_ready.len();
        for _ in 0..scan {
            let request_id = self
                .decode_ready
                .pop_front()
                .ok_or(BatchPlannerErrorV1::UnknownRequest)?;
            if selected.len() < limit && self.record(request_id)?.class == class {
                selected.push(request_id);
            } else {
                self.decode_ready.push_back(request_id);
            }
        }
        for &request_id in &selected {
            self.transition(
                request_id,
                BatchRequestPhaseV1::DecodeReady,
                BatchRequestPhaseV1::DecodeInFlight,
            )?;
        }
        self.decode_rounds_since_prefill = self.decode_rounds_since_prefill.saturating_add(1);
        Ok(Some(BatchRoundV1::Decode(BatchRowMapV1::new(selected)?)))
    }

    pub(crate) fn complete_prefill(
        &mut self,
        request_id: BatchRequestIdV1,
    ) -> Result<(), BatchPlannerErrorV1> {
        let record = self
            .requests
            .get_mut(&request_id)
            .ok_or(BatchPlannerErrorV1::UnknownRequest)?;
        if record.phase != BatchRequestPhaseV1::PrefillInFlight {
            return Err(BatchPlannerErrorV1::InvalidTransition);
        }
        if record.cancel_pending {
            record.phase = BatchRequestPhaseV1::Cancelled;
            record.cancel_pending = false;
            self.active_count -= 1;
        } else {
            record.phase = BatchRequestPhaseV1::DecodeReady;
            self.decode_ready.push_back(request_id);
        }
        Ok(())
    }

    pub(crate) fn complete_decode(
        &mut self,
        rows: &BatchRowMapV1,
        dispositions: &[(BatchRequestIdV1, BatchDecodeDispositionV1)],
    ) -> Result<(), BatchPlannerErrorV1> {
        let expected = rows
            .rows()
            .iter()
            .map(|row| row.request_id)
            .collect::<BTreeSet<_>>();
        let actual = dispositions
            .iter()
            .map(|(request_id, _)| *request_id)
            .collect::<BTreeSet<_>>();
        if expected != actual || actual.len() != dispositions.len() {
            return Err(BatchPlannerErrorV1::StaleRowMap);
        }
        if expected.iter().any(|&request_id| {
            self.record(request_id).map_or(true, |record| {
                record.phase != BatchRequestPhaseV1::DecodeInFlight
            })
        }) {
            return Err(BatchPlannerErrorV1::StaleRowMap);
        }
        for &(request_id, disposition) in dispositions {
            let record = self
                .requests
                .get_mut(&request_id)
                .ok_or(BatchPlannerErrorV1::UnknownRequest)?;
            let next = if record.cancel_pending {
                BatchRequestPhaseV1::Cancelled
            } else {
                match disposition {
                    BatchDecodeDispositionV1::Ready => BatchRequestPhaseV1::DecodeReady,
                    BatchDecodeDispositionV1::Backpressured => BatchRequestPhaseV1::Backpressured,
                    BatchDecodeDispositionV1::Finished => BatchRequestPhaseV1::Finished,
                    BatchDecodeDispositionV1::Cancelled => BatchRequestPhaseV1::Cancelled,
                    BatchDecodeDispositionV1::Failed => BatchRequestPhaseV1::Failed,
                }
            };
            record.phase = next;
            record.service_rounds = record.service_rounds.saturating_add(1);
            record.cancel_pending = false;
            if next == BatchRequestPhaseV1::DecodeReady {
                self.decode_ready.push_back(request_id);
            }
            if next.terminal() {
                self.active_count -= 1;
            }
        }
        Ok(())
    }

    pub(crate) fn resume(
        &mut self,
        request_id: BatchRequestIdV1,
    ) -> Result<(), BatchPlannerErrorV1> {
        self.transition(
            request_id,
            BatchRequestPhaseV1::Backpressured,
            BatchRequestPhaseV1::DecodeReady,
        )?;
        self.decode_ready.push_back(request_id);
        Ok(())
    }

    pub(crate) fn cancel(
        &mut self,
        request_id: BatchRequestIdV1,
    ) -> Result<(), BatchPlannerErrorV1> {
        let record = self
            .requests
            .get_mut(&request_id)
            .ok_or(BatchPlannerErrorV1::UnknownRequest)?;
        if record.phase.terminal() {
            return Err(BatchPlannerErrorV1::InvalidTransition);
        }
        if matches!(
            record.phase,
            BatchRequestPhaseV1::PrefillInFlight | BatchRequestPhaseV1::DecodeInFlight
        ) {
            record.cancel_pending = true;
            return Ok(());
        }
        record.phase = BatchRequestPhaseV1::Cancelled;
        self.active_count -= 1;
        Ok(())
    }

    pub(crate) fn phase(
        &self,
        request_id: BatchRequestIdV1,
    ) -> Result<BatchRequestPhaseV1, BatchPlannerErrorV1> {
        Ok(self.record(request_id)?.phase)
    }

    #[cfg(test)]
    fn service_rounds(&self, request_id: BatchRequestIdV1) -> Result<u64, BatchPlannerErrorV1> {
        Ok(self.record(request_id)?.service_rounds)
    }

    fn record(
        &self,
        request_id: BatchRequestIdV1,
    ) -> Result<&RequestRecordV1, BatchPlannerErrorV1> {
        self.requests
            .get(&request_id)
            .ok_or(BatchPlannerErrorV1::UnknownRequest)
    }

    fn transition(
        &mut self,
        request_id: BatchRequestIdV1,
        expected: BatchRequestPhaseV1,
        next: BatchRequestPhaseV1,
    ) -> Result<(), BatchPlannerErrorV1> {
        let record = self
            .requests
            .get_mut(&request_id)
            .ok_or(BatchPlannerErrorV1::UnknownRequest)?;
        if record.phase != expected {
            return Err(BatchPlannerErrorV1::InvalidTransition);
        }
        record.phase = next;
        Ok(())
    }

    fn compact_queues(&mut self) {
        self.waiting.retain(|request_id| {
            self.requests
                .get(request_id)
                .is_some_and(|record| record.phase == BatchRequestPhaseV1::Queued)
        });
        self.decode_ready.retain(|request_id| {
            self.requests
                .get(request_id)
                .is_some_and(|record| record.phase == BatchRequestPhaseV1::DecodeReady)
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(value: u64) -> BatchRequestIdV1 {
        BatchRequestIdV1::new(value).unwrap()
    }

    fn class(lane: BatchCompatibilityLaneV1) -> BatchCompatibilityClassV1 {
        BatchCompatibilityClassV1::new("sha256:model", lane).unwrap()
    }

    fn planner(max_active: usize, max_batch: usize) -> ContinuousBatchPlannerV1 {
        ContinuousBatchPlannerV1::new(
            ContinuousBatchConfigV1::new(max_active, max_batch, 2).unwrap(),
        )
    }

    fn prefill_all(planner: &mut ContinuousBatchPlannerV1, count: u64) {
        for value in 1..=count {
            let round = planner.next_round().unwrap().unwrap();
            assert_eq!(round, BatchRoundV1::Prefill(id(value)));
            planner.complete_prefill(id(value)).unwrap();
            // Force the next queued prefill for deterministic setup.
            planner.decode_rounds_since_prefill = 2;
        }
    }

    #[test]
    fn bounds_identity_and_duplicate_admission_fail_closed() {
        assert_eq!(
            BatchRequestIdV1::new(0),
            Err(BatchPlannerErrorV1::ZeroRequestId)
        );
        assert_eq!(
            ContinuousBatchConfigV1::new(1, 2, 1),
            Err(BatchPlannerErrorV1::BatchExceedsActive)
        );
        assert_eq!(
            BatchCompatibilityClassV1::new("", BatchCompatibilityLaneV1::DenseBf16Greedy),
            Err(BatchPlannerErrorV1::EmptyCompatibilityIdentity)
        );
        let mut planner = planner(1, 1);
        planner
            .admit(id(1), class(BatchCompatibilityLaneV1::DenseBf16Greedy))
            .unwrap();
        assert_eq!(
            planner.admit(id(1), class(BatchCompatibilityLaneV1::DenseBf16Greedy)),
            Err(BatchPlannerErrorV1::DuplicateRequest)
        );
        assert_eq!(
            planner.admit(id(2), class(BatchCompatibilityLaneV1::DenseBf16Greedy)),
            Err(BatchPlannerErrorV1::ActiveCapacity)
        );
    }

    #[test]
    fn c1_c2_c3_c4_c7_c8_rows_are_unique_and_round_robin_fair() {
        for count in [1_u64, 2, 3, 4, 7, 8] {
            let mut planner = planner(count as usize, count.min(4) as usize);
            for value in 1..=count {
                planner
                    .admit(id(value), class(BatchCompatibilityLaneV1::DenseBf16Greedy))
                    .unwrap();
            }
            prefill_all(&mut planner, count);
            for _ in 0..16 {
                let BatchRoundV1::Decode(rows) = planner.next_round().unwrap().unwrap() else {
                    panic!("expected decode round");
                };
                let ids = rows
                    .rows()
                    .iter()
                    .enumerate()
                    .map(|(expected_row, row)| {
                        assert_eq!(row.row, expected_row);
                        row.request_id
                    })
                    .collect::<Vec<_>>();
                let unique = ids.iter().copied().collect::<BTreeSet<_>>();
                assert_eq!(unique.len(), ids.len());
                planner
                    .complete_decode(
                        &rows,
                        &ids.iter()
                            .map(|&request_id| (request_id, BatchDecodeDispositionV1::Ready))
                            .collect::<Vec<_>>(),
                    )
                    .unwrap();
            }
            let rounds = (1..=count)
                .map(|value| planner.service_rounds(id(value)).unwrap())
                .collect::<Vec<_>>();
            assert!(rounds.iter().max().unwrap() - rounds.iter().min().unwrap() <= 1);
        }
    }

    #[test]
    fn compatibility_backpressure_and_singleton_lanes_are_isolated() {
        let mut planner = planner(4, 4);
        planner
            .admit(id(1), class(BatchCompatibilityLaneV1::DenseBf16Greedy))
            .unwrap();
        planner
            .admit(id(2), class(BatchCompatibilityLaneV1::DenseBf16Greedy))
            .unwrap();
        planner
            .admit(id(3), class(BatchCompatibilityLaneV1::DenseBf16Sampled))
            .unwrap();
        planner
            .admit(id(4), class(BatchCompatibilityLaneV1::Singleton))
            .unwrap();
        prefill_all(&mut planner, 4);

        let BatchRoundV1::Decode(rows) = planner.next_round().unwrap().unwrap() else {
            panic!("expected decode round");
        };
        assert_eq!(rows.rows().len(), 2);
        planner
            .complete_decode(
                &rows,
                &[
                    (id(1), BatchDecodeDispositionV1::Backpressured),
                    (id(2), BatchDecodeDispositionV1::Ready),
                ],
            )
            .unwrap();
        assert_eq!(
            planner.phase(id(1)).unwrap(),
            BatchRequestPhaseV1::Backpressured
        );

        let BatchRoundV1::Decode(sampled) = planner.next_round().unwrap().unwrap() else {
            panic!("expected sampled round");
        };
        assert_eq!(sampled.rows().len(), 1);
        assert_eq!(sampled.rows()[0].request_id, id(3));
        planner
            .complete_decode(&sampled, &[(id(3), BatchDecodeDispositionV1::Finished)])
            .unwrap();
        let BatchRoundV1::Decode(singleton) = planner.next_round().unwrap().unwrap() else {
            panic!("expected singleton round");
        };
        assert_eq!(singleton.rows().len(), 1);
        assert_eq!(singleton.rows()[0].request_id, id(4));
        planner
            .complete_decode(&singleton, &[(id(4), BatchDecodeDispositionV1::Cancelled)])
            .unwrap();
        planner.resume(id(1)).unwrap();
        assert_eq!(
            planner.phase(id(1)).unwrap(),
            BatchRequestPhaseV1::DecodeReady
        );
    }

    #[test]
    fn stale_row_completion_is_rejected_and_inflight_cancel_is_deferred() {
        let mut planner = planner(2, 2);
        for value in 1..=2 {
            planner
                .admit(id(value), class(BatchCompatibilityLaneV1::DenseBf16Greedy))
                .unwrap();
        }
        prefill_all(&mut planner, 2);
        let BatchRoundV1::Decode(rows) = planner.next_round().unwrap().unwrap() else {
            panic!("expected decode round");
        };
        planner.cancel(id(1)).unwrap();
        assert_eq!(
            planner.phase(id(1)).unwrap(),
            BatchRequestPhaseV1::DecodeInFlight
        );
        assert_eq!(
            planner.complete_decode(&rows, &[(id(1), BatchDecodeDispositionV1::Ready)]),
            Err(BatchPlannerErrorV1::StaleRowMap)
        );
        planner
            .complete_decode(
                &rows,
                &[
                    (id(1), BatchDecodeDispositionV1::Ready),
                    (id(2), BatchDecodeDispositionV1::Failed),
                ],
            )
            .unwrap();
        assert_eq!(
            planner.phase(id(1)).unwrap(),
            BatchRequestPhaseV1::Cancelled
        );
        assert_eq!(planner.phase(id(2)).unwrap(), BatchRequestPhaseV1::Failed);
    }

    #[test]
    fn prefill_inflight_cancel_is_published_only_after_completion() {
        let mut planner = planner(1, 1);
        planner
            .admit(id(1), class(BatchCompatibilityLaneV1::DenseBf16Greedy))
            .unwrap();
        assert_eq!(
            planner.next_round().unwrap(),
            Some(BatchRoundV1::Prefill(id(1)))
        );
        planner.cancel(id(1)).unwrap();
        assert_eq!(
            planner.phase(id(1)).unwrap(),
            BatchRequestPhaseV1::PrefillInFlight
        );
        assert_eq!(
            planner.admit(id(2), class(BatchCompatibilityLaneV1::DenseBf16Greedy)),
            Err(BatchPlannerErrorV1::ActiveCapacity)
        );
        planner.complete_prefill(id(1)).unwrap();
        assert_eq!(
            planner.phase(id(1)).unwrap(),
            BatchRequestPhaseV1::Cancelled
        );
        planner
            .admit(id(2), class(BatchCompatibilityLaneV1::DenseBf16Greedy))
            .unwrap();
    }
}

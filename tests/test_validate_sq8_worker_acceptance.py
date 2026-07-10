import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = REPO_ROOT / "tools" / "validate-sq8-worker-acceptance.py"
GIT_COMMIT = "a" * 40
BINARY_BYTES = b"synthetic ullm-sq8-worker release binary\n"
BINARY_SHA256 = hashlib.sha256(BINARY_BYTES).hexdigest()
WORKER_PID = 4242
WORKER_PPID = 4000
WORKER_STARTTIME = 123456
WORKER_EXE = "/tmp/ullm-sq8-worker"
RSS_KB = 200_000
VRAM_BYTES = 18_000_000_000


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_sq8_worker_acceptance", VALIDATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {VALIDATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_validator()


def compact_json(value) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def proc_stat(pid=WORKER_PID, ppid=WORKER_PPID, starttime=WORKER_STARTTIME):
    fields_after_state = ["0"] * 51
    fields_after_state[0] = str(ppid)
    fields_after_state[18] = str(starttime)
    return f"{pid} (ullm worker) S {' '.join(fields_after_state)}\n"


class EvidenceBuilder:
    def __init__(
        self,
        *,
        measured_cancel_delay_ns: int = 1000,
        rss_for_point=None,
        vram_for_point=None,
        threads_for_point=None,
    ):
        self.now = 1_000_000
        self.records = []
        self.measured_cancel_delay_ns = measured_cancel_delay_ns
        self.rss_for_point = rss_for_point or (lambda _: RSS_KB)
        self.vram_for_point = vram_for_point or (lambda _: VRAM_BYTES)
        self.threads_for_point = threads_for_point or (lambda _: 4)

    def tick(self, amount: int = 1000) -> int:
        self.now += amount
        return self.now

    def add(self, record):
        self.records.append(record)

    def header(self):
        amd_list = compact_json(
            [
                {
                    "gpu": 0,
                    "bdf": "0000:03:00.0",
                    "uuid": "76ff73a1-0000-1000-80a0-8c022586fed6",
                    "kfd_id": 8042,
                },
                {
                    "gpu": 2,
                    "bdf": VALIDATOR.GPU_BDF,
                    "uuid": VALIDATOR.GPU_UUID,
                    "kfd_id": VALIDATOR.KFD_GPU_ID,
                },
            ]
        )
        amd_version = (
            "AMDSMI Tool: 26.2.2+e1a6bc5663 | "
            "AMDSMI Library version: 26.2.2 | ROCm version: 7.2.1\n"
        )
        git_status_raw = ""
        self.add(
            {
                "schema_version": VALIDATOR.RAW_SCHEMA_VERSION,
                "record_type": "header",
                "clock": "python.time.monotonic_ns",
                "build": {
                    "git_commit": GIT_COMMIT,
                    "tracked_clean": True,
                    "git_status_raw": git_status_raw,
                    "git_status_raw_sha256": sha256_text(git_status_raw),
                    "binary_sha256": BINARY_SHA256,
                    "artifact_manifest_sha256": VALIDATOR.ARTIFACT_MANIFEST_SHA256,
                    "artifact_content_sha256": VALIDATOR.ARTIFACT_CONTENT_SHA256,
                    "package_manifest_sha256": VALIDATOR.PACKAGE_MANIFEST_SHA256,
                },
                "worker": {
                    "pid": WORKER_PID,
                    "ppid": WORKER_PPID,
                    "starttime_ticks": WORKER_STARTTIME,
                    "exe": WORKER_EXE,
                },
                "device": {
                    "gpu_index": VALIDATOR.GPU_INDEX,
                    "bdf": VALIDATOR.GPU_BDF,
                    "uuid": VALIDATOR.GPU_UUID,
                    "kfd_gpu_id": VALIDATOR.KFD_GPU_ID,
                    "amd_smi_list_raw_json": amd_list,
                    "amd_smi_list_raw_sha256": sha256_text(amd_list),
                },
                "environment": {
                    "hip_visible_devices": "1",
                    "required_hip_guards": {
                        name: "1" for name in sorted(VALIDATOR.REQUIRED_HIP_GUARDS)
                    },
                    "amd_smi_version_raw": amd_version,
                    "amd_smi_version_raw_sha256": sha256_text(amd_version),
                    "preflight_kfd_processes": [],
                },
                "schedule": dict(VALIDATOR.SCHEDULE),
                "thresholds": dict(VALIDATOR.THRESHOLDS),
            }
        )

    def worker_event(self, event):
        observed = self.tick()
        raw = compact_json(event)
        self.add(
            {
                "schema_version": VALIDATOR.RAW_SCHEMA_VERSION,
                "record_type": "worker_event",
                "observed_monotonic_ns": observed,
                "raw_json": raw,
                "raw_sha256": sha256_text(raw),
                "event": event,
            }
        )
        return observed

    def isolation_check(self, phase, request_index, request_id, release_observed):
        self.add(
            {
                "schema_version": VALIDATOR.RAW_SCHEMA_VERSION,
                "record_type": "isolation_check",
                "phase": phase,
                "request_index": request_index,
                "request_id": request_id,
                "release_observed_monotonic_ns": release_observed,
                "captured_monotonic_ns": self.tick(),
                "kfd_processes": [
                    {
                        "pid": WORKER_PID,
                        "vram_raw": str(VRAM_BYTES),
                        "vram_bytes": VRAM_BYTES,
                    }
                ],
            }
        )

    def ready(self):
        observed = self.worker_event(
            {
                "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
                "type": "ready",
                "model": "ullm-qwen3-14b-sq8",
                "model_revision": VALIDATOR.MODEL_REVISION,
                "artifact_content_sha256": VALIDATOR.ARTIFACT_CONTENT_SHA256,
                "package_manifest_sha256": VALIDATOR.PACKAGE_MANIFEST_SHA256,
                "device": "gfx1201",
                "execution_profile": "rdna4_w8a8_block_ck",
                "context_length": 4096,
                "max_new_tokens": 512,
            }
        )
        self.isolation_check("ready", None, None, None)

    def command(self, phase, request_index, request_id, command_type, **extra):
        started = self.tick()
        completed = self.tick()
        if command_type == "generate":
            raw_command = {
                "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
                "type": "generate",
                "request_id": request_id,
                "prompt_token_ids": list(range(1, extra["prompt_tokens"] + 1)),
                "max_new_tokens": extra["max_new_tokens"],
                "sampling": extra["sampling"],
                "eos_token_ids": extra["eos_token_ids"],
            }
        elif command_type == "cancel":
            raw_command = {
                "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
                "type": "cancel",
                "request_id": request_id,
                "reason": extra["cancel_reason"],
            }
        else:
            raw_command = {
                "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
                "type": "shutdown",
            }
        raw = compact_json(raw_command)
        record = {
            "schema_version": VALIDATOR.RAW_SCHEMA_VERSION,
            "record_type": "command",
            "phase": phase,
            "request_index": request_index,
            "request_id": request_id,
            "command_type": command_type,
            "write_started_monotonic_ns": started,
            "write_completed_monotonic_ns": completed,
            "raw_json": raw,
            "raw_sha256": sha256_text(raw),
            **extra,
        }
        self.add(record)
        return started

    def generate(self, phase, index, request_id, prompt_tokens, max_new_tokens):
        return self.command(
            phase,
            index,
            request_id,
            "generate",
            prompt_tokens=prompt_tokens,
            prompt_token_ids_sha256=VALIDATOR.ascending_prompt_sha256(prompt_tokens),
            max_new_tokens=max_new_tokens,
            sampling={"temperature": 0.0, "top_p": 1.0, "top_k": 20, "seed": 0},
            eos_token_ids=[151645, 151643],
        )

    def cancel(self, phase, index, request_id, target):
        return self.command(
            phase,
            index,
            request_id,
            "cancel",
            cancel_reason="operator",
            cancel_target=target,
        )

    def run_request(self, phase, index, request_id, target=None):
        prompt_tokens = 128 if target == "prompt" else 8
        max_new_tokens = 512 if target else 2
        self.generate(phase, index, request_id, prompt_tokens, max_new_tokens)
        self.worker_event(
            {
                "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
                "type": "started",
                "request_id": request_id,
                "prompt_tokens": prompt_tokens,
            }
        )
        cancel_started = None
        completion = 0
        if target == "prompt":
            cancel_started = self.cancel(phase, index, request_id, target)
        else:
            self.worker_event(
                {
                    "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
                    "type": "progress",
                    "request_id": request_id,
                    "phase": "prefill",
                    "processed_prompt_tokens": 8,
                }
            )
            self.worker_event(
                {
                    "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
                    "type": "token",
                    "request_id": request_id,
                    "index": 0,
                    "token_id": 353,
                }
            )
            completion = 1
            if target == "decode":
                cancel_started = self.cancel(phase, index, request_id, target)
            else:
                self.worker_event(
                    {
                        "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
                        "type": "token",
                        "request_id": request_id,
                        "index": 1,
                        "token_id": 10,
                    }
                )
                completion = 2
        if target and phase == "latency_measured":
            self.tick(self.measured_cancel_delay_ns)
        event = {
            "schema_version": VALIDATOR.WORKER_SCHEMA_VERSION,
            "type": "released",
            "request_id": request_id,
            "outcome": "cancelled" if target else "length",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "reset_complete": True,
        }
        if target:
            event["cancel_reason"] = "operator"
        release_observed = self.worker_event(event)
        self.isolation_check(phase, index, request_id, release_observed)
        return {
            "request_index": index,
            "request_id": request_id,
            "outcome": event["outcome"],
            "observed": release_observed,
            "cancel_started": cancel_started,
        }

    def metric(self, boundary):
        raw = compact_json({"gpu_data": [{"gpu": VALIDATOR.GPU_INDEX}]})
        self.add(
            {
                "schema_version": VALIDATOR.RAW_SCHEMA_VERSION,
                "record_type": "gpu_metric",
                "boundary": boundary,
                "captured_monotonic_ns": self.tick(),
                "raw_json": raw,
                "raw_sha256": sha256_text(raw),
            }
        )

    def resource_sample(self, phase, point_index, release, sample_index):
        if sample_index == 0:
            settle = release["observed"]
            self.now = max(self.now + 1, settle + 5_000_000_000)
        else:
            settle = self._point_settle
            self.now += 1_000_000_000
        self._point_settle = settle
        rss_kb = self.rss_for_point(point_index)
        vram = self.vram_for_point(point_index)
        threads = self.threads_for_point(point_index)
        stat_raw = proc_stat()
        status_raw = f"Name:\tullm worker\nVmRSS:\t{rss_kb} kB\nThreads:\t{threads}\n"
        children_raw = ""
        fd_names = [str(index) for index in range(12)]
        process_raw = compact_json(
            [
                {
                    "gpu": VALIDATOR.GPU_INDEX,
                    "process_list": [
                        {
                            "process_info": {
                                "pid": WORKER_PID,
                                "mem_usage": {"value": vram, "unit": "B"},
                            }
                        }
                    ],
                }
            ]
        )
        baseline = phase == "baseline"
        self.add(
            {
                "schema_version": VALIDATOR.RAW_SCHEMA_VERSION,
                "record_type": "resource_sample",
                "phase": phase,
                "request_index": None if baseline else release["request_index"],
                "request_id": None if baseline else release["request_id"],
                "release_outcome": None if baseline else release["outcome"],
                "release_observed_monotonic_ns": None if baseline else release["observed"],
                "settle_started_monotonic_ns": settle,
                "sample_index": sample_index,
                "sample_started_monotonic_ns": self.now,
                "worker": {
                    "pid": WORKER_PID,
                    "ppid": WORKER_PPID,
                    "exe": WORKER_EXE,
                    "starttime_ticks_before": WORKER_STARTTIME,
                    "starttime_ticks_after": WORKER_STARTTIME,
                    "vmrss_kb": rss_kb,
                    "vmrss_bytes": rss_kb * 1024,
                    "threads": threads,
                    "fd_count": 12,
                    "children": [],
                    "stat_before_raw": stat_raw,
                    "stat_before_raw_sha256": sha256_text(stat_raw),
                    "status_raw": status_raw,
                    "status_raw_sha256": sha256_text(status_raw),
                    "exe_target": WORKER_EXE,
                    "fd_names": fd_names,
                    "children_raw": children_raw,
                    "children_raw_sha256": sha256_text(children_raw),
                    "stat_after_raw": stat_raw,
                    "stat_after_raw_sha256": sha256_text(stat_raw),
                },
                "gpu": {
                    "index": VALIDATOR.GPU_INDEX,
                    "bdf": VALIDATOR.GPU_BDF,
                    "uuid": VALIDATOR.GPU_UUID,
                    "kfd_gpu_id": VALIDATOR.KFD_GPU_ID,
                    "process_raw_json": process_raw,
                    "process_raw_sha256": sha256_text(process_raw),
                    "worker_pid": WORKER_PID,
                    "mem_usage_value": vram,
                    "mem_usage_unit": "B",
                    "kfd_vram_bytes": vram,
                    "kfd_processes": [
                        {
                            "pid": WORKER_PID,
                            "vram_raw": str(vram),
                            "vram_bytes": vram,
                        }
                    ],
                    "kfd_positive_processes": [{"pid": WORKER_PID, "vram_bytes": vram}],
                    "unrelated_positive_kfd_pids": [],
                },
            }
        )

    def resource_point(self, phase, point_index, release):
        for sample_index in range(5):
            self.resource_sample(phase, point_index, release, sample_index)

    @staticmethod
    def fixed_id(phase, index):
        if phase == "latency_warmup":
            return f"p8c-latency-warmup-{index:02d}"
        if phase == "latency_measured":
            return f"p8c-latency-measured-{index:02d}"
        if phase == "resource_warmup":
            return f"p8c-resource-warmup-{index:02d}"
        return f"p8c-resource-measured-{index:03d}"

    @staticmethod
    def resource_target(index):
        if index % 5 != 4:
            return None
        ordinal = (index + 1) // 5
        return "prompt" if ordinal % 2 else "decode"

    def build(self):
        self.header()
        self.ready()
        for phase, count in (("latency_warmup", 2), ("latency_measured", 10)):
            for index in range(1, count + 1):
                request_id = self.fixed_id(phase, index)
                target = "prompt" if index % 2 else "decode"
                self.run_request(phase, index, request_id, target)
                self.run_request(phase, index, request_id + "-recovery")
        self.metric("before")
        warmup_release = None
        for index in range(1, 11):
            warmup_release = self.run_request(
                "resource_warmup",
                index,
                self.fixed_id("resource_warmup", index),
                self.resource_target(index),
            )
        self.resource_point("baseline", 0, warmup_release)
        for index in range(1, 101):
            release = self.run_request(
                "resource_measured",
                index,
                self.fixed_id("resource_measured", index),
                self.resource_target(index),
            )
            self.resource_point("post_release", index, release)
        self.metric("after")
        self.command("shutdown", None, None, "shutdown")
        stdout_eof = self.tick()
        self.add(
            {
                "schema_version": VALIDATOR.RAW_SCHEMA_VERSION,
                "record_type": "process_exit",
                "stdout_eof_monotonic_ns": stdout_eof,
                "exit_observed_monotonic_ns": self.tick(),
                "exit_code": 0,
                "stderr_file": "worker-stderr.jsonl",
                "stderr_sha256": None,
                "final_git_commit": GIT_COMMIT,
                "final_git_status_raw": "",
                "final_git_status_raw_sha256": sha256_text(""),
            }
        )
        return self.records


def write_fixture(directory: Path, records):
    worker_path = directory / "ullm-sq8-worker"
    worker_path.write_bytes(BINARY_BYTES)
    stderr_path = directory / "worker-stderr.jsonl"
    stderr_path.write_text('{"event":"worker_test"}\n', encoding="utf-8")
    records = deepcopy(records)
    records[0]["worker"]["exe"] = str(worker_path)
    for record in records:
        if record.get("record_type") == "resource_sample":
            record["worker"]["exe"] = str(worker_path)
            record["worker"]["exe_target"] = str(worker_path)
    records[-1]["stderr_sha256"] = hashlib.sha256(stderr_path.read_bytes()).hexdigest()
    raw_path = directory / "raw.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(compact_json(record))
            handle.write("\n")
    return raw_path, records


def rewrite_fixture(raw_path: Path, records):
    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(compact_json(record))
            handle.write("\n")


def first_record(records, predicate):
    return next(record for record in records if predicate(record))


def shift_timestamps(value, cutoff, delta):
    if isinstance(value, dict):
        for key, item in value.items():
            if key.endswith("_monotonic_ns") and isinstance(item, int) and item >= cutoff:
                value[key] = item + delta
            else:
                shift_timestamps(item, cutoff, delta)
    elif isinstance(value, list):
        for item in value:
            shift_timestamps(item, cutoff, delta)


class ValidateSq8WorkerAcceptanceTests(unittest.TestCase):
    def test_complete_valid_fixture_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, records = write_fixture(Path(temporary), EvidenceBuilder().build())
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

            self.assertTrue(result["passed"])
            self.assertEqual(result["gate_errors"], [])
            self.assertEqual(result["counts"]["commands"], 169)
            self.assertEqual(result["counts"]["releases"], 134)
            self.assertEqual(result["counts"]["resource_samples"], 505)
            self.assertEqual(result["counts"]["resource_points"], 100)
            self.assertEqual(result["counts"]["theil_sen_pairs_per_series"], 4950)
            self.assertEqual(result["cancellation"]["warmup_samples"], 2)
            self.assertEqual(result["cancellation"]["measured_samples"], 10)
            self.assertEqual(result["resources"]["worker_rss_theil_sen_bytes_per_request"], 0)
            self.assertGreater(len(records), 505)

    def test_cli_writes_validation_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw_path, _ = write_fixture(directory, EvidenceBuilder().build())
            output_path = directory / "validation.json"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(VALIDATOR_PATH),
                    str(raw_path),
                    "--expected-git-commit",
                    GIT_COMMIT,
                    "--expected-binary-sha256",
                    BINARY_SHA256,
                    "--output",
                    str(output_path),
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            stdout_result = json.loads(completed.stdout)
            file_result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(stdout_result, file_result)
            self.assertTrue(file_result["passed"])

    def test_unknown_header_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            records[0]["passed"] = True
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "field set differs"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_duplicate_json_key_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, _ = write_fixture(Path(temporary), EvidenceBuilder().build())
            lines = raw_path.read_text(encoding="utf-8").splitlines(keepends=True)
            duplicate = (
                '{"schema_version":"ullm.sq8.worker_acceptance.raw.v1",'
                + lines[0].lstrip()[1:]
            )
            lines[0] = duplicate
            raw_path.write_text("".join(lines), encoding="utf-8", newline="\n")
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "duplicate JSON key"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_expected_build_identity_is_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw_path, _ = write_fixture(Path(temporary), EvidenceBuilder().build())
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "build/model identity"):
                VALIDATOR.validate_evidence(raw_path, "c" * 40, BINARY_SHA256)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "build/model identity"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, "d" * 64)

    def test_amd_smi_list_is_reparsed_instead_of_trusting_header_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            raw_list = json.loads(records[0]["device"]["amd_smi_list_raw_json"])
            raw_list[-1]["uuid"] = "0" * 36
            encoded = compact_json(raw_list)
            records[0]["device"]["amd_smi_list_raw_json"] = encoded
            records[0]["device"]["amd_smi_list_raw_sha256"] = sha256_text(encoded)
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "unique matching GPU index 2"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_fixed_request_id_matrix_is_rejected_when_changed(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            generate = first_record(
                records,
                lambda record: record.get("record_type") == "command"
                and record.get("phase") == "resource_measured"
                and record.get("request_index") == 1,
            )
            generate["request_id"] = "wrong-resource-request"
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "request ID must be"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_noncontiguous_worker_token_event_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            token = first_record(
                records,
                lambda record: record.get("record_type") == "worker_event"
                and record.get("event", {}).get("request_id") == "p8c-latency-warmup-02"
                and record.get("event", {}).get("type") == "token",
            )
            token["event"]["index"] = 1
            token["raw_json"] = compact_json(token["event"])
            token["raw_sha256"] = sha256_text(token["raw_json"])
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "not contiguous"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_missing_resource_sample_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            sample_position = next(
                index
                for index, record in enumerate(records)
                if record.get("record_type") == "resource_sample"
                and record.get("phase") == "post_release"
                and record.get("request_index") == 50
                and record.get("sample_index") == 3
            )
            records.pop(sample_position)
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "sample indices"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_amd_process_raw_json_is_reparsed(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            sample = first_record(
                records,
                lambda record: record.get("record_type") == "resource_sample",
            )
            raw_process = json.loads(sample["gpu"]["process_raw_json"])
            raw_process[0]["process_list"][0]["process_info"]["mem_usage"]["value"] += 1
            encoded = compact_json(raw_process)
            sample["gpu"]["process_raw_json"] = encoded
            sample["gpu"]["process_raw_sha256"] = sha256_text(encoded)
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "parsed AMD SMI"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_kfd_positive_processes_derive_unrelated_owners(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            sample = first_record(
                records,
                lambda record: record.get("record_type") == "resource_sample",
            )
            sample["gpu"]["kfd_positive_processes"].append(
                {"pid": WORKER_PID + 1, "vram_bytes": 4096}
            )
            sample["gpu"]["unrelated_positive_kfd_pids"] = []
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "differs from raw KFD"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_cancel_p95_gate_is_derived_from_ten_measured_bounds(self):
        with tempfile.TemporaryDirectory() as temporary:
            builder = EvidenceBuilder(measured_cancel_delay_ns=2_100_000_000)
            raw_path, _ = write_fixture(Path(temporary), builder.build())
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertFalse(result["passed"])
            self.assertEqual(result["cancellation"]["measured_samples"], 10)
            self.assertGreater(result["cancellation"]["measured_upper_bound_p95_ns"], 2_000_000_000)
            self.assertTrue(any("p95" in error for error in result["gate_errors"]))

    def test_theil_sen_uses_all_4950_pairs_and_gates_positive_growth(self):
        with tempfile.TemporaryDirectory() as temporary:
            builder = EvidenceBuilder(rss_for_point=lambda index: RSS_KB + index * 300)
            raw_path, _ = write_fixture(Path(temporary), builder.build())
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertFalse(result["passed"])
            self.assertEqual(result["counts"]["theil_sen_pairs_per_series"], 4950)
            self.assertGreater(
                result["resources"]["worker_rss_theil_sen_bytes_per_request"], 262_144
            )
            self.assertTrue(any("RSS Theil-Sen" in error for error in result["gate_errors"]))

    def test_final_delta_gate_is_independent_of_robust_slope(self):
        with tempfile.TemporaryDirectory() as temporary:
            builder = EvidenceBuilder(
                rss_for_point=lambda index: RSS_KB + (70_000 if index == 100 else 0)
            )
            raw_path, _ = write_fixture(Path(temporary), builder.build())
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertFalse(result["passed"])
            self.assertGreater(result["resources"]["final_worker_rss_delta_bytes"], 67_108_864)
            self.assertEqual(result["resources"]["worker_rss_theil_sen_bytes_per_request"], 0)
            self.assertTrue(any("RSS final delta" in error for error in result["gate_errors"]))

    def test_thread_fd_child_medians_are_baseline_gated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mutations = {
                "threads": self._mutate_threads,
                "fds": self._mutate_fds,
                "children": self._mutate_children,
            }
            for diagnostic, mutate in mutations.items():
                with self.subTest(diagnostic=diagnostic):
                    records = EvidenceBuilder().build()
                    for record in records:
                        if (
                            record.get("record_type") == "resource_sample"
                            and record.get("request_index") == 25
                        ):
                            mutate(record["worker"])
                    directory = root / diagnostic
                    directory.mkdir()
                    raw_path, _ = write_fixture(directory, records)
                    result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
                    self.assertFalse(result["passed"])
                    self.assertTrue(
                        any(
                            f"request 25 {diagnostic} median" in error
                            for error in result["gate_errors"]
                        )
                    )

    @staticmethod
    def _mutate_threads(worker):
        worker["threads"] = 5
        worker["status_raw"] = worker["status_raw"].replace("Threads:\t4", "Threads:\t5")
        worker["status_raw_sha256"] = sha256_text(worker["status_raw"])

    @staticmethod
    def _mutate_fds(worker):
        worker["fd_count"] = 13
        worker["fd_names"].append("12")

    @staticmethod
    def _mutate_children(worker):
        worker["children"] = [5000]
        worker["children_raw"] = "5000\n"
        worker["children_raw_sha256"] = sha256_text(worker["children_raw"])

    def test_stderr_must_be_exact_sibling_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            raw_path, records = write_fixture(directory, EvidenceBuilder().build())
            records[-1]["stderr_sha256"] = "0" * 64
            rewrite_fixture(raw_path, records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "stderr SHA-256 differs"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

            records[-1]["stderr_file"] = "../worker-stderr.jsonl"
            rewrite_fixture(raw_path, records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "regular sibling"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_final_git_status_may_change_only_within_profiler_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            final_status = "?? .rocprofv3/new-counter.dat\n"
            records[-1]["final_git_status_raw"] = final_status
            records[-1]["final_git_status_raw_sha256"] = sha256_text(final_status)
            raw_path, _ = write_fixture(Path(temporary), records)
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertTrue(result["passed"])

    def test_final_git_status_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            records[-1]["final_git_status_raw"] = "?? .rocprofv3/new-counter.dat\n"
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "raw SHA-256"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_final_git_dirty_tracked_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            final_status = " M tracked.py\n"
            records[-1]["final_git_status_raw"] = final_status
            records[-1]["final_git_status_raw_sha256"] = sha256_text(final_status)
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "forbidden path"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_final_git_commit_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            records[-1]["final_git_commit"] = "b" * 40
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "final git commit"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_stdout_eof_may_race_before_shutdown_write_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            shutdown = records[-2]
            process_exit = records[-1]
            process_exit["stdout_eof_monotonic_ns"] = (
                shutdown["write_started_monotonic_ns"] + 1
            )
            process_exit["exit_observed_monotonic_ns"] = max(
                process_exit["exit_observed_monotonic_ns"],
                process_exit["stdout_eof_monotonic_ns"],
            )
            raw_path, _ = write_fixture(Path(temporary), records)
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertTrue(result["passed"])

    def test_header_git_preflight_kfd_and_thresholds_fail_closed(self):
        mutations = {
            "git": lambda header: header["build"].update(
                {
                    "git_status_raw": " M tracked.py\n",
                    "git_status_raw_sha256": sha256_text(" M tracked.py\n"),
                }
            ),
            "preflight": lambda header: header["environment"].update(
                {
                    "preflight_kfd_processes": [
                        {"pid": 99, "vram_raw": "4096", "vram_bytes": 4096}
                    ]
                }
            ),
            "threshold": lambda header: header["thresholds"].update(
                {"progress_max_ns": 30_000_000_001}
            ),
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, mutate in mutations.items():
                with self.subTest(name=name):
                    records = EvidenceBuilder().build()
                    mutate(records[0])
                    directory = root / name
                    directory.mkdir()
                    raw_path, _ = write_fixture(directory, records)
                    with self.assertRaises(VALIDATOR.ValidationError):
                        VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_command_raw_hash_and_summary_are_independently_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, recompute_hash in (("hash", False), ("summary", True)):
                with self.subTest(name=name):
                    records = EvidenceBuilder().build()
                    command = first_record(
                        records,
                        lambda record: record.get("record_type") == "command"
                        and record.get("command_type") == "generate",
                    )
                    raw = json.loads(command["raw_json"])
                    raw["prompt_token_ids"][0] = 2
                    command["raw_json"] = compact_json(raw)
                    if recompute_hash:
                        command["raw_sha256"] = sha256_text(command["raw_json"])
                    directory = root / name
                    directory.mkdir()
                    raw_path, _ = write_fixture(directory, records)
                    with self.assertRaises(VALIDATOR.ValidationError):
                        VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_raw_generate_bool_values_cannot_equal_numeric_summary_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            command = first_record(
                records,
                lambda record: record.get("record_type") == "command"
                and record.get("command_type") == "generate",
            )
            raw = json.loads(command["raw_json"])
            raw["sampling"]["temperature"] = False
            raw["sampling"]["top_p"] = True
            command["raw_json"] = compact_json(raw)
            command["raw_sha256"] = sha256_text(command["raw_json"])
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "summary differs"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_worker_event_raw_hash_and_decoded_summary_must_match(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            event = first_record(
                records,
                lambda record: record.get("record_type") == "worker_event"
                and record.get("event", {}).get("type") == "token",
            )
            raw = json.loads(event["raw_json"])
            raw["token_id"] += 1
            event["raw_json"] = compact_json(raw)
            event["raw_sha256"] = sha256_text(event["raw_json"])
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "differs from reparsed raw"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_raw_released_integer_cannot_equal_boolean_reset_summary(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            released = first_record(
                records,
                lambda record: record.get("record_type") == "worker_event"
                and record.get("event", {}).get("type") == "released",
            )
            raw = json.loads(released["raw_json"])
            raw["reset_complete"] = 1
            released["raw_json"] = compact_json(raw)
            released["raw_sha256"] = sha256_text(released["raw_json"])
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(
                VALIDATOR.ValidationError, "differs from reparsed raw"
            ):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_proc_raw_and_worker_executable_hash_are_independently_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = EvidenceBuilder().build()
            sample = first_record(
                records, lambda record: record.get("record_type") == "resource_sample"
            )
            sample["worker"]["status_raw"] = sample["worker"]["status_raw"].replace(
                f"VmRSS:\t{RSS_KB}", f"VmRSS:\t{RSS_KB + 1}"
            )
            sample["worker"]["status_raw_sha256"] = sha256_text(
                sample["worker"]["status_raw"]
            )
            proc_dir = root / "proc"
            proc_dir.mkdir()
            raw_path, _ = write_fixture(proc_dir, records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "derived RSS"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

            binary_dir = root / "binary"
            binary_dir.mkdir()
            raw_path, _ = write_fixture(binary_dir, EvidenceBuilder().build())
            (binary_dir / "ullm-sq8-worker").write_bytes(b"tampered binary")
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "executable"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_raw_kfd_value_and_isolation_order_are_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = EvidenceBuilder().build()
            sample = first_record(
                records, lambda record: record.get("record_type") == "resource_sample"
            )
            sample["gpu"]["kfd_processes"][0]["vram_raw"] = "1"
            kfd_dir = root / "kfd"
            kfd_dir.mkdir()
            raw_path, _ = write_fixture(kfd_dir, records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "raw KFD value"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

            records = EvidenceBuilder().build()
            isolation_index = next(
                index
                for index, record in enumerate(records)
                if record.get("record_type") == "isolation_check"
                and record.get("phase") == "latency_measured"
                and record.get("request_index") == 1
            )
            records.pop(isolation_index)
            isolation_dir = root / "isolation"
            isolation_dir.mkdir()
            raw_path, _ = write_fixture(isolation_dir, records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "isolation_check"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_gpu_index_two_must_be_unique_in_raw_list(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            raw_list = json.loads(records[0]["device"]["amd_smi_list_raw_json"])
            duplicate = dict(raw_list[-1])
            duplicate["uuid"] = "f" * 36
            raw_list.append(duplicate)
            encoded = compact_json(raw_list)
            records[0]["device"]["amd_smi_list_raw_json"] = encoded
            records[0]["device"]["amd_smi_list_raw_sha256"] = sha256_text(encoded)
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "unique matching GPU index 2"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_eos_token_cannot_be_followed_by_cancel(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            token = first_record(
                records,
                lambda record: record.get("record_type") == "worker_event"
                and record.get("event", {}).get("request_id") == "p8c-latency-measured-02"
                and record.get("event", {}).get("type") == "token",
            )
            token["event"]["token_id"] = 151645
            token["raw_json"] = compact_json(token["event"])
            token["raw_sha256"] = sha256_text(token["raw_json"])
            raw_path, _ = write_fixture(Path(temporary), records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "EOS token"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

    def test_all_cancellations_use_five_second_upper_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            release = first_record(
                records,
                lambda record: record.get("record_type") == "worker_event"
                and record.get("event", {}).get("request_id") == "p8c-resource-measured-004"
                and record.get("event", {}).get("type") == "released",
            )
            shift_timestamps(records, release["observed_monotonic_ns"], 6_000_000_000)
            raw_path, _ = write_fixture(Path(temporary), records)
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertFalse(result["passed"])
            self.assertEqual(result["cancellation"]["all_samples"], 34)
            self.assertEqual(result["cancellation"]["non_latency_warmup_samples"], 32)
            self.assertTrue(any("resource_measured[4] cancel" in error for error in result["gate_errors"]))

    def test_cancel_target_to_write_gap_uses_progress_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            records = EvidenceBuilder().build()
            cancel = first_record(
                records,
                lambda record: record.get("record_type") == "command"
                and record.get("command_type") == "cancel"
                and record.get("request_id") == "p8c-latency-measured-01",
            )
            shift_timestamps(
                records,
                cancel["write_started_monotonic_ns"],
                31_000_000_000,
            )
            raw_path, _ = write_fixture(Path(temporary), records)
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertFalse(result["passed"])
            self.assertTrue(
                any("cancel trigger gap" in error for error in result["gate_errors"])
            )

    def test_request_and_progress_deadlines_are_derived(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = EvidenceBuilder().build()
            started = first_record(
                records,
                lambda record: record.get("record_type") == "worker_event"
                and record.get("event", {}).get("request_id") == "p8c-latency-measured-01"
                and record.get("event", {}).get("type") == "started",
            )
            shift_timestamps(records, started["observed_monotonic_ns"], 31_000_000_000)
            progress_dir = root / "progress"
            progress_dir.mkdir()
            raw_path, _ = write_fixture(progress_dir, records)
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertFalse(result["passed"])
            self.assertTrue(any("progress gap" in error for error in result["gate_errors"]))

            records = EvidenceBuilder().build()
            release = first_record(
                records,
                lambda record: record.get("record_type") == "worker_event"
                and record.get("event", {}).get("request_id")
                == "p8c-latency-measured-01-recovery"
                and record.get("event", {}).get("type") == "released",
            )
            shift_timestamps(records, release["observed_monotonic_ns"], 181_000_000_000)
            request_dir = root / "request"
            request_dir.mkdir()
            raw_path, _ = write_fixture(request_dir, records)
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertFalse(result["passed"])
            self.assertTrue(any("request duration" in error for error in result["gate_errors"]))

    def test_shutdown_completion_order_and_thirty_second_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = EvidenceBuilder().build()
            shutdown = records[-2]
            records[-1]["exit_observed_monotonic_ns"] = (
                shutdown["write_completed_monotonic_ns"] - 1
            )
            order_dir = root / "order"
            order_dir.mkdir()
            raw_path, _ = write_fixture(order_dir, records)
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "precede shutdown"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)

            records = EvidenceBuilder().build()
            exit_observed = records[-1]["exit_observed_monotonic_ns"]
            shift_timestamps(records[-1], exit_observed, 31_000_000_000)
            deadline_dir = root / "deadline"
            deadline_dir.mkdir()
            raw_path, _ = write_fixture(deadline_dir, records)
            result = VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)
            self.assertFalse(result["passed"])
            self.assertTrue(any("shutdown duration" in error for error in result["gate_errors"]))

    def test_raw_framing_and_hash_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw_path, _ = write_fixture(root, EvidenceBuilder().build())
            payload = raw_path.read_bytes()
            self.assertTrue(payload.endswith(b"\n"))
            raw_path.write_bytes(payload[:-1])
            with self.assertRaisesRegex(VALIDATOR.ValidationError, "not LF terminated"):
                VALIDATOR.validate_evidence(raw_path, GIT_COMMIT, BINARY_SHA256)


if __name__ == "__main__":
    unittest.main()

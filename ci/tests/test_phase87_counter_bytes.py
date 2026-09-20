from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "tools/phase87_decode_profile.py"
SPEC = importlib.util.spec_from_file_location("phase87_decode_profile", SOURCE)
assert SPEC is not None and SPEC.loader is not None
profile = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(profile)


def _write_counter_csv(path: Path, rows: list[list[object]], *, long_form: bool = True) -> None:
    header = [
        "Correlation_Id",
        "Dispatch_Id",
        "Agent_Id",
        "Queue_Id",
        "Process_Id",
        "Thread_Id",
        "Kernel_Id",
        "Kernel_Name",
        "Counter_Name",
        "Counter_Value",
        "Start_Timestamp",
        "End_Timestamp",
    ]
    if not long_form:
        header = ["Bytes", "Dispatch_Id"]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def _row(dispatch: int, name: str, counter: str, value: float) -> list[object]:
    return [1, dispatch, "Agent 3", 1, 77, 77, 42, name, counter, value, 1000 + dispatch, 1010 + dispatch]


class Phase87CounterBytesTests(unittest.TestCase):
    def test_long_form_bins_count_once_and_ignore_aggregate_alias(self) -> None:
        rows = [
            _row(10, "sllm_matmul_nvfp4_v1", "GL2C_EA_RDREQ_32B", 1),
            _row(10, "sllm_matmul_nvfp4_v1", "GL2C_EA_RDREQ_64B", 2),
            _row(10, "sllm_matmul_nvfp4_v1", "GL2C_EA_RDREQ_128B", 3),
            _row(10, "sllm_matmul_nvfp4_v1", "SLLM_GL2C_EA_RDREQ_256B", 4),
            _row(10, "sllm_matmul_nvfp4_v1", "GL2C_EA_RDREQ", 999),
            _row(11, "sllm_matmul_nvfp4_v1", "GL2C_EA_RDREQ_32B", 100),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counter.csv"
            _write_counter_csv(path, rows)
            result = profile._counter_bytes(path, 10, 10)
        self.assertEqual(result["state"], "available_interface_request_estimate")
        self.assertEqual(result["bytes"], 1 * 32 + 2 * 64 + 3 * 128 + 4 * 256)
        self.assertEqual(result["complete_instance_count"], 1)
        self.assertEqual(result["ignored_aggregate_counters"], {"GL2C_EA_RDREQ": 1})
        self.assertEqual(result["per_kernel"]["sllm_matmul_nvfp4_v1"]["complete_instance_count"], 1)

    def test_same_bin_aliases_are_reported_but_not_added(self) -> None:
        rows = [
            _row(10, "kernel", "GL2C_EA_RDREQ_32B", 1),
            _row(10, "kernel", "SLLM_GL2C_EA_RDREQ_32B", 1),
            _row(10, "kernel", "GL2C_EA_RDREQ_64B", 1),
            _row(10, "kernel", "GL2C_EA_RDREQ_128B", 1),
            _row(10, "kernel", "GL2C_EA_RDREQ_256B", 1),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counter.csv"
            _write_counter_csv(path, rows)
            result = profile._counter_bytes(path, 10, 10)
        self.assertEqual(result["state"], "available_interface_request_estimate")
        self.assertEqual(result["bytes"], 32 + 64 + 128 + 256)
        self.assertEqual(len(result["duplicate_aliases"]), 1)

    def test_missing_bin_is_exposed_and_not_counted(self) -> None:
        rows = [
            _row(10, "kernel", "GL2C_EA_RDREQ_32B", 1),
            _row(10, "kernel", "GL2C_EA_RDREQ_64B", 1),
            _row(10, "kernel", "GL2C_EA_RDREQ_256B", 1),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counter.csv"
            _write_counter_csv(path, rows)
            result = profile._counter_bytes(path, 10, 10)
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["bytes"], 0.0)
        self.assertEqual(result["incomplete_instance_count"], 1)
        self.assertEqual(result["missing_bins_by_dispatch"], {"10": 1})

    def test_non_long_form_is_unavailable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "counter.csv"
            _write_counter_csv(path, [[1, 10]], long_form=False)
            result = profile._counter_bytes(path, 10, 10)
        self.assertEqual(result["state"], "unavailable")
        self.assertEqual(result["missing_columns"], ["Agent_Id", "Kernel_Name", "Counter_Name", "Counter_Value"])


if __name__ == "__main__":
    unittest.main()

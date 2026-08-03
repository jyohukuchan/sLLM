#!/usr/bin/env python3
"""Verify that both H3 row jobs preserve the immutable workflow identity."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/h3-compile.yml"
ROW_JOBS = ("h3-gfx1030", "h3-gfx1201")
EXPECTED_ENV = {
    "RUN_ID": "${{ github.run_id }}",
    "RUN_ATTEMPT": "${{ github.run_attempt }}",
}


def load_workflow() -> dict:
    with WORKFLOW.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


class H3WorkflowIdentityTests(unittest.TestCase):
    def _assert_row_identity(self, workflow: dict) -> None:
        jobs = workflow["jobs"]
        row_job_ids = tuple(job_id for job_id in jobs if job_id.startswith("h3-gfx"))
        self.assertEqual(row_job_ids, ROW_JOBS)

        for row_id in ROW_JOBS:
            docker_steps = [
                step for step in jobs[row_id]["steps"]
                if "docker run" in step.get("run", "")
            ]
            self.assertEqual(len(docker_steps), 1)
            step = docker_steps[0]
            self.assertEqual(
                {name: step["env"].get(name) for name in EXPECTED_ENV},
                EXPECTED_ENV,
            )

            run = step["run"]
            self.assertEqual(run.count("--env RUN_ID"), 1)
            self.assertEqual(run.count("--env RUN_ATTEMPT"), 1)
            self.assertIn("--env RUN_ID --env RUN_ATTEMPT", run)
            self.assertEqual(run.count('--run-id "$RUN_ID"'), 1)
            self.assertEqual(run.count('--run-attempt "$RUN_ATTEMPT"'), 1)
            self.assertIn(
                '--run-id "$RUN_ID" --run-attempt "$RUN_ATTEMPT"',
                run,
            )

    def test_both_rows_forward_run_identity_into_container_runner(self) -> None:
        self._assert_row_identity(load_workflow())

    def test_omitting_identity_from_either_row_is_detected(self) -> None:
        workflow = load_workflow()
        for row_id in ROW_JOBS:
            for variable in EXPECTED_ENV:
                with self.subTest(row=row_id, variable=variable):
                    mutated = copy.deepcopy(workflow)
                    docker_step = next(
                        step for step in mutated["jobs"][row_id]["steps"]
                        if "docker run" in step.get("run", "")
                    )
                    del docker_step["env"][variable]
                    with self.assertRaises(AssertionError):
                        self._assert_row_identity(mutated)


if __name__ == "__main__":
    unittest.main()

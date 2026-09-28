"""
The job timeout must outlast the sleep the job is told to take.

market_wait.py parks the runner until the bell. That wait is bounded by
--max-wait-min plus --grace-sec, and those live in the workflow YAML next to a
timeout-minutes that is easy to forget. Get it wrong and nothing looks broken:
the step is simply killed mid-sleep and GitHub reports the job as "cancelled",
which reads like a concurrency collision rather than an arithmetic error.

That is exactly what happened. close-summary.yml asked for a 25-minute
wait inside a 10-minute job, so the 19:45 fire died at 623 seconds on both
2026-09-10 and 09-11 -- roughly seven minutes before the close it was waiting
for. The close-of-day email did not send either day.

No unit test could see it, because neither file is wrong on its own. This
checks them against each other.
"""
import re
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

# Setup, dependency install, the run itself and the commit, generously. The
# rebalance's Yahoo fetch is the slow one at ~7 minutes.
OVERHEAD_MIN = 15.0


def wait_budget(step_run: str):
    """Minutes market_wait.py may sleep, or None if this step does not wait."""
    if "market_wait.py" not in step_run:
        return None
    wait = re.search(r"--max-wait-min\s+([\d.]+)", step_run)
    grace = re.search(r"--grace-sec\s+([\d.]+)", step_run)
    if not wait:
        return None
    return float(wait.group(1)) + (float(grace.group(1)) / 60.0 if grace else 0.0)


def waiting_jobs():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        spec = yaml.safe_load(path.read_text())
        for job_name, job in (spec.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                budget = wait_budget(step.get("run") or "")
                if budget is not None:
                    yield path.name, job_name, budget, job.get("timeout-minutes")


def test_some_job_actually_waits():
    """Guard the guard: if the search stops matching, the test below passes
    vacuously and this whole file becomes decorative."""
    assert list(waiting_jobs()), "no market_wait.py step found — has it moved?"


@pytest.mark.parametrize("wf,job,budget,timeout", list(waiting_jobs()))
def test_timeout_outlasts_the_wait(wf, job, budget, timeout):
    assert timeout is not None, f"{wf}:{job} waits {budget:.1f}m with no timeout set"
    assert timeout >= budget + OVERHEAD_MIN, (
        f"{wf}:{job} sleeps up to {budget:.1f} min but the job is killed at "
        f"{timeout} min. It needs at least {budget + OVERHEAD_MIN:.0f}. "
        "This is the 2026-09-10/11 close-email failure."
    )

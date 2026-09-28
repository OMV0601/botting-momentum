"""
Every step that runs a live-path script must carry all three live settings.

resolve_base() only returns the live endpoint when ALPACA_LIVE is "true" AND
ALPACA_ACCOUNT_ID and MAX_DEPLOY are both present. Omitting one does not
quietly fall back to paper -- it raises, and the step dies.

That is how live day 1 was lost. trade.yml's Preflight step listed
ALPACA_LIVE and ALPACA_ACCOUNT_ID but not MAX_DEPLOY, so on 2026-09-21 all
five network checks failed with "ALPACA_LIVE=*** but MAX_DEPLOY not set", the
Rebalance step was skipped by its `if`, and no orders were placed. The
standalone preflight.yml workflow had all three and had passed four days
running, which is exactly why the gap was invisible: the two files run the
same script with different env.

MAX_DEPLOY is deliberately empty on the dedicated account (deploy real
equity). Empty is fine -- absent is not. Only presence is checked here.
"""
from pathlib import Path

import pytest
import yaml

WORKFLOWS = Path(__file__).resolve().parent.parent / ".github" / "workflows"

# Scripts that can reach resolve_base(). liquidate.py is deliberately excluded:
# assay-liquidate.yml passes no ALPACA_LIVE at all, so it stays pinned to the
# paper endpoint and cannot flatten a live book by accident.
LIVE_PATH = ("run_daily.py", "preflight.py")

REQUIRED = ("ALPACA_LIVE", "ALPACA_ACCOUNT_ID", "MAX_DEPLOY")


def live_steps():
    for path in sorted(WORKFLOWS.glob("*.yml")):
        spec = yaml.safe_load(path.read_text())
        for job_name, job in (spec.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                run = step.get("run") or ""
                if not any(s in run for s in LIVE_PATH):
                    continue
                # Only steps that opt into live at all; a step with no
                # ALPACA_LIVE stays on paper and needs no cap.
                env = step.get("env") or {}
                if "ALPACA_LIVE" not in env:
                    continue
                yield path.name, job_name, step.get("name") or run.strip()[:40], env


def test_some_live_step_exists():
    """Guard the guard: a rename of the scripts would otherwise make the
    parametrised test below pass vacuously."""
    assert list(live_steps()), "no live-path step found -- have the scripts moved?"


@pytest.mark.parametrize(
    "wf,job,step,env", list(live_steps()), ids=lambda v: v if isinstance(v, str) else ""
)
def test_live_step_has_every_setting(wf, job, step, env):
    missing = [k for k in REQUIRED if k not in env]
    assert not missing, (
        f"{wf}:{job}:{step!r} opts into live but omits {', '.join(missing)}. "
        "resolve_base() raises rather than falling back to paper, so this step "
        "fails outright whenever ALPACA_LIVE is true."
    )

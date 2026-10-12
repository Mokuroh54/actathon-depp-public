"""Which stage is a PI fine-tune in? Tells "our job is training" apart from "waiting on PI" (queue, checkpoint
verification, hosting), with timestamps in UTC and local time. Read-only: only get/list calls to PI's API.

    uv run scripts/pi_status.py <run_id>                  # current stage + the timeline so far
    uv run scripts/pi_status.py --latest                  # newest fine-tune in the org
    uv run scripts/pi_status.py <run_id> --watch          # poll every --interval s, a line per stage change;
                                                          # exits when hosted / failed / cancelled or after --max-minutes

Stages (from a real 100-step run: created 09:08:50, started 09:09:00, checkpoint row 09:17:24 while the fine-tune
still said "running" with a lagging `progress`, succeeded + checkpoint available 09:30:39, gateway_url by ~09:46):
    QUEUED       [PI]   fine-tune pending
    STARTING     [job]  running, no progress reported yet (preparing data, first steps)
    TRAINING     [job]  running with progress > 0, or only intermediate (every_n_steps) checkpoints so far
    VERIFYING    [PI]   the final checkpoint row exists but is pending: training is over (trust this over `progress`)
    HOSTING      [PI]   final checkpoint available, no gateway_url yet
    HOSTED              gateway_url set: ready for inference
    FAILED / CANCELLED / CHECKPOINT FAILED  (terminal_reason printed)

PI records no timestamp for gateway_url, so the hosting duration is only known when --watch saw it appear. The same
holds for the start of TRAINING (first progress): one-shot reports STARTING+TRAINING as one "job" span.
Exit status: 0 hosted (or still in progress, one-shot), 1 failed/cancelled, 2 --watch gave up after --max-minutes.
"""
import argparse
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from pi_sdk.finetuning import Client, FinetuningAPIError

from act.policies import pi_api_key

PI, JOB = "waiting on PI", "our training job"
TERMINAL = {"HOSTED", "FAILED", "CANCELLED", "CHECKPOINT FAILED"}


@dataclass(frozen=True)
class Stage:
    key: str        # QUEUED, STARTING, TRAINING, VERIFYING, HOSTING, HOSTED, FAILED, CANCELLED, CHECKPOINT FAILED, UNKNOWN
    side: str       # PI, JOB or "" (done / terminal)
    detail: str
    checkpoint: object = None   # the checkpoint that decided the stage, if any

    def line(self):
        side = f" [{self.side}]" if self.side else ""
        return f"{self.key}{side}: {self.detail}"


def is_final(ck, ft):
    """A checkpoint is the run's last one if the policy keeps only the final one, or it sits at the last step
    (PI numbers steps from 0: the 100-step run's checkpoint is step 99)."""
    if ft.checkpoint_policy is not None and ft.checkpoint_policy.mode == "final_only":
        return True
    return ft.num_steps is not None and ck.step >= ft.num_steps - 1


def step_estimate(ft, checkpoints):
    """(estimated optimizer step or None, where it came from). progress lags; intermediate checkpoints are floors."""
    est, src = None, ""
    if ft.progress is not None and ft.num_steps:
        est, src = ft.progress * ft.num_steps, f"progress {ft.progress:.0%}"
    ck_steps = [c.step for c in checkpoints if not is_final(c, ft)]
    if ck_steps and (est is None or max(ck_steps) > est):
        est, src = float(max(ck_steps)), f"checkpoint at step {max(ck_steps)}"
    return est, src


def classify(ft, checkpoints):
    """Stage of a fine-tune from its Finetune and Checkpoint rows (pure: no API calls)."""
    finals = sorted((c for c in checkpoints if is_final(c, ft)), key=lambda c: c.step)
    final = finals[-1] if finals else None
    if ft.state == "failed":
        return Stage("FAILED", "", ft.terminal_reason or "(no terminal_reason)")
    if ft.state == "cancelled":
        return Stage("CANCELLED", "", ft.terminal_reason or "(no terminal_reason)")
    if ft.state == "pending":
        return Stage("QUEUED", PI, "accepted, waiting for a training slot")
    if ft.state not in ("running", "succeeded"):
        return Stage("UNKNOWN", "", f"unexpected fine-tune state {ft.state!r}")
    if final is not None:
        if final.state == "failed":
            return Stage("CHECKPOINT FAILED", "", f"checkpoint step {final.step} failed: artifact not produced "
                         f"({ft.terminal_reason or 'no terminal_reason'})", final)
        if final.state == "available" and final.gateway_url:
            return Stage("HOSTED", "", f"ready for inference: {final.gateway_url}", final)
        if final.state == "available":
            return Stage("HOSTING", PI, f"checkpoint step {final.step} available, waiting for gateway_url "
                         "(never comes if the org is local-delivery only)", final)
        if final.state == "pending":
            note = " (fine-tune already says succeeded)" if ft.state == "succeeded" else ""
            return Stage("VERIFYING", PI, f"training done, PI is verifying checkpoint step {final.step}{note}", final)
        return Stage("UNKNOWN", "", f"unexpected checkpoint state {final.state!r} (step {final.step})", final)
    if ft.state == "succeeded":
        return Stage("UNKNOWN", "", "fine-tune succeeded but no final checkpoint row is listed")
    est, src = step_estimate(ft, checkpoints)
    if est is not None and est > 0:
        total = f"/{ft.num_steps}" if ft.num_steps else ""
        return Stage("TRAINING", JOB, f"~step {est:.0f}{total} ({src}; progress lags)")
    return Stage("STARTING", JOB, "running, no progress reported yet (preparing data / first steps)")


def fmt_t(t):
    if t is None:
        return "-"
    loc = t.astimezone()
    return f"{t.astimezone(UTC):%H:%M:%S}Z ({loc:%H:%M:%S} {loc:%Z})"


def fmt_d(a, b):
    if a is None or b is None:
        return "?"
    s = (b - a).total_seconds()
    return f"{s / 60:.1f} min" if s >= 60 else f"{s:.0f} s"


def timeline(ft, checkpoints, observed=None):
    """[(time, text)] from PI's timestamps, plus times --watch observed (gateway_url, first progress)."""
    observed = observed or {}
    ev = [(ft.created_at, "fine-tune created, queued [PI]"),
          (ft.started_at, "started: preparing data / training [job]")]
    if observed.get("TRAINING"):
        ev.append((observed["TRAINING"], "first progress seen [job] (observed by --watch)"))
    for c in sorted(checkpoints, key=lambda c: c.step):
        kind = "final" if is_final(c, ft) else "intermediate"
        ev.append((c.created_at, f"checkpoint step {c.step} ({kind}) written, verification starts [PI]"))
        ev.append((c.available_at, f"checkpoint step {c.step} available, hosting starts [PI]"))
        if c.gateway_url:
            t = observed.get(("gateway", c.checkpoint_id))
            ev.append((t, f"checkpoint step {c.step} hosted: gateway_url set"
                          + (" (observed by --watch)" if t else " (PI records no time for this)")))
    if ft.completed_at:
        ev.append((ft.completed_at, f"fine-tune {ft.state}"))
    known = sorted((e for e in ev if e[0] is not None), key=lambda e: e[0])
    return known + [e for e in ev if e[0] is None and "gateway_url" in e[1]]


def durations(ft, checkpoints, stage, now, observed=None):
    """[(label, side, text)] for the PI-wait vs training spans of the run (final checkpoint's path)."""
    observed = observed or {}
    finals = sorted((c for c in checkpoints if is_final(c, ft)), key=lambda c: c.step)
    final = finals[-1] if finals else None
    out = []
    running = lambda t: f"{fmt_d(t, now)} so far"  # noqa: E731
    if ft.created_at:
        out.append(("queue", PI, fmt_d(ft.created_at, ft.started_at) if ft.started_at else running(ft.created_at)))
    if ft.started_at:
        end = final.created_at if final else (ft.completed_at if stage.key in ("FAILED", "CANCELLED") else None)
        if end:
            text = fmt_d(ft.started_at, end)
        else:
            text = "? (no completion time)" if stage.key in TERMINAL else running(ft.started_at)
        if observed.get("TRAINING"):
            text += f" (of which preparing {fmt_d(ft.started_at, observed['TRAINING'])})"
        out.append(("training (incl. data prep)", JOB, text))
    if final and final.created_at:
        out.append(("PI verification", PI, fmt_d(final.created_at, final.available_at) if final.available_at
                    else running(final.created_at)))
    if final and final.available_at:
        if final.gateway_url:
            t = observed.get(("gateway", final.checkpoint_id))
            text = fmt_d(final.available_at, t) + " (observed)" if t else \
                f"<= {fmt_d(final.available_at, now)} (gateway_url time not recorded by PI)"
        else:
            text = running(final.available_at)
        out.append(("hosting", PI, text))
    return out


def report(ft, checkpoints, evals, now, observed=None):
    stage = classify(ft, checkpoints)
    steps = f"{ft.num_steps} steps" if ft.num_steps else "default steps"
    pol = ft.checkpoint_policy
    pol_s = pol.mode + (f" every {pol.every_n_steps} (keep {pol.limit})" if pol and pol.every_n_steps else "") \
        if pol else "?"
    print(f"{ft.name}  run {ft.run_id}")
    print(f"  {ft.base_model or ft.continue_from_checkpoint_id}, {steps}, batch {ft.batch_size}, checkpoints: {pol_s}, "
          f"state={ft.state}, progress={'-' if ft.progress is None else f'{ft.progress:.0%}'}")
    print(f"\nSTAGE  {stage.line()}")
    print("\ntimeline:")
    for t, text in timeline(ft, checkpoints, observed):
        print(f"  {fmt_t(t):<28} {text}")
    print("\ndurations:")
    for label, side, text in durations(ft, checkpoints, stage, now, observed):
        print(f"  {label:<27} {text:<52} [{side}]")
    if checkpoints:
        print("\ncheckpoints:")
        for c in sorted(checkpoints, key=lambda c: c.step):
            kind = "final" if is_final(c, ft) else "intermediate"
            ver = fmt_d(c.created_at, c.available_at) if c.available_at else f"{fmt_d(c.created_at, now)} so far"
            print(f"  step {c.step:<6} {kind:<12} {c.state:<9} verification {ver:<14} "
                  f"gateway {c.gateway_url or '-'}  ({c.checkpoint_id})")
    if evals:
        last = max(evals, key=lambda e: e.step)
        val = f", validation_mae {last.validation_mae:.4f}" if last.validation_mae is not None else ""
        print(f"\nevaluation at step {last.step}: train_mae {last.train_mae:.4f}{val}  ({len(evals)} evaluation rows)")
    return stage


def fetch(client, run_id):
    ft = client.get_finetune(run_id)
    cks = list(client.list_checkpoints(run_id).checkpoints)
    return ft, cks, client.list_finetune_evaluations(run_id)


def watch(client, run_id, interval, max_minutes):
    deadline = time.monotonic() + max_minutes * 60
    observed, prev, prev_rate, seen_evals, seen_gw = {}, None, None, 0, set()
    ft = cks = evals = None
    while True:
        now = datetime.now(UTC)
        try:
            ft, cks, evals = fetch(client, run_id)
        except FinetuningAPIError as e:
            print(f"{fmt_t(now)}  API error (will retry): {e}", flush=True)
        else:
            stage = classify(ft, cks)
            for c in cks:
                if c.gateway_url and c.checkpoint_id not in seen_gw:
                    seen_gw.add(c.checkpoint_id)
                    if prev is not None:     # appeared while watching: now is an upper bound on when it was set
                        observed[("gateway", c.checkpoint_id)] = now
                        print(f"{fmt_t(now)}  checkpoint step {c.step} gateway_url set: {c.gateway_url}", flush=True)
            if prev is None or stage.key != prev.key:
                if prev is not None:         # a transition seen live: now is an upper bound on when it happened
                    observed.setdefault(stage.key, now)
                print(f"{fmt_t(now)}  {stage.line()}", flush=True)
            if stage.key == "TRAINING":
                est, _ = step_estimate(ft, cks)
                if prev_rate and est is not None and est != prev_rate[1]:
                    rate = (est - prev_rate[1]) / max((now - prev_rate[0]).total_seconds() / 60, 1e-9)
                    left = (ft.num_steps - est) / rate if ft.num_steps and rate > 0 else None
                    eta = f", ~{left:.0f} min to last step (progress lags)" if left is not None else ""
                    print(f"{fmt_t(now)}    ~step {est:.0f}/{ft.num_steps}, {rate:.1f} steps/min{eta}", flush=True)
                if est is not None and (prev_rate is None or est != prev_rate[1]):
                    prev_rate = (now, est)
            if len(evals) > seen_evals:
                last = max(evals, key=lambda e: e.step)
                print(f"{fmt_t(now)}    evaluation step {last.step}: train_mae {last.train_mae:.4f}", flush=True)
                seen_evals = len(evals)
            prev = stage
            if stage.key in TERMINAL:
                print()
                report(ft, cks, evals, now, observed)
                return 0 if stage.key == "HOSTED" else 1
        if time.monotonic() + interval > deadline:
            print(f"\ngave up after {max_minutes} min (--max-minutes)")
            if ft is not None:
                print()
                report(ft, cks, evals, datetime.now(UTC), observed)
            return 2
        time.sleep(interval)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_id", nargs="?")
    ap.add_argument("--latest", action="store_true", help="the newest fine-tune in the org")
    ap.add_argument("--watch", action="store_true", help="poll until hosted / failed / cancelled")
    ap.add_argument("--interval", type=float, default=30, help="--watch poll period, seconds")
    ap.add_argument("--max-minutes", type=float, default=180, help="--watch gives up after this long")
    a = ap.parse_args()
    if bool(a.run_id) == a.latest:
        ap.error("give a run_id or --latest (not both)")
    with Client(pi_api_key()) as client:
        run_id = a.run_id
        if a.latest:
            page = client.list_finetunes(page_size=1).finetunes
            if not page:
                raise SystemExit("no fine-tunes in this org")
            run_id = page[0].run_id
        if a.watch:
            return watch(client, run_id, a.interval, a.max_minutes)
        ft, cks, evals = fetch(client, run_id)
        stage = report(ft, cks, evals, datetime.now(UTC))
        return 1 if stage.key in ("FAILED", "CANCELLED", "CHECKPOINT FAILED") else 0


if __name__ == "__main__":
    sys.exit(main())

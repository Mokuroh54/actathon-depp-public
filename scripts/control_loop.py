"""Client-side control loop for remote (chunked) policies.

Runs the machine at a fixed rate from a queue of predicted actions, asks for the next chunk
in the background before the queue runs out, drops the part of each chunk that went stale
while it was in flight, and falls back to a safe action when chunks are late.

    policy.infer(obs, prompt) -> (H, D) array of actions, one per control step
    backends here:
      http      our own policy server (JSON + JPEG frames): modal_jobs/serve_pi05.py serves a pi0.5 checkpoint;
                bearer token PI05_SERVE_TOKEN from the environment or .env (docs/pi05_serving.md)
      pi        a PI-hosted fine-tuned checkpoint (pi07), PI_API_KEY from the environment or .env

    uv run scripts/control_loop.py --machine real --cam front=<src> --cam wrist=<src> --cam side=<src> --policy http \
        --url https://...modal.run --dataset data/<training dataset> --prompt "<training task>" --deadman --show
    uv run scripts/control_loop.py --machine real --cam front=<src> --cam wrist=<src> --policy pi \
        --gateway-url <checkpoint gateway_url> --dataset data/<training dataset> --prompt "<training task>" \
        --request-at 10 --deadman --show
    uv run scripts/control_loop.py --machine real --cam front=<src> --cam wrist=<src> --policy http \
        --url https://...modal.run --dataset data/pilot-best7-nostate --zero-state --prompt "<training task>" \
        --deadman --show                                                                          # zero-state model

--dataset (the LeRobot dataset the checkpoint was trained on) sets the camera crops (<dataset>/crops.json), the image
size and the control rate from the training data, so the policy sees the same pixels at the same rate. --prompt must
be the training task string exactly. --request-at: ask for the next chunk this many steps before the queue runs out;
keep it above round-trip time x fps (the per-request log shows rtt).

ON THE PI, through the dashboard (the deadman is the dashboard page's Policy mode: hold SPACE / L1 / R1 there):
    .venv/bin/python control_loop.py --machine dashboard --policy pi --gateway-url <url> --dataset <training dataset> \
        --prompt "<training task>" --cam front=http://127.0.0.1:8100/stream/<id> --cam wrist=http://127.0.0.1:8100/stream/<id>
    (pi0.5 fallback: --policy http --url <serve_pi05 URL> instead of --policy pi --gateway-url <url>)
It reads /state and camview's streams itself and posts every action to the dashboard (POST /api/policy/act), which is
the only writer to the gate. The runner keeps predicting while the deadman is released; those actions are dropped.

Actions are rig commands {ch}.cmd in -1..1, sent every tick (the gate goes neutral 0.5 s after the last
command); the safe action is neutral (all zeros). Quitting sends /estop on the real machine.

UI (--show): ESC quit; with --deadman hold SPACE / L1 to let the policy act.
Every run writes a per-request CSV log (rtt, stale steps dropped, payload size) to logs/, and a per-tick CSV
(logs/ticks_*.csv: status, deadman, the policy's planned action, what was sent, state_* = the {ch}.cmd readback in obs,
policy_state_* = the state the policy is shown: zeros with --zero-state, else = state_*) while it runs.

--zero-state: for checkpoints trained on a dataset made by scripts/zero_state.py (observation.state = 0 on every frame).
Every request carries state = zeros; the machine, the gate and state_* still use the real readback. Required when
--dataset has meta/zero_state.json, refused when it hasn't (docs/zero_state.md).

--joint-gain JOINT=K (repeatable, e.g. --joint-gain boom=1.5 --joint-gain arm=1.5): multiply the policy's command
on that joint by K and clip to -1..1, before --grip-latch and the deadman. For checkpoints whose commands are too
small to move the real machine. Unlisted joints keep gain 1.
The tick log's plan_* columns keep the raw policy output, sent_* (and a DAgger recording) the scaled command.

--sync aligned|latest (real / dashboard machine): how each observation's frames are picked across cameras
(act/camsync.py, the recorder's rule; docs/runner_sync.md). It must match how the training episodes were recorded
(episode meta.json "sync"). The LeRobot dataset does not record it (convert_pi_episodes.py drops it): unless
<dataset>/pi_sources.json carries it, it defaults to aligned (camsync.DEFAULT_MODE, how the training data was recorded). The tick log gets sync_* columns (t_ref, anchor, skew,
cameras left out) and per camera cap_<cam> (picked frame's capture time, wall s) / age_ms_<cam> (its arrival age).

--subtasks (--policy pi): tell the model its phase of the cone pick-and-place, as in training (docs/subtask_labels.md).
A causal act.phase.PhaseTracker (the labeler's rule and thresholds, carry direction from the prompt, combined
thumb+/bucket+ dump = release) runs over the command actually SENT each tick (after --grip-latch and the deadman /
neutral handling), with the recording's +-0.08 deadband applied first. Each request carries the current phase as
raw_text (robot_task_string stays the prompt): "reach and grip the cone" -> "carry the cone to the target" ->
"release the cone" (latched until the end of the run; a deadman release does not reset it).
Manual overrides, logged: --start-phase reach|carry|release, and the nudge file --phase-file (default phase_next in
the working directory): `touch phase_next` advances one phase (release -> reach = a new attempt), or write a phase
name into it to jump there; the runner deletes the file when it consumes it. The phase is on the status line; the tick
log gets phase (what this tick's request was built with) and phase_event (grip / regrip_open / release / override:...).
--policy http refuses it (our servers take no subtask).

--stop-after-release SECONDS (--subtasks): end the run this long after the tracker enters "release the cone" (its
release event or an override into it), through the normal end (neutral / STOP, logs, DAgger episode saved). The
countdown runs whoever is driving; leaving release first (a nudge to reach) cancels it. Printed,
and the last tick's phase_event is auto_stop. Recommended 1.5 s.

--record-dagger (--machine dashboard): DAgger data collection (docs/dagger.md). Every tick is
recorded as a raw Pi episode (the dashboard recorder's format, act.recorder) in --episodes-dir: control "policy" (the
dashboard passed the policy's action: Policy on + deadman held), "human" (not passed, cab joysticks active: the
operator took over from the cab) or "idle"; the recorded action is the command that drove the machine (policy action /
cab action / zeros), the frames are the camsync-picked JPEGs the policy saw. "human" needs the relay OPEN (gate /state
"armed" false): the cab can't drive the machine while the code path holds the relay, so takeover = STOP in the
dashboard, then the cab; cab input after a mere deadman release is recorded as idle. An episode opens at the first non-idle
tick and is saved on quit, the split file (--dagger-split-file, default dagger_next: `touch` it to save
the current episode and start a new one at the next engagement) or an error (meta "aborted": true). A STOP in the
dashboard does not end the run: the runner keeps posting its heartbeat, but once the dashboard has answered "Policy
off" it posts only neutral actions (the dashboard refuses every action while Policy is off anyway), records the
cab-driven steps as "human", and resumes the policy when Policy is switched on and the deadman held again. With --subtasks the phase tracker follows the recorded action (the human's cab commands
included). All disk I/O is on a writer thread; when more than --dagger-queue steps wait, recording steps are dropped
(counted in meta dropped_steps), never control ticks.

--state-from-drove (real-state checkpoints; --machine dashboard; DEFAULT ON with --record-dagger unless
--zero-state, --no-state-from-drove turns it off): the policy's state input is the command that actually DROVE the
machine on the previous tick -- the passed policy action (policy), the cab action (human, after a STOP), zeros (idle,
and at the first tick) -- instead of the gate's /state "sent". That is how the training datasets define state (the
previous recorded action; convert_pi_episodes.py), so after a STOP -> cab -> re-enable handback the policy sees the
human's last command. state_* in the tick log stays the gate readback; policy_state_* is what the policy got.
Refused with --zero-state (the policy sees zeros there).
"""
import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import cv2
import numpy as np
import pygame

from act import camsync
from act.assist import GripLatch
from act.phase import DEFAULTS as PHASE_DEFAULTS
from act.phase import PHASE_ALIASES, REACH, RELEASE, PhaseTracker, carry_sign, deadband, parse_phase
from act.cameras import camsync_status, tile
from act.control import ChunkedController
from act.recorder import DaggerRecorder, Frame, cab_action, control_label, human_cab_map, relay_closed
from act.policies import (SUBTASK_KEY, HTTPPolicy, PIPolicy, ZeroState, check_zero_state, pi05_serve_token, pi_api_key,
                          zero_state_view)
from act.rig_machine import DEFAULT_DASHBOARD_URL, DEFAULT_RIG_URL, load_crop_file, make_machine
from act.teleop import Gamepad, TeleopConfig


def training_setup(root):
    """<dataset>/meta/info.json + crops.json -> (fps, image_hw, crops) the checkpoint was trained with."""
    root = Path(root)
    info = json.loads((root / "meta" / "info.json").read_text())
    shapes = {tuple(f["shape"][:2]) for k, f in info["features"].items() if k.startswith("observation.images.")}
    if len(shapes) != 1:
        raise SystemExit(f"{root}: expected one image size across cameras, got {shapes}")
    crops = load_crop_file(root / "crops.json") if (root / "crops.json").exists() else {}
    return int(info["fps"]), shapes.pop(), crops


SYNC_HELP = ("must match the recording; session-2 episodes (2026-09-26 16:02+) are aligned, "
             "session-1 (14:01-14:21) latest")


def dataset_sync(root):
    """<dataset>/pi_sources.json -> the camsync mode its episodes were recorded with, or None if not recorded.
    (convert_pi_episodes.py does not carry meta.json "sync" today; accepted as a string or {"mode": ...} per episode.)"""
    path = Path(root) / "pi_sources.json"
    if not path.exists():
        return None
    modes = set()
    for ep in json.loads(path.read_text()).values():
        s = ep.get("sync") if isinstance(ep, dict) else None
        modes.add(s.get("mode") if isinstance(s, dict) else s)
    if not modes or None in modes:
        return None
    if len(modes) > 1:
        raise SystemExit(f"{root}: episodes recorded with different camera sync modes {sorted(modes)}: "
                         "one checkpoint cannot match both")
    return modes.pop()


def resolve_sync(args):
    """--sync, checked against / defaulted from the dataset."""
    recorded = dataset_sync(args.dataset) if args.dataset else None
    if args.sync and recorded and args.sync != recorded:
        raise SystemExit(f"--sync {args.sync} but {args.dataset} was recorded with sync {recorded}")
    mode = args.sync or recorded
    return mode or camsync.DEFAULT_MODE     # aligned: the training data's mode


def parse_joint_gain(specs, joint_names):
    """--joint-gain JOINT=K specs -> per-joint gain array (1 for unlisted joints), or None when none were given."""
    if not specs:
        return None
    joints = list(joint_names)
    gain = np.ones(len(joints))
    for spec in specs:
        joint, sep, k = spec.partition("=")
        joint = joint.strip().removesuffix(".cmd")
        if not sep or joint not in joints:
            raise SystemExit(f"--joint-gain {spec!r}: expected JOINT=K with JOINT one of {', '.join(joints)}")
        try:
            gain[joints.index(joint)] = float(k)
        except ValueError:
            raise SystemExit(f"--joint-gain {spec!r}: K must be a number") from None
        if gain[joints.index(joint)] < 0:
            raise SystemExit(f"--joint-gain {spec!r}: K must be >= 0 (a negative gain flips the joint's direction)")
    return gain


def run(args):
    crops, image_hw, fps = None, None, args.fps or 10
    if args.dataset:
        check_zero_state(args.dataset, args.zero_state)
        data_fps, image_hw, crops = training_setup(args.dataset)
        if args.fps and args.fps != data_fps:
            raise SystemExit(f"--fps {args.fps} but {args.dataset} was recorded at {data_fps} fps")
        fps = data_fps
        print(f"training setup from {args.dataset}: {fps} fps, images {image_hw}, crops {crops or 'centered'}")
    via_dash = args.machine == "dashboard"
    if args.machine == "real" and not args.deadman:
        raise SystemExit("--machine real requires --deadman (hold SPACE / L1 to let the policy act)")
    if via_dash and args.deadman:
        raise SystemExit("--machine dashboard: the deadman is the dashboard page's Policy mode, not --deadman")
    if fps < 5:
        raise SystemExit("--fps must be >= 5 on the real machine (the gate goes neutral 0.5 s after a command)")
    if args.policy == "http" and not args.url:
        raise SystemExit("--policy http needs --url")
    if args.policy == "pi" and not args.gateway_url:
        raise SystemExit("--policy pi needs --gateway-url (the checkpoint's gateway_url)")
    if not args.dataset:
        raise SystemExit(f"--policy {args.policy} on the real machine needs --dataset "
                         "(crops / image size / fps of the training data)")
    if args.subtasks and args.policy == "http":
        raise SystemExit("--subtasks: --policy http can't carry a subtask (serve_pi05.py / serve_smolvla.py take only "
                         "the prompt); subtasks are for --policy pi (PI's raw_text)")
    if args.start_phase and not args.subtasks:
        raise SystemExit("--start-phase needs --subtasks")
    if args.stop_after_release is not None and not args.subtasks:
        raise SystemExit("--stop-after-release needs --subtasks (the release comes from the phase tracker)")
    if args.stop_after_release is not None and args.stop_after_release < 0:
        raise SystemExit("--stop-after-release must be >= 0 s")
    if args.record_dagger and args.machine != "dashboard":
        raise SystemExit("--record-dagger needs --machine dashboard (on the Pi: the dashboard's deadman / STOP decide "
                         "who drives)")
    if args.state_from_drove is None:        # default: on for DAgger runs of real-state checkpoints
        args.state_from_drove = bool(args.record_dagger and not args.zero_state)
    if args.state_from_drove and args.zero_state:
        raise SystemExit("--state-from-drove is for real-state checkpoints: with --zero-state the policy sees zeros")
    if args.state_from_drove and args.machine != "dashboard":
        raise SystemExit("--state-from-drove needs --machine dashboard (who drove comes from the dashboard's verdict "
                         "and the cab readback)")
    sync = resolve_sync(args)
    ok, msg = camsync_status()
    if not ok and not args.allow_camsync_mismatch:
        raise SystemExit(f"{msg}\n(--allow-camsync-mismatch runs anyway: frames may be picked by another rule "
                         "than the training episodes were recorded with)")
    print(("" if ok else "WARNING: ") + msg)
    print(f"camera sync: {sync} (must match the training episodes' meta.json \"sync\")")
    robot = make_machine(args.machine, args.cam, args.rig_url, require_cameras=True, crops=crops, image_hw=image_hw,
                         dashboard_url=args.dashboard_url, sync=sync)
    cfg = robot.cfg
    names = [f"{j}.cmd" for j in cfg.joint_names]

    if args.policy == "pi":
        policy = PIPolicy(args.gateway_url, cfg, pi_api_key(), execute_chunk_size=args.chunk)
        print("connecting to PI (the first connection can take minutes while the model starts) ...")
        meta = policy.connect()
        print(f"PI server: cameras {policy.cam_map}, action horizon {meta.action_horizon}, action dim {meta.action_dim}")
    else:
        token = pi05_serve_token()
        if not token:
            print("no PI05_SERVE_TOKEN (environment or .env): sending no bearer token (serve_pi05.py will answer 401)")
        # with --dataset: frames at the training image size (RigMachine's crops already are), near-lossless JPEG
        quality = args.jpeg_quality or (95 if image_hw else 80)
        policy = HTTPPolicy(args.url, cfg, image_px=args.image_px, jpeg_quality=quality, token=token, image_hw=image_hw,
                            zero_state=args.zero_state)
        print("connecting to the policy server (a cold start loads the model: can take minutes) ...")
        info = policy.connect()
        print(f"policy server: run {info.get('run')}, cameras {info.get('cameras')}, images {info.get('image_hw')}, "
              f"chunk {info.get('chunk_size')}, zero_state {info.get('zero_state')}")
    if args.zero_state:
        policy = ZeroState(policy, cfg.joint_names)   # only the policy's view: obs, gate and state_* logs keep the readback
        print("--zero-state: the policy sees state = zeros")

    safe = lambda obs: np.zeros(len(names))  # neutral on every channel
    ctrl = ChunkedController(policy, safe, request_at=args.request_at, max_use=args.max_use,
                             stop_after_s=args.stop_after, fps=fps)
    # --grip-latch: a policy thumb clamp past the threshold becomes a full clamp for --grip-latch-s (act.assist.GripLatch);
    # cancelled at once by anything that isn't the policy acting (deadman, STOP, starved, neutral)
    latch = (GripLatch(list(cfg.joint_names).index("thumb"), args.grip_latch_threshold, args.grip_latch_s,
                       args.grip_latch_refractory_s) if args.grip_latch else None)
    latch_cols = GripLatch.COLUMNS if latch is not None else []
    gain = parse_joint_gain(args.joint_gain, cfg.joint_names)
    if gain is not None:
        print("joint gain: " + " ".join(f"{j} x{k:g}" for j, k in zip(cfg.joint_names, gain)))
    if latch is not None:
        print(f"grip latch: thumb >= {latch.threshold} -> +1.0 for {latch.hold_s} s, refractory {latch.refractory_s} s")
    # --subtasks: the phase tracker (act.phase) over the SENT commands; a fresh one per run
    phase = (PhaseRunner(fps, list(cfg.joint_names), args.phase_file, args.start_phase, args.stop_after_release)
             if args.subtasks else None)
    phase_cols = PhaseRunner.COLUMNS if phase is not None else []
    if phase is not None:
        phase.new(args.prompt, "start")

    screen = pad = None
    if args.show or args.deadman:
        pygame.init()
        h, w = cfg.image_hw
        screen = pygame.display.set_mode((w * max(1, len(cfg.cameras)), h))
        pygame.display.set_caption("control loop: ESC quit")
        if args.deadman:
            pad = Gamepad(TeleopConfig(), cfg.joint_names)
            pad.connect()
    robot.connect()
    dagger = None
    if args.record_dagger:
        dagger = DaggerRecorder(args.episodes_dir, fps, cfg.cameras, policy_block(args, fps, sync), sync_mode=sync,
                                max_pending=args.dagger_queue)
        print(f"--record-dagger: episodes -> {dagger.base} (opens at the first policy / cab-driven tick; "
              f"`touch {args.dagger_split_file}` saves it and starts a new one)")
    cab_map, cab_db = human_cab_map()
    end_reason, end_aborted = "quit", False
    track_drove = dagger is not None or args.state_from_drove
    prev_drove = {j: 0.0 for j in cfg.joint_names}    # what drove the machine on the previous tick
    if args.state_from_drove:
        print("--state-from-drove: the policy's state = the command that drove the machine on the previous tick "
              "(policy action / cab action after a STOP / zeros), not the gate's `sent`")

    # what the policy is shown as state for this tick's obs (tick log policy_state_*): zeros with --zero-state
    policy_obs = (lambda o: zero_state_view(o, cfg.joint_names)) if args.zero_state else (lambda o: o)
    period, t0, late, statuses = 1.0 / fps, time.monotonic(), 0, {}
    print(f"prompt: {args.prompt}")
    Path("logs").mkdir(exist_ok=True)
    tick_path = Path("logs") / f"ticks_{time.strftime('%m%d_%H%M%S')}.csv"
    tick_file = open(tick_path, "w", newline="")
    ticks = csv.writer(tick_file)
    ticks.writerow(["t", "step", "status", "allowed", "prompt"] + [f"plan_{j}" for j in cfg.joint_names]
                   + [f"sent_{j}" for j in cfg.joint_names] + [f"state_{j}" for j in cfg.joint_names]
                   + [f"policy_state_{j}" for j in cfg.joint_names]
                   + ["sync_mode", "sync_t_ref", "sync_anchor", "sync_skew_ms", "sync_left_out"]
                   + [f"cap_{c}" for c in cfg.cameras] + [f"age_ms_{c}" for c in cfg.cameras] + latch_cols
                   + phase_cols + (["control"] if dagger is not None else []))
    print(f"tick log: {tick_path}")
    try:
        while args.seconds <= 0 or time.monotonic() - t0 < args.seconds:
            tick = time.monotonic()
            if screen is not None:
                for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
                    if e.type == pygame.QUIT or e.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            t_before = time.monotonic()
            obs = robot.get_observation()
            t_obs, wall_obs = time.monotonic(), time.time()
            if phase is not None:
                phase.nudge()                                  # the phase_next file, if someone touched it
                obs = {**obs, SUBTASK_KEY: phase.tracker.phase}  # this tick's request carries the current phase
            allowed = True
            if pad is not None:
                allowed = pad.read()[1]
            if via_dash:
                allowed = robot.allowed          # the dashboard's verdict on the previous action (deadman held?)
            # --state-from-drove: the policy sees the previous tick's driving command as state (obs keeps the readback)
            pobs = {**obs, **{f"{j}.cmd": prev_drove[j] for j in cfg.joint_names}} if args.state_from_drove else obs
            action, status = ctrl.tick(pobs, args.prompt, allowed)
            if gain is not None:
                action = np.clip(np.asarray(action, dtype=float) * gain, -1.0, 1.0)
            if latch is not None:
                action = latch.apply(action, status, allowed)
            action = dict(zip(names, action))
            if dagger is not None and not robot.policy_on:
                # Policy off (STOP): the dashboard writes nothing, and the runner doesn't even offer an action -- the
                # heartbeat is neutral until the dashboard reports Policy on again
                action = {n: 0.0 for n in names}
            # real machine: no /cmd at all until the deadman has been pressed once (first write energises it)
            sent = robot.send_action(action) if via_dash or pad.engaged else action
            control = ""
            if track_drove:             # --machine dashboard only (--record-dagger / --state-from-drove)
                passed = robot.allowed
                raw_state = robot.last_state
                cab = cab_action((raw_state or {}).get("controls"), cab_map, cab_db)
                label = control_label(passed, cab, relay_closed(raw_state))
                # what drove the machine: the passed policy action; the cab (only with the relay open, after a STOP:
                # the cab can't drive while the code holds the relay); else nothing
                drove = ({j: float(sent[f"{j}.cmd"]) for j in cfg.joint_names} if label == "policy" else
                         cab if label == "human" else {j: 0.0 for j in cfg.joint_names})
            if dagger is not None:
                if split_requested(args.dagger_split_file):
                    dagger.end(f"split ({args.dagger_split_file})")
                control = dagger.tick(
                    prompt=args.prompt, now=t_obs, wall=wall_obs, poll_age=t_obs - t_before, raw_state=raw_state,
                    allowed=passed, policy_on=robot.policy_on,
                    drove=drove, cab=cab,
                    policy_action=None if ctrl.planned is None else dict(zip(cfg.joint_names, ctrl.planned)),
                    posted={j: float(action[f"{j}.cmd"]) for j in cfg.joint_names},
                    frames=step_frames(robot, list(cfg.cameras)),
                    sync=getattr(robot, "last_sync", None), phase=phase.tracker.phase if phase is not None else None,
                    latch_active=latch.active if latch is not None else None)
            if phase is not None:   # what actually went out: nothing before the first deadman press on --machine real
                if dagger is not None:   # DAgger: what drove the machine, the human's cab commands included
                    phase.step({f"{j}.cmd": v for j, v in drove.items()}, ctrl.step)
                else:
                    phase.step(sent if via_dash or pad.engaged else {n: 0.0 for n in names}, ctrl.step)
            statuses[status.split(":")[0]] = statuses.get(status.split(":")[0], 0) + 1
            plan = ctrl.planned if ctrl.planned is not None else [float("nan")] * len(names)
            ticks.writerow([round(time.time(), 3), ctrl.step, status, int(bool(allowed)), args.prompt]
                           + [round(float(v), 4) for v in plan] + [round(float(sent[n]), 4) for n in names]
                           + [round(float(obs[n]), 4) for n in names]
                           + [round(float(policy_obs(pobs)[n]), 4) for n in names]
                           + sync_columns(getattr(robot, "last_sync", None), list(cfg.cameras))
                           + (latch.tick_columns() if latch is not None else [])
                           + (phase.tick_columns() if phase is not None else [])
                           + ([control] if dagger is not None else []))
            tick_file.flush()
            if track_drove:
                prev_drove = drove

            if screen is not None:
                img = cv2.cvtColor(tile(obs, list(cfg.cameras), cfg.image_hw), cv2.COLOR_RGB2BGR)
                color = (0, 140, 0) if status == "ok" else (0, 0, 220)
                cv2.putText(img, f"{status}  queue {ctrl.remaining()}", (80, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.putText(img, args.prompt, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 40), 1)
                if phase is not None:
                    cv2.putText(img, f"phase: {phase.tracker.phase}", (8, img.shape[0] - 32), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (160, 60, 0), 1)
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                screen.blit(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
            else:
                state = (" ".join(f"{j} {sent[f'{j}.cmd']:+.2f}" for j in cfg.joint_names)
                         + f" | live {obs.get('live')} veto {obs.get('veto')!r}"
                         + (f" | deadman {'HELD' if robot.allowed else 'released'}" if via_dash else ""))
                shown = status + (" LATCH" if latch is not None and latch.active else "")
                print(f"\r{shown:14s} queue {ctrl.remaining():2d} | {state}"
                      + (f" | {phase.short()}" if phase is not None else "")
                      + (f" | {control or '-'} {dagger.short()}" if dagger is not None else ""), end="", flush=True)

            if phase is not None and phase.stop:   # --stop-after-release: the normal end below (finally: STOP, DAgger save)
                print("\nauto-stop: run ended")
                end_reason = f"auto-stop ({args.stop_after_release:g} s after release)"
                break
            spare = period - (time.monotonic() - tick)
            late += spare < 0
            time.sleep(max(0.0, spare))
        else:
            end_reason = f"--seconds {args.seconds:g}"
    except KeyboardInterrupt:
        pass
    except BaseException as e:  # camera stale, gate / dashboard error, ...: re-raised after finally (STOP first)
        end_reason, end_aborted = f"error: {type(e).__name__}: {e}", True
        raise
    finally:
        tick_file.close()
        try:
            robot.disconnect()  # real machine: /estop; dashboard: STOP
        finally:
            if dagger is not None:   # after the STOP: save what was recorded (an error still saves, meta aborted)
                dagger.close(end_reason, end_aborted)
                print(f"\nDAgger episodes this run: {dagger.saved or 'none (never engaged)'}"
                      + (f" | WRITE ERRORS {dagger.write_errors}: {dagger.last_error}" if dagger.write_errors else ""))
        if hasattr(policy, "close"):
            policy.close()
        if pad is not None:
            pad.disconnect()

    ok = [r for r in ctrl.log if r["ok"]]
    rtt = np.array([r["rtt_ms"] for r in ok]) if ok else np.array([np.nan])
    stale = np.array([r["stale_steps"] for r in ok]) if ok else np.array([np.nan])
    print(f"\n\n{len(ctrl.log)} requests ({ctrl.errors} failed) | rtt median {np.nanmedian(rtt):.0f} ms, "
          f"p95 {np.nanpercentile(rtt, 95):.0f} ms | stale steps dropped median {np.nanmedian(stale):.0f}")
    print(f"ticks: {statuses} | loop overruns: {late}")
    Path("logs").mkdir(exist_ok=True)
    path = Path("logs") / f"control_{time.strftime('%m%d_%H%M%S')}.csv"
    keys = sorted({k for r in ctrl.log for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(ctrl.log)
    print(f"log: {path}")
    return ctrl, statuses


class PhaseRunner:
    """control_loop --subtasks: an act.phase.PhaseTracker fed with the commands actually sent, plus the manual overrides
    (--start-phase for the first tracker of the run, the nudge file every tick). Events go to the tick log
    (phase_event) and are printed. --stop-after-release: .stop turns True once the tracker has been in RELEASE for that
    long (counted in ticks from entering it, cancelled if it leaves RELEASE first)."""

    COLUMNS = ["phase", "phase_event"]
    SHORT = {v: k for k, v in PHASE_ALIASES.items() if k != "grip"}

    def __init__(self, fps, joint_names, nudge_file, start_phase=None, stop_after_release=None):
        self.fps, self.joints = fps, list(joint_names)
        self.nudge_file = Path(nudge_file) if nudge_file else None
        self.start = parse_phase(start_phase) if start_phase else REACH
        self.tracker, self.shown, self.events, self.first = None, None, [], True
        self.stop_after = stop_after_release
        self.stop_ticks = None if stop_after_release is None else round(stop_after_release * fps)
        self.countdown, self.stop = None, False   # ticks left in RELEASE before the auto-stop; True = end the run

    def new(self, prompt, why):
        """A fresh tracker (run start). --start-phase applies to the run's first one only."""
        start = self.start if self.first else REACH
        self.tracker = PhaseTracker(self.fps, carry_sign(prompt), start=start, **PHASE_DEFAULTS)
        self.first = False
        sign = {-1: "swing -", 1: "swing +", None: "either swing direction"}[self.tracker.sign]
        msg = f"phase tracker ({why}): start {self.tracker.phase!r}, carry = {sign}"
        if start != REACH:
            msg += " [--start-phase override]"
            self.events.append(f"override:{start}")
        print(("\n" if why != "start" else "") + msg)

    def nudge(self):
        """Consume the nudge file if it exists: empty -> advance one phase, a phase name -> jump there. Logged."""
        self.shown = None
        if self.nudge_file is None or not self.nudge_file.exists():
            return
        try:
            text = self.nudge_file.read_text().strip()
            self.nudge_file.unlink()
        except OSError as e:
            print(f"\nphase nudge: can't consume {self.nudge_file}: {e}")
            return
        before = self.tracker.phase
        try:
            ev = self.tracker.set_phase(text) if text else self.tracker.advance()
        except ValueError as e:
            print(f"\nphase nudge ignored: {e}")
            return
        self.events.append(ev)
        print(f"\nphase OVERRIDE ({self.nudge_file}): {before!r} -> {self.tracker.phase!r}")

    def step(self, sent, step=None):
        """Feed this tick's sent command ({ch}.cmd -> value), deadbanded like the cab recording. step: the tick log's."""
        self.shown = self.tracker.phase
        before = self.tracker.phase
        ev = self.tracker.step({j: deadband(sent[f"{j}.cmd"]) for j in self.joints})
        if ev:
            self.events.append(ev)
            print(f"\nphase: {ev} -> {self.tracker.phase!r}" + (f" ({self.tracker.last_kind})" if ev != "grip" else "")
                  + f" [was {before!r}]")
        if self.stop_ticks is not None:
            self.count_down(step)

    def count_down(self, step):
        """--stop-after-release, once per tick after this tick's nudge / new tracker / step: whatever put the tracker in
        RELEASE starts the countdown, anything that took it out cancels it."""
        if self.tracker.phase != RELEASE:
            if self.countdown is not None:
                print(f"\nauto-stop cancelled at step {step}: left release")
            self.countdown = None
            return
        if self.countdown is None:
            self.countdown = self.stop_ticks
            print(f"\nrelease detected at step {step}, auto-stop after {self.stop_after:g} s")
        else:
            self.countdown -= 1
        if self.countdown <= 0 and not self.stop:
            self.stop = True
            self.events.append("auto_stop")

    def short(self):
        return "phase " + self.SHORT.get(self.tracker.phase, self.tracker.phase)

    def tick_columns(self):
        """phase = what this tick's request was built with; phase_event = this tick's events ('|'-joined)."""
        cells = [self.shown or self.tracker.phase, "|".join(self.events)]
        self.events = []
        return cells


def policy_block(args, fps, sync):
    """meta.json "policy" of a DAgger episode: what was driving (no secrets)."""
    return {"backend": args.policy, "gateway_url": args.gateway_url, "url": args.url, "dataset": args.dataset,
            "zero_state": args.zero_state, "sync": sync, "chunk": args.chunk,
            "request_at": args.request_at, "fps": fps,
            "grip_latch": ({"threshold": args.grip_latch_threshold, "hold_s": args.grip_latch_s,
                            "refractory_s": args.grip_latch_refractory_s} if args.grip_latch else False),
            "joint_gain": args.joint_gain or None,
            "subtasks": args.subtasks,
            "state_from_drove": args.state_from_drove, "machine": args.machine,
            "dashboard_url": args.dashboard_url}


def split_requested(path):
    """--dagger-split-file: consume (delete) it if it exists -> True."""
    p = Path(path) if path else None
    if p is None or not p.exists():
        return False
    try:
        p.unlink()
    except OSError:
        return False
    return True


def step_frames(robot, cams):
    """{camera: act.recorder.Frame} of the frames THIS tick's observation used: the camsync-picked RingFrames (camview
    JPEG bytes, as received)."""
    picked, per = getattr(robot, "last_picked", {}) or {}, (getattr(robot, "last_sync", None) or {}).get("cams", {})
    out = {}
    for c in cams:
        item, info = picked.get(c), per.get(c)
        if item is not None and info is not None:
            out[c] = Frame(item.data, item.arrival, info["cap"], item.seq, info["age"])
    return out


def sync_columns(info, cams):
    """RigMachine.last_sync -> tick-log cells (blank when there is none)."""
    if not info:
        return [""] * (5 + 2 * len(cams))
    per = info.get("cams", {})
    return ([info["mode"], "" if info["t_ref"] is None else round(info["t_ref"], 4), info["anchor"] or "",
             round(info["skew"] * 1e3, 1), "|".join(info["left_out"])]
            + [round(per[c]["cap"], 4) if c in per else "" for c in cams]
            + [round(per[c]["age"] * 1e3, 1) if c in per else "" for c in cams])


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--machine", choices=["real", "dashboard"], required=True,
                    help="dashboard: on the Pi, actions go through dashboard.py's Policy mode")
    ap.add_argument("--dashboard-url", default=DEFAULT_DASHBOARD_URL, help="--machine dashboard: dashboard.py address")
    ap.add_argument("--rig-url", default=DEFAULT_RIG_URL, help="gate base URL (--machine real)")
    ap.add_argument("--policy", choices=["http", "pi"], required=True)
    ap.add_argument("--url", help="policy server base URL (for --policy http)")
    ap.add_argument("--gateway-url", help="PI checkpoint gateway_url (for --policy pi)")
    ap.add_argument("--chunk", type=int, default=15, help="--policy pi: actions per request (execute_chunk_size)")
    ap.add_argument("--grip-latch", action="store_true",
                    help="assist: once the policy clamps the thumb past --grip-latch-threshold, send a full clamp for "
                         "--grip-latch-s (cancelled at once by deadman release / STOP / any neutral tick)")
    ap.add_argument("--grip-latch-threshold", type=float, default=0.3, help="policy thumb value that triggers the latch")
    ap.add_argument("--grip-latch-s", type=float, default=3.5, help="how long the latched full clamp is held (s)")
    ap.add_argument("--grip-latch-refractory-s", type=float, default=5.0, help="no new latch for this long after one ends")
    ap.add_argument("--joint-gain", action="append", default=[], metavar="JOINT=K",
                    help="multiply the policy's command on JOINT by K, then clip to -1..1 (repeatable; e.g. boom=1.5)")
    ap.add_argument("--subtasks", action="store_true",
                    help="--policy pi: send the cone pick-and-place phase (act.phase, tracked from the sent commands) "
                         "as raw_text, like the training labels (docs/subtask_labels.md)")
    ap.add_argument("--start-phase", choices=sorted(PHASE_ALIASES), default=None,
                    help="--subtasks: start the run in this phase (manual override, logged)")
    ap.add_argument("--stop-after-release", type=float, default=None, metavar="SECONDS",
                    help="--subtasks: end the run this long after the phase enters release (tracker event or "
                         "override); leaving release first cancels it. Recommended 1.5")
    ap.add_argument("--phase-file", default="phase_next",
                    help="--subtasks: nudge file; touch it to advance one phase (or write a phase name into it); "
                         "consumed and deleted by the runner")
    ap.add_argument("--dataset", help="training dataset root: crops, image size and fps come from it")
    ap.add_argument("--zero-state", action="store_true",
                    help="the policy sees state = zeros (checkpoint trained on a scripts/zero_state.py dataset); "
                         "required iff --dataset has meta/zero_state.json")
    ap.add_argument("--prompt", required=True, help="task string (must match the training task exactly)")
    ap.add_argument("--fps", type=int, default=None, help="control rate (default: the --dataset fps, else 10)")
    ap.add_argument("--seconds", type=float, default=0, help="0 = until ESC / Ctrl-C")
    ap.add_argument("--request-at", type=int, default=5, help="ask for the next chunk when this many actions remain")
    ap.add_argument("--max-use", type=int, default=25, help="use at most this many steps of each chunk")
    ap.add_argument("--stop-after", type=float, default=1.0, help="watchdog: stop if no chunk for this long (s)")
    ap.add_argument("--image-px", type=int, default=224,
                    help="--policy http without --dataset: long side of frames sent (with --dataset: its image size)")
    ap.add_argument("--jpeg-quality", type=int, default=None, help="--policy http: default 95 with --dataset, else 80")
    ap.add_argument("--cam", action="append", help="name=source (repeatable; at least one)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--deadman", action="store_true", help="policy acts only while SPACE / L1 is held (required for --machine real)")
    ap.add_argument("--sync", choices=camsync.SYNC_MODES, default=None,
                    help="real / dashboard machine: how one observation's frames are picked across cameras "
                         "(act/camsync.py); " + SYNC_HELP + ". Default: the dataset's recorded mode, else aligned")
    ap.add_argument("--allow-camsync-mismatch", action="store_true",
                    help="real / dashboard machine: run even if act/camsync.py differs from the recorder's camsync.py")
    ap.add_argument("--record-dagger", action="store_true",
                    help="--machine dashboard: record DAgger episodes (policy + human cab takeover steps) in the "
                         "dashboard recorder's raw format into --episodes-dir (docs/dagger.md)")
    ap.add_argument("--state-from-drove", action=argparse.BooleanOptionalAction, default=None,
                    help="real-state checkpoints (--machine dashboard): the policy's state = the command that "
                         "drove the machine on the previous tick (policy action / cab action / zeros) instead of the "
                         "gate's `sent`; default on with --record-dagger unless --zero-state")
    ap.add_argument("--episodes-dir", default="~/act-athon/episodes", help="--record-dagger: where episodes go")
    ap.add_argument("--dagger-queue", type=int, default=300,
                    help="--record-dagger: max steps waiting for the disk before recording steps are dropped")
    ap.add_argument("--dagger-split-file", default="dagger_next",
                    help="--record-dagger: touch this file to save the current episode (a new one opens at the next "
                         "engagement); consumed and deleted by the runner")
    return ap.parse_args(argv)


if __name__ == "__main__":
    run(parse_args())

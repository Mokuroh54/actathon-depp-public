"""The subtask phase of a cone pick-and-place, derived CAUSALLY from the joystick commands (docs/subtask_labels.md).

One rule for both sides:
  training   scripts/subtask_labels.py runs PhaseTracker over each recorded episode's cab commands -> the
             language_segment spans (subtask = the phase) uploaded by scripts/pi_upload_spans.py
  inference  scripts/control_loop.py --subtasks runs a PhaseTracker over the commands it actually sends
             -> raw_text of every PI request
so the model is told the phase exactly the way it was labelled in training.

Phases (exact strings; the subtask instruction):
  REACH    "reach and grip the cone"       from the start until the grip is complete
  CARRY    "carry the cone to the target"  from grip complete until the release starts
  RELEASE  "release the cone"              from the release until the end of the episode (latched)

Rule (thresholds in seconds, converted with fps; commands in -1..1 rig signs: thumb + = clamp in, - = open;
bucket + = dump; values inside the recording's +-0.08 deadband count as 0, see deadband()):
  REACH -> CARRY    grip complete: the first step with thumb <= thr after a clamp of >= min_clamp s of accumulated
                    thumb > thr (clamp pieces separated by gaps <= clamp_gap s count as one clamp)
  in CARRY          "carried" once the swing has accumulated >= min_carry s of |swing| > swing_thr in the task's
                    carry direction (sign; None = either direction)
  CARRY -> ...      at the first step after an OPEN: a thumb-open run (thumb < -thr) of >= min_open s, or -- only once
                    carried -- a bucket-dump run (bucket > dump_thr) of >= min_dump s (with combined_dump, also while
                    the thumb clamps). Once carried the open is the RELEASE (-> RELEASE, latched); before the carry it
                    is a regrip (-> REACH, the next full clamp completes a new grip).
Every transition fires when the command returns to neutral after the qualifying run, so the policy issues the grip /
open under the old phase and sees the new phase right after.
"""

REACH = "reach and grip the cone"
CARRY = "carry the cone to the target"
RELEASE = "release the cone"
PHASES = (REACH, CARRY, RELEASE)
# short names for CLIs / the nudge file
PHASE_ALIASES = {"reach": REACH, "grip": REACH, "carry": CARRY, "release": RELEASE}

RECORDING_DEADBAND = 0.08   # the cab recording's deadband (meta.json cab_deadband): |x| <= 0.08 was recorded as 0

# task text -> swing sign of the carry (the session-2/3 layout: the blue square is house-left of the red one)
CARRY_SIGN = {"from the blue square to the red square": -1, "from the red square to the blue square": +1}

DEFAULTS = dict(thr=0.3, min_clamp=1.2, clamp_gap=0.5, min_open=0.2, dump_thr=0.7, min_dump=0.4, swing_thr=0.3,
                min_carry=1.0, combined_dump=True)


def carry_sign(task):
    """The carry's swing sign for a task string, None if the task names no known pair of squares (either direction)."""
    for key, sign in CARRY_SIGN.items():
        if key in (task or ""):
            return sign
    return None


def parse_phase(text):
    """'reach' / 'carry' / 'release' or an exact phase string -> the phase string. ValueError otherwise."""
    t = (text or "").strip()
    if t in PHASES:
        return t
    if t.lower() in PHASE_ALIASES:
        return PHASE_ALIASES[t.lower()]
    raise ValueError(f"unknown phase {text!r}: one of {', '.join(PHASE_ALIASES)} or {PHASES}")


def deadband(value, width=RECORDING_DEADBAND):
    """A command as the cab recording would have stored it: 0 inside +-width, else unchanged (clipped to -1..1)."""
    v = float(value)
    return 0.0 if abs(v) <= width else max(-1.0, min(1.0, v))


class PhaseTracker:
    """Feed one command per control step (a mapping with thumb, bucket, swing; any extra keys ignored) to step();
    read .phase. step() returns the event of this step: "grip" (REACH -> CARRY), "regrip_open" (CARRY -> REACH),
    "release" (CARRY -> RELEASE) or None; the new phase applies from the next step on.
    Manual overrides (the runner's nudge): advance(), set_phase(). Their events are "override:<phase>"."""

    def __init__(self, fps, sign=None, thr=0.3, min_clamp=1.2, clamp_gap=0.5, min_open=0.2, dump_thr=0.7,
                 min_dump=0.4, swing_thr=0.3, min_carry=1.0, combined_dump=True, start=REACH):
        self.fps, self.sign = float(fps), sign
        self.thr, self.dump_thr, self.swing_thr = float(thr), float(dump_thr), float(swing_thr)
        self.min_clamp = round(min_clamp * self.fps)
        self.clamp_gap = round(clamp_gap * self.fps)
        self.min_open = max(1, round(min_open * self.fps))
        self.min_dump = max(1, round(min_dump * self.fps))
        self.min_carry = round(min_carry * self.fps)
        self.combined_dump = bool(combined_dump)
        self.params = dict(fps=self.fps, sign=sign, thr=thr, min_clamp=min_clamp, clamp_gap=clamp_gap,
                           min_open=min_open, dump_thr=dump_thr, min_dump=min_dump, swing_thr=swing_thr,
                           min_carry=min_carry, combined_dump=self.combined_dump)
        self.last_kind = None   # what the last CARRY exit was: thumb_open, bucket_dump, combined_dump
        self._enter(parse_phase(start))

    def _enter(self, phase):
        self.phase = phase
        self.carried = False
        self.clamp_acc, self.since_clamp = 0, 10 ** 9
        self.open_run = self.dump_run = self.carry_acc = 0
        self.dump_had_clamp = False

    @property
    def done(self):
        return self.phase == RELEASE

    def step(self, a):
        thumb, bucket, swing = float(a["thumb"]), float(a["bucket"]), float(a["swing"])
        clamping, opening = thumb > self.thr, thumb < -self.thr
        dumping = bucket > self.dump_thr and (self.combined_dump or not clamping)
        if self.phase == RELEASE:
            return None
        if self.phase == REACH:
            if clamping:
                if self.since_clamp > self.clamp_gap:
                    self.clamp_acc = 0
                self.clamp_acc += 1
                self.since_clamp = 0
                return None
            fire = self.since_clamp == 0 and self.clamp_acc >= self.min_clamp
            self.since_clamp += 1
            if fire:
                self._enter(CARRY)
                return "grip"
            return None
        # CARRY
        if (swing * self.sign if self.sign else abs(swing)) > self.swing_thr:
            self.carry_acc += 1
            if self.carry_acc >= self.min_carry:
                self.carried = True
        ended_open = self.open_run >= self.min_open and not opening
        ended_dump = self.carried and self.dump_run >= self.min_dump and not dumping
        kind = None
        if ended_open or ended_dump:
            kind = "thumb_open" if ended_open else ("combined_dump" if self.dump_had_clamp else "bucket_dump")
        self.open_run = self.open_run + 1 if opening else 0
        self.dump_had_clamp = (self.dump_had_clamp or clamping) if dumping and self.dump_run else clamping
        self.dump_run = self.dump_run + 1 if dumping else 0
        if kind is None:
            return None
        self.last_kind = kind
        if self.carried:
            self._enter(RELEASE)
            return "release"
        self._enter(REACH)
        return "regrip_open"

    def set_phase(self, phase):
        """Manual override: jump to `phase` (a phase string or alias) with fresh counters -> the event string."""
        self._enter(parse_phase(phase))
        return f"override:{self.phase}"

    def advance(self):
        """Manual override: REACH -> CARRY -> RELEASE -> REACH (a new attempt) -> the event string."""
        return self.set_phase(PHASES[(PHASES.index(self.phase) + 1) % len(PHASES)])

"""Gamepad teleop (step 2), with a keyboard fallback.

Works with any controller SDL recognizes (Xbox, PlayStation, Switch Pro, most
generic USB/Bluetooth pads) through SDL's standard layout, so you don't need
to know which controller you have or its raw axis numbers.

Outputs {ch}.cmd joystick commands in -1..1 with the rig's signs (act/machine.py):
swing + = house right, boom + = down, arm + = out, bucket + = dump/open, thumb + = clamp in.

Default mapping is the ISO excavator pattern (change TeleopConfig.mapping to suit):
    left stick  right / left     -> swing right / left
    left stick  forward / back   -> arm out / in
    right stick forward / back   -> boom down / up
    right stick left / right     -> bucket curl / dump
    R2/RT                        -> thumb clamp    L2/LT -> thumb open
Hold a shoulder button (L1/LB or R1/RB) as the deadman: release it and every channel is
commanded neutral (0). `engaged` becomes True the first time the deadman is pressed; on the real
machine the scripts send nothing at all until then (the first /cmd write energises the relay).

No controller? Keyboard in the pygame window: A/D swing (D = right), W/S arm (W = out),
I/K boom (I = down), J/L bucket (J = curl), U/O thumb (U = open, O = clamp), hold SPACE as deadman.

    uv run scripts/teleop.py --probe             # see what the pad reports
    uv run scripts/teleop.py --machine real      # the excavator (tunnel to the gate first; see README)
    uv run scripts/teleop.py --machine real --show --video teleop.mp4
"""
import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import numpy as np
import pygame
from pygame._sdl2 import controller as sdl_controller

from act.machine import CHANNELS

SDL_AXES = {
    "leftx": pygame.CONTROLLER_AXIS_LEFTX,
    "lefty": pygame.CONTROLLER_AXIS_LEFTY,
    "rightx": pygame.CONTROLLER_AXIS_RIGHTX,
    "righty": pygame.CONTROLLER_AXIS_RIGHTY,
}
# triggers are 0..1 in SDL; read() combines them into one signed axis "triggers" = RT - LT
SDL_TRIGGERS = {
    "lefttrigger": pygame.CONTROLLER_AXIS_TRIGGERLEFT,
    "righttrigger": pygame.CONTROLLER_AXIS_TRIGGERRIGHT,
}
DEADMAN_BUTTONS = (pygame.CONTROLLER_BUTTON_LEFTSHOULDER, pygame.CONTROLLER_BUTTON_RIGHTSHOULDER)


@dataclass
class TeleopConfig:
    # channel -> (stick axis, sign), giving rig-signed commands. SDL reports "up" (forward) as negative y.
    mapping: dict = field(default_factory=lambda: {
        "swing": ("leftx", 1.0),      # right (D) = + = house right
        "arm": ("lefty", -1.0),       # forward (W) = + = out
        "boom": ("righty", -1.0),     # forward (I) = + = down
        "bucket": ("rightx", 1.0),    # left (J) = - = curl, right (L) = + = dump
        "thumb": ("triggers", 1.0),   # RT (O) = + = clamp in, LT (U) = - = open
    })
    deadzone: float = 0.12
    expo: float = 0.5  # 0 = linear, 1 = cubic; more expo = finer control near center
    # keyboard fallback: axis -> (key for -1, key for +1)
    keys: dict = field(default_factory=lambda: {
        "leftx": (pygame.K_a, pygame.K_d),
        "lefty": (pygame.K_w, pygame.K_s),
        "rightx": (pygame.K_j, pygame.K_l),
        "righty": (pygame.K_i, pygame.K_k),
        "triggers": (pygame.K_u, pygame.K_o),  # U = LT (open), O = RT (clamp)
    })


def _norm(v):
    """SDL controller axes come back as ints in [-32768, 32767]."""
    return max(-1.0, min(1.0, v / 32767.0)) if abs(v) > 1 else float(v)


class Gamepad:
    """LeRobot-teleoperator-shaped: connect(), get_action(obs) -> dict, disconnect()."""

    def __init__(self, cfg: TeleopConfig, joint_names=CHANNELS):
        self.cfg, self.joint_names = cfg, list(joint_names)
        self.pad, self.deadman = None, False
        self.engaged = False  # deadman pressed at least once since connect()

    @property
    def action_features(self):
        return {f"{j}.cmd": float for j in self.joint_names}

    @property
    def uses_keyboard(self):
        return self.pad is None

    def connect(self):
        pygame.init()
        sdl_controller.init()
        for i in range(sdl_controller.get_count()):
            if sdl_controller.is_controller(i):
                self.pad = sdl_controller.Controller(i)
                self.name = sdl_controller.name_forindex(i)
                break
        else:
            self.name = "keyboard (no controller found)"
        print(f"teleop input: {self.name}")

    def disconnect(self):
        if self.pad is not None:
            self.pad.quit()
        pygame.quit()

    def read(self):
        """-> ({axis: value in [-1, 1]}, deadman_held)"""
        pygame.event.pump()
        if self.pad is not None:
            axes = {k: _norm(self.pad.get_axis(a)) for k, a in SDL_AXES.items()}
            trig = {k: _norm(self.pad.get_axis(a)) for k, a in SDL_TRIGGERS.items()}
            axes["triggers"] = trig["righttrigger"] - trig["lefttrigger"]
            deadman = any(self.pad.get_button(b) for b in DEADMAN_BUTTONS)
        else:
            pressed = pygame.key.get_pressed()
            axes = {k: float(pressed[pos]) - float(pressed[neg]) for k, (neg, pos) in self.cfg.keys.items()}
            deadman = bool(pressed[pygame.K_SPACE])
        self.engaged = self.engaged or deadman
        return axes, deadman

    def _shape(self, x):
        dz = self.cfg.deadzone
        if abs(x) < dz:
            return 0.0
        x = np.sign(x) * (abs(x) - dz) / (1 - dz)
        return (1 - self.cfg.expo) * x + self.cfg.expo * x ** 3

    def get_action(self, obs: dict = None) -> dict:
        """Rig-signed commands in -1..1; all zeros (neutral) while the deadman is released."""
        axes, deadman = self.read()
        a = np.clip([self._shape(axes[ax]) * sign for ax, sign in (self.cfg.mapping[j] for j in self.joint_names)],
                    -1.0, 1.0)
        if not deadman:
            a = np.zeros(len(self.joint_names))
        self.deadman = deadman
        return {f"{j}.cmd": float(a[i]) for i, j in enumerate(self.joint_names)}


def probe():
    pygame.init()
    sdl_controller.init()
    pygame.joystick.init()
    n = pygame.joystick.get_count()
    print(f"{n} joystick(s) found")
    if n == 0:
        print("none detected: pair it in System Settings > Bluetooth, or plug it in, then rerun")
        return
    js = pygame.joystick.Joystick(0)
    std = sdl_controller.Controller(0) if sdl_controller.is_controller(0) else None
    print(f"name: {js.get_name()} | axes: {js.get_numaxes()} | buttons: {js.get_numbuttons()} "
          f"| standard layout: {'yes' if std else 'NO (tell Claude the name)'}")
    print("move sticks / press buttons; Ctrl-C to stop")
    try:
        while True:
            pygame.event.pump()
            raw = " ".join(f"{js.get_axis(i):+.2f}" for i in range(js.get_numaxes()))
            btn = [i for i in range(js.get_numbuttons()) if js.get_button(i)]
            line = f"raw axes [{raw}] buttons {btn}"
            if std:
                line += " | " + " ".join(f"{k} {_norm(std.get_axis(a)):+.2f}" for k, a in {**SDL_AXES, **SDL_TRIGGERS}.items())
                line += f" deadman {any(std.get_button(b) for b in DEADMAN_BUTTONS)}"
            print("\r" + line + "   ", end="", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print()

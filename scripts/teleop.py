"""Drive the real excavator with a gamepad or keyboard (see act/teleop.py for controls).

Commands are rig-signed joystick values in -1..1, sent every tick (the gate goes neutral 0.5 s after the
last command). On the real machine nothing is sent until the deadman is first pressed (the first /cmd
energises the relay); after that, releasing it sends neutral (all zeros). Quitting sends /estop.
Tunnel to the gate first: ssh -L 8096:127.0.0.1:8096 excavator

    uv run scripts/teleop.py --probe
    uv run scripts/teleop.py --machine real [--cam front=0] [--show] [--video teleop.mp4]
"""
import argparse
import os
import time

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import cv2
import pygame

from act.cameras import tile
from act.rig_machine import DEFAULT_RIG_URL, make_machine
from act.teleop import Gamepad, TeleopConfig, probe
from act.video import open_video_writer


def run(machine, rig_url, seconds, fps, show, video, cameras=None):
    if fps < 5:
        raise SystemExit("--fps must be >= 5 on the real machine (the gate goes neutral 0.5 s after a command)")
    robot = make_machine(machine, cameras, rig_url)
    mcfg = robot.cfg
    pad = Gamepad(TeleopConfig(), mcfg.joint_names)
    robot.connect()
    pad.connect()
    h, w = mcfg.image_hw
    width = w * max(1, len(mcfg.cameras))
    screen = None
    if show or pad.uses_keyboard:  # keyboard input needs a focused pygame window
        screen = pygame.display.set_mode((width, h))
        pygame.display.set_caption(f"teleop ({machine}): hold deadman to move, Esc to quit")
    writer = open_video_writer(video, fps, width, h)[0] if video else None

    period, t0, n, late = 1.0 / fps, time.monotonic(), 0, 0
    try:
        while seconds <= 0 or time.monotonic() - t0 < seconds:
            tick = time.monotonic()
            if screen is not None:
                for e in pygame.event.get((pygame.QUIT, pygame.KEYDOWN)):
                    if e.type == pygame.QUIT or e.key == pygame.K_ESCAPE:
                        raise KeyboardInterrupt
            obs = robot.get_observation()
            action = pad.get_action(obs)  # neutral while the deadman is released
            # real machine: no /cmd at all until the deadman has been pressed once (first write energises it)
            sent = robot.send_action(action) if pad.engaged else action
            rgb = tile(obs, list(mcfg.cameras), mcfg.image_hw)
            if writer is not None:
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if screen is not None:
                screen.blit(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)), (0, 0))
                pygame.display.flip()
            cmds = " ".join(f"{j} {sent[f'{j}.cmd']:+.2f}" for j in mcfg.joint_names)
            extra = f" | live {obs['live']} veto {obs['veto']!r}"
            mode = "MOVING " if pad.deadman else ("deadman" if pad.engaged else "idle (nothing sent)")
            print(f"\r{mode} | {cmds}{extra}   ", end="", flush=True)
            n += 1
            spare = period - (time.monotonic() - tick)
            late += spare < 0
            time.sleep(max(0.0, spare))
    except KeyboardInterrupt:
        pass
    finally:
        robot.disconnect()  # real machine: /estop
        pad.disconnect()
        if writer is not None:
            writer.release()
            print(f"\nsaved {video}")
    print(f"\n{n} ticks, {late} late (>{1e3 * period:.0f} ms)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="print what the controller reports")
    ap.add_argument("--machine", choices=["real"], default="real")
    ap.add_argument("--rig-url", default=DEFAULT_RIG_URL, help="gate base URL")
    ap.add_argument("--seconds", type=float, default=0, help="0 = run until Esc / Ctrl-C")
    ap.add_argument("--fps", type=int, default=10, help="control rate; >= 5 (0.5 s gate hold)")
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--video", help="save frames to this .mp4")
    ap.add_argument("--cam", action="append", help="name=source, e.g. front=0 wrist=1 (repeatable); default: none")
    args = ap.parse_args()
    if args.probe:
        probe()
    else:
        run(args.machine, args.rig_url, args.seconds, args.fps, args.show, args.video, args.cam)

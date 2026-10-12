"""Read-only: one observation through the runner's RigMachine path (crops + size from the training dataset). Never
writes to the gate (no send_action, no disconnect: that would /estop); releases the cameras directly."""
import sys

import cv2

sys.path.insert(0, ".")
sys.argv = ["x"]
from control_loop import training_setup
from act.rig_machine import make_machine

fps, hw, crops = training_setup("datasets/rig_cab_nighttest")
base = "http://127.0.0.1:8100/stream/"
cams = ["front=" + base + "usb-3-1-3-1-0", "wrist=" + base + "lux-1944301051A0764800-color", "side=" + base + "usb-4-1-1-1-0"]
m = make_machine("real", cams, require_cameras=True, crops=crops, image_hw=hw)
m.connect()
obs = m.get_observation()
for c in ("front", "wrist", "side"):
    print(c, obs[c].shape, obs[c].dtype, "mean %.0f" % obs[c].mean())
    cv2.imwrite("/tmp/camcheck_%s.jpg" % c, cv2.cvtColor(obs[c], cv2.COLOR_RGB2BGR))
print("state", {k: v for k, v in obs.items() if k.endswith(".cmd")}, "live", obs["live"])
for cap in m.caps.values():
    cap.release()

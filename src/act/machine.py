"""The excavator's action / observation schema (the machine itself is act/rig_machine.py).

Action space:
    {ch}.cmd for ch in CHANNELS, joystick commands in -1..1 (NOT positions), rig signs:
        swing  + = house right       boom   + = down (lower)     arm + = out (extend)
        bucket + = dump / open       thumb  + = clamp in
Observation state: the same 5 {ch}.cmd values (the real gate's readback), plus cameras. There is no joint-angle
feedback on the machine.
"""

CHANNELS = ("swing", "boom", "arm", "bucket", "thumb")  # order of every per-joint list / array

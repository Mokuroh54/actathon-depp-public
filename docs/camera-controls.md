# Camera controls (locked exposure and white balance)

The USB webcams on the Pi (Logitech C270) run with fixed image controls. `pi/act-athon/camctl.py` writes them and
keeps them in place, and `pi/act-athon/start_dashboard.sh` runs it every time the dashboard starts.

## Why the controls are locked

- **Steady 15 fps.** With `exposure_dynamic_framerate=1` (the C270 default) the camera lengthens its frame time
  when it wants more light and drops below the 15 fps camview asks for. The recording rate must be steady
  (AGENTS.md, Data rules).
- **Same images in training and in the demo.** AGENTS.md requires fixed exposure and white balance between
  collection and the demo. Auto exposure and auto white balance change the picture with every cloud and with
  whatever is in view, including the arm itself.
- **Values reset on re-enumeration.** V4L2 control values are stored in the camera. A replug, a USB reset or a Pi
  reboot puts them back to the camera defaults (auto everything), and the `/dev/videoN` number can change too.
  That is why `camctl.py watch` runs next to the dashboard. It finds cameras by their camview stream id, which is
  derived from the physical USB port, so it doesn't depend on `/dev/videoN`.

## What is locked

`pi/act-athon/cam_controls.json`, tuned in outdoor daylight on the C270:

| control | value | meaning |
|---|---|---|
| `exposure_dynamic_framerate` | 0 | never drop the frame rate to get more light |
| `backlight_compensation` | 0 | off |
| `auto_exposure` | 1 | Manual Mode (3 = auto, "Aperture Priority") |
| `exposure_time_absolute` | 5 | units of 100 µs, so 0.5 ms. On the C270, values 1 to 5 give identical images; 8 and above clips in sun |
| `white_balance_automatic` | 0 | off |
| `white_balance_temperature` | 5000 | K, daylight |

**Write order matters.** A UVC camera rejects writes to an inactive control. `exposure_time_absolute` is
inactive while auto exposure is on, and `white_balance_temperature` is inactive while auto white balance is on.
camctl always writes the mode controls (`auto_exposure`, `white_balance_automatic`, …) first, whatever their
order in the file.

## Run

Copy the files to the Pi next to `dashboard.py` (`~/act-athon`): `camctl.py`, `cam_controls.json` and
`start_dashboard.sh`. Then start the dashboard through the script instead of `python3 dashboard.py`:

```bash
cd ~/act-athon && ./start_dashboard.sh                 # = python3 dashboard.py --episode-s 90
cd ~/act-athon && ./start_dashboard.sh --episode-s 45  # your --episode-s wins (either --episode-s N or --episode-s=N)
```

The script works like this:

1. It runs `python3 camctl.py apply`, which writes the controls to every USB camera, reads them back and prints
   want vs got per camera. If something doesn't take, it prints a loud warning and **starts the dashboard anyway**.
2. It starts `python3 camctl.py watch` in the background, logging to `logs/camctl.log`. Every 5 s the watcher
   reads each camera's controls. It writes only when a value differs from the config: after a replug or reboot, or
   when a camera is plugged into a new port. Each re-apply is logged with a timestamp. The watcher also picks up
   edits to `cam_controls.json` without a restart.
3. It runs `python3 dashboard.py` in the foreground with all your flags. Every dashboard flag works unchanged.
   `--episode-s` defaults to 90 (`EPISODE_S_DEFAULT` at the top of the script) unless you pass it yourself.
   Ctrl-C and closing the ssh session stop the dashboard as before (it sends STOP), and the watcher stops with
   it. If the script is killed, the watcher exits on its own within one interval.

camview can keep streaming the whole time, because camctl only changes controls and never opens a stream.
After every apply, `logs/camctl_status.json` records, per camera id, the `/dev` node, `applied_at`, the actual
values, `ok`, and any mismatched or skipped controls.

Other commands:

```bash
python3 camctl.py show              # read-only: every control, its range, flags, and the configured value
python3 camctl.py apply --dry-run   # what apply would write; writes nothing
python3 camctl.py apply             # exit 0 = all applied, 1 = a control didn't take (MISMATCH), 2 = bad config
```

## Changing values

Edit `cam_controls.json`. The file has two kinds of keys:

- `"*"` applies to every USB camera.
- A camview stream id (`usb-3-1-1-1-0`, as shown on the dashboard tile and by `camctl.py show`) overrides single
  controls for the camera in that USB port. Setting a control to `null` there means "leave this control alone
  on that camera".

The running watcher applies the change within 5 s. `python3 camctl.py apply` does it immediately and shows the
result.

**Changing exposure or white balance in the middle of a dataset makes its episodes inconsistent.** If you change
them, start a new batch or collection and write down when and why. Don't mix old and new episodes in one
training set unless you mean to.

## Finding a good exposure

1. `python3 camctl.py show` shows the current values and each control's range.
2. Try a value on the running camera: `v4l2-ctl -d /dev/videoN -c exposure_time_absolute=8`. `show` prints the
   `/dev` node for each camera id. With the watcher running, a manual change is reverted within 5 s. For longer
   experiments, edit `cam_controls.json` instead, or stop the dashboard first.
3. Check a frame on the camview / dashboard tile, in the real light, with the arm and trash in view. Look for
   clipped highlights (sky, white or shiny trash) and whether the dark bucket interior is still readable.
4. Put the value you chose in `cam_controls.json`.

The C270 can't go below about 5 in practice: values 1 to 5 all give the same image. In full sun even 5 is on
the bright side. If it clips, shade the camera (hood or tape); the exposure can't go lower. Re-check in the
demo-day light: a sunny noon and an overcast morning need different values, and a value tuned at night means
nothing outdoors.

## Network cameras

camctl only controls the USB cameras attached to the Pi. A network camera, such as the `world` camera streamed
from a friend's laptop (`docs/remote-camera.md`), must have its exposure and white balance locked on the computer
that sends it, with that computer's own camera tools. The Pi only receives JPEGs, so it can't change them.

## Checking on the Pi

- `./start_dashboard.sh` should print one block per camera with every line `ok`.
- `tail -f logs/camctl.log`, then unplug and replug a camera. Within about 5 s there should be a `camera … at
  /dev/videoN` line and a `re-apply … ok` line. Then `python3 camctl.py show` should show no `DIFFERS`.
- If a control is listed as unsupported (`skipped`), that camera model or kernel uses a different control name.
  Use the name `v4l2-ctl -d /dev/videoN -l` prints.

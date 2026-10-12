# Excavator Pi: access

The excavator's onboard computer is a Raspberry Pi 5 (`<robot-id>`, Debian 13, aarch64, no GPU, 4 GB RAM).

- Address: `<pi-host>` (Tailscale; your laptop must be on the event tailnet). Also `<pi-host>.local` on the same LAN.
- User: the organisers' unprivileged account. Password: ask the organisers. Don't commit it anywhere.

## One-time setup (per laptop)

1. Make a dedicated key without a passphrase (skip if `~/.ssh/id_ed25519_actathon` exists):
   ```bash
   ssh-keygen -t ed25519 -N "" -C "actathon-excavator" -f ~/.ssh/id_ed25519_actathon
   ```
2. Add a host alias to `~/.ssh/config`:
   ```
   Host excavator
       HostName <pi-host>
       User <user>
       IdentityFile ~/.ssh/id_ed25519_actathon
       IdentitiesOnly yes
   ```
3. Install the key on the Pi (asks for the account password once; run it on your laptop, not on the Pi):
   ```bash
   ssh-copy-id -i ~/.ssh/id_ed25519_actathon.pub excavator
   ```
4. Check: `ssh excavator hostname` should print `<robot-id>` with no password prompt.

If login fails with `Permission denied (publickey,password)`: check Tailscale is up (`tailscale status`), and that step 3 was done for the key the alias uses.

## Everyday use

```bash
ssh excavator                                        # shell on the Pi
ssh excavator 'cd act-athon && .venv/bin/python3 camview.py'   # camera server (teammate's), http://<pi-host>:8100
ssh excavator 'cd act-athon && python3 dashboard.py --cam front=<camview id>'  # status + teleop + recording, http://<pi-host>:8080 (docs/dashboard.md)
ssh -N -L 8096:127.0.0.1:8096 excavator              # tunnel the control gate to the laptop (for --machine real)
scp pi/act-athon/dashboard.py excavator:act-athon/   # copy a file over
```

While the tunnel is open, anything on the laptop that talks to `127.0.0.1:8096` moves the real machine.

## What's on the Pi

- `~/act-athon`: organisers' repo (`rig/` Python API, `example.py`, `ui.py`, plus our `dashboard.py`, `camview.py`, ...).
- Control gate `127.0.0.1:8096` (service `actor-gate`, runs as `actor`, not readable): the only way to move the arm. `GET /state`, `POST /cmd`, `POST /estop`.
- `rig-ui` test page: `http://<pi-host>:8097` (always running).
- Dashboard / red KILL: port 8095 (not reachable from the organisers' user).
- CAN `reader_a/b` (listen-only), `writer_a/b` (down, not ours).

## Safety

Someone holds the physical kill switch for every run, nobody within 3 m of the arm. `rig.stop()`, STOP/ESC in the teleop pages, and the kill switch all open the relay. Every command is held 0.5 s then neutral, so a crashed script stops the arm within half a second.

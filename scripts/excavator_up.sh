#!/usr/bin/env bash
# One laptop command for a dashboard session: world camera + tunnel on this laptop, dashboard on the Pi.
#
#   scripts/excavator_up.sh [dashboard.py flags...]       e.g. scripts/excavator_up.sh --rec-fps 15
#   scripts/excavator_up.sh --no-world [flags...]         without the laptop world camera
#
# 1. worldcam.py --oak --fps 15 on this laptop (port 8100; log logs/worldcam.log), kept awake with caffeinate
# 2. ssh -R 18100:127.0.0.1:8100 excavator: the Pi's camview reads it as stream "world" (remote_cams.json on the Pi)
# 3. on the Pi, in the foreground: ./start_dashboard.sh (camctl + dashboard.py, which runs camview.py itself)
#    Then open http://<pi-host>:8080.
#
# Ctrl-C quits the dashboard (it sends STOP on the way out) and then stops what this script started here.
# A worldcam or tunnel that is already running is reused and left running.
# Env: WORLDCAM_PYTHON (default ~/.venvs/worldcam/bin/python: needs depthai + opencv), PI_HOST (default excavator).
# No dashboard flags given -> --rec-fps 15.

PI_HOST="${PI_HOST:-excavator}"
WORLDCAM_PYTHON="${WORLDCAM_PYTHON:-$HOME/.venvs/worldcam/bin/python}"
WORLD_PORT=8100          # worldcam on this laptop
PI_WORLD_PORT=18100      # where the Pi reaches it (remote_cams.json: http://127.0.0.1:18100/stream/world)

cd "$(dirname "$0")/.." || exit 1
mkdir -p logs

world=1
if [ "${1:-}" = "--no-world" ]; then
    world=0
    shift
fi
if [ $# -eq 0 ]; then
    set -- --rec-fps 15
fi

WORLD_PID=""
TUNNEL_PID=""
cleanup() {
    for pid in $TUNNEL_PID $WORLD_PID; do
        kill -TERM "$pid" 2>/dev/null
    done
    TUNNEL_PID=""
    WORLD_PID=""
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# 0 = something holds :8080, 1 = free, 255 = ssh itself failed (Pi unreachable): only 1 may go on
rc=0
ssh -o ConnectTimeout=8 "$PI_HOST" 'ss -ltn | grep -q ":8080 "' || rc=$?
if [ "$rc" = 0 ]; then
    echo "a dashboard is already running on the Pi (port 8080): quit it first (Ctrl-C in its ssh session)" >&2
    exit 1
elif [ "$rc" != 1 ]; then
    echo "can't check the Pi ($PI_HOST): ssh exit $rc -- is it on / on Tailscale?" >&2
    exit 1
fi

if [ "$world" = 1 ]; then
    # world camera
    if curl -s -m 2 "http://127.0.0.1:$WORLD_PORT/api/cameras" >/dev/null; then
        echo "world camera: already answering on :$WORLD_PORT, reusing it"
    else
        if [ ! -x "$WORLDCAM_PYTHON" ]; then
            echo "no python at $WORLDCAM_PYTHON (set WORLDCAM_PYTHON, or run with --no-world)" >&2
            exit 1
        fi
        echo "world camera: starting worldcam.py --oak --fps 15 (log logs/worldcam.log)"
        (cd pi/act-athon/worldcam && exec caffeinate -dimsu "$WORLDCAM_PYTHON" worldcam.py --oak --fps 15 \
            --port "$WORLD_PORT") >> logs/worldcam.log 2>&1 &
        WORLD_PID=$!
        for _ in $(seq 30); do
            curl -s -m 1 "http://127.0.0.1:$WORLD_PORT/api/cameras" >/dev/null && break
            if ! kill -0 "$WORLD_PID" 2>/dev/null; then
                echo "worldcam.py exited: see logs/worldcam.log (or run with --no-world)" >&2
                tail -5 logs/worldcam.log >&2
                exit 1
            fi
            sleep 1
        done
        curl -s -m 1 "http://127.0.0.1:$WORLD_PORT/api/cameras" >/dev/null ||
            { echo "worldcam.py not answering after 30 s: see logs/worldcam.log" >&2; exit 1; }
    fi

    # tunnel: the Pi's 127.0.0.1:18100 -> this laptop's :8100
    if pgrep -f -- "-R $PI_WORLD_PORT:127.0.0.1:$WORLD_PORT $PI_HOST" >/dev/null; then
        echo "tunnel: already open, reusing it"
    else
        echo "tunnel: Pi :$PI_WORLD_PORT -> laptop :$WORLD_PORT"
        ssh -N -o ServerAliveInterval=5 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes \
            -R "$PI_WORLD_PORT:127.0.0.1:$WORLD_PORT" "$PI_HOST" &
        TUNNEL_PID=$!
        sleep 2
        if ! kill -0 "$TUNNEL_PID" 2>/dev/null; then
            echo "tunnel failed (is port $PI_WORLD_PORT already taken on the Pi?)" >&2
            exit 1
        fi
    fi
fi

# dashboard on the Pi, in the foreground (-t: Ctrl-C reaches it, and it sends STOP when it quits)
args=$(printf ' %q' "$@")
echo "Pi: ./start_dashboard.sh$args   ->  http://<pi-host>:8080"
ssh -t "$PI_HOST" "cd act-athon && ./start_dashboard.sh$args"

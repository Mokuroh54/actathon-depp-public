#!/usr/bin/env bash
# Start the dashboard with the USB camera controls locked (docs/camera-controls.md).
#
#   ./start_dashboard.sh [dashboard.py flags...]      e.g. ./start_dashboard.sh --rec-fps 15
#
# 1. camctl.py apply: writes cam_controls.json to every USB camera (a failure only warns: the dashboard still starts)
# 2. camctl.py watch in the background (log: logs/camctl.log): re-applies after a replug / USB reset / reboot
# 3. dashboard.py in the foreground with your flags (Ctrl-C works as before). The watcher is stopped when the
#    dashboard exits, and exits by itself if this script dies.
# --episode-s defaults to EPISODE_S_DEFAULT unless you pass --episode-s yourself.

EPISODE_S_DEFAULT=90

cd "$(dirname "$0")" || exit 1
mkdir -p logs

has_episode_s=0
for a in "$@"; do
    case "$a" in
        --episode-s|--episode-s=*) has_episode_s=1 ;;
    esac
done
if [ "$has_episode_s" = 0 ]; then
    set -- --episode-s "$EPISODE_S_DEFAULT" "$@"
fi

if ! python3 camctl.py apply; then
    echo "" >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "!!  WARNING: camera controls NOT fully applied (see camctl output above).     !!" >&2
    echo "!!  Exposure / white balance may differ from the training data.               !!" >&2
    echo "!!  Starting the dashboard anyway; the watcher keeps retrying (logs/camctl.log)!!" >&2
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" >&2
    echo "" >&2
fi

python3 camctl.py watch --parent $$ >> logs/camctl.log 2>&1 &
WATCH_PID=$!

stop_watcher() {
    if [ -n "$WATCH_PID" ] && kill -0 "$WATCH_PID" 2>/dev/null; then
        kill -TERM "$WATCH_PID" 2>/dev/null
        for _ in 1 2 3 4 5 6 7 8 9 10; do
            kill -0 "$WATCH_PID" 2>/dev/null || break
            sleep 0.2
        done
        kill -KILL "$WATCH_PID" 2>/dev/null
    fi
    WATCH_PID=""
}
# Note: while dashboard.py runs in the foreground, bash runs these traps only after it exits. Ctrl-C and an ssh
# hangup reach the dashboard directly (same process group); a SIGTERM sent to this script's pid alone waits for the
# dashboard to exit (the watcher lives exactly as long as the dashboard either way).
trap stop_watcher EXIT
trap 'stop_watcher; exit 130' INT
trap 'stop_watcher; exit 143' TERM
trap 'stop_watcher; exit 129' HUP

echo "camctl watch running (pid $WATCH_PID, log logs/camctl.log); starting: python3 dashboard.py $*"
python3 dashboard.py "$@"
rc=$?
stop_watcher
exit $rc

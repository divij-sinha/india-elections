#!/usr/bin/env bash
# Downloads several states side by side: N slots, each with its own network namespace + NordVPN tunnel
# (scripts/vpn_ns.sh). A slot takes the next state from the queue, runs it, and when it's done takes the next one
# on the same tunnel. Each state gets its own dir under .runner/<STATE>/ (log, stdout, API cache).
#   nohup scripts/run_states.sh -n 4 PUNJAB BIHAR KERALA GOA ASSAM > .runner.out 2>&1 &
#   nohup scripts/run_states.sh -n 4 > .runner.out 2>&1 &   ## no states: resume what's left in .runner/queue
# Options: -n slots (4), -p passes per state (2; later passes fill parts that failed), -r retries after a crash (3)
# Env: NO_VPN=1 runs without namespaces, ROLL_CMD replaces ".venv/bin/python main.py" (testing),
#      GEROLL_SEMAPHORE etc. are passed through to every download.
# Progress: .runner/status.log, ntfy pushes on start/finish/failure, `scripts/monitor_download.sh --all` from cron.
# Stop: kill the runner's pid (in .runner/pid); it stops the downloads and takes the tunnels down. Unfinished states
# stay in the queue (the one a slot was on goes back to the front).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

SLOTS=4 PASSES=2 RETRIES=3
while getopts "n:p:r:" opt; do
    case "$opt" in
    n) SLOTS=$OPTARG ;; p) PASSES=$OPTARG ;; r) RETRIES=$OPTARG ;;
    *) sed -n '2,15p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))

RUN=.runner
QUEUE=$RUN/queue
ROLL_CMD="${ROLL_CMD:-.venv/bin/python main.py}"
NTFY_TOPIC="${NTFY_TOPIC:-india-elections-5c37080255a56fb714}"
mkdir -p "$RUN"
touch "$QUEUE"

if [ -e "$RUN/pid" ] && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then
    echo "runner already running (pid $(cat "$RUN/pid"))" >&2; exit 1
fi
echo $$ > "$RUN/pid"

for s in "$@"; do  ## validate against StateCode names before queueing
    grep -qE "^    ${s^^} = \"" electoral_rolls/get.py || { echo "unknown state $s" >&2; exit 1; }
    echo "${s^^}"
done > "$QUEUE.new"
[ $# -gt 0 ] && mv "$QUEUE.new" "$QUEUE" || rm -f "$QUEUE.new"

log() {
    echo "$(date '+%F %T') $*" | tee -a "$RUN/status.log"
    curl -s -m 20 -H "Title: Roll runner ($(hostname))" -H "Priority: ${PRIORITY:-low}" -d "$*" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null
}

## Takes the first state off the queue (locked: slots pop concurrently)
pop() {
    flock "$QUEUE.lock" bash -c 'line=$(grep -m1 . "$1") || exit 1; sed -i "0,/^$line\$/{//d}" "$1"; echo "$line"' _ "$QUEUE"
}
push_front() { flock "$QUEUE.lock" sed -i "1i $1" "$QUEUE"; [ -s "$QUEUE" ] || echo "$1" > "$QUEUE"; }

## One state, PASSES times, retrying crashes. Returns 0 done, 3 not published, 1 gave up
run_state() {
    local state=$1 ns=$2 dir=$RUN/$1 pass tries rc
    mkdir -p "$dir/cache"
    [ -d .cache ] && cp -rn .cache/. "$dir/cache/" 2>/dev/null  ## seed the API cache from earlier single runs
    echo "$ns" > "$dir/running"
    for ((pass = 1; pass <= PASSES; pass++)); do
        for ((tries = 0; ; tries++)); do
            echo "=== pass $pass try $((tries + 1)) $(date '+%F %T')" >> "$dir/out.log"
            # shellcheck disable=SC2086  ## ROLL_CMD is a command line
            ${NETNS_EXEC[$ns]:-} env VOTER_LOG="$dir/voter_scraper.log" CACHE_DIR="$dir/cache" \
                $ROLL_CMD "$state" >> "$dir/out.log" 2>&1 &
            local pid=$!
            echo "$pid" > "$dir/pid"
            wait "$pid"; rc=$?
            rm -f "$dir/pid"
            [ "$rc" = 0 ] && break
            [ "$rc" = 3 ] && { rm -f "$dir/running"; return 3; }
            [ -e "$RUN/stopping" ] && { rm -f "$dir/running"; return 2; }
            if [ "$tries" -ge "$RETRIES" ]; then rm -f "$dir/running"; return 1; fi
            PRIORITY=high log "$state crashed (exit $rc, pass $pass), retry $((tries + 1))/$RETRIES: $(grep -v -i warn "$dir/out.log" | tail -1 | cut -c1-100)"
            [ -z "${NETNS_EXEC[$ns]:-}" ] || scripts/vpn_ns.sh up "$ns" >> "$RUN/status.log" 2>&1  ## fresh tunnel, in case it was the cause
            sleep 30
        done
    done
    rm -f "$dir/running"
    return 0
}

slot() {
    local ns=$1 state rc
    while [ ! -e "$RUN/stopping" ] && state=$(pop); do
        log "$state started on $ns"
        local t0=$SECONDS
        run_state "$state" "$ns"; rc=$?
        local took="$(((SECONDS - t0) / 3600))h$((((SECONDS - t0) % 3600) / 60))m"
        case $rc in
        0) log "$state finished ($took, $(cat "$RUN/$state/voter_scraper.log" 2>/dev/null | grep -c 'Failed to get\|Failed to save') failures logged)"
           echo "$state" >> "$RUN/done" ;;
        3) PRIORITY=high log "$state skipped: roll not published"; echo "$state" >> "$RUN/skipped" ;;
        2) push_front "$state"; return ;;  ## runner stopping
        *) PRIORITY=high log "$state gave up after $RETRIES retries, see $RUN/$state/out.log"; echo "$state" >> "$RUN/failed" ;;
        esac
    done
}

declare -A NETNS_EXEC
rm -f "$RUN/stopping"
stop() {
    touch "$RUN/stopping"  ## a file, not a variable: the slots are subshells and wouldn't see the change
    for d in "$RUN"/*/pid; do [ -e "$d" ] && kill "$(cat "$d")" 2>/dev/null; done  ## the downloads
}
trap stop INT TERM

namespaces=()
for ((i = 1; i <= SLOTS; i++)); do
    ns="roll$i"
    if [ "${NO_VPN:-0}" != 1 ]; then
        scripts/vpn_ns.sh up "$ns" | tee -a "$RUN/status.log" || { PRIORITY=high log "tunnel $ns failed to come up, skipping slot"; continue; }
        NETNS_EXEC[$ns]="ip netns exec $ns"
        namespaces+=("$ns")
    fi
    slot "$ns" &
done
wait  ## returns early on a signal: stop() has run, wait again for the slots to finish up
wait

for ns in "${namespaces[@]}"; do scripts/vpn_ns.sh down "$ns"; done
count() { cat "$1" 2>/dev/null | grep -c .; }
log "runner exiting: done $(count "$RUN/done"), failed $(count "$RUN/failed"), skipped $(count "$RUN/skipped"), still queued $(count "$QUEUE")"
rm -f "$RUN/pid" "$RUN/stopping"

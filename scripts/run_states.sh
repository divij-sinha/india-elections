#!/usr/bin/env bash
# Downloads many states side by side: N slots, each with its own network namespace + NordVPN tunnel on its own server
# (scripts/vpn_ns.sh). A slot takes the next (round, state) from the queue, runs it, and when it's done takes the next.
# Each state is done in ROUNDS rounds: when a state finishes round r, round r+1 of it goes to the back of the queue, so
# every state gets round 1 before any gets round 2. Later rounds only fetch parts that have no pdf/NOT_IN_ROLL yet.
# Each state gets its own dir under .runner/<STATE>/ (log, stdout, API cache).
#   nohup scripts/run_states.sh -n 4 -A > .runner.out 2>&1 &              ## all states, in ALL_STATES order
#   nohup scripts/run_states.sh -n 4 PUNJAB BIHAR > .runner.out 2>&1 &    ## just these, in this order
#   nohup scripts/run_states.sh -n 4 > .runner.out 2>&1 &                 ## no states: resume what's left in .runner/queue
# Options: -n slots (4), -p rounds per state (2), -r retries after a crash (3), -A all states
# Env: NO_VPN=1 runs without namespaces, ROLL_CMD replaces ".venv/bin/python main.py" (testing), NTFY_TOPIC= (empty)
#      turns pushes off, GEROLL_SEMAPHORE etc. are passed through to every download.
# Needs the nordvpn app disconnected: the tunnels would run inside its tunnel, and could land on its server.
# Progress: .runner/status.log, ntfy pushes on start/finish/failure, `scripts/monitor_download.sh --all` from cron.
# Stop: kill the runner's pid (in .runner/pid); it stops the downloads and takes the tunnels down. Unfinished states
# stay in the queue (the one a slot was on goes back to the front, same round).
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

## Order for -A. First the states already downloaded once by single runs (their round 1 here only fills gaps),
## then largest electorate first (2024, approx.) so a big state doesn't end up as one slot's long tail.
## Left out: NCT_OF_DELHI (no 2024 roll published), ASSAM_BYE_ELECTION.
## JHARKHAND and JAMMU_AND_KASHMIR only have assembly-election rolls for 2024, MAHARASHTRA's is labelled Vidhan Sabha.
ALL_STATES=(
    JHARKHAND ODISHA PUNJAB
    UTTAR_PRADESH MAHARASHTRA BIHAR WEST_BENGAL TAMIL_NADU MADHYA_PRADESH KARNATAKA RAJASTHAN GUJARAT ANDHRA_PRADESH
    TELANGANA KERALA ASSAM CHHATTISGARH HARYANA JAMMU_AND_KASHMIR UTTARAKHAND HIMACHAL_PRADESH TRIPURA MEGHALAYA
    MANIPUR NAGALAND GOA PUDUCHERRY ARUNACHAL_PRADESH MIZORAM CHANDIGARH SIKKIM DADRA_NAGAR_HAVELI_DAMAN_DIU
    ANDAMAN_NICOBAR_ISLANDS LADAKH LAKSHADWEEP
)

SLOTS=4 ROUNDS=2 RETRIES=3 ALL=0
while getopts "n:p:r:A" opt; do
    case "$opt" in
    n) SLOTS=$OPTARG ;; p) ROUNDS=$OPTARG ;; r) RETRIES=$OPTARG ;; A) ALL=1 ;;
    *) sed -n '2,17p' "$0"; exit 1 ;;
    esac
done
shift $((OPTIND - 1))
[ "$ALL" = 1 ] && set -- "${ALL_STATES[@]}"

RUN=.runner
QUEUE=$RUN/queue  ## one "<round> <STATE>" per line
ROLL_CMD="${ROLL_CMD:-.venv/bin/python main.py}"
NTFY_TOPIC="${NTFY_TOPIC-india-elections-5c37080255a56fb714}"
VPN=1; [ "${NO_VPN:-0}" = 1 ] && VPN=0
mkdir -p "$RUN"
touch "$QUEUE"

if [ "$VPN" = 1 ] && command -v nordvpn > /dev/null && nordvpn status 2>/dev/null | grep -q "Status: Connected"; then
    echo "the nordvpn app is connected, disconnect it first: nordvpn disconnect" >&2; exit 1
fi
if [ -e "$RUN/pid" ] && kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then
    echo "runner already running (pid $(cat "$RUN/pid"))" >&2; exit 1
fi
if [ $# -gt 0 ]; then  ## validate against StateCode names, then queue round 1 of each
    for s in "$@"; do
        grep -qE "^    ${s^^} = \"" electoral_rolls/get.py || { echo "unknown state $s" >&2; exit 1; }
    done
    for s in "$@"; do echo "1 ${s^^}"; done > "$QUEUE"
fi
echo $$ > "$RUN/pid"

log() {
    echo "$(date '+%F %T') $*" | tee -a "$RUN/status.log"
    [ -z "$NTFY_TOPIC" ] || curl -s -m 20 -H "Title: Roll runner ($(hostname))" -H "Priority: ${PRIORITY:-low}" -d "$*" \
        "https://ntfy.sh/$NTFY_TOPIC" > /dev/null
}

## Queue ops, locked: slots use the queue concurrently
pop() { flock "$QUEUE.lock" bash -c 'head -1 "$1" | grep . && sed -i 1d "$1"' _ "$QUEUE"; }
push_front() { flock "$QUEUE.lock" bash -c '{ echo "$2"; cat "$1"; } > "$1.tmp" && mv "$1.tmp" "$1"' _ "$QUEUE" "$1"; }
push_back() { flock "$QUEUE.lock" bash -c 'echo "$2" >> "$1"' _ "$QUEUE" "$1"; }

## Brings up ns's tunnel: on the slot's current server first (reconnecting there doesn't use up another of the
## account's simultaneous connections, NordVPN allows ~10 and seems to hold dropped sessions for a while), else on up to
## 3 of the least-loaded servers no slot is on. .runner/slots/<ns> holds the slot's server, <ns>.failed the servers that
## didn't work for it; both count as taken for every slot.
tunnel_up() {
    local ns=$1 server _
    server=$(cat "$RUN/slots/$ns" 2>/dev/null)
    if [ -n "$server" ]; then
        scripts/vpn_ns.sh up "$ns" "$server" && return 0
        echo "$server" >> "$RUN/slots/$ns.failed"
    fi
    for _ in 1 2 3; do
        server=$(flock "$RUN/slots.lock" bash -c '
            used=$(cat "$2"/* 2>/dev/null)
            for s in $(scripts/vpn_ns.sh servers); do
                grep -qxF "$s" <<< "$used" || { echo "$s" > "$2/$1"; echo "$s"; exit 0; }
            done
            exit 1' _ "$ns" "$RUN/slots") || { echo "no free vpn server for $ns" >&2; return 1; }
        scripts/vpn_ns.sh up "$ns" "$server" && return 0
        echo "$server" >> "$RUN/slots/$ns.failed"
    done
    return 1
}

## One round of one state, retrying crashes on a fresh tunnel. Returns 0 done, 3 not published, 2 runner stopping,
## 1 gave up
run_state() {
    local round=$1 state=$2 ns=$3 dir=$RUN/$2 tries rc pid
    mkdir -p "$dir/cache"
    [ -d .cache ] && cp -rn .cache/. "$dir/cache/" 2>/dev/null  ## seed the API cache from earlier single runs
    echo "$ns" > "$dir/running"
    for ((tries = 0; ; tries++)); do
        [ -e "$RUN/stopping" ] && { rc=2; break; }  ## also checked here: a stop can come during the pause before a retry
        echo "=== round $round try $((tries + 1)) $(date '+%F %T')" >> "$dir/out.log"
        # shellcheck disable=SC2086  ## NETNS_EXEC and ROLL_CMD are command lines
        $NETNS_EXEC env VOTER_LOG="$dir/voter_scraper.log" CACHE_DIR="$dir/cache" \
            $ROLL_CMD "$state" >> "$dir/out.log" 2>&1 &
        pid=$!
        echo "$pid" > "$dir/pid"
        wait "$pid"; rc=$?
        rm -f "$dir/pid"
        [ "$rc" = 0 ] || [ "$rc" = 3 ] && break
        [ -e "$RUN/stopping" ] && { rc=2; break; }
        [ "$tries" -ge "$RETRIES" ] && { rc=1; break; }
        PRIORITY=high log "$state crashed (exit $rc, round $round), retry $((tries + 1))/$RETRIES: $(grep -v -i warn "$dir/out.log" | tail -1 | cut -c1-100)"
        for _ in {1..30}; do [ -e "$RUN/stopping" ] && break; sleep 1; done  ## pause, but don't hold up a stop
        [ -e "$RUN/stopping" ] && { rc=2; break; }
        [ "$VPN" = 0 ] || tunnel_up "$ns" >> "$RUN/status.log" 2>&1  ## fresh tunnel, in case it was the cause
    done
    rm -f "$dir/running"
    return "$rc"
}

slot() {
    local ns=$1 item round state rc t0 took lines0
    NETNS_EXEC=""; [ "$VPN" = 0 ] || NETNS_EXEC="ip netns exec $ns"  ## this slot's subshell only
    while [ ! -e "$RUN/stopping" ] && item=$(pop); do
        read -r round state <<< "$item"
        log "$state round $round started on $ns"
        t0=$SECONDS
        lines0=$(cat "$RUN/$state/voter_scraper.log" 2>/dev/null | wc -l)
        run_state "$round" "$state" "$ns"; rc=$?
        took="$(((SECONDS - t0) / 3600))h$((((SECONDS - t0) % 3600) / 60))m"
        case $rc in
        0) log "$state round $round finished ($took, $(tail -n +$((lines0 + 1)) "$RUN/$state/voter_scraper.log" 2>/dev/null | grep -c 'Failed to get\|Failed to save') failures logged)"
           echo "$state $round" >> "$RUN/done" ;;
        3) PRIORITY=high log "$state skipped: roll not published"; echo "$state" >> "$RUN/skipped"; continue ;;
        2) push_front "$item"; return ;;  ## runner stopping
        *) PRIORITY=high log "$state round $round gave up after $RETRIES retries, see $RUN/$state/out.log"
           echo "$state $round" >> "$RUN/failed" ;;
        esac
        [ "$round" -lt "$ROUNDS" ] && push_back "$((round + 1)) $state"  ## behind every state still waiting for this round
    done
}

rm -f "$RUN/stopping"
## Each slot keeps its server across runner restarts: reconnecting to it doesn't take another of the account's
## connections while the old sessions are still held. Servers that failed get another chance.
mkdir -p "$RUN/slots"; rm -f "$RUN"/slots/*.failed
stop() {
    touch "$RUN/stopping"  ## a file, not a variable: the slots are subshells and wouldn't see the change
    for d in "$RUN"/*/pid; do [ -e "$d" ] && kill "$(cat "$d")" 2>/dev/null; done  ## the downloads
}
trap stop INT TERM

slot_pids=()
for ((i = 1; i <= SLOTS; i++)); do
    [ -e "$RUN/stopping" ] && break
    ns="roll$i"
    if [ "$VPN" = 1 ]; then
        tunnel_up "$ns" 2>&1 | tee -a "$RUN/status.log" || { PRIORITY=high log "tunnel $ns failed to come up, skipping slot"; continue; }
    fi
    slot "$ns" &
    slot_pids+=($!)
done
## wait returns early on every signal (stop() has run): keep waiting until every slot has really finished, so the
## tunnels aren't taken down and the stop flag isn't cleared under a slot that is still running
for pid in "${slot_pids[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do wait "$pid"; done
done

## Every slot's namespace, also one whose tunnel failed at startup or on a restart
[ "$VPN" = 0 ] || for ((i = 1; i <= SLOTS; i++)); do scripts/vpn_ns.sh down "roll$i"; done
count() { cat "$1" 2>/dev/null | grep -c .; }
log "runner exiting: rounds done $(count "$RUN/done"), failed $(count "$RUN/failed"), skipped $(count "$RUN/skipped"), still queued $(count "$QUEUE")"
rm -f "$RUN/pid" "$RUN/stopping"

#!/usr/bin/env bash
# Cron health check for a running electoral-roll download (main.py). Sends an ntfy.sh push every run:
# "ok" or the list of problems, plus the success rate since the previous push.
#   */30 * * * * /root/projects/india-elections/scripts/monitor_download.sh
# With scripts/run_states.sh, cron runs `monitor_download.sh --all` instead: one check per state currently running,
# each from that state's .runner/<STATE>/ dir.
cd "$(dirname "$0")/.." || exit 1

if [ "${1:-}" = "--all" ]; then
    for f in .runner/*/running; do [ -e "$f" ] && "$0" "$(dirname "$f")"; done
    exit 0
fi
RUN_DIR="${1:-}"  ## a .runner/<STATE> dir, or empty for a single `nohup uv run main.py` run

NTFY_TOPIC="${NTFY_TOPIC-india-elections-5c37080255a56fb714}"  ## empty: no pushes (testing)
if [ -n "$RUN_DIR" ]; then
    LOG=$RUN_DIR/voter_scraper.log NOHUP=$RUN_DIR/out.log STATE=$RUN_DIR/monitor.state
    PROC_PAT="main.py $(basename "$RUN_DIR")\$" TITLE="$(basename "$RUN_DIR") on $(hostname)"
else
    LOG=voter_scraper.log NOHUP=nohup.out STATE=.tmp/monitor_download.state
    PROC_PAT="python main.py" TITLE=$(hostname)
fi
[ -f "$LOG" ] || exit 0  ## nothing logged yet
## Roll and state of the current run, from its latest "Using roll S27-2024-GEN-FIR for StateCode.JHARKHAND" line
roll_line=$(grep "Using roll " "$LOG" | tail -1)
state_name=$(awk '{print $NF}' <<< "$roll_line")
MIN_FREE_GB=20
MAX_FAIL_PCT=20
MAX_MEM_PCT=80

mkdir -p .tmp
now=$(date +%s)
running=$(pgrep -f "$PROC_PAT" >/dev/null && echo 1 || echo 0)
## From the log, not `find` on the storage box: listing tens of thousands of part dirs over CIFS takes minutes per state
pdfs=$(grep -c " - Saved " "$LOG")  ## saved by the runs in this log
marked=$(grep -c " - Not in roll " "$LOG")
log_lines=$(wc -l < "$LOG")
free_gb=$(df -BG --output=avail /mnt/storage-box-1 | tail -1 | tr -dc '0-9')
mem_pct=$(free | awk '/^Mem:/ {printf "%d", ($2 - $7) * 100 / $2}')  ## system-wide, (total - available) / total
proc_mem_mb=$(ps -o rss= -p "$(pgrep -f "$PROC_PAT" | head -1)" 2>/dev/null | awk '{printf "%d", $1 / 1024}')

## First run, or a state file from before counts came from the log (prev_saved missing): record a baseline only
if [ ! -f "$STATE" ] || ! grep -q '^prev_saved=' "$STATE"; then
    printf 'prev_saved=%q\nprev_marked=%q\nprev_lines=%q\nprev_alert=%q\n' "$pdfs" "$marked" "$log_lines" "" > "$STATE"
    exit 0
fi
. "$STATE"

new_log=$(tail -n +$((prev_lines + 1)) "$LOG")
new_pdfs=$((pdfs - prev_saved))
new_marked=$((marked - prev_marked))
failed=$(grep -c "Failed to get: \|Failed to save " <<< "$new_log")
high_fail=$(grep -c "High failure rate" <<< "$new_log")
position=$(grep "Running ac " "$LOG" | tail -1 | sed 's/.*Running ac //')

problems=()
if [ "$running" = 0 ]; then
    if grep -q "Finished running full state for $state_name" "$NOHUP" 2>/dev/null; then
        problems+=("download finished")
    else
        problems+=("download process not running: $(grep -v -i warn "$NOHUP" 2>/dev/null | tail -1 | cut -c1-80)")
    fi
elif [ $((new_pdfs + new_marked)) -le 0 ]; then
    problems+=("stalled, no new pdfs or NOT_IN_ROLL markers since last check")
fi
attempted=$((new_pdfs + failed))
if [ "$attempted" -gt 0 ] && [ $((failed * 100 / attempted)) -gt "$MAX_FAIL_PCT" ]; then
    problems+=("$failed/$attempted parts failed")
fi
[ "$high_fail" -gt 0 ] && problems+=("$high_fail ACs with high failure rate")
[ "${free_gb:-0}" -lt "$MIN_FREE_GB" ] && problems+=("storage low: ${free_gb}G free")
[ "$mem_pct" -gt "$MAX_MEM_PCT" ] && problems+=("memory high: ${mem_pct}% used, download ${proc_mem_mb:-?}MB")

alert=$(IFS='; '; echo "${problems[*]}")
success_rate=$([ "$attempted" -gt 0 ] && echo "$((new_pdfs * 100 / attempted))%" || echo "n/a")
summary="success $success_rate ($new_pdfs ok, $failed failed, $new_marked not in roll) | saved this run $pdfs | at ac $position | mem ${mem_pct}%"
message="${alert:-ok} | $summary"
echo "$(date '+%F %T') ${RUN_DIR:+$(basename "$RUN_DIR") }$message" >> .tmp/monitor_download.log

## Problems get high priority so they stand out from the routine "ok" pushes
priority=$([ -n "$alert" ] && echo high || echo low)
[ -z "$NTFY_TOPIC" ] || curl -s -m 20 -H "Title: Electoral roll download ($TITLE)" -H "Priority: $priority" -d "$message" "https://ntfy.sh/$NTFY_TOPIC" > /dev/null

printf 'prev_saved=%q\nprev_marked=%q\nprev_lines=%q\nprev_alert=%q\n' "$pdfs" "$marked" "$log_lines" "$alert" > "$STATE"

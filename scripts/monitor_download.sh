#!/usr/bin/env bash
# Cron health check for a running electoral-roll download (main.py). Sends an ntfy.sh push every run:
# "ok" or the list of problems, plus the success rate since the previous push.
#   */30 * * * * /root/projects/india-elections/scripts/monitor_download.sh
cd "$(dirname "$0")/.." || exit 1

NTFY_TOPIC="${NTFY_TOPIC:-india-elections-5c37080255a56fb714}"
ROLL_DIR="${ROLL_DIR:-data/pdf/voter_rolls/S18-2024-GEN-FIR}"
LOG=voter_scraper.log
STATE=.tmp/monitor_download.state
MIN_FREE_GB=20
MAX_FAIL_PCT=20

mkdir -p .tmp
now=$(date +%s)
running=$(pgrep -f "python main.py" >/dev/null && echo 1 || echo 0)
pdfs=$(find "$ROLL_DIR" -name '*.pdf' 2>/dev/null | wc -l)
log_lines=$(wc -l < "$LOG")
free_gb=$(df -BG --output=avail /mnt/storage-box-1 | tail -1 | tr -dc '0-9')

if [ ! -f "$STATE" ]; then  ## first run: record a baseline only
    printf 'prev_pdfs=%q\nprev_lines=%q\nprev_alert=%q\n' "$pdfs" "$log_lines" "" > "$STATE"
    exit 0
fi
. "$STATE"

new_log=$(tail -n +$((prev_lines + 1)) "$LOG")
new_pdfs=$((pdfs - prev_pdfs))
failed=$(grep -c "Failed to get: \|Failed to save " <<< "$new_log")
high_fail=$(grep -c "High failure rate" <<< "$new_log")
position=$(grep "Running ac " "$LOG" | tail -1 | sed 's/.*Running ac //')

problems=()
if [ "$running" = 0 ]; then
    if grep -q "Finished running full state" nohup.out 2>/dev/null; then
        problems+=("download finished")
    else
        problems+=("download process not running: $(grep -v -i warn nohup.out 2>/dev/null | tail -1 | cut -c1-80)")
    fi
elif [ "$new_pdfs" -le 0 ]; then
    problems+=("stalled, no new pdfs since last check")
fi
attempted=$((new_pdfs + failed))
if [ "$attempted" -gt 0 ] && [ $((failed * 100 / attempted)) -gt "$MAX_FAIL_PCT" ]; then
    problems+=("$failed/$attempted parts failed")
fi
[ "$high_fail" -gt 0 ] && problems+=("$high_fail ACs with high failure rate")
[ "${free_gb:-0}" -lt "$MIN_FREE_GB" ] && problems+=("storage low: ${free_gb}G free")

alert=$(IFS='; '; echo "${problems[*]}")
success_rate=$([ "$attempted" -gt 0 ] && echo "$((new_pdfs * 100 / attempted))%" || echo "n/a")
summary="success $success_rate ($new_pdfs ok, $failed failed) | total pdfs $pdfs | at ac $position"
message="${alert:-ok} | $summary"
echo "$(date '+%F %T') $message" >> .tmp/monitor_download.log

## Problems get high priority so they stand out from the routine "ok" pushes
priority=$([ -n "$alert" ] && echo high || echo low)
curl -s -m 20 -H "Title: Electoral roll download" -H "Priority: $priority" -d "$message" "https://ntfy.sh/$NTFY_TOPIC" > /dev/null

printf 'prev_pdfs=%q\nprev_lines=%q\nprev_alert=%q\n' "$pdfs" "$log_lines" "$alert" > "$STATE"

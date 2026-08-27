#!/usr/bin/env bash
#
# Run one search per scenario configuration, in parallel and resumably.
#
# Each job is `python -m src.main optimize` on one file in $CONFIG_DIR.  Jobs are
# independent and the simulator is single-threaded, but measured throughput plateaus at
# about four workers, which is why JOBS defaults to 4.  Raising it past that stretches
# each individual search instead of finishing the campaign sooner, and a search writes
# nothing until it completes.
#
# Prerequisite:  .venv/bin/python scripts/generate_search_configs.py
#
# Usage:
#     ./run_experiments.sh                  # everything, resuming any partial run
#     DRY_RUN=1 ./run_experiments.sh        # list the jobs and exit
#     LIMIT=2 ./run_experiments.sh          # smoke test the first two jobs
#     JOBS=14 ./run_experiments.sh          # pin the worker count
#     FORCE=1 ./run_experiments.sh          # re-run jobs that already have results
#     FILTER='*_a-p_*' ./run_experiments.sh # only the configs matching this glob
#     CONFIG_DIR=config/search_baseline MODE=human_only ./run_experiments.sh
#                                           # enumerate the zero-agent baseline instead
#     nohup ./run_experiments.sh > campaign.log 2>&1 &     # long unattended run
#
# Further environment overrides: PYTHON, CONFIG_DIR, OUT_ROOT, SAMPLES, START_STATE,
# PROGRESS_EVERY.
#
# Re-running is safe: a job whose search_summary.json already exists is skipped unless
# FORCE=1.  Per-job stdout goes to $OUT_ROOT/<scenario_id>/<mode>/log.txt and one
# TSV record per finished job is appended to $OUT_ROOT/index.tsv.

set -euo pipefail

# The header comment block: everything from line 3 up to the first non-comment line.
usage() { sed -n '3,/^[^#]/p' "$0" | sed -n 's/^# \{0,1\}//p'; exit 0; }
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

# ---------------------------------------------------------------------------
# Configuration (all overridable from the environment)
# ---------------------------------------------------------------------------
cd "$(dirname "$0")"

PYTHON=${PYTHON:-.venv/bin/python}
CONFIG_DIR=${CONFIG_DIR:-config/search}
OUT_ROOT=${OUT_ROOT:-results/experiments}
# MODE picks between running the search and enumerating the zero-agent baseline, and
# names the result subdirectory either way.
MODE=${MODE:-search}
SAMPLES=${SAMPLES:-best}
START_STATE=${START_STATE:-config/startstate_default.json}
PROGRESS_EVERY=${PROGRESS_EVERY:-500}
JOBS=${JOBS:-4}
FILTER=${FILTER:-'*'}
LIMIT=${LIMIT:-0}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

case "$MODE" in
    search|human_only) ;;
    *) echo "error: MODE must be 'search' or 'human_only', got '$MODE'" >&2; exit 1 ;;
esac

INDEX="$OUT_ROOT/index.tsv"

[[ -x "$PYTHON" ]] || command -v "$PYTHON" >/dev/null \
    || { echo "error: python not found at $PYTHON" >&2; exit 1; }
[[ -d "$CONFIG_DIR" ]] || {
    # The baseline set comes from a separate invocation of the generator, so name the
    # flag that writes whichever directory is missing.
    gen="$PYTHON scripts/generate_search_configs.py"
    [[ "$CONFIG_DIR" == *baseline* ]] && gen="$gen --baseline"
    echo "error: $CONFIG_DIR not found — run '$gen' first" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Job list: one line per search config
# ---------------------------------------------------------------------------
job_list() {
    local cfg id
    for cfg in "$CONFIG_DIR"/search_$FILTER.json; do
        [[ -e "$cfg" ]] || continue
        id=$(basename "$cfg" .json); id=${id#search_}
        printf '%s\t%s\n' "$cfg" "$id"
    done
}

JOBS_TSV=$(job_list)
[[ -n "$JOBS_TSV" ]] || { echo "error: no configs matched $CONFIG_DIR/search_$FILTER.json" >&2; exit 1; }
if (( LIMIT > 0 )); then
    JOBS_TSV=$(printf '%s\n' "$JOBS_TSV" | head -n "$LIMIT")
fi
N_JOBS=$(printf '%s\n' "$JOBS_TSV" | wc -l)

echo "Campaign: $N_JOBS job(s), $JOBS parallel worker(s)"
echo "  configs   $CONFIG_DIR/search_$FILTER.json"
echo "  mode      $MODE"
echo "  output    $OUT_ROOT"
echo "  samples   $SAMPLES"
echo "  resume    $([[ $FORCE == 1 ]] && echo 'no (FORCE=1, re-running everything)' || echo yes)"
echo

if (( DRY_RUN == 1 )); then
    printf '%s\n' "$JOBS_TSV"
    echo
    echo "(DRY_RUN=1: nothing executed)"
    exit 0
fi

# ---------------------------------------------------------------------------
# One job
# ---------------------------------------------------------------------------
run_job() {
    local cfg=$1 id=$2
    local out="$OUT_ROOT/$id/$MODE"
    local started=$SECONDS

    if [[ -s "$out/search_summary.json" && "$FORCE" != 1 ]]; then
        printf 'SKIP  %-22s %-13s (results present)\n' "$id" "$MODE"
        printf '%s\t%s\t%s\t%s\n' "$id" "$MODE" "skip" "0" >> "$INDEX"
        return 0
    fi

    mkdir -p "$out"
    local status=0
    # The search config's `human_only` key selects the enumerator; $MODE only names the
    # output directory.  `optimize` refuses to run when the two disagree, so a mismatched
    # MODE/CONFIG_DIR pair fails loudly rather than filing results under the wrong mode.
    "$PYTHON" -m src.main optimize "$cfg" \
        --samples "$SAMPLES" \
        --progress-every "$PROGRESS_EVERY" \
        --start-state "$START_STATE" \
        --out "$out" > "$out/log.txt" 2>&1 || status=$?

    local elapsed=$(( SECONDS - started ))
    if (( status == 0 )); then
        printf 'OK    %-22s %-13s %5ds\n' "$id" "$MODE" "$elapsed"
        printf '%s\t%s\t%s\t%s\n' "$id" "$MODE" "0" "$elapsed" >> "$INDEX"
    else
        printf 'FAIL  %-22s %-13s %5ds  exit=%d  see %s\n' \
            "$id" "$MODE" "$elapsed" "$status" "$out/log.txt" >&2
        printf '%s\t%s\t%s\t%s\n' "$id" "$MODE" "$status" "$elapsed" >> "$INDEX"
    fi
    return 0    # a single failed search must not abort the campaign
}
export -f run_job
export PYTHON OUT_ROOT SAMPLES START_STATE PROGRESS_EVERY FORCE INDEX MODE

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ROOT"
[[ -f "$INDEX" ]] || printf 'scenario_id\tmode\texit_code\tseconds\n' > "$INDEX"

campaign_start=$SECONDS
# NUL-delimit the two fields of every job line, so xargs hands each worker exactly
# one (config, scenario_id) pair regardless of the characters in them.
printf '%s\n' "$JOBS_TSV" | tr '\t\n' '\0\0' \
    | xargs -0 -P "$JOBS" -n 2 bash -c 'run_job "$@"' _
campaign_elapsed=$(( SECONDS - campaign_start ))

# ---------------------------------------------------------------------------
# Summary (over the records this invocation appended)
# ---------------------------------------------------------------------------
n_ok=$(tail -n "$N_JOBS" "$INDEX" | awk -F'\t' '$3 == "0"' | wc -l)
n_skip=$(tail -n "$N_JOBS" "$INDEX" | awk -F'\t' '$3 == "skip"' | wc -l)
n_fail=$(( N_JOBS - n_ok - n_skip ))

echo
printf 'Campaign finished in %dh %dm %ds\n' \
    $(( campaign_elapsed / 3600 )) $(( campaign_elapsed % 3600 / 60 )) $(( campaign_elapsed % 60 ))
echo "  completed  $n_ok"
echo "  skipped    $n_skip"
echo "  failed     $n_fail"
echo "  index      $INDEX"
if (( n_fail > 0 )); then
    echo
    echo "Failed jobs (re-run this script to retry them):" >&2
    tail -n "$N_JOBS" "$INDEX" | awk -F'\t' '$3 != "0" && $3 != "skip" {print "  " $1 " " $2 " exit=" $3}' >&2
    exit 1
fi
echo
echo "Next: $PYTHON -m src.main campaign $OUT_ROOT"

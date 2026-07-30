#!/usr/bin/env bash
#
# Run the full experiment campaign of paper Section 4.2.
#
# 243 scenarios (5 orthogonal parameter groups x 3 tiers) are each solved twice, once
# with uniform random and once with evolutionary search, giving 486 searches of 10,000
# candidate evaluations each.  Jobs are independent, so they run in parallel and the
# simulator itself is single-threaded — but throughput does NOT scale to the core count.
# Measured on an i7-1260P (4 P-cores + 8 E-cores, 28 W), one search of 10,000 candidates:
#
#     JOBS   throughput   speedup   wall-clock per search
#        1    5.7 cand/s     1.0x      0.49 h
#        2   10.2 cand/s     1.8x      0.54 h
#        4   13.3 cand/s     2.4x      0.83 h
#        6   14.5 cand/s     2.6x      1.12 h
#        8   13.3 cand/s     2.3x      1.66 h
#       14   13.7 cand/s     2.5x      2.81 h
#
# Throughput is flat from 4 workers on, so raising $JOBS beyond that does not finish the
# campaign any sooner — it only stretches each individual search (and since a search
# writes nothing until it completes, that is lost work if the run is interrupted).
# Hence the default of 4.  Expect roughly 250 core-hours in total.
#
# Prerequisite:  .venv/bin/python scripts/generate_search_configs.py
#
# Usage:
#     ./run_experiments.sh                  # full campaign, resuming any partial run
#     DRY_RUN=1 ./run_experiments.sh        # list the jobs and exit
#     LIMIT=2 ./run_experiments.sh          # smoke test the first two jobs
#     JOBS=14 ./run_experiments.sh          # pin the worker count
#     FORCE=1 ./run_experiments.sh          # re-run jobs that already have results
#     nohup ./run_experiments.sh > campaign.log 2>&1 &     # long unattended run
#
# Re-running is safe: a job whose search_summary.json already exists is skipped unless
# FORCE=1.  Per-job stdout goes to $OUT_ROOT/<scenario_id>/<strategy>/log.txt and one
# TSV record per finished job is appended to $OUT_ROOT/index.tsv.

set -euo pipefail

usage() { sed -n '3,36p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
[[ "${1:-}" == "-h" || "${1:-}" == "--help" ]] && usage

# ---------------------------------------------------------------------------
# Configuration (all overridable from the environment)
# ---------------------------------------------------------------------------
cd "$(dirname "$0")"

PYTHON=${PYTHON:-.venv/bin/python}
CONFIG_DIR=${CONFIG_DIR:-config/search}
OUT_ROOT=${OUT_ROOT:-results/experiments}
STRATEGIES=${STRATEGIES:-"random evolutionary"}
SAMPLES=${SAMPLES:-frontier}
START_STATE=${START_STATE:-config/startstate_default.json}
PROGRESS_EVERY=${PROGRESS_EVERY:-500}
JOBS=${JOBS:-4}
FILTER=${FILTER:-'*'}
LIMIT=${LIMIT:-0}
FORCE=${FORCE:-0}
DRY_RUN=${DRY_RUN:-0}

INDEX="$OUT_ROOT/index.tsv"

[[ -x "$PYTHON" || $(command -v "$PYTHON") ]] || { echo "error: python not found at $PYTHON" >&2; exit 1; }
[[ -d "$CONFIG_DIR" ]] || {
    echo "error: $CONFIG_DIR not found — run '$PYTHON scripts/generate_search_configs.py' first" >&2
    exit 1
}

# ---------------------------------------------------------------------------
# Job list: one line per (search config, strategy) pair
# ---------------------------------------------------------------------------
job_list() {
    local cfg id strategy
    for cfg in "$CONFIG_DIR"/search_$FILTER.json; do
        [[ -e "$cfg" ]] || continue
        id=$(basename "$cfg" .json); id=${id#search_}
        for strategy in $STRATEGIES; do
            printf '%s\t%s\t%s\n' "$cfg" "$id" "$strategy"
        done
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
    local cfg=$1 id=$2 strategy=$3
    local out="$OUT_ROOT/$id/$strategy"
    local started=$SECONDS

    if [[ -s "$out/search_summary.json" && "$FORCE" != 1 ]]; then
        printf 'SKIP  %-22s %-13s (results present)\n' "$id" "$strategy"
        printf '%s\t%s\t%s\t%s\n' "$id" "$strategy" "skip" "0" >> "$INDEX"
        return 0
    fi

    mkdir -p "$out"
    local status=0
    "$PYTHON" -m src.main optimize "$cfg" \
        --strategy "$strategy" \
        --samples "$SAMPLES" \
        --progress-every "$PROGRESS_EVERY" \
        --start-state "$START_STATE" \
        --out "$out" > "$out/log.txt" 2>&1 || status=$?

    local elapsed=$(( SECONDS - started ))
    if (( status == 0 )); then
        printf 'OK    %-22s %-13s %5ds\n' "$id" "$strategy" "$elapsed"
        printf '%s\t%s\t%s\t%s\n' "$id" "$strategy" "0" "$elapsed" >> "$INDEX"
    else
        printf 'FAIL  %-22s %-13s %5ds  exit=%d  see %s\n' \
            "$id" "$strategy" "$elapsed" "$status" "$out/log.txt" >&2
        printf '%s\t%s\t%s\t%s\n' "$id" "$strategy" "$status" "$elapsed" >> "$INDEX"
    fi
    return 0    # a single failed search must not abort the campaign
}
export -f run_job
export PYTHON OUT_ROOT SAMPLES START_STATE PROGRESS_EVERY FORCE INDEX

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
mkdir -p "$OUT_ROOT"
[[ -f "$INDEX" ]] || printf 'scenario_id\tstrategy\texit_code\tseconds\n' > "$INDEX"

campaign_start=$SECONDS
# NUL-delimit the three fields of every job line, so xargs hands each worker exactly
# one (config, scenario_id, strategy) triple regardless of the characters in them.
printf '%s\n' "$JOBS_TSV" | tr '\t\n' '\0\0' \
    | xargs -0 -P "$JOBS" -n 3 bash -c 'run_job "$@"' _
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
echo "Next: $PYTHON scripts/collect_results.py --results-dir $OUT_ROOT"

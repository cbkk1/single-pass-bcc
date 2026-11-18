#!/bin/bash
# Targeted re-measure of the 24 cells whose timeout verdict did not survive
# scrutiny (2026-11-17).
#
#   setsid nohup bash experiments/run-regate.sh >> experiments/results/regate.out 2>&1 < /dev/null &
#   check:  ps -o pid,cmd -C bash | grep run-regate | grep -v grep
#
# Two faults were found in the first sweep's timeout handling:
#   * the 4x bound from the 1/b law was too tight for a shared GPU -- it cut
#     off two honest kmer_U1a runs at 1.4x and produced six 1.5-4x "hangs";
#   * the early-skip breaks after the FIRST timeout, so 21 of 31 "total hangs"
#     rested on a single rep and could not be told from one flaky run.
# cell-plan.py now bounds at 8x and only accepts a hang as grounds for
# skipping when it overran by >=8x or timed out on >=2 separate reps. The
# suspect rows were removed from the csv (backup alongside it), so these cells
# are simply missing and get re-derived under the corrected rules.
#
# Cheapest batch sizes first: a wrong verdict at 125k/64k/32k re-gates whole
# rows below it, so settling those first can remove 8k work entirely.
set -u
cd "$(dirname "$0")/.." || exit 1
export REPS=3
CSV=experiments/results/bssweep-twopass.csv

run() {  # bs  graphs...
  local bs=$1; shift
  echo "###### regate bs=$bs  $(date '+%F %T')"
  BS_LIST=$bs TMO=7200 CSV=$CSV GRAPHS="$*" \
    bash experiments/run-twopass.sh medium
}

run 125000 higgs-twitter soc-LiveJournal1
run  64000 uk-2002
run  32000 uk-2002
run  16000 coPapersCiteseer eu-2005 europe_osm GAP-road germany_osm \
           hollywood-2009 road_usa sx-stackoverflow
run   8000 as-Skitter coPapersCiteseer delaunay_n24 eu-2005 europe_osm \
           GAP-road germany_osm hollywood-2009 kron_g500-logn21 road_usa \
           soc-LiveJournal1 sx-stackoverflow
echo "###### regate done $(date '+%F %T')"

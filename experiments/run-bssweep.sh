#!/bin/bash
# Batch-size curve: 125k -> 8M, all three arms, on six graphs chosen to span
# the whole range of outcomes in the fixed-size experiment -- including
# eu-2005 (parity) and higgs-twitter (the one loss), so the figure shows where
# the advantage ends rather than only where it is largest.
#
#   setsid nohup bash experiments/run-bssweep.sh >> experiments/results/bssweep.out 2>&1 < /dev/null &
#
# 8M is the ceiling: V_fb = 2*numVert + 2*batchSize OOMs from ~20M up.
# Arms run one after another, never together -- the GPU is shared.
set -u
cd "$(dirname "$0")/.." || exit 1

export GRAPHS="uk-2002 europe_osm rgg_n_2_24_s0 in-2004 eu-2005 higgs-twitter"
export BS_LIST="125000 250000 500000 1000000 2000000 4000000 8000000"
export REPS=3

echo "###### ours (default + filter) $(date '+%F %T')"
CSV=experiments/results/bssweep.csv         bash experiments/run-full.sh    medium
echo "###### two-pass baseline $(date '+%F %T')"
CSV=experiments/results/bssweep-twopass.csv bash experiments/run-twopass.sh medium
echo "###### all done $(date '+%F %T')"

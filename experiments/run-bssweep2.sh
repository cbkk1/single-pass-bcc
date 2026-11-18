#!/bin/bash
# Extended batch-size sweep: ALL 27 medium graphs, 8k -> 8M.
#
#   setsid nohup bash experiments/run-bssweep2.sh >> experiments/results/bssweep2.out 2>&1 < /dev/null &
#   check with:  pgrep -a -x bash | grep run-bssweep2
#
# Ordered CHEAP SIZES FIRST, one batch size at a time across all graphs, so
# the speedup table gains whole columns early instead of whole rows. Cost per
# rep over 27 graphs scales as 1/b, measured: 39 s at 500k, 59x that at 8k.
#   >=1M  : ~40 s/rep      125k-500k: ~270 s/rep
#   64k   : ~305 s/rep     32k: ~610    16k: ~1220    8k: ~2440
# Both modes x 3 reps = 6 passes, so ~8 h in total for our arm.
#
# Writes to the SAME csv as run-bssweep.sh; every finished
# (graph,bs,mode,rep) row is skipped, so the 252 rows already collected are
# reused and only the new cells run.
#
# The two-pass arm is deliberately limited to b >= 125k here: below that it
# hangs (rgg_n_2_24_s0 at 125k burned 1800 s on two of three reps) and a
# legitimate 8k run would itself need ~1200 s, so timeouts stop being
# distinguishable from slowness. Decide that arm separately.
set -u
cd "$(dirname "$0")/.." || exit 1

export REPS=3

# Timeout bounded per batch size. Compute scales as 1/b, so a flat value is
# either too tight at small b (mislabels a slow run as a hang) or wastes
# hours at large b. These sit ~2x above the slowest legitimate run measured
# at each size, so a hang is caught quickly and an honest run always fits.
# Fallback bound, used only until one rep of a cell succeeds; after that the
# adaptive per-cell bound (8x the first good run) takes over. Sized ~2x the
# slowest HONEST run measured at each size -- kmer_V2a's two-pass run was
# 495 s at 64k and scales as 1/b, so the old 8k value of 3000 s would have
# cut off a legitimate ~66 min run and manufactured a false timeout.
tmo_for() { case $1 in
    8000) echo 7200 ;; 16000) echo 4200 ;; 32000) echo 2400 ;;
   64000) echo 1200 ;; 125000) echo 700 ;; 250000) echo 450 ;;
       *) echo 300 ;; esac; }

echo "###### extended sweep start $(date '+%F %T')"

# ---- leg order, by cost-per-cell -----------------------------------------
# Reordered after the cell-plan rules landed. Under those rules the large-b
# two-pass legs cost ~0.6 h each and fill most of the table's blank columns,
# while the 8k leg costs ~6 h for one column -- so the cheap columns go first
# and 8k goes last. Every finished (graph,bs,mode,rep) row is skipped on
# restart, so reordering loses no measurement.
#
#   16k   ours 2.0 h  two-pass 0.9 h   (ours already most of the way through)
#   large ours 0.5 h  two-pass 3.6 h   7 columns
#   8k    ours 4.1 h  two-pass 1.8 h   1 column
for bs in 16000 250000 125000 1000000 2000000 4000000 8000000 500000 8000; do
  echo "###### ours bs=$bs tmo=$(tmo_for $bs)s $(date '+%F %T')"
  BS_LIST=$bs TMO=$(tmo_for $bs) CSV=experiments/results/bssweep.csv \
    bash experiments/run-full.sh medium
  echo "###### two-pass bs=$bs tmo=$(tmo_for $bs)s $(date '+%F %T')"
  BS_LIST=$bs TMO=$(tmo_for $bs) CSV=experiments/results/bssweep-twopass.csv \
    bash experiments/run-twopass.sh medium
done

echo "###### all done $(date '+%F %T')"

#!/bin/bash
# Fill the cells that the OLD allocation made impossible.
#
#   setsid nohup bash experiments/run-kmer.sh >> experiments/results/kmer.out 2>&1 < /dev/null &
#   check:  ps -o pid,cmd -C bash | grep run-kmer | grep -v grep
#
# The three sizing coefficients were cut from 5b+6n / V_fb=2n to 3b+1.5n /
# V_fb=1.0n after the aligned-pair justification turned out to be dead; 29
# graphs x {64k,500k} all came back exact against serial Tarjan with 49% less
# device memory. kmer_U1a and kmer_V2a, previously oom at every batch size,
# now run -- so their 22 \emph{m} cells and europe_osm@8M are measurable.
#
# LIMITED TO b >= 64k ON PURPOSE. Cost scales as 1/b and kmer is the most
# expensive pair in the set; measured from their 500k times, the full grid is
# ~32 h while b>=64k is ~4.1 h for 16 of the 22 cells. The tail is the whole
# bill: 32k costs 4 h, 16k 8 h, 8k 16 h, for two cells each. Those three sizes
# are left unmeasured ("--") rather than spending 28 h on six cells.
#
# Largest b first: those are the cheap cells, and they are also where the
# two-pass arm already has data to join against.
set -u
cd "$(dirname "$0")/.." || exit 1
export REPS=3

for bs in 8000000 4000000 2000000 1000000 500000 250000 125000 64000; do
  echo "###### ours kmer bs=$bs $(date '+%F %T')"
  BS_LIST=$bs TMO=7200 CSV=experiments/results/bssweep.csv \
    GRAPHS="kmer_U1a kmer_V2a" bash experiments/run-full.sh medium
  echo "###### two-pass kmer bs=$bs $(date '+%F %T')"
  BS_LIST=$bs TMO=7200 CSV=experiments/results/bssweep-twopass.csv \
    GRAPHS="kmer_U1a kmer_V2a" bash experiments/run-twopass.sh medium
done

# europe_osm at 8M was the one non-kmer m cell: it needed 41.6 GB of a 45.5 GB
# device under the old sizing and now needs 19.8 GB.
echo "###### europe_osm bs=8M $(date '+%F %T')"
BS_LIST=8000000 TMO=1800 CSV=experiments/results/bssweep.csv \
  GRAPHS="europe_osm" bash experiments/run-full.sh medium
BS_LIST=8000000 TMO=1800 CSV=experiments/results/bssweep-twopass.csv \
  GRAPHS="europe_osm" bash experiments/run-twopass.sh medium

echo "###### kmer fill done $(date '+%F %T')"

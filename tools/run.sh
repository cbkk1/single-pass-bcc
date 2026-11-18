#!/usr/bin/env bash
# Run the single-pass BCC over a set of graphs and print one row each.
#
#   tools/run.sh <batch-size> <graph> [graph ...]
#   tools/run.sh 2000000 /mnt/hdd/CS22S501/datasets/medium_datasets/ecl_graphs/*.egr
#
# Set SWEEP to a space-separated list of batch sizes to run each graph at
# several batch sizes instead (the last one should be >= |E| so it runs as a
# single batch, which is the exact answer to compare the others against):
#
#   SWEEP="100000 1000000 10000000" tools/run.sh 0 web-BerkStan.egr

set -u

BIN="$(dirname "$0")/../build/External_BCC"
[ -x "$BIN" ] || { echo "build first: cmake -S . -B build && cmake --build build -j" >&2; exit 1; }

[ $# -ge 2 ] || { echo "usage: $0 <batch-size> <graph> [graph ...]" >&2; exit 1; }

BATCH=$1; shift
SIZES=${SWEEP:-$BATCH}

printf "%-24s %-10s %-10s %-8s %-9s %s\n" graph V E batches time supernodes

for f in "$@"; do
    name=$(basename "$f"); name=${name%.*}
    for b in $SIZES; do
        out=$(timeout "${TIMEOUT:-900}" "$BIN" "$f" "$b" 2>&1)
        if [ $? -ne 0 ]; then
            printf "%-24s %s\n" "$name" "FAILED  $(echo "$out" | tail -3 | tr '\n' ' ')"
            continue
        fi
        V=$(echo "$out"   | grep '^|V|'                    | awk '{print $2}')
        E=$(echo "$out"   | grep '^|E|'                    | awk '{print $2}')
        nb=$(echo "$out"  | grep 'Total Number of Batches' | awk '{print $NF}')
        ms=$(echo "$out"  | grep 'computeBCG finished'     | awk '{print $(NF-1)}')
        sn=$(echo "$out"  | grep 'number of supernodes'    | awk '{print $NF}')
        printf "%-24s %-10s %-10s %-8s %-9s %s\n" "$name" "$V" "$E" "$nb" "${ms}ms" "$sn"
    done
done

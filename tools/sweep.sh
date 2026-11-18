#!/bin/bash
# Batch-count sweep against known-exact BCC counts.
#
#   tools/sweep.sh          split-range ids, host Tarjan for the BCC step
#   tools/sweep.sh -g       dense ids, Compress-Graph on the GPU
#
# Batch size is derived from each graph's edge count so the batch COUNT is what
# varies -- that is the axis the de-merger is exercised along, since a single
# batch never fires it at all.

set -u
cd "$(dirname "$0")/.." || exit 1

FLAG="${1:-}"
D=/mnt/hdd/CS22S501/datasets/medium_datasets/ecl_graphs
BATCHES="${BATCHES:-64 8 1}"

declare -A EXACT=(
  [web-BerkStan]=62970      [as-Skitter]=239476    [coPapersDBLP]=17066
  [in-2004]=299221          [higgs-twitter]=24910  [kron_g500-logn18]=79315
  [germany_osm]=2524313
)
declare -A M=(
  [web-BerkStan]=6650145    [as-Skitter]=11096053  [coPapersDBLP]=15245729
  [in-2004]=13591606        [higgs-twitter]=12508568 [kron_g500-logn18]=10634682
  [germany_osm]=12369181
)

printf "%-18s %-8s %-9s %-9s %-6s %-8s %s\n" graph batches got exact diff time n_s
for g in web-BerkStan as-Skitter coPapersDBLP in-2004 kron_g500-logn18 higgs-twitter germany_osm; do
  for nb in $BATCHES; do
    bs=$(( ${M[$g]} / nb + 1 ))
    out=$(timeout "${TMO:-600}" ./build/External_BCC "$D/$g.egr" "$bs" $FLAG 2>&1)
    [ $? -eq 124 ] && out="TIMEOUT"
    got=$(grep -oP 'is: *\K[0-9]+'                 <<<"$out")
    t=$(grep   -oP 'computeBCG finished in: *\K[0-9]+' <<<"$out")
    ns=$(grep  -oP 'BCG nodes \(n_s\): *\K[0-9]+'  <<<"$out")
    ex=${EXACT[$g]}
    printf "%-18s %-8s %-9s %-9s %-6s %-8s %s\n" \
      "$g" "$nb" "${got:-ERR}" "$ex" "$(( ${got:-0} - ex ))" "${t:-?}ms" "${ns:--}"
  done
done

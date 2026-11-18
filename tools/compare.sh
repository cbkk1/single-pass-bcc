#!/bin/bash
# Two-pass GEM-BCC (baseline) vs single-pass (ours), same graphs, same batch counts.
set -u
OURS=/home/chinmaya/Ex_BCC/External-BCC-for-GPU/build/External_BCC
BASE=/home/chinmaya/Ex_BCC/AbhijeethBaseline/External-BCC-for-GPU/build/External_BCC
D=/mnt/hdd/CS22S501/datasets/medium_datasets/ecl_graphs
TMO=${TMO:-900}
BATCHES="${BATCHES:-64 8 1}"

declare -A EXACT=( [web-BerkStan]=62970 [as-Skitter]=239476 [coPapersDBLP]=17066
                   [in-2004]=299221 [higgs-twitter]=24910 [kron_g500-logn18]=79315
                   [germany_osm]=2524313 )
declare -A M=( [web-BerkStan]=6650145 [as-Skitter]=11096053 [coPapersDBLP]=15245729
               [in-2004]=13591606 [higgs-twitter]=12508568 [kron_g500-logn18]=10634682
               [germany_osm]=12369181 )

printf "%-18s %-4s | %-9s %-9s | %-9s %-9s | %s\n" graph bat "two-pass" "time" "ours" "time" "speedup"
for g in web-BerkStan as-Skitter coPapersDBLP in-2004 kron_g500-logn18 higgs-twitter germany_osm; do
  for nb in $BATCHES; do
    bs=$(( ${M[$g]} / nb + 1 ))
    b=$(timeout $TMO $BASE -i "$D/$g.egr" -b "$bs" 2>&1)
    bc=$(grep -oP 'is: *\K[0-9]+' <<<"$b"); bt=$(grep -oP 'computeBCG finished in: *\K[0-9]+' <<<"$b")
    bst=$(grep -oP 'Spanning Tree construction.*|Spanning Tree finished in \K[0-9]+' <<<"$b" | tail -1)
    [ -n "${bt:-}" ] && [ -n "${bst:-}" ] && bt=$((bt+bst))
    o=$(timeout $TMO $OURS "$D/$g.egr" "$bs" -g 2>&1)
    oc=$(grep -oP 'is: *\K[0-9]+' <<<"$o"); ot=$(grep -oP 'computeBCG finished in: *\K[0-9]+' <<<"$o")
    sp="-"; [ -n "${bt:-}" ] && [ -n "${ot:-}" ] && [ "${ot:-0}" -gt 0 ] && sp=$(awk "BEGIN{printf \"%.1fx\", $bt/$ot}")
    printf "%-18s %-4s | %-9s %-9s | %-9s %-9s | %s\n" \
       "$g" "$nb" "${bc:-TIMEOUT}" "${bt:-?}ms" "${oc:-ERR}" "${ot:-?}ms" "$sp"
  done
done

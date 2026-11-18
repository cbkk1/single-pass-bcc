#!/bin/bash
# Does the reduced allocation still give the RIGHT answer, on every graph?
#
#   setsid nohup bash experiments/run-allotcheck.sh >> experiments/results/allotcheck.out 2>&1 < /dev/null &
#
# The three sizing coefficients trace to the removed aligned-pair scheme:
#   max_allot = A_B*b + A_N*n     shipped 5b + 6n
#   V_fb      = F_N*n + 2b + 2    shipped 2n
# A_B=5 was justified as "a leaver joins both halves of its block" -- under the
# no-pair design a leaver emits ONE link, so 3 is the true bound. A_N=6 and
# F_N=2 came from blocks owning an aligned PAIR of ids. H_i is a forest, so
# |E(H_i)| < n_s <= n; and n_s starts at n and compression is monotone
# non-increasing, so peak n_t <= n + 2b. Measured n_s/n is 0.06-0.46.
#
# The risk is NOT a crash: a too-small V_fb could give a silently wrong answer
# on some graph. So every cell runs with -c, which compares against an exact
# serial Tarjan on the same input, and the verdict is the comparison -- not
# merely "it did not abort". Two batch sizes, since the b-terms interact with
# the n-terms differently at each.
set -u
cd "$(dirname "$0")/.." || exit 1

MED=/mnt/hdd/CS22S501/datasets/medium_datasets/ecl_graphs
CSV=experiments/results/allotcheck.csv
[ -f "$CSV" ] || echo "graph,bs,cfg,status,bcc_gpu,bcc_serial,bridges_gpu,bridges_serial,blocks_gpu,blocks_serial,used_mb,ms" > "$CSV"

GRAPHS="as-Skitter coPapersCiteseer coPapersDBLP delaunay_n24 eu-2005
  europe_osm GAP-road germany_osm higgs-twitter hollywood-2009
  hugebubbles-00020 in-2004 indochina-2004 kmer_U1a kmer_V2a
  kron_g500-logn18 kron_g500-logn19 kron_g500-logn20 kron_g500-logn21
  ljournal-2008 rgg_n_2_23_s0 rgg_n_2_24_s0 road_usa soc-LiveJournal1
  sx-stackoverflow uk-2002 web-BerkStan wiki-topcats com-Orkut"

# cfg label : env settings
CFGS=("shipped:" "tight:BCC_ALLOT_B=3 BCC_ALLOT_N=1.5 BCC_VFB_N=1.0")

for bs in 500000 64000; do
for cfg in "${CFGS[@]}"; do
  lbl=${cfg%%:*}; envs=${cfg#*:}
  echo "###### cfg=$lbl bs=$bs $(date '+%F %T')"
  for g in $GRAPHS; do
    f=$MED/$g.egr
    [ -f "$f" ] || { echo "  -- $g: no file"; continue; }
    grep -q "^$g,$bs,$lbl," "$CSV" && continue
    t0=$(date +%s)
    out=$(env $envs timeout 3600 ./build/External_BCC -i "$f" -a ext-bcc -g -b "$bs" -c 2>&1)
    t1=$(date +%s)
    # first occurrence = gpu, second = serial reference
    mapfile -t B < <(sed -n 's/^Biconnected components: *\([0-9]*\).*/\1/p' <<<"$out")
    mapfile -t R < <(sed -n 's/^  of which bridges: *\([0-9]*\).*/\1/p' <<<"$out")
    mapfile -t K < <(sed -n 's/^  2-connected blocks: *\([0-9]*\).*/\1/p' <<<"$out")
    mem=$(sed -n 's/^Used: *\([0-9.]*\).*/\1/p' <<<"$out" | head -1)
    ms=$(sed -n 's/.*computeBCG finished in: *\([0-9]*\).*/\1/p' <<<"$out" | head -1)

    if   grep -qiE "Assertion" <<<"$out";                     then st=assert
    elif grep -qiE "out of memory|bad_alloc" <<<"$out";       then st=oom
    elif grep -qiE "illegal memory access" <<<"$out";         then st=fault
    elif [ -z "${B[0]:-}" ];                                  then st=nooutput
    elif [ -z "${B[1]:-}" ];                                  then st=nooracle
    elif [ "${B[0]}" = "${B[1]}" ] && [ "${R[0]}" = "${R[1]}" ] \
      && [ "${K[0]}" = "${K[1]}" ];                           then st=exact
    else                                                       st=WRONG
    fi
    printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
      "$g" "$bs" "$lbl" "$st" "${B[0]:-}" "${B[1]:-}" "${R[0]:-}" "${R[1]:-}" \
      "${K[0]:-}" "${K[1]:-}" "${mem:-}" "${ms:-}" >> "$CSV"
    printf '  %-20s bs=%-7s %-8s %-9s bcc=%-11s oracle=%-11s %9s MB %7s ms  %ss\n' \
      "$g" "$bs" "$lbl" "$st" "${B[0]:-?}" "${B[1]:-?}" "${mem:-?}" "${ms:-?}" "$((t1-t0))"
    [ "$st" = WRONG ] && echo "     !!! MISMATCH -- log kept" && \
      printf '%s\n' "$out" > "experiments/logs/allot.$g.$bs.$lbl.log"
  done
done
done
echo "###### allotcheck done $(date '+%F %T')"

#!/bin/bash
# Full experiment: single-pass default (parent-is-block fence rule) vs the v2
# filter path (BCC_FENCE_KEEP=0), on real graphs.
#
#   bash experiments/run-full.sh medium        29 medium graphs
#   bash experiments/run-full.sh large         feasible large graphs
#   bash experiments/run-full.sh both
#
# Knobs:  REPS (default 3)  BS_LIST  TMO (per-run timeout, s)  CSV
#         GRAPHS (override the built-in list, for a smoke test or a re-run)
#
# Serialised by construction -- the GPU is shared, so never run two of these.
# Resumable: every finished (graph,bs,mode,rep) row is skipped on restart, so
# a kill or a crash costs one run, not the sweep.
# computeBCG_ms is the metric (pure compute); wall_s includes HDD load, which
# is why it is recorded separately and never used as the headline number.

set -u
cd "$(dirname "$0")/.." || exit 1

MED=/mnt/hdd/CS22S501/datasets/medium_datasets/ecl_graphs
LRG=/mnt/hdd/CS22S501/datasets/large_datasets/ecl_graphs
BIN=./build/External_BCC

REPS="${REPS:-3}"
BS_LIST="${BS_LIST:-500000 2000000}"
TMO="${TMO:-1800}"
CSV="${CSV:-experiments/results/full.csv}"
LOGD=experiments/logs

MEDIUM_GRAPHS="as-Skitter coPapersCiteseer coPapersDBLP delaunay_n24 eu-2005
  europe_osm GAP-road germany_osm higgs-twitter hollywood-2009
  hugebubbles-00020 in-2004 indochina-2004 kmer_U1a kmer_V2a
  kron_g500-logn18 kron_g500-logn19 kron_g500-logn20 kron_g500-logn21
  ljournal-2008 rgg_n_2_23_s0 rgg_n_2_24_s0 road_usa soc-LiveJournal1
  sx-stackoverflow uk-2002 web-BerkStan wiki-topcats com-Orkut"

# Ordered smallest-first: the sweep produces usable rows early, and the ones
# that may not finish at all sit at the end.
LARGE_GRAPHS="arabic-2005 uk-2005 webbase-2001 it-2004 twitter7 GAP-twitter
  com-Friendster GAP-web sk-2005 GAP-kron GAP-urand MOLIERE_2016"

[ -f "$CSV" ] || echo "graph,category,bs,mode,rep,status,exit,wall_s,computeBCG_ms,bcc,bridges,blocks2,supernodes,n_s,labels,comps_with_head,distinct_heads,max_headed" > "$CSV"

done_row() { grep -q "^$1,[^,]*,$2,$3,$4," "$CSV"; }

num() { sed -n "s/.*$1[^0-9-]*\([0-9][0-9]*\).*/\1/p" <<<"$2" | head -1; }

run_one() {   # graph category file bs mode rep
  local g=$1 cat=$2 f=$3 bs=$4 mode=$5 rep=$6
  local tag="$g.$bs.${mode:-def}.$rep" t0 t1 out rc st

  t0=$(date +%s.%N)
  out=$(env ${mode:+BCC_FENCE_KEEP=$mode} timeout "$TMO" $BIN -i "$f" -a ext-bcc -g -b "$bs" 2>&1)
  rc=$?
  t1=$(date +%s.%N)

  local bcc; bcc=$(sed -n 's/.*components in .* is: *\([0-9]*\).*/\1/p' <<<"$out" | head -1)
  if   [ -n "$bcc" ] && [ "$rc" -eq 0 ]; then st=ok
  elif [ "$rc" -eq 124 ];                then st=timeout
  elif grep -qiE "out of memory|bad_alloc|cudaErrorMemoryAllocation" <<<"$out"; then st=oom
  else st=crash; fi

  [ "$st" = ok ] || printf '%s\n' "$out" > "$LOGD/$tag.log"

  printf '%s,%s,%s,%s,%s,%s,%s,%.1f,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$g" "$cat" "$bs" "${mode:-default}" "$rep" "$st" "$rc" \
    "$(echo "$t1 - $t0" | bc)" \
    "$(num 'computeBCG finished in:' "$out")" \
    "${bcc:-}" \
    "$(num 'of which bridges:' "$out")" \
    "$(num '2-connected blocks:' "$out")" \
    "$(num 'Supernodes in M:' "$out")" \
    "$(num 'BCG nodes (n_s):' "$out")" \
    "$(num 'components (labels):' "$out")" \
    "$(num 'components with a head:' "$out")" \
    "$(num 'distinct head vertices:' "$out")" \
    "$(num 'max BCCs headed by one v:' "$out")" >> "$CSV"

  printf '  %-22s bs=%-9s %-9s rep%-2s %-8s %6s ms  %8.1f s  bcc=%s\n' \
    "$g" "$bs" "${mode:-default}" "$rep" "$st" \
    "$(num 'computeBCG finished in:' "$out")" "$(echo "$t1 - $t0" | bc)" "${bcc:-?}"
  LAST_ST=$st
  LAST_WALL=$(echo "$t1 - $t0" | bc)
  # Retry ONLY a genuine crash. A timeout is a hang and retrying it just
  # hangs again (rgg_n_2_24_s0 @125k timed out on both attempts); an oom is
  # deterministic. Retrying either doubled the cost for no information.
  [ "$st" != crash ]
}

sweep() {   # category dir graphs...
  local cat=$1 dir=$2; shift 2
  for g in $*; do
    local f="$dir/$g.egr"
    [ -f "$f" ] || { echo "  -- $g: no file, skipped"; continue; }
    for bs in $BS_LIST; do
      # Mode "0" -- the v2 FILTER path -- is gone. fence-keep=down replaced it
      # (un-fence only tree edges whose parent is a block): exact, 4-15%
      # faster, and the only thing gpu_compress.cu's do_filter still responds
      # to is fence mode 0, i.e. dead code. Benchmarking it cost 894 runs and
      # 4.81 h across this project, slightly MORE than the default path it was
      # being compared against. Set BCC_MODES="  0" to bring it back for a
      # one-off fence-keep comparison.
      # BCC_MODES is a word list; "default" is the sentinel for the empty mode,
      # since an unquoted empty string word-splits away to nothing. To run the
      # old comparison: BCC_MODES="default 0".
      for mode in ${BCC_MODES:-default}; do
        [ "$mode" = default ] && mode=""
        local mname="${mode:-default}"
        for rep in $(seq 1 "$REPS"); do
          done_row "$g" "$bs" "$mname" "$rep" && continue
          run_one "$g" "$cat" "$f" "$bs" "$mode" "$rep" || {
            # one retry: the baseline is known to crash ~9% of the time
            sed -i "\#^$g,[^,]*,$bs,$mname,$rep,#d" "$CSV"
            echo "     retry once"
            run_one "$g" "$cat" "$f" "$bs" "$mode" "$rep"
          }
          if [ "${LAST_ST:-}" = timeout ]; then
            echo "     timed out -- skipping remaining reps for this cell"
            break
          fi
        done
      done
    done
  done
}

echo "=== full experiment  reps=$REPS  bs={$BS_LIST}  tmo=${TMO}s  csv=$CSV"
echo "=== started $(date '+%F %T')"
case "${1:-both}" in
  medium) sweep medium "$MED" ${GRAPHS:-$MEDIUM_GRAPHS} ;;
  large)  sweep large  "$LRG" ${GRAPHS:-$LARGE_GRAPHS} ;;
  both)   sweep medium "$MED" $MEDIUM_GRAPHS; sweep large "$LRG" $LARGE_GRAPHS ;;
  *) echo "usage: $0 [medium|large|both]"; exit 1 ;;
esac
echo "=== finished $(date '+%F %T')"

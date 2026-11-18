#!/bin/bash
# The Euro-Par 2025 two-pass baseline, on the same graph x batch-size grid as
# experiments/run-full.sh, so the rows join on (graph,bs).
#
#   bash experiments/run-twopass.sh medium|large
#
# Its printed computeBCG already INCLUDES the pass-1 global spanning tree: the
# outer `dur` holding the ST time is still live at the print (the `dur` inside
# the batch loop is a shadowed local for copy time).  So its number and ours
# bracket the same thing and need no adjustment.  st_ms is recorded anyway, to
# show how much of the baseline's time is the pass we deleted.
#
# The baseline has no -g flag and no LCH readout, so it yields a BCC COUNT
# only -- no block-cut structure to compare.  That is a property of the
# baseline, not a gap in this harness.

# Launch DETACHED or a session teardown kills it:
#   setsid nohup bash experiments/run-twopass.sh medium >> out 2>&1 < /dev/null &
# and check it with  pgrep -a -x bash | grep run-twopass  -- a plain
# `pgrep -f run-twopass` also matches the shell running the pgrep itself,
# which reads as alive when the sweep is long dead.

set -u
cd "$(dirname "$0")/.." || exit 1

MED=/mnt/hdd/CS22S501/datasets/medium_datasets/ecl_graphs
LRG=/mnt/hdd/CS22S501/datasets/large_datasets/ecl_graphs
BIN="${BIN:-../AbhijeethBaseline/External-BCC-for-GPU/build/External_BCC}"

REPS="${REPS:-3}"
BS_LIST="${BS_LIST:-500000 2000000}"
BASE_TMO="${TMO:-1800}"
TMO="$BASE_TMO"
CSV="${CSV:-experiments/results/twopass.csv}"
OURS_CSV="${OURS_CSV:-experiments/results/bssweep.csv}"
LOGD=experiments/logs

MEDIUM_GRAPHS="as-Skitter coPapersCiteseer coPapersDBLP delaunay_n24 eu-2005
  europe_osm GAP-road germany_osm higgs-twitter hollywood-2009
  hugebubbles-00020 in-2004 indochina-2004 kmer_U1a kmer_V2a
  kron_g500-logn18 kron_g500-logn19 kron_g500-logn20 kron_g500-logn21
  ljournal-2008 rgg_n_2_23_s0 rgg_n_2_24_s0 road_usa soc-LiveJournal1
  sx-stackoverflow uk-2002 web-BerkStan wiki-topcats com-Orkut"

LARGE_GRAPHS="arabic-2005 uk-2005 webbase-2001 it-2004 twitter7 GAP-twitter
  com-Friendster GAP-web sk-2005 GAP-kron GAP-urand MOLIERE_2016"

[ -f "$CSV" ] || echo "graph,category,bs,rep,status,exit,wall_s,computeBCG_ms,st_ms,ibcc_ms,repair_ms,transform_ms,bcc" > "$CSV"

num() { sed -n "s/.*$1[^0-9-]*\([0-9][0-9]*\).*/\1/p" <<<"$2" | head -1; }
fnum() { sed -n "s/.*$1[^0-9-]*\([0-9][0-9]*\)\.[0-9]*.*/\1/p" <<<"$2" | head -1; }

run_one() {   # graph category file bs rep
  local g=$1 cat=$2 f=$3 bs=$4 rep=$5 t0 t1 out rc st
  t0=$(date +%s.%N)
  out=$(timeout "$TMO" $BIN -i "$f" -a ext-bcc -b "$bs" 2>&1)
  rc=$?
  t1=$(date +%s.%N)

  # the baseline words it "Biconnected Components (BCCs) in X is: N"
  local bcc; bcc=$(sed -n 's/.*omponents.* in .* is: *\([0-9]*\).*/\1/p' <<<"$out" | head -1)
  if   [ -n "$bcc" ] && [ "$rc" -eq 0 ]; then st=ok
  elif [ "$rc" -eq 124 ]; then st=timeout
  elif grep -qiE "out of memory|bad_alloc|cudaErrorMemoryAllocation" <<<"$out"; then st=oom
  # The baseline's dominant failure is NOT a crash: it aborts on
  #   extern_bcc.cu:130  Assertion `totalSize <= g_bcg_ds.max_allot\' failed
  # which is its own pre-allocated budget (2b+2n edges) running out -- the
  # direct counterpart of our oom, and 203 of 227 non-ok runs. Lumping it
  # under "crash" made a capacity result look like instability, so it gets
  # its own status. A separate 24 runs are genuine CUDA faults (illegal
  # memory access, sync failures) from the uninitialised d_a1/d_a2 that our
  # fork fixes with init_a -- those may be our build, not their design.
  elif grep -q "max_allot. failed" <<<"$out"; then st=cap
  elif grep -qiE "illegal memory access|Failed to synchronize" <<<"$out"; then st=fault
  else st=crash; fi
  [ "$st" = ok ] || printf '%s\n' "$out" > "$LOGD/tp.$g.$bs.$rep.log"

  printf '%s,%s,%s,%s,%s,%s,%.1f,%s,%s,%s,%s,%s,%s\n' \
    "$g" "$cat" "$bs" "$rep" "$st" "$rc" "$(echo "$t1 - $t0" | bc)" \
    "$(num  'computeBCG finished in:' "$out")" \
    "$(num  'Spanning Tree finished in' "$out")" \
    "$(fnum 'Time of i-BCC :' "$out")" \
    "$(fnum 'Time of Repair :' "$out")" \
    "$(fnum 'Time of Transform :' "$out")" \
    "${bcc:-}" >> "$CSV"

  printf '  %-22s bs=%-9s rep%-2s %-8s %6s ms (st %s)  %7.1f s  bcc=%s\n' \
    "$g" "$bs" "$rep" "$st" "$(num 'computeBCG finished in:' "$out")" \
    "$(num 'Spanning Tree finished in' "$out")" "$(echo "$t1 - $t0" | bc)" "${bcc:-?}"
  LAST_ST=$st
  LAST_WALL=$(echo "$t1 - $t0" | bc)
  # Retry ONLY a genuine fault. A capacity abort (cap) is deterministic --
  # the budget does not change between runs -- so retrying it wastes time.
  # A timeout is a hang and retrying it just
  # hangs again (rgg_n_2_24_s0 @125k timed out on both attempts); an oom is
  # deterministic. Retrying either doubled the cost for no information.
  [ "$st" != crash ] && [ "$st" != fault ]
}

sweep() {
  local cat=$1 dir=$2; shift 2
  for g in $*; do
    local f="$dir/$g.egr"
    [ -f "$f" ] || { echo "  -- $g: no file, skipped"; continue; }
    for bs in $BS_LIST; do
      # What to do with this cell is decided before any GPU time is spent:
      # experiments/cell-plan.py returns "skip <reason>" when the table marker
      # is already known (our arm ooms here, or the baseline totally hung at a
      # larger b -- hangs are monotone in b and re-deriving one costs 20-60
      # min), else a bound from the 1/b law. See that file for the rules.
      PLAN=$(python3 experiments/cell-plan.py "$g" "$bs" 2>/dev/null)
      case "$PLAN" in
        skip*)
          echo "  -- $g bs=$bs: ${PLAN#skip } -- not run"
          # Recorded so the table generator can mark the cell, with a status
          # that can never be mistaken for a measurement.
          grep -q "^$g,[^,]*,$bs,0,infer-" "$CSV" || \
            printf '%s,%s,%s,0,infer-%s,0,0.0,,,,,,\n' \
              "$g" "$cat" "$bs" "${PLAN#skip }" >> "$CSV"
          continue ;;
        "tmo -") PRED_TMO="" ;;
        tmo*)    PRED_TMO="${PLAN#tmo }"
                 echo "  -- $g bs=$bs bound ${PRED_TMO}s (1/b law)" ;;
        *)       PRED_TMO="" ;;
      esac
      # Once ONE rep of this cell succeeds we know its honest runtime exactly,
      # so a later rep exceeding a small multiple of it is a hang. This caught
      # coPapersDBLP (1.4 s, 1.4 s, then 1800 s) in seconds, not half an hour.
      CELL_TMO=""
      for rep in $(seq 1 "$REPS"); do
        grep -q "^$g,[^,]*,$bs,$rep," "$CSV" && continue
        if   [ -n "$CELL_TMO" ]; then TMO=$CELL_TMO
        elif [ -n "$PRED_TMO" ]; then TMO=$PRED_TMO
        else TMO=$BASE_TMO; fi
        run_one "$g" "$cat" "$f" "$bs" "$rep" || {
          sed -i "\#^$g,[^,]*,$bs,$rep,#d" "$CSV"
          echo "     retry once"
          run_one "$g" "$cat" "$f" "$bs" "$rep"
        }
        # A timed-out cell yields no median, so reps 2..N are pure waiting.
        # One timeout is enough to mark the cell -- 25 of these at 64k cost
        # 8.3 h of the 9 h that leg took.
        if [ "${LAST_ST:-}" = ok ] && [ -z "$CELL_TMO" ]; then
          # 8x the first good run, floored at 60 s to absorb load variance
          CELL_TMO=$(echo "w=${LAST_WALL:-0}*8; if (w<60) 60 else w" | bc | cut -d. -f1)
          echo "     cell bound -> ${CELL_TMO}s (first good run ${LAST_WALL}s)"
        fi
        if [ "${LAST_ST:-}" = timeout ]; then
          echo "     timed out -- skipping remaining reps for this cell"
          break
        fi
      done
    done
  done
}

echo "=== two-pass baseline  reps=$REPS  bs={$BS_LIST}  bin=$BIN"
echo "=== started $(date '+%F %T')"
case "${1:-medium}" in
  medium) sweep medium "$MED" ${GRAPHS:-$MEDIUM_GRAPHS} ;;
  large)  sweep large  "$LRG" ${GRAPHS:-$LARGE_GRAPHS} ;;
  *) echo "usage: $0 [medium|large]"; exit 1 ;;
esac
echo "=== finished $(date '+%F %T')"

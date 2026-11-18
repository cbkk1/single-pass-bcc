#!/bin/bash
# Re-measure the 18 baseline cells that were SKIPPED, never measured.
#
#   setsid nohup bash experiments/run-unskip18.sh >> experiments/results/unskip18.out 2>&1 < /dev/null &
#   check:  pgrep -a -x bash | grep run-unskip18
#
# WHY. These 18 render as "--" in Table 1. Our arm has all of them; only the
# baseline is missing, and it is missing because cell-plan.py rule 2 declined
# to run it -- on the LOOSE form of that rule, before it was tightened to
# require a >=8x overrun or >=2 timeout reps. That loose rule is known wrong:
#   * hangs are not monotone in b. rgg_n_2_24_s0 times out at b=1M and runs in
#     12 s at b=500k -- 1 clean counter-example in the whole dataset, but it is
#     the exact shape the rule assumes away.
#   * rgg_n_2_24_s0 @ 500k was itself skipped by this rule, then measured
#     anyway: ok 3/3, and it is 4.51 in the table -- the third-best cell in the
#     paper, which the skip had hidden.
# All 18 sit at b=8k/16k, where the sparse graphs peak (rgg_n_2_23 4.18 @ 64k,
# rgg_n_2_24 5.03 @ 125k, both still climbing as b falls). So these blanks are
# where unclaimed wins are most likely, not where hangs are most certain.
#
# BCC_PLAN_NOSKIP=1 drops cell-plan rules 1-2 and keeps rule 3, so every cell
# still gets the 8x-the-1/b-law bound that stopped the old 4x bar from cutting
# off honest runs. One timeout still early-skips reps 2-3 of that cell.
#
# ORDER IS CHEAPEST-FIRST, by the 1/b prediction. The 7 cheapest finish inside
# ~50 min even if all 7 hang, so the answer starts arriving immediately and a
# long tail can be killed without losing the cheap results. Budget: 5.2 h if
# every cell is honest, 14.0 h if every cell hangs. uk-2002 @ 8k alone is
# 4.3 h of that worst case and runs last for exactly that reason.
set -u
cd "$(dirname "$0")/.." || exit 1

export BCC_PLAN_NOSKIP=1
export REPS=3
CSV=experiments/results/bssweep-twopass.csv

# graph:bs, cheapest predicted first
CELLS="
higgs-twitter:16000
higgs-twitter:8000
wiki-topcats:16000
wiki-topcats:8000
com-Orkut:16000
ljournal-2008:16000
rgg_n_2_23_s0:16000
com-Orkut:8000
ljournal-2008:8000
hugebubbles-00020:16000
indochina-2004:16000
rgg_n_2_23_s0:8000
rgg_n_2_24_s0:16000
hugebubbles-00020:8000
indochina-2004:8000
rgg_n_2_24_s0:8000
uk-2002:16000
uk-2002:8000
"

i=0
for c in $CELLS; do
  g=${c%:*}; bs=${c#*:}
  i=$((i+1))
  echo "###### [$i/18] $g bs=$bs  $(date '+%F %T')"
  GRAPHS="$g" BS_LIST="$bs" CSV="$CSV" bash experiments/run-twopass.sh medium
done

echo "###### unskip18 done $(date '+%F %T')"
echo "--- outcome per cell ---"
for c in $CELLS; do
  g=${c%:*}; bs=${c#*:}
  printf '  %-22s bs=%-7s %s\n' "$g" "$bs" \
    "$(awk -F, -v g="$g" -v b="$bs" '$1==g && $3==b && $4!=0 {printf "%s ", $5}' "$CSV")"
done

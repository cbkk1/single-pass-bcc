#!/usr/bin/env python3
"""Decide what to do with one two-pass cell, before spending GPU time on it.

    python3 experiments/cell-plan.py <graph> <bs>
        -> "skip <reason>"      do not run; the table marker is already known
        -> "tmo <seconds>"      run, bounded at <seconds>
        -> "tmo -"              run, bounded by the caller's per-size fallback

Three rules, in order.

1. OUR arm already ooms here.  A speedup cell needs both arms, so the
   baseline's number can never reach the table; the cell stays \\emph{m}.

2. The baseline TOTALLY HUNG at some larger b -- timed out and never once
   succeeded there.  Hangs are monotone in b: smaller b means more batches and
   strictly more work, and across every cell measured so far there is not one
   counter-example (0 cases of hang-at-larger-b, ok-at-smaller-b over the full
   grid).  Re-running such a cell costs 20-60 min to re-derive a marker we have
   already measured twice.  NOTE the cell is then marked \\emph{h}, NOT
   \\emph{t} -- it is inferred, not measured, and the table says so.
   A single flaky rep does not trigger this: the larger-b cell must have no ok
   run at all, which is why eu-2005 and delaunay_n24 (one slow rep each, cell
   otherwise fine) still get measured here.

3. Otherwise bound the run at 8x the time the 1/b law predicts for this cell
   from the same graph at a larger b.  computeBCG is linear in 1/b to within
   [0.69, 1.13] over 52 measured points (median 0.88); 4x was tried first and
   proved too tight on a shared GPU -- it cut off two honest kmer_U1a runs at
   1.4x and produced six 1.5-4x "hangs" that were probably contention.  Floored at 60 s for load variance and capped at 4 h,
   except the cap never cuts below 2x the prediction so a genuinely slow cell
   keeps its room.
"""
import csv, os, sys, collections, statistics as st

# BCC_PLAN_NOSKIP=1 keeps rule 3 (the 1/b bound) and drops rules 1-2,
# the two that decline to run a cell at all. Rule 2 rests on hangs being
# monotone in b, and that is not quite true: rgg_n_2_24_s0 times out at
# b=1M yet runs in 12 s at b=500k. That cell was skipped on exactly this
# inference, and when it was finally measured it came back ok 3/3 -- a
# 4.51x speedup, the third-best in the table, hidden by a skip. 18 cells
# at b=8k/16k were skipped by the LOOSE form of rule 2 (before it was
# tightened to >=8x or >=2 reps), so they are re-measured under this flag
# rather than left inferred.
NOSKIP = os.getenv("BCC_PLAN_NOSKIP") == "1"

OURS = "experiments/results/bssweep.csv"
TWO  = "experiments/results/bssweep-twopass.csv"
graph, bs = sys.argv[1], int(sys.argv[2])


def read(path, mode=None):
    times, status = collections.defaultdict(list), collections.defaultdict(set)
    reps, wall = collections.defaultdict(list), collections.defaultdict(list)
    try:
        for r in csv.DictReader(open(path)):
            if r["graph"] != graph:
                continue
            if mode is not None and r.get("mode") != mode:
                continue
            b = int(r["bs"])
            status[b].add(r["status"])
            reps[b].append(r["status"])
            try:
                wall[b].append(float(r["wall_s"]))
            except (ValueError, KeyError):
                pass
            if r["status"] == "ok" and r.get("computeBCG_ms"):
                times[b].append(float(r["computeBCG_ms"]) / 1000)
    except FileNotFoundError:
        pass
    return times, status, reps, wall


_, ours_st, _, _ = read(OURS, "default")
two_t, two_st, two_reps, two_wall = read(TWO)

if not NOSKIP and "oom" in ours_st.get(bs, set()):
    print("skip our-arm-ooms-here")
    sys.exit(0)

# Only a STRONG hang at a larger b justifies skipping this cell. Measured
# 2026-11-17: of 31 cells that merely "timed out and never succeeded", 21 had
# run exactly ONE rep -- the early-skip breaks after the first timeout, so that
# phrase cannot distinguish a reliable hang from one flaky rep. And judging
# each against the 1/b law showed 6 overran by only 1.5-4x (germany_osm 2.8x,
# road_usa 3.7x), well inside shared-GPU contention, while 2 kmer_U1a cells at
# 1.4x were honest runs this harness cut off. A hang counts as strong only if
# it overran its predicted time by >=8x, or timed out on >=2 separate reps.
def strong_hang(b):
    s_ = two_st.get(b, set())
    if "timeout" not in s_ or "ok" in s_:
        return False
    if sum(1 for x in two_reps.get(b, []) if x == "timeout") >= 2:
        return True
    p = [st.median(v) * bb / b for bb, v in two_t.items() if bb > b]
    w = max(two_wall.get(b, [0]))
    return bool(p) and w >= 8 * max(p)

hung = [] if NOSKIP else [b for b in two_st if b > bs and strong_hang(b)]
if hung:
    print(f"skip strong-hang-at-b={min(hung)}")
    sys.exit(0)

# A graph the baseline crashes on deterministically crashes in seconds, so it
# must never be given the generous per-size fallback: at 8k that fallback is
# 7200 s, and seven graphs (the four krons, ljournal-2008, soc-LiveJournal1,
# sx-stackoverflow) crash at every b tested. Run them -- the crash is the
# datum -- but bounded tightly.
crashed = [b for b, s_ in two_st.items()
           if b > bs and "crash" in s_ and "ok" not in s_]
if crashed and not any("ok" in s_ for s_ in two_st.values()):
    print("tmo 60")
    sys.exit(0)

preds = [st.median(v) * b / bs for b, v in two_t.items() if b > bs]
if not preds:
    print("tmo -")
    sys.exit(0)
pred = max(preds)
print(f"tmo {int(max(120, min(max(8 * pred, 120), max(28800, 3 * pred))))}")

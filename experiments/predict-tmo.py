#!/usr/bin/env python3
"""Predicted timeout for one two-pass cell, from the 1/b scaling law.

    python3 experiments/predict-tmo.py <csv> <graph> <bs>

The baseline's computeBCG is almost exactly linear in 1/b: validated over 52
(graph, b) predictions made from each graph's largest measured b, the ratio
actual/predicted sits in [0.69, 1.13] with median 0.88 (slightly sub-linear
because fixed costs amortise).  So any cell with an ok run at some LARGER b
has a known honest runtime at this b, and a run exceeding a few multiples of
it is a hang, not slowness.

Prints a bound in whole seconds, or nothing when no larger-b datum exists
(the caller then falls back to its per-size bound).  The bound is 4x the
most pessimistic prediction -- 3.5x clear of the worst ratio observed --
floored at 60 s for load variance and capped at 4 h, except that the cap
never cuts below 2x the prediction, so a genuinely slow cell keeps its room.
"""
import csv, sys, statistics as st, collections

csvf, graph, bs = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = collections.defaultdict(list)
try:
    for r in csv.DictReader(open(csvf)):
        if r["graph"] == graph and r["status"] == "ok" and r.get("computeBCG_ms"):
            d[int(r["bs"])].append(float(r["computeBCG_ms"]) / 1000)
except FileNotFoundError:
    sys.exit(0)

preds = [st.median(v) * b / bs for b, v in d.items() if b > bs]
if not preds:
    sys.exit(0)
pred = max(preds)
print(int(max(60, min(max(4 * pred, 60), max(14400, 2 * pred)))))

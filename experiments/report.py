#!/usr/bin/env python3
"""Turn experiments/results/full.csv into the two paper tables.

  python3 experiments/report.py [csv ...]

Table 1 (correctness): every structural column of the block-cut forest, new
default vs the v2 filter path.  A single differing column is a real defect, so
this table reports per-column equality, not just the BCC count.

Table 2 (timing): median computeBCG over the reps, and the gain.  wall_s is
ignored -- it is dominated by the HDD read and says nothing about the
algorithm.
"""
import csv, sys, statistics as st
from collections import defaultdict

STRUCT = ["bcc", "bridges", "blocks2", "supernodes", "n_s",
          "labels", "comps_with_head", "distinct_heads", "max_headed"]

rows = []
for p in (sys.argv[1:] or ["experiments/results/full.csv"]):
    with open(p) as f:
        rows += list(csv.DictReader(f))

ok   = [r for r in rows if r["status"] == "ok"]
bad  = [r for r in rows if r["status"] != "ok"]

# ---- structural equality -------------------------------------------------
struct = {}                       # (graph,bs,mode) -> tuple of columns
times  = defaultdict(list)        # (graph,bs,mode) -> [ms]
for r in ok:
    k = (r["graph"], int(r["bs"]), r["mode"])
    struct.setdefault(k, tuple(r[c] for c in STRUCT))
    if r["computeBCG_ms"]:
        times[k].append(int(r["computeBCG_ms"]))

keys = sorted({(g, bs) for (g, bs, _) in struct})

print("=" * 96)
print("TABLE 1  block-cut structure: default (parent-is-block) vs BCC_FENCE_KEEP=0 (v2 filter)")
print("=" * 96)
print(f"{'graph':<22}{'bs':>9}  {'bcc':>10} {'bridges':>10} {'blocks':>8} "
      f"{'n_s':>10} {'heads':>8} {'maxhead':>8}  verdict")
nmatch = ndiff = nsolo = 0
diffs = []
for g, bs in keys:
    a = struct.get((g, bs, "default")); b = struct.get((g, bs, "0"))
    d = dict(zip(STRUCT, a or b))
    if a and b:
        if a == b:
            v = "IDENTICAL"; nmatch += 1
        else:
            bad_cols = [c for c, x, y in zip(STRUCT, a, b) if x != y]
            v = "DIFFERS: " + ",".join(bad_cols); ndiff += 1
            diffs.append((g, bs, bad_cols, a, b))
    else:
        v = "default only" if a else "filter only"; nsolo += 1
    print(f"{g:<22}{bs:>9}  {d['bcc']:>10} {d['bridges']:>10} {d['blocks2']:>8} "
          f"{d['n_s']:>10} {d['distinct_heads']:>8} {d['max_headed']:>8}  {v}")
print(f"\n  {nmatch} configs identical on all {len(STRUCT)} columns, "
      f"{ndiff} differing, {nsolo} with only one path's result.")
for g, bs, cols, a, b in diffs:
    print(f"  !! {g} bs={bs}")
    for c in cols:
        print(f"       {c:<18} default={dict(zip(STRUCT,a))[c]:<14} filter={dict(zip(STRUCT,b))[c]}")

# ---- timing --------------------------------------------------------------
print()
print("=" * 96)
print("TABLE 2  computeBCG, median of reps (ms).  Lower is better.")
print("=" * 96)
print(f"{'graph':<22}{'bs':>9}  {'filter':>8} {'default':>8}  {'gain':>7}   reps")
gains = []
for g, bs in keys:
    f_ = times.get((g, bs, "0")); d_ = times.get((g, bs, "default"))
    if not (f_ and d_):
        continue
    mf, md = st.median(f_), st.median(d_)
    gain = (mf - md) / mf * 100 if mf else 0.0
    gains.append(gain)
    print(f"{g:<22}{bs:>9}  {mf:>8.0f} {md:>8.0f}  {gain:>6.1f}%   {len(f_)}/{len(d_)}")
if gains:
    print(f"\n  {len(gains)} configs: median gain {st.median(gains):.1f}%, "
          f"range {min(gains):.1f}% .. {max(gains):.1f}%, "
          f"{sum(1 for x in gains if x > 0)} faster / {sum(1 for x in gains if x < 0)} slower")

# ---- failures ------------------------------------------------------------
if bad:
    print()
    print("=" * 96)
    print("NOT OK")
    print("=" * 96)
    seen = defaultdict(list)
    for r in bad:
        seen[(r["graph"], r["bs"], r["mode"], r["status"])].append(r["rep"])
    for (g, bs, m, s), reps in sorted(seen.items()):
        print(f"  {g:<22} bs={bs:<9} {m:<9} {s:<8} reps {','.join(reps)}"
              f"   log: experiments/logs/{g}.{bs}.{'def' if m=='default' else m}.{reps[0]}.log")

# ---- Table 3: the two-pass baseline ------------------------------------
import os
TP = "experiments/results/twopass.csv"
if os.path.exists(TP):
    tp = [r for r in csv.DictReader(open(TP))]
    tpok = [r for r in tp if r["status"] == "ok"]
    tptime = defaultdict(list); tpst = defaultdict(list); tpbcc = {}
    for r in tpok:
        k = (r["graph"], int(r["bs"]))
        if r["computeBCG_ms"]: tptime[k].append(int(r["computeBCG_ms"]))
        if r["st_ms"]:         tpst[k].append(int(r["st_ms"]))
        tpbcc.setdefault(k, r["bcc"])

    print()
    print("=" * 96)
    print("TABLE 3  two-pass (Euro-Par 2025) vs single-pass (ours).  computeBCG, median of reps (ms).")
    print("         Both numbers bracket the whole algorithm: the baseline's includes its pass-1")
    print("         spanning tree (the outer `dur` is still live at the print), ours has no pass 1.")
    print("=" * 96)
    print(f"{'graph':<22}{'bs':>9}  {'2-pass':>8} {'of which':>9} {'ours':>8}  {'speedup':>8} {'gain':>7}  bcc")
    print(f"{'':<22}{'':>9}  {'':>8} {'pass-1':>9} {'':>8}  {'':>8} {'':>7}")
    sp = []; cnt_ok = cnt_bad = 0
    for g, bs in keys:
        t2 = tptime.get((g, bs)); t1 = times.get((g, bs, "default"))
        if not (t2 and t1):
            continue
        m2, m1 = st.median(t2), st.median(t1)
        stm = st.median(tpst[(g, bs)]) if tpst.get((g, bs)) else 0
        gain = (m2 - m1) / m2 * 100 if m2 else 0.0
        sp.append(m2 / m1 if m1 else 0)
        ours_bcc = struct[(g, bs, "default")][0]
        agree = tpbcc.get((g, bs)) == ours_bcc
        cnt_ok += agree; cnt_bad += (not agree)
        print(f"{g:<22}{bs:>9}  {m2:>8.0f} {stm:>9.0f} {m1:>8.0f}  "
              f"{m2/m1 if m1 else 0:>7.2f}x {gain:>6.1f}%  "
              f"{ours_bcc if agree else f'MISMATCH 2p={tpbcc.get((g,bs))} ours={ours_bcc}'}")
    if sp:
        print(f"\n  {len(sp)} configs: median speedup {st.median(sp):.2f}x, "
              f"range {min(sp):.2f}x .. {max(sp):.2f}x")
        print(f"  BCC count agrees with the two-pass baseline on {cnt_ok}/{cnt_ok+cnt_bad} configs"
              + ("" if not cnt_bad else f"  -- {cnt_bad} MISMATCH, see above"))
    tpbad = [r for r in tp if r["status"] != "ok"]
    if tpbad:
        seen = defaultdict(list)
        for r in tpbad: seen[(r["graph"], r["bs"], r["status"])].append(r["rep"])
        print("\n  baseline not ok:")
        for (g, bs, s), reps in sorted(seen.items()):
            print(f"    {g:<22} bs={bs:<9} {s:<8} reps {','.join(reps)}")

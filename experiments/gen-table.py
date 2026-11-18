#!/usr/bin/env python3
"""Regenerate paper/results.tex from the sweep CSVs.

    python3 experiments/gen-table.py

Speedup = median two-pass computeBCG / median ours (default mode), per
(graph, batch size). Cells are only emitted when BOTH arms have an ok run;
everything else gets a marker explaining why, so a gap is never silently a
blank. Re-run whenever the sweep advances -- never hand-edit results.tex.
"""
import csv, json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES  = os.path.join(HERE, "results")
OUT  = os.path.join(HERE, "..", "paper", "results.tex")
SIZES= json.load(open(os.path.join(RES, "graph-sizes.json")))

BSS = [8000, 16000, 32000, 64000, 125000, 250000, 500000,
       1000000, 2000000, 4000000, 8000000]
HDR = ["8k", "16k", "32k", "64k", "125k", "250k", "500k", "1M", "2M", "4M", "8M"]

CATS = [
    ("Web", ["uk-2002", "indochina-2004", "in-2004", "eu-2005", "web-BerkStan"]),
    ("Social", ["soc-LiveJournal1", "ljournal-2008", "com-Orkut",
                "hollywood-2009", "wiki-topcats", "sx-stackoverflow",
                "higgs-twitter", "as-Skitter"]),
    ("Road", ["europe_osm", "road_usa", "GAP-road", "germany_osm"]),
    ("Citation \\& bio", ["coPapersDBLP", "coPapersCiteseer",
                          "kmer_U1a", "kmer_V2a"]),
    ("Synthetic", ["rgg_n_2_24_s0", "rgg_n_2_23_s0", "delaunay_n24",
                   "hugebubbles-00020", "kron_g500-logn21",
                   "kron_g500-logn20", "kron_g500-logn19",
                   "kron_g500-logn18"]),
]


def load(path, mode=None):
    ok, status = {}, {}
    if not os.path.exists(path):
        return ok, status
    for r in csv.DictReader(open(path)):
        if mode is not None and r.get("mode") != mode:
            continue
        k = (r["graph"], int(r["bs"]))
        status.setdefault(k, []).append(r["status"])
        if r["status"] == "ok" and r.get("computeBCG_ms"):
            ok.setdefault(k, []).append(float(r["computeBCG_ms"]))
    return ok, status


ours, ours_st = load(os.path.join(RES, "bssweep.csv"), "default")
two,  two_st  = load(os.path.join(RES, "bssweep-twopass.csv"))


def cell(g, b):
    k = (g, b)
    if k in ours and k in two:
        return f"{st.median(two[k]) / st.median(ours[k]):.2f}"
    o, t = set(ours_st.get(k, [])), set(two_st.get(k, []))
    if "oom" in o:
        return r"\emph{m}"
    # A cell the sweep deliberately did not run because the baseline totally
    # hung at a LARGER b. Marked h, never t: t is measured here, h is inferred
    # from this graph's own larger-b hang. Kept separate so the table never
    # shows an untested cell as a measurement.
    # An inferred hang is NOT reported. The sweep skips a cell when the graph
    # hung at a larger b, and those rows stay in the csv as provenance, but the
    # table prints only what was measured here. The premise behind the
    # inference does not fully hold -- rgg_n_2_24_s0 hangs at b=1M yet succeeds
    # at 500k, 250k and 125k -- every such skip traced to a single timed-out
    # rep, and this baseline is nondeterministic in 21% of its cells, so a
    # one-rep verdict is too thin to print as a result. Such cells read "--".
    t = {x for x in t if not x.startswith("infer-")}
    if t and "ok" not in t:
        # Report the most explanatory failure. A capacity abort is
        # deterministic and fully explains the cell, so it outranks a fault,
        # which outranks a timeout (which explains nothing).
        if "cap" in t:
            return r"\emph{a}"
        if "fault" in t:
            return r"\emph{f}"
        if "crash" in t:
            return r"\emph{c}"
        return r"\emph{t}"
    return "--"


rows, n_done, n_cells = [], 0, 0
for cat, graphs in CATS:
    rows.append(r"\multicolumn{%d}{l}{\emph{%s}} \\" % (len(BSS) + 2, cat))
    for g in graphs:
        cs = [cell(g, b) for b in BSS]
        n_cells += len(cs)
        n_done += sum(1 for c in cs if c[0].isdigit())
        m = SIZES.get(g, {}).get("Eund")
        msz = f"{m/1e6:.1f}" if m else "--"
        rows.append("~~" + g.replace("_", r"\_") + " & " + msz + " & "
                    + " & ".join(cs) + r" \\")

body = "\n".join(rows)

# Only gloss the markers that actually occur. After the allocation was cut
# from 5b+6n to 3b+1.5n our arm stopped running out of memory anywhere in
# the grid, so the static legend was left defining \emph{m} -- a symbol no
# cell uses any more, which reads as a failure mode being hidden. Driven off
# the rendered body so caption and table cannot drift apart.
LEGEND = [
    (r"\emph{a}", r"\emph{a}: the baseline exceeded its own allocation budget."),
    (r"\emph{f}", r"\emph{f}: the baseline faulted."),
    (r"\emph{t}", r"\emph{t}: the baseline exceeded its timeout."),
    (r"\emph{m}", r"\emph{m}: our implementation exhausted device memory."),
    ("--", "--: not measured (either not run, or skipped because the baseline "
           "hung at a\nlarger $b$ for the same graph)."),
]
legend = "\n".join(txt for mark, txt in LEGEND if mark in body)
tex = r"""%% GENERATED by experiments/gen-table.py -- do not hand-edit.
%% %d of %d cells filled.
\begin{table*}[t]
\centering
\caption{Speedup of Single-Pass \textsc{gem-bcc} over two-pass
\textsc{gem-bcc}~\cite{europar25}, by batch size $b$. Each entry is the ratio
of median \textsc{computeBCG} times; larger is better, and below $1.00$ we are
slower. $|E|$ is the undirected edge count.
%s}
\label{tab:speedup}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lr%s}
\toprule
 & & \multicolumn{%d}{c}{batch size $b$} \\
\cmidrule(l){3-%d}
Dataset & $|E|$\,(M) & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table*}
""" % (n_done, n_cells, legend, "r" * len(BSS), len(BSS), len(BSS) + 2,
       " & ".join(HDR), body)

open(OUT, "w").write(tex)
print(f"wrote {OUT}: {n_done}/{n_cells} cells filled")

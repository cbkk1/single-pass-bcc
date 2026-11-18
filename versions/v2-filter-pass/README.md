# v2-filter-pass — the per-round filter as a separate phase

Snapshot taken 2026-11-15, immediately before the block-aware fence test
(`Fast-BCC+`) became the default path.

In this version Compress-Graph carries **Step D'**, the filter: after beta
exists, union every beta class incident to a node `< n_b_prev`, then re-derive
edge counts, anchors and the cut set from the merged labels. Fast_BCC's step 4
is the stock Fast_BCC fence test, unmodified.

Replaced by: never fence a tree edge whose **parent** is a carried block
(`par(c) < n_b`), evaluated inside Fast_BCC's union-find. Same result, one
phase fewer, 4-15% faster. See `evidence/fence-keep-down-replaces-filter.sh`.

The filter path is still reachable in the current tree with
`BCC_FENCE_KEEP=0` (or `BCC_NOFILTER=1` for the no-filter baseline), so this
snapshot is for reference and for reconstructing the two-phase timings.

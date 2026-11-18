# Single-Pass GEM-BCC

Biconnected components of graphs larger than device memory, on one GPU, in a
**single streaming pass** over the edge list.

This is a fork of the Euro-Par 2025 artifact
([Abhijeetkumar96/External-BCC-for-GPU](https://github.com/Abhijeetkumar96/External-BCC-for-GPU),
"GEM-BCC"), which reads the stream **twice**: pass 1 builds a global spanning
tree, pass 2 computes the BCCs. This version deletes pass 1. The spanning
forest is maintained incrementally inside the single pass, in the
Biconnectivity Compressed Graph `H`, so the stream is touched once.

The win is not the saved stream pass — that is only 2–7 ms. It is that each
round's work is bounded by `H`'s size instead of the whole graph.

## Build

```bash
mkdir -p build && cd build && cmake .. && make -j$(nproc)
```

Needs CUDA ≥ 11.0, compute capability ≥ 6.0, CMake ≥ 3.10, GCC ≥ 7.

## Run

```bash
# single-pass path (this work)
./build/External_BCC -i graph.egr -a ext-bcc -g -b 500000

# add -c to verify against an exact serial Tarjan on the same input
./build/External_BCC -i graph.egr -a ext-bcc -g -b 500000 -c
```

`-g` selects dense ids + GPU `Compress-Graph` (this work). Without it the
binary falls back to split id ranges and a host Tarjan for the BCC step.
`-b` is the batch size in edges. Formats: `.txt`/`.edges`/`.eg2` text edge
list, `.egr`/`.bin`/`.csr` binary CSR.

## Layout

| path | what |
|---|---|
| `src/extern_bcc/` | the single pass: `single_pass.cu`, `gpu_compress.cu`, `lch.cu`, allocation in `bcg_memory_utils.cu` |
| `src/fast_bcc/` | `fast-bcc+`: Euler tour, sparse-table RMQ, skeleton CC |
| `experiments/` | sweep harness and the result CSVs behind Table 1 |
| `evidence/` | one script per design claim, each reproducing the counter-example that settled it |
| `paper/` | `algorithms.tex`, generated `results.tex`, standalone builds |
| `versions/` | superseded designs, kept for the record |

## Results

Speedup over the two-pass baseline, median `computeBCG` ratio, 29 graphs ×
11 batch sizes (8k…8M). Regenerate with `python3 experiments/gen-table.py`.

- **196 of 319 cells measured**: median **1.37×**, range 0.64–5.03, ahead in
  **88%** (172/196).
- Best: `rgg_n_2_24_s0` 5.03, `rgg_n_2_23_s0` 4.18, `kmer_U1a` 2.93.
- The two classes move in **opposite** directions: sparse graphs (road, rgg,
  kmer) gain as `b` falls; dense ones (kron, social) gain as `b` rises.
- The remaining cells are baseline failures, not gaps in ours: 72 where it
  exceeded its own allocation budget, 35 timeouts, 4 CUDA faults.

**If you want one batch size, use `-b 500000`.** It is the only size that
maximises both coverage and median speedup at once (22 of 29 graphs, median
1.51, geomean 1.62, ahead on 18/22), and the only one where no cell is a
timeout or a fault -- every baseline failure there is its own allocation cap,
which is the clean capacity comparison. 13-524 batches per graph, our peak
memory 1.0-25.4 GB of a 45.5 GB card. Fixing it costs 16% against tuning `b`
per graph.

A rule of the form `b = alpha * |E|` is worse at every alpha tried
(0.5-10%, best median 1.47): the direction of the `b` trend is set by density
`|E|/|V|`, not by `|E|`, so a rule keyed on edge count moves the sparse and
dense graphs the same way when they want opposite things. Sparse graphs
(`|E|/|V|` below ~20) speed up as `b` falls; dense ones speed up as `b` rises.

Cutting `max_allot` from `5b+6n` to `3b+1.5n` and `V_fb` from `2n` to `1.0n`
(the old coefficients were sized for an aligned-pair id scheme that no longer
exists) saved **49%** of device memory — 496 GB → 252 GB across the suite,
runtime unchanged, and verified exact against serial Tarjan on 58/58
graph×batch configurations. After that our implementation does not run out of
memory anywhere in the grid.

## Caveats

- **The baseline is nondeterministic.** 21% of the cells where it ever
  succeeded also crashed or hung on another attempt; several cells produced one
  honest run and two timeouts on identical input. Single-replicate verdicts
  about it are not reliable.
- **The 4 `fault` cells are not yet attributed.** They may be this build rather
  than the baseline's design: `fast_bcc.cu` reads `d_a1`/`d_a2` uninitialised,
  which this fork fixes with `init_a`. Until that is settled those cells cannot
  be cited as Euro-Par's behaviour.
- Harness scripts carry absolute paths for the machine they were run on
  (`/mnt/hdd/.../datasets`); edit the `MED`/`LRG` variables at the top.

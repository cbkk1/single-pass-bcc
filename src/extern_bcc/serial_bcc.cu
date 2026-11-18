//---------------------------------------------------------------------
// Single-pass BCC, phase 4 -- host-side biconnected components.
//---------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <vector>

#include "extern_bcc/serial_bcc.cuh"

namespace {

struct Frame {
    int v;    // vertex
    int pe;   // edge id we entered v through (-1 at a root)
    int it;   // cursor into v's adjacency slice
};

// The Tarjan core, shared by the phase-4 entry point and the reference
// counter, so the two can never drift apart.
//
// Fills eu/ev (endpoints by edge id), off (CSR offsets), iscut, and
// bcc_of_edge / bcc_edges (which component each edge landed in, and how many
// edges each component holds). Returns the number of biconnected components;
// isolated nodes are skipped and contribute none.
int tarjan_bcc(const uint64_t* edges, long m, int num_nodes,
               std::vector<int>& eu, std::vector<int>& ev,
               std::vector<int>& off,
               std::vector<char>& iscut,
               std::vector<int>& bcc_of_edge,
               std::vector<long>& bcc_edges) {

    iscut.assign(num_nodes, 0);
    off.assign(num_nodes + 1, 0);
    eu.resize(m); ev.resize(m);
    bcc_of_edge.assign(m, -1);
    bcc_edges.clear();

    if (m == 0) return 0;

    // ---- CSR adjacency, carrying edge ids so the parent edge can be skipped
    // exactly once (skipping by parent *vertex* would drop genuine parallel
    // edges, and mis-handle a self-loop).
    for (long i = 0; i < m; ++i) {
        int u = (int)(edges[i] >> 32);
        int v = (int)(edges[i] & 0xFFFFFFFF);
        eu[i] = u; ev[i] = v;
        off[u + 1]++; off[v + 1]++;
    }
    for (int i = 0; i < num_nodes; ++i) off[i + 1] += off[i];

    std::vector<int> cur(off.begin(), off.begin() + num_nodes);
    std::vector<int> adj_v(2 * m), adj_e(2 * m);
    for (long i = 0; i < m; ++i) {
        adj_v[cur[eu[i]]] = ev[i]; adj_e[cur[eu[i]]] = (int)i; cur[eu[i]]++;
        adj_v[cur[ev[i]]] = eu[i]; adj_e[cur[ev[i]]] = (int)i; cur[ev[i]]++;
    }

    // ---- iterative Tarjan
    std::vector<int>  disc(num_nodes, 0), low(num_nodes, 0);
    std::vector<char> visited(num_nodes, 0);
    std::vector<int>  estack;
    std::vector<Frame> st;
    estack.reserve(m);

    int timer = 0, num_bcc = 0;

    for (int r = 0; r < num_nodes; ++r) {
        if (visited[r] || off[r] == off[r + 1]) continue;   // unseen, non-isolated

        visited[r] = 1;
        disc[r] = low[r] = ++timer;
        st.push_back({r, -1, off[r]});
        int root_children = 0;

        while (!st.empty()) {
            // Do not hold a reference across push_back -- it would dangle.
            const int v = st.back().v;

            if (st.back().it < off[v + 1]) {
                const int idx = st.back().it++;
                const int w   = adj_v[idx];
                const int eid = adj_e[idx];

                if (eid == st.back().pe) continue;   // the edge we came in on

                if (!visited[w]) {
                    estack.push_back(eid);
                    visited[w] = 1;
                    disc[w] = low[w] = ++timer;
                    st.push_back({w, eid, off[w]});
                } else if (disc[w] < disc[v]) {
                    estack.push_back(eid);
                    if (disc[w] < low[v]) low[v] = disc[w];
                }
            } else {
                const int pe = st.back().pe;
                st.pop_back();
                if (st.empty()) break;

                const int p = st.back().v;
                if (low[v] < low[p]) low[p] = low[v];

                if (low[v] >= disc[p]) {
                    // p separates v's subtree from the rest: close a component.
                    const int id = num_bcc++;
                    long cnt = 0;
                    while (!estack.empty()) {
                        const int e = estack.back();
                        estack.pop_back();
                        bcc_of_edge[e] = id;
                        cnt++;
                        if (e == pe) break;
                    }
                    bcc_edges.push_back(cnt);

                    // The root is a cut vertex only if it has >= 2 children.
                    if (st.size() > 1) iscut[p] = 1;
                    else root_children++;
                }
            }
        }
        if (root_children >= 2) iscut[r] = 1;
    }

    return num_bcc;
}

} // namespace

int serial_bcc(const std::vector<uint64_t>& edges,
               int n, int num_nodes,
               const std::vector<char>& live,
               std::vector<int>& I, std::vector<char>& iscut,
               std::vector<uint64_t>* out_edges) {

    const long m = (long)edges.size();

    // Default: every node is its own supernode. Isolated nodes keep this.
    I.resize(num_nodes);
    for (int i = 0; i < num_nodes; ++i) I[i] = i;
    if (out_edges) out_edges->clear();

    std::vector<int>  eu, ev, off, bcc_of_edge;
    std::vector<long> bcc_edges;

    const int num_bcc = tarjan_bcc(edges.data(), m, num_nodes,
                                   eu, ev, off, iscut, bcc_of_edge, bcc_edges);
    if (m == 0) return 0;

    // ---- allocate block ids
    // EVERY component with >= 2 edges gets one, even one whose members are all
    // cut vertices and which therefore absorbs nobody. Skipping those was a
    // divergence from the prototype (core_fix.hpp mints `representative` for
    // every comp.size() > 1): a block with no id is invisible in the rebuilt
    // BCG, and its edges get mistaken for bridges.
    std::vector<char> compresses(num_bcc, 0);
    for (int b = 0; b < num_bcc; ++b) compresses[b] = (bcc_edges[b] >= 2);

    // Which component swallows each node (-1 = the node survives as itself).
    std::vector<int> absorbed_into(num_nodes, -1);
    for (long i = 0; i < m; ++i) {
        const int b = bcc_of_edge[i];
        if (b < 0 || !compresses[b]) continue;
        if (!iscut[eu[i]]) absorbed_into[eu[i]] = b;
        if (!iscut[ev[i]]) absorbed_into[ev[i]] = b;
    }

    // A live block that is NOT absorbed keeps its id, so that id is still taken
    // and a new block must skip over it. Getting this wrong silently fuses two
    // unrelated blocks that happen to draw the same number.
    std::vector<char> taken(num_nodes, 0);
    for (int s = n; s < num_nodes; ++s)
        if (s < (int)live.size() && live[s] && absorbed_into[s] < 0) taken[s] = 1;

    // A block takes TWO ids, allocated as an aligned pair, so that (id - n)
    // being even identifies the primary. Residents map to the primary, and the
    // block count keys on primaries.
    //
    // Two nodes, not one, because a single node cannot express a 2-connection.
    // A block with one attachment point c would emit the lone edge (B, c) -- a
    // bridge -- although c lies inside the block and is 2-connected to the rest
    // of it. That makes B a false articulation point and strands its residents.
    // With a pair, B-B'-c is a triangle, which is 2-connected and correct.
    std::vector<int> block_id(num_bcc, -1);
    int next = n, nblocks = 0;
    for (int b = 0; b < num_bcc; ++b) {
        if (!compresses[b]) continue;
        while (next + 1 < num_nodes && (taken[next] || taken[next + 1])) next += 2;
        assert(next + 1 < num_nodes);      // blocks are edge-disjoint and hold
        block_id[b] = next;                // >= 3 vertices each, so <= (n-1)/2
        next += 2;
        nblocks++;
    }

    for (int s = 0; s < num_nodes; ++s)
        if (absorbed_into[s] >= 0) I[s] = block_id[absorbed_into[s]];

    if (!out_edges) return nblocks;

    // ---- rebuild the block-cut graph for the next round
    // Built from scratch: a bridge is kept (mapped through I), and a compressed
    // block becomes its node pair joined by an edge, with each of its cut
    // vertices joined to BOTH halves. Every edge internal to a block disappears
    // -- which relabelling in place could not do, because an internal edge
    // between two cut vertices maps to itself and survives looking like a
    // bridge.
    //
    // Joining a cut vertex to both halves is what preserves 2-connectivity:
    // c-B-B'-c is a triangle, so no single node separates c from the block, and
    // two cut vertices c1, c2 are 2-connected through it without needing an
    // edge between them.
    //
    // Edges are walked grouped by component so one `mark` slot per node dedups
    // (block, cut vertex) exactly.
    std::vector<int> start(num_bcc + 1, 0);
    for (long i = 0; i < m; ++i)
        if (bcc_of_edge[i] >= 0) start[bcc_of_edge[i] + 1]++;
    for (int b = 0; b < num_bcc; ++b) start[b + 1] += start[b];

    std::vector<int> order(start[num_bcc]);
    {
        std::vector<int> pos(start.begin(), start.end() - 1);
        for (long i = 0; i < m; ++i)
            if (bcc_of_edge[i] >= 0) order[pos[bcc_of_edge[i]]++] = (int)i;
    }

    auto emit = [&](int a, int c) {
        out_edges->push_back(((uint64_t)std::min(a, c) << 32) | (uint32_t)std::max(a, c));
    };

    std::vector<int> mark(num_nodes, -1);
    for (int b = 0; b < num_bcc; ++b) {
        if (!compresses[b]) {                       // a bridge: keep it
            for (int k = start[b]; k < start[b + 1]; ++k) {
                const int i = order[k];
                emit(I[eu[i]], I[ev[i]]);
            }
            continue;
        }

        const int p = block_id[b], s = p + 1;
        emit(p, s);

        for (int k = start[b]; k < start[b + 1]; ++k) {
            const int i = order[k];
            for (int x : {eu[i], ev[i]}) {
                if (!iscut[x] || mark[x] == b) continue;  // absorbed, or already attached
                mark[x] = b;
                emit(x, p);
                emit(x, s);
            }
        }
    }

    return nblocks;
}

long bcg_bridge_count(const uint64_t* edges, long m, int n, int num_nodes) {

    std::vector<int>  eu, ev, off, bcc_of_edge;
    std::vector<long> bcc_edges;
    std::vector<char> iscut;

    tarjan_bcc(edges, m, num_nodes, eu, ev, off, iscut, bcc_of_edge, bcc_edges);

    // A bridge of the original graph stays a cut edge of the block-cut graph,
    // so it is a component of exactly one edge, and it never touches a block:
    // a block member lying on a bridge is by definition a cut vertex, so it is
    // not absorbed and the bridge keeps its own endpoint id.
    //
    // Both halves of that test are needed. Dropping "one edge" would also count
    // the ring edges between two cut vertices of one block; dropping "no block
    // endpoint" would also count a link from a block to its single cut vertex,
    // where the ring degenerates to one edge.
    long bridges = 0;
    for (long i = 0; i < m; ++i) {
        const int b = bcc_of_edge[i];
        if (b < 0 || bcc_edges[b] != 1) continue;
        if (eu[i] < n && ev[i] < n) bridges++;
    }
    return bridges;
}

long serial_bcc_count(const uint64_t* edges, long m, int n,
                      long* out_bridges, long* out_isolated) {

    std::vector<int>  eu, ev, off, bcc_of_edge;
    std::vector<long> bcc_edges;
    std::vector<char> iscut;

    const long num_bcc = tarjan_bcc(edges, m, n,
                                    eu, ev, off, iscut, bcc_of_edge, bcc_edges);

    // A single-edge component is a bridge; anything larger is 2-connected.
    if (out_bridges) {
        long bridges = 0;
        for (long c : bcc_edges) if (c < 2) bridges++;
        *out_bridges = bridges;
    }

    // Reported separately rather than folded in: whether a lone vertex counts
    // as a trivial BCC is a convention, and the recursive reference this was
    // validated against enumerates edges, so it counts none.
    if (out_isolated) {
        long iso = 0;
        for (int i = 0; i < n; ++i) if (off[i] == off[i + 1]) iso++;
        *out_isolated = iso;
    }

    return num_bcc;
}

// Reference edge->block partition and cut set, for differential testing of the
// GPU Compress-Graph. Shares tarjan_bcc with everything else here, so the two
// cannot drift.
int serial_bcc_partition(const uint64_t* edges, long m, int num_nodes,
                         std::vector<int>& bcc_of_edge,
                         std::vector<char>& iscut,
                         std::vector<long>& bcc_edges) {
    std::vector<int> eu, ev, off;
    return tarjan_bcc(edges, m, num_nodes, eu, ev, off, iscut, bcc_of_edge, bcc_edges);
}

//---------------------------------------------------------------------
// LCH readout. Contract and reasoning in lch.cuh.
//---------------------------------------------------------------------

#include <algorithm>
#include <cassert>
#include <map>
#include <vector>

#include "extern_bcc/lch.cuh"

namespace {

// A block owns ONE id, in [0, n_b).
inline bool is_block(int x, int n_b) { return x < n_b; }

// ... and is its own component.
inline int block_of(int x) { return x; }

}  // namespace

int compute_lch(const std::vector<int>& M, int n,
                const std::vector<uint64_t>& edges, int n_s, int n_b,
                std::vector<int>& label, std::vector<int>& head) {

    const int two_nb = n_b;   // one id per block
    const long m = static_cast<long>(edges.size());

    // ---- which original vertex lives in each node -------------------------
    // A node that keeps its own identity holds exactly one original vertex:
    // residents only accumulate through absorption into a block, and a block
    // primary is never given an own id. So this inverse is well defined
    // wherever we read it. Block halves get many writers and we never read
    // them, so no guard is needed.
    std::vector<int> resident(n_s, -1);
    for (int v = 0; v < n; ++v)
        if (M[v] >= 0 && M[v] < n_s) resident[M[v]] = v;

    // ---- adjacency of the final block-cut forest --------------------------
    std::vector<int> deg(n_s + 1, 0);
    for (uint64_t e : edges) {
        ++deg[(int)(e >> 32)];
        ++deg[(int)(e & 0xFFFFFFFF)];
    }
    std::vector<int> off(n_s + 1, 0);
    for (int i = 0; i < n_s; ++i) off[i + 1] = off[i] + deg[i];
    std::vector<int> adj(2 * m), fill(off.begin(), off.end());
    for (uint64_t e : edges) {
        int a = (int)(e >> 32), b = (int)(e & 0xFFFFFFFF);
        adj[fill[a]++] = b;
        adj[fill[b]++] = a;
    }

    // ---- root every tree of the forest ------------------------------------
    // Iterative BFS: the forest can be a long path (a road network is close to
    // one), and recursion would overflow.
    std::vector<int> parent(n_s, -1);
    std::vector<char> seen(n_s, 0);
    std::vector<int> queue;
    queue.reserve(n_s);
    for (int r = 0; r < n_s; ++r) {
        if (seen[r]) continue;
        seen[r] = 1;
        queue.clear();
        queue.push_back(r);
        for (size_t qi = 0; qi < queue.size(); ++qi) {
            int x = queue[qi];
            for (int k = off[x]; k < off[x + 1]; ++k) {
                int y = adj[k];
                if (seen[y]) continue;
                seen[y] = 1;
                parent[y] = x;
                queue.push_back(y);
            }
        }
    }

    // ---- component ids ----------------------------------------------------
    // A block is one component, numbered by its id. A bridge -- an edge
    // between two own-nodes -- is a component in its own right, numbered after
    // the blocks.
    int n_comp = n_b;
    std::map<std::pair<int, int>, int> bridge_id;
    for (uint64_t e : edges) {
        int a = (int)(e >> 32), b = (int)(e & 0xFFFFFFFF);
        if (is_block(a, two_nb) || is_block(b, two_nb)) continue;
        auto key = std::make_pair(a, b);
        if (!bridge_id.count(key)) bridge_id[key] = n_comp++;
    }

    // ---- the component each node is LABELLED by ---------------------------
    // Uniformly: the component holding the node's parent edge.
    //
    //   a block half        -> its own block (its parent edge is an attachment
    //                          to that block, and if it is a tree root the
    //                          answer is the same)
    //   an own node         -> the parent edge is either an attachment to a
    //                          block, or a bridge
    //   a tree root own-node-> nothing. It sits above everything it touches,
    //                          so it is only ever a head.
    std::vector<int> comp_of(n_s, -1);
    for (int x = 0; x < n_s; ++x) {
        if (is_block(x, two_nb)) { comp_of[x] = block_of(x); continue; }
        int p = parent[x];
        if (p < 0) continue;                                   // tree root
        if (is_block(p, two_nb)) { comp_of[x] = block_of(p); continue; }
        int a = std::min(p, x), b = std::max(p, x);
        auto it = bridge_id.find(std::make_pair(a, b));
        if (it != bridge_id.end()) comp_of[x] = it->second;
    }

    // ---- label[] over ORIGINAL vertices ------------------------------------
    label.assign(n, -1);
    for (int v = 0; v < n; ++v)
        if (M[v] >= 0 && M[v] < n_s) label[v] = comp_of[M[v]];

    // ---- head[] ------------------------------------------------------------
    // The head of a component is the vertex sitting directly above it. For a
    // block that is whichever half is topmost -- its parent is an own-node, or
    // it is a tree root and the component has no head at all.
    head.assign(n_comp, -1);
    for (int p = 0; p < two_nb; ++p) {
        int pa = parent[p];
        if (pa >= 0 && !is_block(pa, two_nb)) head[block_of(p)] = resident[pa];
    }
    for (auto& [key, j] : bridge_id) {
        // The child of the two carries the label; the parent is the head.
        int a = key.first, b = key.second;
        if (parent[b] == a)      head[j] = resident[a];
        else if (parent[a] == b) head[j] = resident[b];
        // else: the edge is not a tree edge of the BFS forest, which cannot
        // happen for a bridge in a forest -- but leave head = -1 rather than
        // assert, so a malformed input degrades instead of aborting.
    }

    return n_comp;
}

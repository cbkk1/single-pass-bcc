# Hand-trace worksheet: why the filter cannot be deferred

Graph: 12 vertices, 12 edges. One edge per batch.

    9-cycle  : 0-1-8-7-6-5-2-3-4-0
    pendants : 1-9, 6-10, 4-11
    truth    : 4 BCCs = {0..8}, {1,9}, {6,10}, {4,11}

## Input, batch by batch

| batch | edge | note |
|-------|------|------|
| @0  | 0-1   | |
| @1  | 2-3   | |
| @2  | 4-3   | |
| @3  | 4-0   | |
| @4  | 5-6   | |
| @5  | 6-7   | |
| @6  | 8-1   | |
| @7  | 8-7   | |
| @8  | 2-5   | closes the 9-cycle -> first compression |
| @9  | 9-1   | |
| @10 | 10-6  | **the round that decides everything** |
| @11 | 4-11  | |

## Answers (filter every round -- correct)

    @0   0-1
    @1   0-1  2-3
    @2   0-1  2-3  3-4
    @3   0-1  0-4  2-3  3-4
    @4   + 5-6
    @5   + 6-7
    @6   + 1-8
    @7   + 7-8
    @8   n_b=1, B0 <- {0,1,2,3,4,5,6,7,8}, no edges left
    @9   B0{0,2,3,4,5,6,7,8}--1      9--1                        (1 de-merges)
    @10  B0{0,2,3,4,5,7,8}--1   --6  9--1  10--6                 (6 de-merges)
    @11  B0{0,2,3,5,7,8}--1  --6  --4  9--1  10--6  11--4        (4 de-merges)

Read-off: residents(B0) u neighbours(B0) = {0,2,3,5,7,8} u {1,6,4} = {0..8},
plus the three own-own edges 9-1, 10-6, 11-4.  = 4 BCCs.

## Answers (filter only on the last batch, pred=dis -- wrong)

Identical through @9.  The graph handed to the compression step at @10 is the
TREE   9--1--B0--6--10  : four edges, all four of them bridges.

A beta class becomes a BLOCK if it has >= 2 edges OR it touches a carried
block node (d_blk_struct).  The second clause is what lets a carried block
survive a round in which its class holds a single edge -- and it is why the
classes at a carried block MUST be merged before the decision is taken.

    filter OFF at @10                        filter ON at @10
    (9,1)  |E|=1 struct=0 -> no              (9,1)  |E|=1 struct=0 -> no
    (10,6) |E|=1 struct=0 -> no              (10,6) |E|=1 struct=0 -> no
    (B0,1) |E|=1 struct=1 -> BLOCK           (B0,1)+(B0,6)
    (B0,6) |E|=1 struct=1 -> BLOCK                  |E|=2 struct=1 -> BLOCK
      two BRIDGES become two blocks;           one block, residents kept
      B0 is then in two blocks, so it
      reads as a cut vertex and is
      demoted to an OWN-NODE:

    @10  n_b=2, both blocks empty, the real BCC is now an own-node:
         B0{}--O2{0,2,3,4,5,7,8}  B0{}--1  B1{}--O2{0,2,3,4,5,7,8}  B1{}--6
         9--1  10--6
    @11  O2 has 7 residents, so |f^-1| != 1 makes it "false" and the final
         filter unions everything at it, 11's class included:
         B0{0,2,3,4,5,7,8,11}--1  --6   9--1  10--6     = 3 BCCs

Note: a hand trace CANNOT reproduce this.  By hand one says "all four edges
are bridges, B0 is already a BCC so it stays one node" -- which is exactly
what the filter enforces.  The filter is not a relabelling that can be
deferred: it runs BEFORE the per-round, irreversible block-vs-own-node
decision.

## Reproduce

    ./tools/build-case.sh                       # builds build/case
    E="0 1  2 3  4 3  4 0  5 6  6 7  8 1  8 7  2 5  9 1  10 6  4 11"
    BCC_NAMES=1 BCC_SHOW=1 ./build/case 12 12 $E                       # correct
    BCC_PRED=dis BCC_FILTER_LAST=1 BCC_NAMES=1 BCC_SHOW=1 ./build/case 12 12 $E
    for f in 9 10 11; do BCC_PRED=dis BCC_FILTER_FROM=$f ./build/case 12 12 $E | grep 'count'; done

    # BCC_BETA=1 shows, per round, why each beta class was or was not made a block
    BCC_BETA=1 BCC_NAMES=1 BCC_SHOW=1 ./build/case 12 12 $E

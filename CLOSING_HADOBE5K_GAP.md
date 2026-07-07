# Samsung Pro Test — Practice Questions (with online judges & test cases)

For each problem in this repo, the closest online problem(s) you can practice on with real test cases.
Legend: 🟢 easy · 🟡 medium · 🔴 hard · ⭐ = closest/best match.

---

## Tier 1 — Dynamic Programming (do these first)

### Subset-sum / min partition difference
Files: `min_subset_diff.cpp`, `Min_Subset_Diff(2025_Test_2).cpp`
- ⭐ [LeetCode 2035 — Partition Array Into Two Arrays to Minimize Sum Difference](https://leetcode.com/problems/partition-array-into-two-arrays-to-minimize-sum-difference/) 🔴 (exact bitset/meet-in-middle version)
- ⭐ [GFG — Minimum sum partition](https://www.geeksforgeeks.org/problems/minimum-sum-partition3317/1) 🟡 (classic DP version)
- [LeetCode 1049 — Last Stone Weight II](https://leetcode.com/problems/last-stone-weight-ii/) 🟡 (same problem, reworded)
- [LeetCode 416 — Partition Equal Subset Sum](https://leetcode.com/problems/partition-equal-subset-sum/) 🟡 (warm-up)

### Digit DP — count numbers with digit sum S
File: `7_2_digit_sum.cpp`
- ⭐ [GFG — Count numbers ≤ N with given digit sum](https://www.geeksforgeeks.org/dsa/count-numbers-smaller-than-or-equal-to-n-with-given-digit-sum/) 🟡 (nearly identical)
- [LeetCode 2719 — Count of Integers](https://leetcode.com/problems/count-of-integers/) 🔴 (digit sum in a range — same technique)
- [LeetCode 902 — Numbers At Most N Given Digit Set](https://leetcode.com/problems/numbers-at-most-n-given-digit-set/) 🔴 (digit DP + tight bound)
- Practice more: [Codeforces — Digit DP tutorial + problem list](https://codeforces.com/blog/entry/53960)

### Interval / linear DP — stone removal cost by neighbours
File: `4_3_stones.cpp`
- ⭐ [Codeforces blog — the original problem/discussion](https://codeforces.com/blog/entry/117311) (approach by vgtcross, see comments)
- Similar drill: [LeetCode 1000 — Minimum Cost to Merge Stones](https://leetcode.com/problems/minimum-cost-to-merge-stones/) 🔴

### 2D grid DP — robot garbage cleaning
File: `7_1_robot.cpp`
- Samsung-original DP (no exact LC). Closest practice for the "deploy vs. carry cost" DP pattern:
  [LeetCode 1553 — Minimum Number of Days to Eat N Oranges](https://leetcode.com/problems/minimum-number-of-days-to-eat-n-oranges/) 🔴 (decision DP)

### State-machine DP — string merge (first char == last char)
File: `5_1_strings.cpp`
- Samsung-original DP over digit-start/end states. No exact judge match — drill it directly from this repo (write your own test cases). Conceptual cousin: [LeetCode 2317 style state DP].

---

## Tier 1 — Binary Search on the Answer

### Threshold / minimize-the-maximum
Files: `4_2_tiles.cpp` (2D prefix + BS), `5_2_scores.cpp`
- ⭐ [LeetCode 410 — Split Array Largest Sum](https://leetcode.com/problems/split-array-largest-sum/) 🔴 (canonical "minimize the max")
- ⭐ [LeetCode 875 — Koko Eating Bananas](https://leetcode.com/problems/koko-eating-bananas/) 🟡 (canonical BS-on-answer)
- [LeetCode 1283 — Find the Smallest Divisor Given a Threshold](https://leetcode.com/problems/find-the-smallest-divisor-given-a-threshold/) 🟡
- [LeetCode 1631 — Path With Minimum Effort](https://leetcode.com/problems/path-with-minimum-effort/) 🟡 (BS on answer + grid, close to the tiles idea)

---

## Tier 1 — BFS / DFS + grid simulation

### Grid + collect items under a cost (warehouse truck)
File: `4_1_warehouse.cpp`
- ⭐ [LeetCode 847 — Shortest Path Visiting All Nodes](https://leetcode.com/problems/shortest-path-visiting-all-nodes/) 🔴 (bitmask BFS — the core technique)
- ⭐ [LeetCode 864 — Shortest Path to Get All Keys](https://leetcode.com/problems/shortest-path-to-get-all-keys/) 🔴 (grid + bitmask BFS — closest overall)

### Grid movement simulation (right-turn-only apples)
File: `apples.cpp`
- Samsung-original simulation. Practice the family on Samsung sets:
  [SW Expert Academy](https://swexpertacademy.com/) · [Baekjoon "삼성 SW 역량테스트 기출" workbook](https://www.acmicpc.net/step) · [GitHub — SWEA solutions](https://github.com/sjnov11/SW-expert-academy)
- Similar grid-sim on LC: [LeetCode 885 — Spiral Matrix III](https://leetcode.com/problems/spiral-matrix-iii/) 🟡

### Robot cutting/loading with ordering (logging trees)
File: `logging_trees.cpp`
- Samsung-original. Same source sets as above (SWEA / Baekjoon Samsung workbook).

---

## Tier 2 — Graphs & Trees

### Tree rerooting — sum over all nodes
File: `min_cost.cpp`
- ⭐ [LeetCode 834 — Sum of Distances in Tree](https://leetcode.com/problems/sum-of-distances-in-tree/) 🔴 (exact rerooting technique)
- Reference: [Codeforces — rerooting blog](https://codeforces.com/blog/entry/63962)

### Tree DFS balancing (equal sibling subtree sums)
File: `soldiers.cpp`
- Samsung-original tree DFS. Warm-up on tree DFS: [GFG — Tree traversals / subtree sum](https://www.geeksforgeeks.org/problems/) ; drill directly from repo.

### Prefix sum + hashmap — balanced necklace
File: `3_RB.cpp`
- ⭐ [LeetCode 525 — Contiguous Array](https://leetcode.com/problems/contiguous-array/) 🟡 (exact: longest subarray with equal counts → answer = N − that length)

---

## Tier 2 — Backtracking (from `text.txt` notes)
- [LeetCode 51 — N-Queens](https://leetcode.com/problems/n-queens/) 🔴
- [LeetCode 52 — N-Queens II](https://leetcode.com/problems/n-queens-ii/) 🔴
- [LeetCode 37 — Sudoku Solver](https://leetcode.com/problems/sudoku-solver/) 🔴

## Tier 2 — Stock buy/sell DP (from `text.txt` notes)
- [LeetCode 123 — Best Time to Buy and Sell Stock III](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iii/) 🔴 (≤2 transactions)
- [LeetCode 188 — Best Time to Buy and Sell Stock IV](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-iv/) 🔴 (≤k transactions)

---

## Tier 3 — Segment Tree (flagged "Imp" in notes)
- [GFG — Range Minimum Query (practice)](https://www.geeksforgeeks.org/problems/range-minimum-query/1) 🟡
- [SPOJ GSS1 — Can you answer these queries I](https://www.spoj.com/problems/GSS1/) 🔴
- [Codeforces EDU — Segment Tree course](https://codeforces.com/edu/course/2/lesson/4) (best structured practice)

---

## Repo problems with no exact online judge (drill from this repo)
These are Samsung-original — write your own test cases from the statements in `Test - 1.docx`:
- **Test 1** — points on a rectilinear lattice path (coordinate compression + interval search)
- **Test 2** — warehouse inventory min days (greedy + sorting, `2_days.cpp`)
- **Test 6** — cars to a point, drive t moves on turn t (math/parity + triangular numbers, `6_1_cars.cpp`)

---

## Suggested mapping to the 4-day plan
- **Day 1 (DP):** LC 416 → 1049 → 2035, then GFG Minimum sum partition; LC 1000 for interval DP.
- **Day 2 (BS-on-answer + digit DP):** LC 875 → 410 → 1283 → 1631; GFG digit-sum → LC 2719/902.
- **Day 3 (graphs/trees/grids):** LC 864 → 847; LC 834; LC 525; a couple SWEA grid sims.
- **Day 4 (backtracking + review):** LC 51 → 52 → 37; LC 123 → 188; timed redo of weak spots.

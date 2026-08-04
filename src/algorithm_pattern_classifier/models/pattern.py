from enum import StrEnum


class AlgorithmPattern(StrEnum):
    """Supported algorithmic design patterns for classification taxonomy.

    Members:
        TWO_POINTER: Two pointers moving inwards or at different speeds across sequences.
        SLIDING_WINDOW: Dynamic or fixed sub-array/substring window tracking.
        DYNAMIC_PROGRAMMING: Memoization or tabular state transitions for overlapping subproblems.
        BACKTRACKING: Recursive state-space search with pruning.
        BFS_DFS: Breadth-first or depth-first graph and tree traversals.
        GREEDY: Locally optimal choices at each stage.
        DIVIDE_AND_CONQUER: Dividing a problem into independent sub-problems and combining results.
        BINARY_SEARCH: Logarithmic search space reduction on sorted structures.
    """

    TWO_POINTER = "two-pointer"
    SLIDING_WINDOW = "sliding-window"
    DYNAMIC_PROGRAMMING = "dynamic-programming"
    FAST_SLOW_POINTERS = "fast-slow-pointers"
    BACKTRACKING = "backtracking"
    BFS_DFS = "bfs-dfs"
    GREEDY = "greedy"
    DIVIDE_AND_CONQUER = "divide-and-conquer"
    BINARY_SEARCH = "binary-search"

from enum import StrEnum


class AlgorithmPattern(StrEnum):
    """Supported algorithmic design patterns for classification."""

    TWO_POINTER = "two-pointer"
    SLIDING_WINDOW = "sliding-window"
    DYNAMIC_PROGRAMMING = "dynamic-programming"
    BACKTRACKING = "backtracking"
    BFS_DFS = "bfs-dfs"
    GREEDY = "greedy"
    DIVIDE_AND_CONQUER = "divide-and-conquer"
    BINARY_SEARCH = "binary-search"

def knapsack(w_max: int, wt: list[int], val: list[int], n: int) -> int:
    dp = [[0 for x in range(w_max + 1)] for x in range(n + 1)]
    for i in range(n + 1):
        for w in range(w_max + 1):
            if i == 0 or w == 0:
                dp[i][w] = 0
            elif wt[i - 1] <= w:
                dp[i][w] = max(val[i - 1] + dp[i - 1][w - wt[i - 1]], dp[i - 1][w])
            else:
                dp[i][w] = dp[i - 1][w]
    return dp[n][w_max]

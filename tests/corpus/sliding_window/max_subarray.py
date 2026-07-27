def max_subarray(arr: list[int], k: int) -> int:
    if len(arr) < k:
        return 0
    curr_sum = sum(arr[:k])
    max_sum = curr_sum
    for i in range(k, len(arr)):
        curr_sum += arr[i] - arr[i - k]
        max_sum = max(max_sum, curr_sum)
    return max_sum

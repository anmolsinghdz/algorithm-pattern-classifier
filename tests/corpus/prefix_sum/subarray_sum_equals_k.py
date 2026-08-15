def subarray_sum(nums: list[int], k: int) -> int:
    prefix_counts = {0: 1}
    current_sum = 0
    total_subarrays = 0
    for num in nums:
        current_sum += num
        if current_sum - k in prefix_counts:
            total_subarrays += prefix_counts[current_sum - k]
        prefix_counts[current_sum] = prefix_counts.get(current_sum, 0) + 1
    return total_subarrays

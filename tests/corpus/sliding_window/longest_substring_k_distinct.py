def longest_substring_k_distinct(s: str, k: int) -> int:
    counts: dict[str, int] = {}
    left = 0
    max_len = 0
    for right in range(len(s)):
        char = s[right]
        counts[char] = counts.get(char, 0) + 1
        while len(counts) > k:
            left_char = s[left]
            counts[left_char] -= 1
            if counts[left_char] == 0:
                del counts[left_char]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len

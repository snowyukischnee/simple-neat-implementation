def clamp(x: float, min_val: float, max_val: float) -> float:
    return min(max(x, min_val), max_val)
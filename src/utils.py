from typing import List, Any


def clamp(x: float, min_val: float, max_val: float) -> float:
    return min(max(x, min_val), max_val)


def without_keys(d: dict, keys: List[Any]):
    return {k: v for k, v in d.items() if k not in keys}
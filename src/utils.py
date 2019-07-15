from typing import List, Any, Tuple


def clamp(x: float, min_val: float, max_val: float) -> float:
    return min(max(x, min_val), max_val)


def without_keys(d: dict, keys: List[Any]):
    return {k: v for k, v in d.items() if k not in keys}


def required_for_output(input_keys: List[int], output_keys: List[int], connections: List[Tuple[int, int]]) -> List[int]:
    required_nodes = set(output_keys)
    required_w_input_keys = set(output_keys)
    while True:
        # set of node that connected to the nodes in set s
        layer_w_input_keys = set(i for (i, o) in connections if (o in required_w_input_keys) and (i not in required_w_input_keys))
        if len(layer_w_input_keys) == 0:
            break
        layer_nodes = set(x for x in layer_w_input_keys if x not in input_keys)
        if len(layer_nodes) == 0:
            break
        # print(layer_nodes)
        required_nodes = required_nodes.union(layer_nodes)
        required_w_input_keys = required_w_input_keys.union(layer_w_input_keys)
    return required_nodes

# def create_cycle(connections: List[Tuple[int, int]], test: Tuple[int, int]) -> bool:
#     pass
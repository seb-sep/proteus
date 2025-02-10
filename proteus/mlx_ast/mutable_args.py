from typing import Dict, List, Callable
import logging

import torch
from torch.fx import Graph, Node

aten = torch.ops.aten

logger = logging.getLogger(__file__)

# aten ops mapped to which args they mutate
mutating_aten_ops: Dict[Callable, List[int]] = {
    aten.mul_.Tensor: (0,),
    aten.index_copy_.default: (0,),
    aten.copy_.default: (0,),
}


def get_mut_arg_indices(graph: Graph) -> List[int]:
    """
    Return the argument indices which will be mutated by the FX graph.
    """

    # graph.find_nodes should return in topological order
    graph_args: Dict[Node, int] = {
        node: idx for idx, node in enumerate(graph.find_nodes(op="placeholder"))
    }
    mut_arg_idxs = []

    for node in graph.nodes:
        if node.op != "call_function":
            continue
        mut_args = mutating_aten_ops.get(node.target, ())
        for i in mut_args:
            arg_node = node.args[i]
            if arg_node in graph_args:
                mut_arg_idxs.append(graph_args[arg_node])

    return mut_arg_idxs

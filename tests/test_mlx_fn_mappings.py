import unittest
from typing import Callable, Any, Tuple, Dict, Iterable

import torch
from torch.fx import GraphModule, Graph, Node
import mlx.core as mx

from proteus.utils import coerce_torch_to_mx, coerce_mx_to_torch
from proteus.mlx_builder import MLXASTBuilder, aten_to_mlx

aten = torch.ops.aten


class TestMLXFunctionMappings(unittest.TestCase):

    def create_simple_graph(
        self, torch_op: Callable, num_args: int = 0, example_kwargs: Iterable[str] = ()
    ) -> GraphModule:
        graph = Graph()
        arg_nodes = tuple(graph.placeholder(f"_{i}") for i in range(num_args))
        kwarg_nodes = {key: graph.placeholder(f"_{key}") for key in example_kwargs}
        call_site = graph.call_function(torch_op, arg_nodes, kwarg_nodes)
        ret = graph.output((call_site,))
        gm = GraphModule({}, graph)
        gm.recompile()
        return gm

    def _test_op(
        self,
        torch_op: Callable,
        example_args: Tuple[Any] = (),
        example_kwargs: Dict[str, Any] = {},
        rtol: float = 1e-4,
        atol: float = 1e-4,
    ):
        """
        Generic test function for validating torch->mlx operator mappings

        Args:
            torch_op: The PyTorch operator to test
            mlx_op: The corresponding MLX operator
            marshaler: Function to handle any argument preprocessing
            example_args: Example positional arguments for the operator
            example_kwargs: Example keyword arguments for the operator
            rtol: Relative tolerance for numerical comparison
            atol: Absolute tolerance for numerical comparison
        """

        test_gm = self.create_simple_graph(
            torch_op, len(example_args), example_kwargs.keys()
        )
        # Flatten args and kwargs into a single tuple and coerce tensors to MLX

        torch_results = test_gm(*example_args, **example_kwargs)

        builder = MLXASTBuilder()
        builder.ingest_graph(test_gm.graph)
        mlx_fn = builder.export()

        flattened_mlx_args = tuple(
            coerce_torch_to_mx(arg) if isinstance(arg, torch.Tensor) else arg
            for arg in (*example_args, *example_kwargs.values())
        )
        mlx_results = tuple(
            coerce_mx_to_torch(out) for out in mlx_fn(*flattened_mlx_args)
        )
        # Compare results
        for torch_result, mlx_result in zip(torch_results, mlx_results):
            self.assertTrue(
                torch.allclose(torch_result, mlx_result, rtol=rtol, atol=atol),
                f"Output mismatch for operator {torch_op.__name__}",
            )

    def test_mm(self):
        a = torch.randn(32, 32, dtype=torch.float16)
        b = torch.randn_like(a)
        op = aten.mm.default

        self._test_op(op, (a, b))


if __name__ == "__main__":
    unittest.main()

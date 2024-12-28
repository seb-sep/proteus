import unittest
from typing import Callable, Any, Tuple, Dict, Iterable, List

import torch
from torch.fx import GraphModule, Graph, Node
import mlx.core as mx

from proteus.utils import coerce_torch_to_mx, coerce_mx_to_torch
from proteus.mlx_builder import MLXASTBuilder, aten_to_mlx
from torch.fx.experimental.proxy_tensor import make_fx

aten = torch.ops.aten


class TestMLXFunctionMappings(unittest.TestCase):

    def create_simple_graph(
        self,
        torch_op: Callable,
        example_args: List,
        example_kwargs: Dict[str, Any] = (),
    ) -> Tuple[GraphModule, List]:

        # dynamo will fail to produce a graph module if the cache is full, so just reset each time, this should
        # be fine because graphs should never be reused anyway
        torch._dynamo.reset_code_caches()

        ret_gm: GraphModule = None
        ret_example_inputs = None

        # dynamo is evil and will swap the order of inputs???
        def _capture_graph_backend(gm: GraphModule, example_inputs):
            nonlocal ret_gm
            ret_gm = gm
            nonlocal ret_example_inputs
            ret_example_inputs = example_inputs
            return gm.forward

        def foo(*args, **kwargs):
            return torch_op(*args, **kwargs)

        compiled_fn = torch.compile(foo, backend=_capture_graph_backend)
        out = compiled_fn(*example_args, **example_kwargs)
        # note that the end to end function is still correct, it's just the intermediary graph module which does unexpected things
        assert torch.allclose(out, torch_op(*example_args, **example_kwargs))
        assert isinstance(ret_gm, GraphModule)
        # # strip all unused args out of the graphmodule
        # for node in ret_gm.graph.find_nodes(op="placeholder"):
        #     assert isinstance(node, Node)
        #     if len(node.users) == 0:
        #         ret_gm.graph.erase_node(node)

        # ret_gm.recompile()
        return ret_gm, ret_example_inputs

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

        test_gm, example_inputs = self.create_simple_graph(
            torch_op, example_args, example_kwargs
        )

        torch_results = test_gm(*example_inputs)
        torch_results = (
            torch_results if isinstance(torch_results, tuple) else (torch_results,)
        )

        # Flatten args and kwargs into a single tuple and coerce tensors to MLX

        builder = MLXASTBuilder()
        builder.ingest_graph(test_gm.graph)
        mlx_fn = builder.export()

        flattened_mlx_args = tuple(
            (
                coerce_torch_to_mx(arg)
                if isinstance(arg, torch.Tensor)
                else int(arg) if isinstance(arg, torch.SymInt) else arg
            )
            for arg in example_inputs
        )
        # print("Flattened MLX args:", flattened_mlx_args)
        # print("oriignal PyTorch args:", example_args, example_kwargs)
        mlx_results = mlx_fn(*flattened_mlx_args)
        mx.eval(mlx_results)
        # print("results", mlx_results)

        mlx_results = tuple(coerce_mx_to_torch(out) for out in mlx_results)
        # print("torchified results:", mlx_results)
        # Compare results
        for torch_result, mlx_result in zip(torch_results, mlx_results):
            if torch_result.dtype.is_floating_point:
                self.assertTrue(
                    torch.allclose(mlx_result, torch_result, rtol=rtol, atol=atol),
                    f"Output mismatch for operator {torch_op.__name__}:\ntorch output {torch_result}\n\nmlx output {mlx_result}\ndifference: {torch_result - mlx_result}, biggest diff {torch.abs(torch_result - mlx_result).max()}",
                )
            else:
                self.assertTrue(
                    (mlx_result == torch_result).all(),
                    f"Output mismatch for operator {torch_op.__name__}:\ntorch output {torch_result}\n\nmlx output {mlx_result}",
                )

    def test_mm(self):
        a = torch.randn((32, 16), dtype=torch.float16)
        b = torch.randn((16, 32), dtype=torch.float16)
        op = aten.mm.default

        self._test_op(op, (a, b), rtol=1e-3)

    def test_t(self):
        """Test simple transpose operator"""
        a = torch.randn((4, 4), dtype=torch.float16)
        op = aten.t.default

        self._test_op(op, (a,))

    def test_transpose(self):
        """Test transpose with dimension arguments"""
        a = torch.randint(0, 10, (32, 64, 16), dtype=torch.int32)
        op = aten.transpose.int

        # Test transposing different dimensions
        self._test_op(op, (a, 0, 1))
        self._test_op(op, (a, 1, 2))

    def test_expand(self):
        """Test expand/broadcast operator"""
        a = torch.randn((1, 64, 1), dtype=torch.float16)
        op = aten.expand.default

        # Test expanding to larger dimensions
        self._test_op(op, (a, (32, 64, 16)))
        # Test expanding with -1 to keep original dimension
        self._test_op(op, (a, (-1, 64, 16)))

    def test_activations(self):
        """Test various activation functions"""
        a = torch.randn((32, 64), dtype=torch.float16)

        # Test ReLU
        self._test_op(aten.relu.default, (a,))

        # these tolerances are kinda bad but there's nothing I can do about it because there are no knobs
        # to tweak

        # Test SiLU/Swish
        self._test_op(aten.silu.default, (a,), atol=3e-3, rtol=1e-3)

        # Test GELU
        self._test_op(aten.gelu.default, (a,), atol=1e-3, rtol=1e-3)

    def test_triangular(self):
        """
        Test triangular matrix operators. only support ops where diagonal=0 for now
        """
        a = torch.randn((32, 32), dtype=torch.float16)

        # Test upper triangular
        self._test_op(aten.triu.default, (a,))
        # self._test_op(aten.triu.default, (a,), {"diagonal": 1})
        # self._test_op(aten.triu.default, (a,), {"diagonal": -1})

        # Test lower triangular
        self._test_op(aten.tril.default, (a,))
        # self._test_op(aten.tril.default, (a,), {"diagonal": 1})
        # self._test_op(aten.tril.default, (a,), {"diagonal": -1})

    def test_multiply(self):
        """Test multiplication operator"""
        # Test same-shape multiplication
        a = torch.randn((32, 64), dtype=torch.float16)
        b = torch.randn_like(a)
        self._test_op(aten.mul.Tensor, (a, b))

        # Test broadcasting multiplication
        c = torch.randn((1, 64), dtype=torch.float16)
        self._test_op(aten.mul.Tensor, (a, c))

        # Test scalar multiplication
        d = torch.tensor(2.0, dtype=torch.float16)
        self._test_op(aten.mul.Tensor, (a, d))

    def test_divide(self):
        """Test division operator"""
        # Test same-shape division
        a = torch.randn((32, 64), dtype=torch.float16)
        b = torch.randn_like(a)
        self._test_op(aten.div.Tensor, (a, b))

        # Test broadcasting division
        c = torch.randn((1, 64), dtype=torch.float16)
        self._test_op(aten.div.Tensor, (a, c))

        # Test scalar division
        d = torch.tensor(2.0, dtype=torch.float16)
        self._test_op(aten.div.Tensor, (a, d))

    def test_add(self):
        """Test addition operator"""
        # Test same-shape addition
        a = torch.randn((32, 64), dtype=torch.float16)
        b = torch.randn_like(a)
        self._test_op(aten.add.Tensor, (a, b))

        # Test broadcasting addition
        c = torch.randn((1, 64), dtype=torch.float16)
        self._test_op(aten.add.Tensor, (a, c))

        # Test scalar addition
        d = torch.tensor(2.0, dtype=torch.float16)
        self._test_op(aten.add.Tensor, (a, d))

    def test_exp(self):
        """Test exponential operator"""
        # Test basic exponential
        a = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.exp.default, (a,))

        # Test different shape
        b = torch.randn((16, 8), dtype=torch.float16)
        self._test_op(aten.exp.default, (b,))

        # Test 1D tensor
        c = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.exp.default, (c,))

    def test_greater_than(self):
        """Test greater than operator"""
        # Test same-shape comparison
        a = torch.randn((32, 64), dtype=torch.float16)
        b = torch.randn_like(a)
        self._test_op(aten.gt.Tensor, (a, b))

        # Test broadcasting comparison
        c = torch.randn((1, 64), dtype=torch.float16)
        self._test_op(aten.gt.Tensor, (a, c))

        # Test scalar comparison
        d = torch.tensor(0.5, dtype=torch.float16)
        self._test_op(aten.gt.Tensor, (a, d))

    def test_equals_scalar(self):
        """Test equals scalar operator"""
        # Test basic scalar equality
        a = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.eq.Scalar, (a, 0.0))

        # Test different shape
        b = torch.randn((16, 8), dtype=torch.float16)
        self._test_op(aten.eq.Scalar, (b, 1.0))

        # Test 1D tensor
        c = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.eq.Scalar, (c, -1.0))

    def test_neg(self):
        """Test negation operator"""
        # Test basic negation
        a = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.neg.default, (a,))

        # Test different shape
        b = torch.randn((16, 8), dtype=torch.float16)
        self._test_op(aten.neg.default, (b,))

        # Test 1D tensor
        c = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.neg.default, (c,))

    def test_trig(self):
        """Test trigonometric operators"""
        # Test basic cosine
        a = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.cos.default, (a,))

        # Test different shape
        b = torch.randn((16, 8), dtype=torch.float16)
        self._test_op(aten.cos.default, (b,))

        # Test 1D tensor
        c = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.cos.default, (c,))

        # Test basic sine
        self._test_op(aten.sin.default, (a,))

        # Test different shape
        self._test_op(aten.sin.default, (b,))

        # Test 1D tensor
        self._test_op(aten.sin.default, (c,))

    def test_rsqrt(self):
        """Test reciprocal square root operator"""
        # Test basic rsqrt
        a = (
            torch.rand((32, 64), dtype=torch.float32) + 1e-6
        )  # Add small constant to avoid division by zero
        self._test_op(aten.rsqrt.default, (a,))

        # Test different shape
        b = torch.rand((16, 8), dtype=torch.float32) + 1e-6
        self._test_op(aten.rsqrt.default, (b,))

        # Test 1D tensor
        c = torch.rand(100, dtype=torch.float32) + 1e-6
        self._test_op(aten.rsqrt.default, (c,))

    def test_cat(self):
        """Test concatenation operator"""
        # Test basic concatenation along default dim (0)
        a = torch.randn((32, 64), dtype=torch.float16)
        b = torch.randn((16, 64), dtype=torch.float16)
        self._test_op(aten.cat.default, ((a, b),))

        # Test concatenation along dim 1
        c = torch.randn((32, 64), dtype=torch.float16)
        d = torch.randn((32, 32), dtype=torch.float16)
        self._test_op(aten.cat.default, ((c, d), 1))

        # Test concatenation of 3 tensors
        e = torch.randn((8, 16), dtype=torch.float16)
        f = torch.randn((8, 8), dtype=torch.float16)
        g = torch.randn((8, 24), dtype=torch.float16)
        self._test_op(aten.cat.default, ((e, f, g), 1))

        # Test concatenation of 1D tensors
        h = torch.randn(50, dtype=torch.float16)
        i = torch.randn(30, dtype=torch.float16)
        self._test_op(aten.cat.default, ((h, i),))

    def test_select(self):
        """Test select operator"""
        # Test selecting from 2D tensor along dim 0
        a = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.select.int, (a, 0, 5))

        # Test selecting from 2D tensor along dim 1
        self._test_op(aten.select.int, (a, 1, 10))

        # Test selecting from 3D tensor
        b = torch.randn((16, 8, 4), dtype=torch.float16)
        self._test_op(aten.select.int, (b, 0, 5))
        self._test_op(aten.select.int, (b, 1, 3))
        self._test_op(aten.select.int, (b, 2, 2))

        # Test selecting from 1D tensor
        c = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.select.int, (c, 0, 50))

    def test_arange(self):
        """Test arange operators"""
        # Test arange.default with just end
        self._test_op(aten.arange.default, (10,))
        self._test_op(aten.arange.default, (100,))

        # Test arange.start with start and end
        self._test_op(aten.arange.start, (5, 25))
        self._test_op(aten.arange.start, (-10, 10))

        # Test with different dtypes
        self._test_op(aten.arange.default, (10,), {"dtype": torch.float16})
        self._test_op(aten.arange.start, (0, 10), {"dtype": torch.int32})


if __name__ == "__main__":
    unittest.main()
    # TestMLXFunctionMappings().test_arange()

# import mlx.core


# def cool_mlx_fn(_0):
#     t_default = mlx.core.transpose(_0)
#     return (t_default,)


# a = mlx.core.random.normal((4, 4))
# print(a, cool_mlx_fn(a))

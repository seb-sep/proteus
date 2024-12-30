import unittest
from typing import Callable, Any, Tuple, Dict, Iterable, List

import torch
from torch.fx import GraphModule, Graph, Node
import mlx.core as mx
from torch._subclasses.fake_tensor import unset_fake_temporarily

from proteus.utils import coerce_torch_to_mx, coerce_mx_to_torch
from proteus.mlx_builder import MLXASTBuilder
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
        op_res = torch_op(*example_args, **example_kwargs)
        # in case the graph returns some iterable of tensors like torch.split() does
        if isinstance(out, (tuple, List)):
            assert all(
                torch.allclose(out_elem, op_elem)
                for out_elem, op_elem in zip(out, op_res)
            )
        else:
            assert torch.allclose(out, op_res)
        assert isinstance(ret_gm, GraphModule)
        # # strip all unused args out of the graphmodule
        # for node in ret_gm.graph.find_nodes(op="placeholder"):
        #     assert isinstance(node, Node)
        #     if len(node.users) == 0:
        #         ret_gm.graph.erase_node(node)

        # ret_gm.recompile()
        print(ret_gm.graph)
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
                close = torch.allclose(mlx_result, torch_result, rtol=rtol, atol=atol)
                self.assertTrue(
                    close,
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
        self._test_op(aten.triu.default, (a,), {"diagonal": 1})
        self._test_op(aten.triu.default, (a,), {"diagonal": -1})

        # Test lower triangular
        self._test_op(aten.tril.default, (a,))
        self._test_op(aten.tril.default, (a,), {"diagonal": 1})
        self._test_op(aten.tril.default, (a,), {"diagonal": -1})

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

    def test_unsqueeze(self):
        """Test unsqueeze operator"""
        # Test unsqueezing 2D tensor along dim 0
        a = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.unsqueeze.default, (a, 0))

        # Test unsqueezing 2D tensor along dim 1
        self._test_op(aten.unsqueeze.default, (a, 1))

        # Test unsqueezing 2D tensor along dim 2
        self._test_op(aten.unsqueeze.default, (a, 2))

        # Test unsqueezing 1D tensor
        b = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.unsqueeze.default, (b, 0))
        self._test_op(aten.unsqueeze.default, (b, 1))

        # Test unsqueezing 3D tensor
        c = torch.randn((16, 8, 4), dtype=torch.float16)
        self._test_op(aten.unsqueeze.default, (c, 0))
        self._test_op(aten.unsqueeze.default, (c, 1))
        self._test_op(aten.unsqueeze.default, (c, 2))
        self._test_op(aten.unsqueeze.default, (c, 3))

    def test_full(self):
        """Test full operator"""
        # Test basic 2D tensor creation
        self._test_op(aten.full.default, ((32, 64), 1.0))
        self._test_op(aten.full.default, ((32, 64), 0.0))
        self._test_op(aten.full.default, ((32, 64), -1.0))

        # Test 1D tensor creation
        self._test_op(aten.full.default, ((100,), 5.0))

        # Test 3D tensor creation
        self._test_op(aten.full.default, ((16, 8, 4), 2.0))

        # Test with different dtypes
        self._test_op(aten.full.default, ((32, 64), 1.0), {"dtype": torch.float16})
        self._test_op(aten.full.default, ((32, 64), 1.0), {"dtype": torch.int32})
        self._test_op(aten.full.default, ((32, 64), 1.0), {"dtype": torch.float32})

        # Test with integer fill values
        self._test_op(aten.full.default, ((32, 64), 1))
        self._test_op(aten.full.default, ((32, 64), 0))
        self._test_op(aten.full.default, ((32, 64), -1))

    def test_view(self):
        """Test view operator"""
        # Test basic 2D tensor reshaping
        a = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.view.default, (a, (64, 32)))
        self._test_op(aten.view.default, (a, (2048,)))
        self._test_op(aten.view.default, (a, (8, 8, 32)))

        # Test 1D tensor reshaping
        b = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.view.default, (b, (10, 10)))
        self._test_op(aten.view.default, (b, (4, 25)))
        self._test_op(aten.view.default, (b, (2, 2, 25)))

        # Test 3D tensor reshaping
        c = torch.randn((16, 8, 4), dtype=torch.float16)
        self._test_op(aten.view.default, (c, (512,)))
        self._test_op(aten.view.default, (c, (32, 16)))
        self._test_op(aten.view.default, (c, (8, 8, 8)))
        self._test_op(aten.view.default, (c, (4, 4, 4, 8)))

        # Test with different dtypes
        d = torch.randn((32, 64), dtype=torch.float32)
        self._test_op(aten.view.default, (d, (64, 32)))

        e = torch.randint(0, 10, (32, 64), dtype=torch.int32)

        self._test_op(aten.view.default, (e, (2048,)))

    def test_clone(self):
        """Test clone operator"""
        # Test basic 2D tensor cloning
        a = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.clone.default, (a,))

        # Test 1D tensor cloning
        b = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.clone.default, (b,))

        # Test 3D tensor cloning
        c = torch.randn((16, 8, 4), dtype=torch.float16)
        self._test_op(aten.clone.default, (c,))

        # Test with different dtypes
        d = torch.randn((32, 64), dtype=torch.float32)
        self._test_op(aten.clone.default, (d,))

        e = torch.randint(0, 10, (32, 64), dtype=torch.int32)
        self._test_op(aten.clone.default, (e,))

    def test_copy(self):
        """Test copy operator"""
        # Test basic 2D tensor copying
        a = torch.randn((32, 64), dtype=torch.float16)
        b = torch.randn((32, 64), dtype=torch.float16)
        self._test_op(aten.copy.default, (a, b))

        # Test 1D tensor copying
        c = torch.randn(100, dtype=torch.float16)
        d = torch.randn(100, dtype=torch.float16)
        self._test_op(aten.copy.default, (c, d))

        # Test 3D tensor copying
        e = torch.randn((16, 8, 4), dtype=torch.float16)
        f = torch.randn((16, 8, 4), dtype=torch.float16)
        self._test_op(aten.copy.default, (e, f))

        # Test with different dtypes
        g = torch.randn((32, 64), dtype=torch.float32)
        h = torch.randn((32, 64), dtype=torch.float32)
        self._test_op(aten.copy.default, (g, h))

        i = torch.randint(0, 10, (32, 64), dtype=torch.int32)
        j = torch.randint(0, 10, (32, 64), dtype=torch.int32)
        self._test_op(aten.copy.default, (i, j))

    # know what graphs calling this look like
    # def test_conv2d(self):
    #     """Test 2D convolution operator"""
    #     # Test basic 2D convolution with square kernel
    #     batch_size = 8
    #     in_channels = 3
    #     out_channels = 16
    #     input_height = 32
    #     input_width = 32
    #     kernel_size = 3

    #     # Input tensor [batch, channels, height, width]
    #     x = torch.randn(
    #         (batch_size, in_channels, input_height, input_width), dtype=torch.float32
    #     )
    #     # Weight tensor [out_channels, in_channels, kernel_height, kernel_width]
    #     w = torch.randn(
    #         (out_channels, in_channels, kernel_size, kernel_size), dtype=torch.float32
    #     )
    #     # Optional bias [out_channels]
    #     b = torch.randn(out_channels, dtype=torch.float32)

    #     # Test with bias and default stride/padding
    #     self._test_op(aten.conv2d.default, (x, w, b), rtol=1e-4, atol=1e-4)

    #     # Test without bias
    #     self._test_op(aten.conv2d.default, (x, w, None), rtol=1e-4, atol=1e-4)

    #     # Test with explicit stride and padding
    #     self._test_op(
    #         aten.conv2d.default,
    #         (x, w, None),
    #         {"stride": (2, 2), "padding": (1, 1)},
    #         rtol=1e-4,
    #         atol=1e-4,
    #     )

    #     # Test with rectangular kernel
    #     w_rect = torch.randn((out_channels, in_channels, 3, 5), dtype=torch.float32)
    #     self._test_op(aten.conv2d.default, (x, w_rect, None), rtol=1e-4, atol=1e-4)

    #     # Test with different input size
    #     x_large = torch.randn((4, in_channels, 64, 64), dtype=torch.float32)
    #     self._test_op(aten.conv2d.default, (x_large, w, b), rtol=1e-4, atol=1e-4)

    def test_unsafe_view(self):
        """Test unsafe view operator"""
        x = torch.randn((2, 3, 4))
        self._test_op(aten._unsafe_view.default, (x, (2, 12)))
        self._test_op(aten._unsafe_view.default, (x, (3, 8)))
        self._test_op(aten._unsafe_view.default, (x, (-1, 4)))

    def test_split(self):
        """Test split operator"""
        # Test basic split
        torch.split
        x = torch.randn((10, 4))
        self._test_op(aten.split.Tensor, (x, 2))  # Split into sections of size 2

        # Test uneven split
        # TODO: uneven split does not work for now, circle back if this is
        # necessary for inferencing models
        # x = torch.randn((5, 3))
        # self._test_op(aten.split.Tensor, (x, 2))  # Last section will be size 1

        # Test split along different dimension
        x = torch.randn((6, 6))
        self._test_op(aten.split.Tensor, (x, 3), {"dim": 1})

        # Test with larger tensor
        x = torch.randn((8, 8, 8))
        self._test_op(aten.split.Tensor, (x, 4))  # Split first dim into size 4 chunks
        self._test_op(
            aten.split.Tensor, (x, 2), {"dim": 2}
        )  # Split last dim into size 2 chunks

    def test_layer_norm(self):
        """Test layer normalization operator"""
        # Test basic layer norm
        x = torch.randn((32, 64), dtype=torch.float32)
        normalized_shape = [64]
        weight = torch.randn(normalized_shape, dtype=torch.float32)
        bias = torch.randn(normalized_shape, dtype=torch.float32)
        self._test_op(
            aten.layer_norm.default,
            (x, normalized_shape, weight, bias),
            rtol=1e-4,
            atol=1e-4,
        )

        # Test without bias
        self._test_op(
            aten.layer_norm.default,
            (x, normalized_shape, weight, None),
            rtol=1e-4,
            atol=1e-4,
        )

        # Test without weight and bias
        self._test_op(
            aten.layer_norm.default,
            (x, normalized_shape, None, None),
            rtol=1e-4,
            atol=1e-4,
        )

        # Test with 3D input
        x = torch.randn((8, 16, 32), dtype=torch.float32)
        normalized_shape = [32]
        weight = torch.randn(normalized_shape, dtype=torch.float32)
        bias = torch.randn(normalized_shape, dtype=torch.float32)
        self._test_op(
            aten.layer_norm.default,
            (x, normalized_shape, weight, bias),
            rtol=1e-4,
            atol=1e-4,
        )

        # Test with multiple normalized dimensions
        # Unsupported for now but perhaps later if necessary
        # x = torch.randn((8, 16, 32, 64), dtype=torch.float32)
        # normalized_shape = [32, 64]
        # weight = torch.randn(normalized_shape, dtype=torch.float32)
        # bias = torch.randn(normalized_shape, dtype=torch.float32)
        # self._test_op(
        #     aten.layer_norm.default,
        #     (x, normalized_shape, weight, bias),
        #     rtol=1e-4,
        #     atol=1e-4,
        # )

    def test_scaled_dot_product_attention(self):
        """Test scaled dot product attention operator"""
        # Test basic case
        # batch_size = 2
        # num_heads = 4
        # seq_len = 8
        # head_dim = 16

        batch_size = 1
        num_heads = 1
        seq_len = 2
        head_dim = 2

        query = torch.randn(
            (batch_size, num_heads, seq_len, head_dim), dtype=torch.float32
        )
        key = torch.randn(
            (batch_size, num_heads, seq_len, head_dim), dtype=torch.float32
        )
        value = torch.randn(
            (batch_size, num_heads, seq_len, head_dim), dtype=torch.float32
        )

        self._test_op(
            aten.scaled_dot_product_attention.default,
            (query, key, value),
            rtol=1e-4,
            atol=1e-4,
        )

        torch.masked_fill

        # Test with attention mask
        attn_mask = torch.ones((seq_len, seq_len), dtype=torch.bool).tril()
        self._test_op(
            aten.scaled_dot_product_attention.default,
            (query, key, value),
            {"attn_mask": attn_mask},
            rtol=1e-4,
            atol=1e-4,
        )

        # Test different sequence lengths for key/value
        kv_seq_len = 16
        key = torch.randn(
            (batch_size, num_heads, kv_seq_len, head_dim), dtype=torch.float32
        )
        value = torch.randn(
            (batch_size, num_heads, kv_seq_len, head_dim), dtype=torch.float32
        )

        self._test_op(
            aten.scaled_dot_product_attention.default,
            (query, key, value),
            rtol=1e-4,
            atol=1e-4,
        )

        # Test with custom scale
        query = torch.randn(
            (batch_size, num_heads, seq_len, head_dim), dtype=torch.float32
        )
        key = torch.randn(
            (batch_size, num_heads, seq_len, head_dim), dtype=torch.float32
        )
        value = torch.randn(
            (batch_size, num_heads, seq_len, head_dim), dtype=torch.float32
        )
        scale = 2.0

        self._test_op(
            aten.scaled_dot_product_attention.default,
            (query, key, value),
            {"scale": scale},
            rtol=1e-4,
            atol=1e-4,
        )

    def test_pow(self):
        """Test power operator with scalar exponents"""
        # Test with various shapes and exponents
        a = torch.randn((32, 64), dtype=torch.float16)

        # weirdly specific powers are occasionally just wrong for fp16;
        # relax the tolerances here or test in fp32/bf16
        # Test squaring
        self._test_op(aten.pow.Tensor_Scalar, (a, 2), atol=1e-3, rtol=1e-3)

        # Test cubing
        self._test_op(aten.pow.Tensor_Scalar, (a.float(), 3), atol=1e-3, rtol=1e-3)

        # Test fractional power (obviously only works on positive numbers)
        self._test_op(aten.pow.Tensor_Scalar, (a.abs(), 0.5))

        # Test negative power
        self._test_op(aten.pow.Tensor_Scalar, (a.float(), -1))

        # Test power of 1 (identity)
        self._test_op(aten.pow.Tensor_Scalar, (a, 1))

        # Test power of 0 (should all be 1s)
        self._test_op(aten.pow.Tensor_Scalar, (a, 0))

    def test_mean(self):
        """Test mean reduction operator"""

        # correct enough on float16?
        a = torch.randn((32, 64, 16), dtype=torch.float32)

        # Test mean over all dimensions
        self._test_op(aten.mean.default, (a,))

        # Test mean over specific dimensions
        # Single dimension
        self._test_op(aten.mean.dim, (a, (0,)))
        self._test_op(aten.mean.dim, (a, (1,)))
        self._test_op(aten.mean.dim, (a, (2,)))

        # Multiple dimensions
        self._test_op(aten.mean.dim, (a, (0, 1)))
        self._test_op(aten.mean.dim, (a, (1, 2)))
        self._test_op(aten.mean.dim, (a, (0, 2)))

        # Test with keepdim=True
        self._test_op(aten.mean.dim, (a, (1,)), {"keepdim": True})
        self._test_op(aten.mean.dim, (a, (0, 2)), {"keepdim": True})

    def test_einsum(self):
        """Test einsum operator"""
        # Test matrix multiplication
        # don't worry, all tests are still using real tensors! issue running aten einsum
        # on faketensors must be for creating the fx graph
        a = torch.randn((32, 16), dtype=torch.float16)
        b = torch.randn((16, 32), dtype=torch.float16)
        self._test_op(aten.einsum.default, ("ik,kj->ij", (a, b)), rtol=1e-3)

        # Test batch matrix multiplication
        a = torch.randn((8, 32, 16), dtype=torch.float16)
        b = torch.randn((8, 16, 32), dtype=torch.float16)
        self._test_op(aten.einsum.default, ("bik,bkj->bij", (a, b)), rtol=1e-3)

        # Test inner product
        a = torch.randn((32,), dtype=torch.float16)
        b = torch.randn((32,), dtype=torch.float16)
        self._test_op(aten.einsum.default, ("i,i->", (a, b)), rtol=1e-3)

        # Test outer product
        a = torch.randn((32,), dtype=torch.float16)
        b = torch.randn((16,), dtype=torch.float16)
        self._test_op(aten.einsum.default, ("i,j->ij", (a, b)), rtol=1e-3)

        # Test trace
        a = torch.randn((32, 32), dtype=torch.float16)
        self._test_op(aten.einsum.default, ("ii->", (a,)), rtol=1e-3)

        # Test diagonal
        a = torch.randn((32, 32), dtype=torch.float16)
        self._test_op(aten.einsum.default, ("ii->i", (a,)), rtol=1e-3)

        # Test transpose
        a = torch.randn((32, 16), dtype=torch.float16)
        self._test_op(aten.einsum.default, ("ij->ji", (a,)), rtol=1e-3)


if __name__ == "__main__":
    unittest.main()
    # TestMLXFunctionMappings().test_einsum()

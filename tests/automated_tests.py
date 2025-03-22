import unittest
from typing import Tuple

import torch
import torch.nn as nn
from test_modules import (
    SimpleModule,
    EmbeddingModule,
    IndexCopyModule,
    TestModule,
    TestGraphBreakModule,
)
from proteus import proteus
from proteus.utils.utils import coerce_torch_to_mx, coerce_mx_to_torch


def flatten(obj) -> tuple:
    acc: list = []
    if not isinstance(obj, (tuple, list)):
        return (obj,)
    else:
        for o in obj:
            acc.extend(flatten(o))
    return tuple(acc)


class TestProteus(unittest.TestCase):
    def test_models(self):
        in_dim, h_dim, out_dim = 16, 64, 32
        device = "cpu"
        dtype = torch.float16
        test_in = torch.randn((out_dim, in_dim), device=device, dtype=dtype)
        lookup_tensor = torch.randint(0, in_dim, (1, in_dim), device=device)

        self.compile_model_test(
            TestGraphBreakModule(in_dim, h_dim, out_dim, dtype),
            (test_in,),
        )

        self.compile_model_test(
            TestModule(in_dim, h_dim, out_dim, dtype),
            (test_in,),
        )

        self.compile_model_test(
            SimpleModule(in_dim, h_dim, out_dim, dtype),
            (test_in,),
        )

        self.compile_model_test(
            IndexCopyModule(in_dim, in_dim, dtype),
            (test_in,),
        )

    def compile_model_test(
        self,
        model: nn.Module,
        test_input: Tuple[torch.Tensor] = (),
        test_kwargs: dict = {},
        atol=1e-4,
        rtol=1e-4,
    ):
        """Test that a model can be compiled and produces the same output"""
        # Get baseline output from original model
        test_out = model(*test_input, **test_kwargs)
        # Compile the model
        compiled_model = proteus(model)
        compiled_out = compiled_model(*test_input, **test_kwargs)

        print(f"testing {model._get_name()}")
        self.compare_outs(test_out, compiled_out)

    def compare_outs(self, test_out, compiled_out, rtol=1e-4, atol=1e-4):
        if not isinstance(test_out, tuple):
            test_out = (test_out,)
            compiled_out = (compiled_out,)

        for baseline, compiled in zip(flatten(test_out), flatten(compiled_out)):
            if baseline is None and compiled is None:
                continue
            # Compare outputs
            is_close = torch.allclose(baseline, compiled, rtol=rtol, atol=atol)
            if not is_close:
                print(f"\nBaseline output:\n{baseline}")
                print(f"\nCompiled output:\n{compiled}")
                diff = torch.abs(baseline - compiled)
                print(f"\nDifference:\n{diff}")
                top_5_diffs = torch.topk(diff.flatten(), k=5)
                print("\nTop 5 biggest differences:")
                for i, val in enumerate(top_5_diffs.values, 1):
                    print(f"{i}. {val.item()}")
            self.assertTrue(
                is_close,
                f"Compiled output does not match original output",
            )

    def compile_indexcopy_test(self):
        i, j = 4, 4
        dtype = torch.float32
        model = IndexCopyModule(i, j, dtype)
        input = torch.randn((i, j))
        input_copy = input.clone()
        output = model(input)  # this should mutate input
        compiled = proteus(model)
        compiled_out = compiled(input_copy)
        print(input, input_copy)


if __name__ == "__main__":
    TestProteus().test_models()

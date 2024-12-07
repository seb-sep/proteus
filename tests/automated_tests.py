import unittest

import torch
import torch.nn as nn

from test_modules import (
    SimpleModule,
    EmbeddingModule,
    SimpleTransformer,
    TestModule,
    TestGraphBreakModule,
)
from proteus.proteus import proteus, proteus_v3, proteus_v4
from proteus.utils import coerce_torch_to_mx, coerce_mx_to_torch


class TestProteus(unittest.TestCase):
    def test_models(self):

        in_dim, h_dim, out_dim = 16, 64, 32
        test_in = torch.randn((out_dim, in_dim))
        lookup_tensor = torch.randint(0, in_dim, (1, in_dim))
        testcases = [
            # (TestGraphBreakModule(in_dim, h_dim, out_dim), test_in),
            (TestModule(in_dim, h_dim, out_dim), test_in),
            # (SimpleModule(in_dim, h_dim, out_dim), test_in),
            # (EmbeddingModule(in_dim, h_dim, out_dim), lookup_tensor),
        ]
        for model, test_input in testcases:
            self.compile_model_test(model, test_input)

    def compile_model_test(self, model: nn.Module, test_input: torch.Tensor):
        """Test that a model can be compiled and produces the same output"""
        # Get baseline output from original model
        test_out = model(test_input)

        # Compile the model
        compiled_model = proteus_v4(model)
        compiled_out = compiled_model(test_input)
        # compiled_out = compiled_model(coerce_torch_to_mx(test_input))

        # Compare outputs
        self.assertTrue(
            torch.allclose(test_out, (compiled_out), rtol=1e-4, atol=1e-4),
            f"Compiled {model._get_name()} model output does not match original output",
        )


if __name__ == "__main__":
    unittest.main()

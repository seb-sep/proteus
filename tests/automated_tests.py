import unittest
from typing import Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import LlamaConfig

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
        device = "cpu"
        dtype = torch.float16
        test_in = torch.randn((out_dim, in_dim), device=device, dtype=dtype)
        lookup_tensor = torch.randint(0, in_dim, (1, in_dim), device=device)
        llama_config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        testcases = [
            # (TestGraphBreakModule(in_dim, h_dim, out_dim), (test_in,), {}),
            # (TestModule(in_dim, h_dim, out_dim), (test_in,), {}),
            # (SimpleModule(in_dim, h_dim, out_dim), (test_in,), {}),
            # (EmbeddingModule(in_dim, h_dim, out_dim), (lookup_tensor,), {}),
            (
                LlamaAttention(llama_config, layer_idx=0).to(device).to(dtype),
                (
                    torch.randn(
                        1,
                        h_dim,
                        llama_config.hidden_size,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                {
                    "position_ids": torch.arange(h_dim, device=device)
                    .unsqueeze(0)
                    .expand(1, -1),
                    "use_cache": False,
                },
            )
        ]
        for model, test_input, test_kwargs in testcases:
            self.compile_model_test(model, test_input, test_kwargs)

    def compile_model_test(
        self, model: nn.Module, test_input: Tuple[torch.Tensor], test_kwargs: dict
    ):
        """Test that a model can be compiled and produces the same output"""
        # Get baseline output from original model
        test_out = model(*test_input, **test_kwargs)
        # Compile the model
        compiled_model = proteus_v4(model)
        compiled_out = compiled_model(*test_input, **test_kwargs)
        # compiled_out = compiled_model(coerce_torch_to_mx(test_input))

        # Compare outputs
        self.assertTrue(
            torch.allclose(test_out, (compiled_out), rtol=1e-4, atol=1e-4),
            f"Compiled {model._get_name()} model output does not match original output",
        )


if __name__ == "__main__":
    unittest.main()

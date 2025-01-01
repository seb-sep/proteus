import unittest
from typing import Tuple

import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaDecoderLayer,
    LlamaModel,
)
from transformers import LlamaConfig

from test_modules import (
    SimpleModule,
    EmbeddingModule,
    SimpleTransformer,
    TestModule,
    TestGraphBreakModule,
)
from proteus.proteus import proteus_v4
from proteus.utils import coerce_torch_to_mx, coerce_mx_to_torch


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
            TestGraphBreakModule(in_dim, h_dim, out_dim, dtype), (test_in,), {}
        )

        self.compile_model_test(
            TestModule(in_dim, h_dim, out_dim, dtype), (test_in,), {}
        )

        self.compile_model_test(
            SimpleModule(in_dim, h_dim, out_dim, dtype), (test_in,), {}
        )

        self.compile_model_test(
            EmbeddingModule(in_dim, h_dim, out_dim, dtype), (lookup_tensor,), {}
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
        compiled_model = proteus_v4(model)
        compiled_out = compiled_model(*test_input, **test_kwargs)

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
                f"Compiled {model._get_name()} model output does not match original output",
            )


class TestLlama(TestProteus):

    def test_models(self):
        device = "cpu"
        dtype = torch.float16
        seq_len = 24
        batch_size = 2
        llama_config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
        hidden_size = llama_config.hidden_size
        position_ids = torch.vstack(
            [torch.arange(seq_len, device=device).unsqueeze(0)] * batch_size
        )
        hidden_states = torch.randn(
            (batch_size, seq_len, hidden_size),
            device=device,
            dtype=dtype,
        )
        # Test LlamaAttention
        self.compile_model_test(
            LlamaAttention(llama_config, layer_idx=0).to(device).to(dtype),
            (hidden_states,),
            {
                "position_ids": position_ids,
                "use_cache": False,
            },
            atol=1e-3,
        )

        self.compile_model_test(LlamaRMSNorm(hidden_size), (hidden_states,), atol=1e-3)

        self.compile_model_test(
            LlamaRotaryEmbedding(config=llama_config), (hidden_states, position_ids)
        )

        self.compile_model_test(
            LlamaDecoderLayer(llama_config, 1),
            (hidden_states,),
            {"position_ids": position_ids},
        )

        input_ids = torch.randint(
            0, llama_config.vocab_size, (batch_size, seq_len), device="cpu"
        )
        llama = LlamaModel(llama_config)
        self.compile_model_test(
            llama,
            (input_ids,),
            {"position_ids": position_ids, "return_dict": False},
        )


if __name__ == "__main__":
    unittest.main()
    # TestLlama().test_models()

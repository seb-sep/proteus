import unittest
import os
from copy import deepcopy
from typing import Tuple

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaModel,
    LlamaForCausalLM,
    LlamaConfig,
    GenerationConfig,
    StaticCache,
    BatchEncoding,
)

from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
    LlamaDecoderLayer,
    LlamaModel,
)

from transformers.generation.utils import GenerateDecoderOnlyOutput


import torch

from proteus import proteus
from automated_tests import TestProteus

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence warnings when compiling


class TestLlama(TestProteus):

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    llama_config = LlamaConfig.from_pretrained("meta-llama/Llama-3.2-1B")
    device = "cpu"
    dtype = torch.float16
    seq_len = 24
    batch_size = 2
    hidden_size = llama_config.hidden_size
    position_ids = torch.vstack(
        [torch.arange(seq_len, device=device).unsqueeze(0)] * batch_size
    )
    hidden_states = torch.randn(
        (batch_size, seq_len, hidden_size),
        device=device,
        dtype=dtype,
    )

    def test_llama_modules(self):

        # Test LlamaAttention
        self.compile_model_test(
            LlamaAttention(self.llama_config, layer_idx=0)
            .to(self.device)
            .to(self.dtype),
            (self.hidden_states,),
            {
                "position_ids": self.position_ids,
                "use_cache": False,
            },
            atol=1e-3,
        )

        self.compile_model_test(
            LlamaRMSNorm(self.hidden_size), (self.hidden_states,), atol=1e-3
        )

        self.compile_model_test(
            LlamaRotaryEmbedding(config=self.llama_config),
            (self.hidden_states, self.position_ids),
        )

        self.compile_model_test(
            LlamaDecoderLayer(self.llama_config, 1),
            (self.hidden_states,),
            {"position_ids": self.position_ids, "past_key_values": None},
        )

        input_ids = torch.randint(
            0,
            self.llama_config.vocab_size,
            (self.batch_size, self.seq_len),
            device="cpu",
        )

        llama = LlamaModel(self.llama_config)

        self.compile_model_test(
            llama,
            (input_ids,),
            {
                "position_ids": self.position_ids,
                "return_dict": False,
            },
        )

    def test_llama_no_kv(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, legacy=False)
        prompt = ["what is a compiler?"] * self.batch_size
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)

        # llama = LlamaForCausalLM(self.llama_config)
        llama = AutoModelForCausalLM.from_pretrained(self.model_name)

        outs = llama(
            **inputs,
            return_dict=False,
        )

        compiled = proteus(llama)
        compiled_outs = compiled(
            **inputs,
            return_dict=False,
        )

        # don't compare returned StaticCache
        self.compare_outs(outs[:1], compiled_outs[:1])

    def test_llama_empty_kv(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, legacy=False)
        prompt = ["what is a compiler?"] * self.batch_size
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        cache = StaticCache(
            config=self.llama_config,
            batch_size=self.batch_size,
            max_cache_len=2 * self.seq_len,
            dtype=torch.float32,
            device="cpu",
        )

        # llama = LlamaModel(self.llama_config)

        llama = LlamaForCausalLM(self.llama_config)

        outs = llama(
            **inputs,
            past_key_values=cache,
            return_dict=False,
        )

        cache.reset()

        # compiles for better perf on macos
        compiled = proteus(llama)
        compiled_outs = compiled(
            **inputs,
            past_key_values=cache,
            return_dict=False,
        )

        # don't compare returned StaticCache
        self.compare_outs(outs[:1], compiled_outs[:1])

    def get_kv_for_model(
        self, model: LlamaForCausalLM, gen_config: GenerationConfig
    ) -> Tuple[BatchEncoding, GenerateDecoderOnlyOutput, StaticCache]:
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, legacy=False)
        prompt = ["what is a compiler?"]
        inputs = tokenizer(prompt, return_tensors="pt").to(self.device)
        mask = inputs["attention_mask"]

        cache = StaticCache(
            config=self.llama_config,
            batch_size=1,
            max_cache_len=128,
            dtype=torch.float16,
            device="cpu",
        )

        # model.register_buffer("kv_cache", cache.buffers)

        # repeatedly decode to populate kv cache
        generated_output: GenerateDecoderOnlyOutput = model.generate(
            **inputs, past_key_values=cache, generation_config=gen_config
        )

        return inputs, generated_output, cache

    def test_kv_decoderblock(self):
        model: LlamaModel = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        model.to(torch.float16)
        model.to(self.device)

        gen_config = GenerationConfig(
            num_beams=1,
            max_length=9,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
        )

        inputs, generated_output, cache = self.get_kv_for_model(model, gen_config)
        mask = inputs["attention_mask"]
        cache_copy = deepcopy(cache)

        decoder = LlamaDecoderLayer(self.llama_config, 0)
        outs = decoder(
            hidden_states=generated_output.hidden_states[-1][0],
            attention_mask=mask,
            past_key_value=cache,
            use_cache=True,
        )

        compiled_decoder = proteus(decoder)
        compiled_outs = compiled_decoder(
            hidden_states=generated_output.hidden_states,
            attention_mask=mask,
            past_key_value=cache_copy,
            use_cache=True,
        )

        self.compare_outs(outs, compiled_outs)

    def test_populated_kv(self):
        # if we add the kv cache tensors to the dict of mlx digital twins,
        # this might fix our issue and prevent the extra copies!
        model: LlamaModel = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float16
        )
        # model = LlamaForCausalLM(self.llama_config)

        gen_config = GenerationConfig(
            num_beams=1,
            max_length=8,
            do_sample=False,
            return_dict_in_generate=True,
        )

        # the way it SEEMS to work w/staticcache:
        # list of tensors of size [B, KV, L, D]
        # the list length is the num of attn blocks in the model
        # B is batch size (1)
        # KV is the number of KV attention heads (its multi query attention so multiple projected queries share a subset of KV activations)?
        # in llama's case, 32 attn heads and 8 kv heads per layer
        # L is max sequence length to compute
        # D is head dimension, how many numbers per K/V vector

        inputs, generated_output, cache = self.get_kv_for_model(model, gen_config)
        mask = inputs["attention_mask"]

        input_ids = generated_output.sequences

        gen_config.max_length = 21
        cache_copy = deepcopy(cache)
        outs = model.generate(
            inputs=input_ids,
            attention_mask=mask,
            past_key_values=cache,
            generation_config=gen_config,
        )

        model = proteus(model)
        compiled_outs = model.generate(
            inputs=input_ids,
            attention_mask=mask,
            past_key_values=cache_copy,
            generation_config=gen_config,
        )
        print("should be printing mlx cache bufs")

        # compiled_outs = model(**inputs, return_dict=False)

        self.compare_outs(outs.sequences, compiled_outs.sequences)

    # follow the huggingface compile recipe here:
    # https://github.com/huggingface/huggingface-llama-recipes/blob/main/performance_optimization/torch_compile.py
    def test_compile_llama(self):
        model: LlamaModel = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.float32
        )

        device = "cpu"
        model.to(device)

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, legacy=False)
        prompt = "what is a compiler?"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        torch._dynamo.config.cache_size_limit = 10
        gen_config = GenerationConfig(
            cache_implementation="static",
            cache_config={"batch_size": 1, "max_cache_len": 24},
            # use_cache=False,
            num_beams=1,
            max_length=25,
            do_sample=False,
        )

        # Set the model as our test model
        outputs = model.generate(**inputs, generation_config=gen_config)
        response = tokenizer.batch_decode(outputs)[0]
        print(f"Non-compiled response: {response}")
        # outputs = model(**inputs)
        # print(f"Non-compiled: {outputs.logits}")

        model.to("cpu")
        compiled = proteus(model)

        inputs.to("cpu")
        compiled_outputs = compiled.generate(**inputs, generation_config=gen_config)
        response = tokenizer.batch_decode(compiled_outputs)[0]
        print(f"Compiled response: {response}")
        # compiled_outputs = compiled(**inputs)
        # print(f"Non-compiled: {compiled_outputs.logits}")

        # Get sorted indices for comparison
        # torch_sorted_indices = torch.argsort(outputs.logits.flatten(), descending=True)
        # compiled_sorted_indices = torch.argsort(
        #     compiled_outputs.logits.flatten(), descending=True
        # )
        # print(f"\nTop 10 indices torch: {torch_sorted_indices[:10]}")
        # print(f"Top 10 indices compiled: {compiled_sorted_indices[:10]}")

    # def export_llama_getattr():
    #     model_name = "meta-llama/Llama-3.2-1B"
    #     model = AutoModelForCausalLM.from_pretrained(model_name)
    #     tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

    #     text = "Proteus is a cracked PyTorch compiler"
    #     test_in = tokenizer(text, return_tensors="pt")

    #     # Set the model as our test model

    #     # use export().run_decompositions() to get core aten ir graph
    #     # without lifing model params into inputs
    #     # compiled = aot_module(model, aten_opset_compiler)
    #     compiled = export(
    #         model,
    #         (test_in["input_ids"],),
    #         {"attention_mask": test_in["attention_mask"]},
    #         strict=False,
    #     ).module()
    #     named_params = dict(model.named_parameters())
    #     named_buffers = dict(model.named_buffers())
    #     DefaultInterpreter(compiled).boxed_run(
    #         [named_params, named_buffers, test_in["input_ids"]]
    #     )
    #     # compiled_graph = proteus(mod)
    #     compiled_out = compiled(
    #         test_in["input_ids"], attention_mask=test_in["attention_mask"]
    #     )
    #     # print(compiled_out[0])


if __name__ == "__main__":
    # TestLlama().test_llama_no_kv()
    # TestLlama().test_populated_kv()
    # TestLlama().test_kv_decoderblock()
    # unittest.main()
    TestLlama().test_compile_llama()

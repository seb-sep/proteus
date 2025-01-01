import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LlamaModel,
    GenerationConfig,
)
import torch

from proteus.proteus import proteus_v4

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # silence warnings when compiling


# follow the huggingface compile recipe here:
# https://github.com/huggingface/huggingface-llama-recipes/blob/main/performance_optimization/torch_compile.py
def compile_llama():
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    model: LlamaModel = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16
    )
    device = "cpu"
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)
    prompt = "what is a compiler?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    torch._dynamo.config.cache_size_limit = 10
    gen_config = GenerationConfig(
        use_cache=False,
        # cache_implementation="static",
        # cache_config={"batch_size": 1, "max_cache_len": 24},
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
    compiled = proteus_v4(model)

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


compile_llama()

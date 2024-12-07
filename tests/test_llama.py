from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from proteus.proteus import proteus, proteus_v3


def compile_llama():
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

    text = "Proteus is a cracked PyTorch compiler"
    test_in = tokenizer(text, return_tensors="pt")

    # Set the model as our test model
    test_out = model(test_in["input_ids"], attention_mask=test_in["attention_mask"])

    # use export().run_decompositions() to get core aten ir graph
    # without lifing model params into inputs
    compiled = torch.compile(model, fullgraph=True, backend=proteus_v3)
    compiled_out = compiled(
        test_in["input_ids"], attention_mask=test_in["attention_mask"]
    )
    print(compiled_out[0], test_out[0])


def export_llama_getattr():
    model_name = "meta-llama/Llama-3.2-1B"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, legacy=False)

    text = "Proteus is a cracked PyTorch compiler"
    test_in = tokenizer(text, return_tensors="pt")

    # Set the model as our test model

    # use export().run_decompositions() to get core aten ir graph
    # without lifing model params into inputs
    # compiled = aot_module(model, aten_opset_compiler)
    compiled = export(
        model,
        (test_in["input_ids"],),
        {"attention_mask": test_in["attention_mask"]},
        strict=False,
    ).module()
    named_params = dict(model.named_parameters())
    named_buffers = dict(model.named_buffers())
    DefaultInterpreter(compiled).boxed_run(
        [named_params, named_buffers, test_in["input_ids"]]
    )
    # compiled_graph = proteus(mod)
    compiled_out = compiled(
        test_in["input_ids"], attention_mask=test_in["attention_mask"]
    )
    # print(compiled_out[0])


compile_llama()

def compile_sd():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.float16
    )
    pipe = pipe.to("mps")
    pipe.transformer = proteus_no_compile(pipe.transformer)
    # pipe.transformer = torch.compile(pipe.transformer, backend=aten_opset_compiler)
    # compiled_pipe = proteus_no_compile(pipe)

    image = pipe(
        "A capybara holding a sign that reads Hello World",
        num_inference_steps=1,
        guidance_scale=4.5,
    ).images[0]
    image.save("capybara.png")

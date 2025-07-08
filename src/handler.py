import runpod
from diffusers import StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, DPMSolverMultistepScheduler
import torch
import base64
import io
import time

try:
    base_pipe = StableDiffusionXLPipeline.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    base_pipe.scheduler = DPMSolverMultistepScheduler.from_config(base_pipe.scheduler.config, use_karras_sigmas=True)
    base_pipe.to("cuda")

    refiner_pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    refiner_pipe.scheduler = DPMSolverMultistepScheduler.from_config(refiner_pipe.scheduler.config, use_karras_sigmas=True)
    refiner_pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    job_input = job['input']
    prompt = job_input['prompt']

    neg_prompt = "bad hands, bad anatomy, ugly, deformed, (face asymmetry, eyes asymmetry, deformed eyes, deformed mouth, open mouth)"

    time_start = time.time()

    # 1. Generate base image
    base_result = base_pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=50,
        guidance_scale=1.0,
        output_type="latent",         # Output latents for refiner
        denoising_end=0.7,            # Where base stops, refiner takes over
    )

    # 2. Refine image
    refined_image = refiner_pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        image=base_result.images,     # Pass latents from base
        num_inference_steps=50,
        guidance_scale=1.0,
        denoising_start=0.7,          # Refiner starts from here
    ).images[0]

    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    refined_image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')

runpod.serverless.start({"handler": handler})
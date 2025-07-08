""" Example handler file. """

import runpod
from diffusers import AutoPipelineForText2Image, DPMSolverMultistepScheduler
import torch
import base64
import io
import time

# If your handler runs inference on a model, load the model here.
# You will want models to be loaded into memory before starting serverless.

try:
    pipe = AutoPipelineForText2Image.from_pretrained(
        "SG161222/RealVisXL_V5.0",
        torch_dtype=torch.float16,
        variant="fp16",
        add_watermarker=False,
        use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
    pipe.to("cuda")
except RuntimeError:
    quit()

def handler(job):
    """ Handler function that will be used to process jobs. """
    job_input = job['input']
    prompt = job_input['prompt']

    neg_prompt = "(octane render, render, drawing, anime," \
    "bad photo, bad photography:1.3), (worst quality," \
    "low quality, blurry:1.2), (bad teeth, deformed teeth," \
    "deformed lips), (bad anatomy, bad proportions:1.1)," \
    "(deformed iris, deformed pupils), (deformed eyes, bad eyes)," \
    "(deformed face, ugly face, bad face), (deformed hands, bad hands," \
    "fused fingers), morbid, mutilated, mutation, disfigured"

    time_start = time.time()
    image = pipe(
        prompt=prompt,
        negative_prompt=neg_prompt,
        num_inference_steps=35,
        guidance_scale=1.0).images[0]
    print(f"Time taken: {time.time() - time_start}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_bytes = buffer.getvalue()

    return base64.b64encode(image_bytes).decode('utf-8')


runpod.serverless.start({"handler": handler})

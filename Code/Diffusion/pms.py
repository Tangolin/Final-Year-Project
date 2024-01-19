"""For pipelines, models and schedulers tutorial."""

# Using a pipeline directly
from diffusers import DDPMPipeline

ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256", use_safetensors=True)
image = ddpm(num_inference_steps=25).images[0]
image.save("test.png")

# Creating the same pipeline but on a granular level
import torch
from diffusers import DDPMScheduler, UNet2DModel

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True)

scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size))
inputs = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(inputs, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, inputs).prev_sample

from diffusers import DiffusionPipeline
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from typing import Dict


def generate_images(
    lora_model_id: str,
    model_base: str,
    prompts: Dict[str, str],
    negative_prompt: str,
    inference_dir: str,
    n_samples_per_prompt: int = 4,
    width: int = 768,
    height: int = 512,
    num_inference_steps: int = 100,
    guidance_scale: float = 7.5,
    cross_attention_scale: float = 0.85
):
    """
    Generates images using a pretrained diffusion model.

    Args:
        model_base (str): Base model name or path.
        prompts (Dict[str, str]): Dictionary of prompts for image generation.
        negative_prompt (str): Negative prompt to avoid certain features.
        inference_dir (str): Directory to save generated images.
        n_samples_per_prompt (int, optional): Number of samples per prompt. Defaults to 4.
        width (int, optional):  Width of generated images. Defaults to 768.
        height (int, optional): Height of generated images. Defaults to 512.
        num_inference_steps (int, optional): Number of inference steps. Defaults to 100.
        guidance_scale (float, optional): Guidance scale for the generation process. Defaults to 7.5.
        cross_attention_scale (float, optional): Scale for cross attention. Defaults to 0.85.
    """
    # Create inference dir to save images
    inference_dir.mkdir(exist_ok=True)

    pipe = DiffusionPipeline.from_pretrained(model_base, torch_dtype=torch.float16, safety_checker=None)
    pipe = pipe.to("cuda")

    fig = plt.figure(figsize=(20, 12))
    grid = ImageGrid(fig, 111, nrows_ncols=(len(prompts), n_samples_per_prompt), axes_pad=1, aspect=False)
    ax_count = 0
    for token, prompt in prompts.items():
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            num_images_per_prompt=n_samples_per_prompt,
            guidance_scale=guidance_scale,
            cross_attention_kwargs={"scale": cross_attention_scale},
        ).images

        grid[ax_count].set_title(prompt, fontsize=10, color='blue')

        for i in range(n_samples_per_prompt):
            images[i].save(f"{inference_dir}/tank-{token}-{i}.png")
            grid[ax_count].imshow(images[i])
            grid[ax_count].axis("off")
            ax_count += 1

    plt.show()

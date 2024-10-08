import gradio as gr
import torch
import torchvision
from diffusers import I2VGenXLPipeline, DiffusionPipeline
from torchvision.transforms.functional import to_tensor
from PIL import Image
from utils import create_progress_updater

if gr.NO_RELOAD:
    n_sdxl_steps = 50
    n_i2v_steps = 50
    high_noise_frac = 0.8
    negative_prompt = "Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms"
    generator = torch.manual_seed(8888)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_steps = n_sdxl_steps + n_i2v_steps
    print("Device:", device)

    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True,
    )
    # refiner = DiffusionPipeline.from_pretrained(
    #     "stabilityai/stable-diffusion-xl-refiner-1.0",
    #     text_encoder_2=base.text_encoder_2,
    #     vae=base.vae,
    #     torch_dtype=torch.float16,
    #     use_safetensors=True,
    #     variant="fp16",
    # )
    pipeline = I2VGenXLPipeline.from_pretrained("ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16")

    # base.to("cuda")
    # refiner.to("cuda")
    # pipeline.to("cuda")

    # base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)
    # refiner.unet = torch.compile(refiner.unet, mode="reduce-overhead", fullgraph=True)
    # pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)
    base.enable_model_cpu_offload()
    pipeline.enable_model_cpu_offload()
    pipeline.unet.enable_forward_chunking()

def generate(prompt: str, progress=gr.Progress()):
    progress((0, 100), desc="Generating first frame...")
    image = base(
        prompt=prompt,
        num_inference_steps=n_sdxl_steps,
        callback_on_step_end=create_progress_updater(
            start=0, 
            total=total_steps, 
            desc="Generating first frame...", 
            progress=progress,
        ),
    ).images[0]
    # progress((n_sdxl_steps * high_noise_frac, total_steps), desc="Refining first frame...")
    # image = refiner(
    #     prompt=prompt,
    #     num_inference_steps=n_sdxl_steps,
    #     denoising_start=high_noise_frac,
    #     image=image,
    #     callback_on_step_end=create_progress_updater(
    #         start=n_sdxl_steps * high_noise_frac,
    #         total=total_steps,
    #         desc="Refining first frame...",
    #         progress=progress,
    #     ),
    # ).images[0]
    image = to_tensor(image)
    progress((n_sdxl_steps, total_steps), desc="Generating video...")
    frames: list[Image.Image] = pipeline(
        prompt=prompt,
        image=image,
        num_inference_steps=50,
        negative_prompt=negative_prompt,
        guidance_scale=9.0,
        generator=generator,
        decode_chunk_size=2,
        num_frames=32,
    ).frames[0]
    progress((total_steps - 1, total_steps), desc="Finalizing...")
    frames = [to_tensor(frame.convert("RGB")).mul(255).byte().permute(1, 2, 0) for frame in frames]
    frames = torch.stack(frames)
    torchvision.io.write_video("video.mp4", frames, fps=8)
    return "video.mp4"

app = gr.Interface(
    fn=generate,
    inputs=["text"],
    outputs=gr.Video()
)

if __name__ == "__main__":
    app.launch()

# Vidcraft

This application uses advanced AI models to generate a video based on a text prompt. It combines Stable Diffusion XL for initial image generation and I2VGenXL for video generation.

## Description

This Gradio-based application allows users to input a text prompt, which is then used to generate a short video. The process involves two main steps:

1. Generating an initial image using Stable Diffusion XL based on the input prompt.
2. Creating a video from the initial image using I2VGenXL.

The app uses PyTorch and leverages GPU acceleration when available.

## Requirements

- Python 3.7+
- PyTorch
- torchvision
- Gradio
- Diffusers
- Pillow

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/text-to-video-generation.git
   cd text-to-video-generation
   ```

2. Install the required packages:
   ```
   pip install torch torchvision gradio diffusers Pillow
   ```

3. Download the necessary model weights (this will happen automatically on first run).

## Usage

1. Run the application:
   ```
   python app.py
   ```

2. Open your web browser and go to the URL provided in the terminal (usually `http://127.0.0.1:7860`).

3. Enter a text prompt in the input field and click "Submit".

4. Wait for the video generation process to complete. The progress will be displayed in the interface.

5. Once finished, the generated video will be displayed and can be downloaded.

## Notes

- The application uses CUDA if available, falling back to CPU if not.
- Video generation can be computationally intensive and may take some time, especially on CPU.
- The generated video is saved as "video.mp4" in the same directory as the script.

## Customization

You can adjust various parameters in the script, such as:

- `n_sdxl_steps`: Number of inference steps for Stable Diffusion XL
- `n_i2v_steps`: Number of inference steps for I2VGenXL
- `negative_prompt`: Negative prompt to guide the generation process
- `generator`: Random seed for reproducibility
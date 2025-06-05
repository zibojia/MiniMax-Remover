<h1 align="center">
  <span style="color:#2196f3;"><b>MiniMax</b></span><span style="color:#f06292;"><b>-Remover</b></span>: Taming Bad Noise Helps Video Object Removal
</h1>

<p align="center">
  <a href="https://github.com/zibojia"><b>Bojia Zi</b></a><sup>*</sup>,
  Weixuan Peng<sup>*</sup>,
  Xianbiao Qi<sup>†</sup>,
  Jianan Wang, Shihao Zhao, Rong Xiao, Kam-Fai Wong<br>
  <sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.
</p>

<p align="center">
  <a href="https://huggingface.co/zibojia/MiniMaxRemover"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/zibojia/MiniMax-Remover"><img alt="Github" src="https://img.shields.io/badge/MiniMaxRemover-github-grey"></a>
  <a href="https://huggingface.co/spaces/zibojia/MiniMaxRemover"><img alt="Huggingface Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Space-1e90ff"></a>
  <a href="https://arxiv.org/abs/2505.24873"><img alt="arXiv" src="https://img.shields.io/badge/MiniMaxRemover-arXiv-b31b1b"></a>
  <a href="https://www.youtube.com/watch?v=KaU5yNl6CTc"><img alt="YouTube" src="https://img.shields.io/badge/Youtube-video-ff0000"></a>
</p>

---

## 🚀 Overview

**MiniMax-Remover** is an advanced video object removal pipeline designed to robustly inpaint video regions while taming noise. It achieves state-of-the-art results by carefully handling temporal consistency and mask corruption, as described in our [arXiv paper](https://arxiv.org/abs/2505.24873).

- **[Online demo (Hugging Face Spaces)](https://huggingface.co/spaces/zibojia/MiniMaxRemover)**
- **[Demo Video (YouTube)](https://www.youtube.com/watch?v=KaU5yNl6CTc)**
- **[Model & Code (GitHub)](https://github.com/zibojia/minimax_remover)**

---

## 🛠️ Installation

All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## 📺 Demo Video

Click the image to watch the demo on YouTube:

[![Demo](https://img.youtube.com/vi/KaU5yNl6CTc/0.jpg)](https://www.youtube.com/watch?v=KaU5yNl6CTc)

---

## ⚡ Quick Start

### Minimal Example

```python
import torch
from diffusers.utils import export_to_video
from decord import VideoReader
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

random_seed = 42
video_length = 81
device = torch.device("cuda:0")

# Load model weights separately
vae = AutoencoderKLWan.from_pretrained("./vae", torch_dtype=torch.float16)
transformer = Transformer3DModel.from_pretrained("./transformer", torch_dtype=torch.float16)
scheduler = UniPCMultistepScheduler.from_pretrained("./scheduler")

# Load video and mask
def load_video(path):
    vr = VideoReader(path)
    imgs = vr.get_batch(list(range(video_length))).asnumpy()
    return torch.from_numpy(imgs) / 127.5 - 1.0  # [-1, 1]

def load_mask(path):
    vr = VideoReader(path)
    masks = vr.get_batch(list(range(video_length))).asnumpy()
    masks = torch.from_numpy(masks)[:, :, :, :1]
    masks[masks > 20] = 255
    masks[masks < 255] = 0
    return masks / 255.0  # [0, 1]

images = load_video("./video.mp4")      # images in range [-1, 1]
masks = load_mask("./mask.mp4")         # masks in range [0, 1]

# Initialize the pipeline (pass the loaded weights as objects)
pipe = Minimax_Remover_Pipeline(
    vae=vae,
    transformer=transformer,
    scheduler=scheduler,
    torch_dtype=torch.float16
).to(device)

result = pipe(
    images=images,
    masks=masks,
    num_frames=video_length,
    height=480,
    width=832,
    num_inference_steps=12,
    generator=torch.Generator(device=device).manual_seed(random_seed),
    iterations=6
).frames[0]
export_to_video(result, "./output.mp4")
```

- `images`: Video frames, must be normalized to the range `[-1, 1]`
- `masks`: Mask frames, must be normalized to the range `[0, 1]`

---

## 🔑 Important Parameters

When calling the pipeline, **do not change these unless you know what you're doing**:

- `height=480`
- `width=832`
- `num_inference_steps=12`

These control output resolution and the speed/quality tradeoff. Changing them may cause shape errors or degrade results.

---

## 📥 Input & Output

- **Input Video:** `./video.mp4`
- **Mask Video:** `./mask.mp4` (single channel, auto-binarized)
- **Output Video:** `./output.mp4`

**Note:** The input and mask video must have the same number of frames (default: 81).

---

## ⚙️ Parameter Descriptions

| Parameter             | Description                                                    | Default |
|-----------------------|----------------------------------------------------------------|---------|
| `iterations`          | Mask dilation hyperparameter (controls inpainting margin)      | 6       |
| `num_frames`          | Number of frames to process                                   | 81      |
| `height`, `width`     | Output video resolution                                       | 480,832 |
| `num_inference_steps` | Diffusion steps per frame (higher = better quality, slower)   | 12      |

Other parameters can be adjusted in the pipeline call.

---

## 📂 Model Weights

Place model weights in the following directories:

- `./vae/`
- `./transformer/`
- `./scheduler/`

You should load each component separately (as shown above) and pass the loaded objects to the pipeline.

---

## 💡 Notes

- `transformer_minimax_remover` and `pipeline_minimax_remover` are included as project modules and do **not** require separate installation.
- This repository is intended for **academic research only**. **Commercial use is strictly prohibited.**

---

## 🙏 Acknowledgements

This project is for academic research purposes only and must **not** be used for any commercial activities.

---

## 📧 Contact

Feel free to send an email to [19210240030@fudan.edu.cn](mailto:19210240030@fudan.edu.cn) if you have any questions or suggestions.

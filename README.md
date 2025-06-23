<h1 align="center">
  <span style="color:#2196f3;"><b>MiniMax</b></span><span style="color:#f06292;"><b>-Remover</b></span>: Taming Bad Noise Helps Video Object Removal
</h1>

<p align="center">
  Bojia Zi<sup>*</sup>,
  Weixuan Peng<sup>*</sup>,
  Xianbiao Qi<sup>†</sup>,
  Jianan Wang, Shihao Zhao, Rong Xiao, Kam-Fai Wong<br>
  <sup>*</sup> Equal contribution. <sup>†</sup> Corresponding author.
</p>

<p align="center">
  <a href="https://huggingface.co/zibojia/minimax-remover"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/zibojia/MiniMax-Remover"><img alt="Github" src="https://img.shields.io/badge/MiniMaxRemover-github-black"></a>
  <a href="https://huggingface.co/spaces/zibojia/MiniMax-Remover"><img alt="Huggingface Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Space-1e90ff"></a>
  <a href="https://arxiv.org/abs/2505.24873"><img alt="arXiv" src="https://img.shields.io/badge/MiniMaxRemover-arXiv-b31b1b"></a>
  <a href="https://www.youtube.com/watch?v=KaU5yNl6CTc"><img alt="YouTube" src="https://img.shields.io/badge/Youtube-video-ff0000"></a>
  <a href="https://minimax-remover.github.io"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
  <a href="https://replicate.com/ayushunleashed/minimax-remover"><img alt="Replicate" src="https://replicate.com/cjwbw/i2vgen-xl/badge"></a>
</p>

---

## 🚀 Overview

**MiniMax-Remover** is a fast and effective video object remover based on minimax optimization. It operates in two stages: the first stage trains a remover using a simplified DiT architecture, while the second stage distills a robust remover with CFG removal and fewer inference steps.

---

## ✨ Features: 

* **Fast:** Requires only 6 inference steps and does not use CFG, making it highly efficient.

* **Effective:** Seamlessly removes objects from videos and generates high-quality visual content.

* **Robust:** Maintains robustness by preventing the regeneration of undesired objects or artifacts within the masked region, even under varying noise conditions.

---

## 🛠️ Installation

All dependencies are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Gradio Demo

<p align="center">
  <a href="https://youtu.be/1V7Ov4vmnBc" target="_blank">
    <img src="./imgs/gradio_demo.gif" alt="firstpage" style="width:80%;" />
  </a>
</p>

You can use this gradio demo to remove objects. Note that you don't need to compile the sam2.
```bash
cd gradio_demo
python3 test.py
```

---

## 📂 Download

```shell
huggingface-cli download zibojia/minimax-remover --include vae transformer scheduler --local-dir .
```

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

images = # images in range [-1, 1]
masks = # masks in range [0, 1]

# Initialize the pipeline (pass the loaded weights as objects)
pipe = Minimax_Remover_Pipeline(vae=vae, transformer=transformer, \
    scheduler=scheduler, torch_dtype=torch.float16
).to(device)

result = pipe(images=images, masks=masks, num_frames=video_length, height=480, width=832, \
    num_inference_steps=12, generator=torch.Generator(device=device).manual_seed(random_seed), iterations=6 \
).frames[0]
export_to_video(result, "./output.mp4")
```
---

## 📧 Contact

Feel free to send an email to [19210240030@fudan.edu.cn](mailto:19210240030@fudan.edu.cn) if you have any questions or suggestions.

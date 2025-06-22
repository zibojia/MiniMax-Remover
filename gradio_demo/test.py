import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image

os.makedirs("./SAM2-Video-Predictor/checkpoints/", exist_ok=True)
os.makedirs("./model/", exist_ok=True)

from huggingface_hub import snapshot_download

def download_sam2():
    snapshot_download(repo_id="facebook/sam2-hiera-large", local_dir="./SAM2-Video-Predictor/checkpoints/")
    print("Download sam2 completed")

def download_remover():
    snapshot_download(repo_id="zibojia/minimax-remover", local_dir="./model/")
    print("Download minimax remover completed")

download_sam2()
download_remover()

import torch
import argparse
import random

import torch.nn.functional as F
import time
import random
from omegaconf import OmegaConf
from einops import rearrange
from diffusers.models import AutoencoderKLWan
import scipy
from transformer_minimax_remover import Transformer3DModel
from einops import rearrange
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline

from diffusers.utils import export_to_video
from decord import VideoReader, cpu
from moviepy.editor import ImageSequenceClip

from sam2 import load_model

from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0)
]

random_seed = 42
video_length = 201
W = 1024
H = W
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pipe_image_and_video_predictor():
    vae = AutoencoderKLWan.from_pretrained("./model/vae", torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained("./model/transformer", torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained("./model/scheduler")

    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.to(device)

    sam2_checkpoint = "./SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"
    config = "sam2_hiera_l.yaml"

    video_predictor = build_sam2_video_predictor(config, sam2_checkpoint, device=device)
    model = build_sam2(config, sam2_checkpoint, device=device)
    model.image_size = 1024
    image_predictor = SAM2ImagePredictor(sam_model=model)

    return pipe, image_predictor, video_predictor

def get_video_info(video_path, video_state):
    video_state["input_points"] = []
    video_state["scaled_points"] = []
    video_state["input_labels"] = []
    video_state["frame_idx"] = 0
    vr = VideoReader(video_path, ctx=cpu(0))
    first_frame = vr[0].asnumpy()
    del vr

    if first_frame.shape[0] > first_frame.shape[1]:
        W_ = W
        H_ = int(W_ * first_frame.shape[0] / first_frame.shape[1])
    else:
        H_ = H
        W_ = int(H_ * first_frame.shape[1] / first_frame.shape[0])

    first_frame = cv2.resize(first_frame, (W_, H_))
    video_state["origin_images"] = np.expand_dims(first_frame, axis=0)
    video_state["inference_state"] = None
    video_state["video_path"] = video_path
    video_state["masks"] = None
    video_state["painted_images"] = None
    image = Image.fromarray(first_frame)
    return image

def segment_frame(evt: gr.SelectData, label, video_state):
    if video_state["origin_images"] is None:
        gr.Warning("Please click \"Extract First Frame\" to extract the first frame first, then click the annotation")
        return None
    x, y = evt.index
    new_point = [x, y]
    label_value = 1 if label == "Positive" else 0

    video_state["input_points"].append(new_point)
    video_state["input_labels"].append(label_value)
    height, width = video_state["origin_images"][0].shape[0:2]
    scaled_points = []
    for pt in video_state["input_points"]:
        sx = pt[0] / width
        sy = pt[1] / height
        scaled_points.append([sx, sy])

    video_state["scaled_points"] = scaled_points

    image_predictor.set_image(video_state["origin_images"][0])
    mask, _, _ = image_predictor.predict(
        point_coords=video_state["scaled_points"],
        point_labels=video_state["input_labels"],
        multimask_output=False,
        normalize_coords=False,
    )

    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (width, height))
    mask = mask[:,:,None]

    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    org_image = video_state["origin_images"][0].astype(np.float32) / 255.0
    painted_image = (1 - mask * 0.5) * org_image + mask * 0.5 * color
    painted_image = np.uint8(np.clip(painted_image * 255, 0, 255))
    video_state["painted_images"] = np.expand_dims(painted_image, axis=0)
    video_state["masks"] = np.expand_dims(mask[:,:,0], axis=0)

    for i in range(len(video_state["input_points"])):
        point = video_state["input_points"][i]
        if video_state["input_labels"][i] == 0:
            cv2.circle(painted_image, point, radius=3, color=(0, 0, 255), thickness=-1)  # 红色点，半径为3
        else:
            cv2.circle(painted_image, point, radius=3, color=(255, 0, 0), thickness=-1)

    return Image.fromarray(painted_image)

def clear_clicks(video_state):
    video_state["input_points"] = []
    video_state["input_labels"] = []
    video_state["scaled_points"] = []
    video_state["inference_state"] = None
    video_state["masks"] = None
    video_state["painted_images"] = None
    return Image.fromarray(video_state["origin_images"][0]) if video_state["origin_images"] is not None else None


def preprocess_for_removal(images, masks):
    out_images = []
    out_masks = []
    for img, msk in zip(images, masks):
        if img.shape[0] > img.shape[1]:
            img_resized = cv2.resize(img, (480, 832), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = cv2.resize(img, (832, 480), interpolation=cv2.INTER_LINEAR)
        img_resized = img_resized.astype(np.float32) / 127.5 - 1.0  # [-1, 1]
        out_images.append(img_resized)
        if msk.shape[0] > msk.shape[1]:
            msk_resized = cv2.resize(msk, (480, 832), interpolation=cv2.INTER_NEAREST)
        else:
            msk_resized = cv2.resize(msk, (832, 480), interpolation=cv2.INTER_NEAREST)
        msk_resized = msk_resized.astype(np.float32)
        msk_resized = (msk_resized > 0.5).astype(np.float32)
        out_masks.append(msk_resized)
    arr_images = np.stack(out_images)
    arr_masks = np.stack(out_masks)
    return torch.from_numpy(arr_images).half().to(device), torch.from_numpy(arr_masks).half().to(device)


def inference_and_return_video(dilation_iterations, num_inference_steps, video_state=None):
    if video_state["origin_images"] is None or video_state["masks"] is None:
        return None
    images = video_state["origin_images"]
    masks = video_state["masks"]

    images = np.array(images)
    masks = np.array(masks)
    img_tensor, mask_tensor = preprocess_for_removal(images, masks)
    mask_tensor = mask_tensor[:,:,:,:1]

    if mask_tensor.shape[1] < mask_tensor.shape[2]:
        height = 480
        width = 832
    else:
        height = 832
        width = 480

    with torch.no_grad():
        out = pipe(
                images=img_tensor,
                masks=mask_tensor,
                num_frames=mask_tensor.shape[0],
                height=height,
                width=width,
                num_inference_steps=int(num_inference_steps),
                generator=torch.Generator(device=device).manual_seed(random_seed),
                iterations=int(dilation_iterations)
        ).frames[0]

        out = np.uint8(out * 255)
        output_frames = [img for img in out]

    video_file = f"/tmp/{time.time()}-{random.random()}-removed_output.mp4"
    clip = ImageSequenceClip(output_frames, fps=15)
    clip.write_videofile(video_file, codec='libx264', audio=False, verbose=False, logger=None)
    return video_file


def track_video(n_frames, video_state):
    if video_state["origin_images"] is None or video_state["masks"] is None:
        gr.Warning("Please complete target segmentation on the first frame first, then click Tracking")
        return None

    input_points = video_state["input_points"]
    input_labels = video_state["input_labels"]
    frame_idx = video_state["frame_idx"]
    obj_id = video_state["obj_id"]
    scaled_points = video_state["scaled_points"]

    vr = VideoReader(video_state["video_path"], ctx=cpu(0))
    height, width = vr[0].shape[0:2]
    images = [vr[i].asnumpy() for i in range(min(len(vr), n_frames))]
    del vr

    if images[0].shape[0] > images[0].shape[1]:
        W_ = W
        H_ = int(W_ * images[0].shape[0] / images[0].shape[1])
    else:
        H_ = H
        W_ = int(H_ * images[0].shape[1] / images[0].shape[0])

    images = [cv2.resize(img, (W_, H_)) for img in images]
    video_state["origin_images"] = images
    images = np.array(images)
    inference_state = video_predictor.init_state(images=images/255, device=device)
    video_state["inference_state"] = inference_state

    if len(torch.from_numpy(video_state["masks"][0]).shape) == 3:
        mask = torch.from_numpy(video_state["masks"][0])[:,:,0]
    else:
        mask = torch.from_numpy(video_state["masks"][0])

    video_predictor.add_new_mask(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=obj_id,
        mask=mask
    )

    output_frames = []
    mask_frames = []
    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        frame = images[out_frame_idx].astype(np.float32) / 255.0
        mask = np.zeros((H, W, 3), dtype=np.float32)
        for i, logit in enumerate(out_mask_logits):
            out_mask = logit.cpu().squeeze().detach().numpy()
            out_mask = (out_mask[:,:,None] > 0).astype(np.float32)
            mask += out_mask
        mask = np.clip(mask, 0, 1)
        mask = cv2.resize(mask, (W_, H_))
        mask_frames.append(mask)
        painted = (1 - mask * 0.5) * frame + mask * 0.5 * color
        painted = np.uint8(np.clip(painted * 255, 0, 255))
        output_frames.append(painted)
    video_state["masks"] = mask_frames
    video_file = f"/tmp/{time.time()}-{random.random()}-tracked_output.mp4"
    clip = ImageSequenceClip(output_frames, fps=15)
    clip.write_videofile(video_file, codec='libx264', audio=False, verbose=False, logger=None)
    return video_file

text = """
<div style='text-align:center; font-size:32px; font-family: Arial, Helvetica, sans-serif;'>
  Minimax-Remover: Taming Bad Noise Helps Video Object Removal
</div>
<div style="display: flex; justify-content: center; align-items: center; gap: 10px; flex-wrap: nowrap;">
  <a href="https://huggingface.co/zibojia/minimax-remover"><img alt="Huggingface Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Model-brightgreen"></a>
  <a href="https://github.com/zibojia/MiniMax-Remover"><img alt="Github" src="https://img.shields.io/badge/MiniMaxRemover-github-black"></a>
  <a href="https://huggingface.co/spaces/zibojia/MiniMaxRemover"><img alt="Huggingface Space" src="https://img.shields.io/badge/%F0%9F%A4%97%20Huggingface-Space-1e90ff"></a>
  <a href="https://arxiv.org/abs/2505.24873"><img alt="arXiv" src="https://img.shields.io/badge/MiniMaxRemover-arXiv-b31b1b"></a>
  <a href="https://www.youtube.com/watch?v=KaU5yNl6CTc"><img alt="YouTube" src="https://img.shields.io/badge/Youtube-video-ff0000"></a>
  <a href="https://minimax-remover.github.io"><img alt="Demo Page" src="https://img.shields.io/badge/Website-Demo%20Page-yellow"></a>
</div>
<div style='text-align:center; font-size:20px; margin-top: 10px; font-family: Arial, Helvetica, sans-serif;'>
  Bojia Zi<sup>*</sup>, Weixuan Peng<sup>*</sup>, Xianbiao Qi<sup>†</sup>, Jianan Wang, Shihao Zhao, Rong Xiao, Kam-Fai Wong
</div>
<div style='text-align:center; font-size:14px; color: #888; margin-top: 5px; font-family: Arial, Helvetica, sans-serif;'>
  <sup>*</sup> Equal contribution &nbsp; &nbsp; <sup>†</sup> Corresponding author
</div>
"""

pipe, image_predictor, video_predictor = get_pipe_image_and_video_predictor()

with gr.Blocks() as demo:
    video_state = gr.State({
        "origin_images": None,
        "inference_state": None,
        "masks": None,  # Store user-generated masks
        "painted_images": None,
        "video_path": None,
        "input_points": [],
        "scaled_points": [],
        "input_labels": [],
        "frame_idx": 0,
        "obj_id": 1
    })
    gr.Markdown(f"<div style='text-align:center;'>{text}</div>")

    with gr.Column():
        video_input = gr.Video(label="Upload Video", elem_id="my-video1")
        get_info_btn = gr.Button("Extract First Frame", elem_id="my-btn")

        gr.Examples(
            examples=[
                ["./cartoon/0.mp4"],
                ["./cartoon/1.mp4"],
                ["./cartoon/2.mp4"],
                ["./cartoon/3.mp4"],
                ["./cartoon/4.mp4"],
                ["./normal_videos/0.mp4"],
                ["./normal_videos/1.mp4"],
                ["./normal_videos/3.mp4"],
                ["./normal_videos/4.mp4"],
                ["./normal_videos/5.mp4"],
            ],
            inputs=[video_input],
            label="Choose a video to remove.",
            elem_id="my-btn2"
        )

        image_output = gr.Image(label="First Frame Segmentation", interactive=True, elem_id="my-video")#, height="35%", width="60%")
        demo.css = """
        #my-btn {
           width: 60% !important;
           margin: 0 auto;
        }

        #my-video1 {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #my-video {
           width: 60% !important;
           height: 35% !important;
           margin: 0 auto;
        }
        #my-md {
           margin: 0 auto;
        }
        #my-btn2 {
            width: 60% !important;
            margin: 0 auto;
        }
        #my-btn2 button {
            width: 120px !important;
            max-width: 120px !important;
            min-width: 120px !important;
            height: 70px !important;
            max-height: 70px !important;
            min-height: 70px !important;
            margin: 8px !important;
            border-radius: 8px !important;
            overflow: hidden !important;
            white-space: normal !important;
        }
        """
        with gr.Row(elem_id="my-btn"):
            point_prompt = gr.Radio(["Positive", "Negative"], label="Click Type", value="Positive")
            clear_btn = gr.Button("Clear All Clicks")

        with gr.Row(elem_id="my-btn"):
            n_frames_slider = gr.Slider(minimum=1, maximum=201, value=81, step=1, label="Tracking Frames N")
            track_btn = gr.Button("Tracking")
        video_output = gr.Video(label="Tracking Result", elem_id="my-video")

        with gr.Column(elem_id="my-btn"):
            dilation_slider = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="Mask Dilation")
            inference_steps_slider = gr.Slider(minimum=1, maximum=100, value=6, step=1, label="Num Inference Steps")

        remove_btn = gr.Button("Remove", elem_id="my-btn")
        remove_video = gr.Video(label="Remove Results", elem_id="my-video")
        remove_btn.click(
            inference_and_return_video,
            inputs=[dilation_slider, inference_steps_slider, video_state],
            outputs=remove_video
        )
        get_info_btn.click(get_video_info, inputs=[video_input, video_state], \
                       outputs=image_output)
        image_output.select(fn=segment_frame, inputs=[point_prompt, video_state], outputs=image_output)
        clear_btn.click(clear_clicks, inputs=video_state, outputs=image_output)
        track_btn.click(track_video, inputs=[n_frames_slider, video_state], outputs=video_output)

demo.launch(server_name="0.0.0.0", server_port=8000)

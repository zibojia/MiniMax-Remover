# create gradio demo to input one image and output one image
import os
import gradio as gr
import cv2
import numpy as np
from PIL import Image
import torch
import time
import random
from huggingface_hub import snapshot_download
from diffusers.models import AutoencoderKLWan
from transformer_minimax_remover import Transformer3DModel
from diffusers.schedulers import UniPCMultistepScheduler
from pipeline_minimax_remover import Minimax_Remover_Pipeline
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Create directories for models
os.makedirs("./SAM2-Video-Predictor/checkpoints/", exist_ok=True)
os.makedirs("./model/", exist_ok=True)

# Download models from Hugging Face Hub
def download_sam2():
    snapshot_download(repo_id="facebook/sam2-hiera-large", local_dir="./SAM2-Video-Predictor/checkpoints/")
    print("Download sam2 completed")

def download_remover():
    snapshot_download(repo_id="zibojia/minimax-remover", local_dir="./model/")
    print("Download minimax remover completed")

download_sam2()
download_remover()

COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
    (0, 255, 255), (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0)
]

random_seed = 42
W = 1024
H = W
device = "cuda" if torch.cuda.is_available() else "cpu"

def get_pipe_and_predictor():
    vae = AutoencoderKLWan.from_pretrained("./model/vae", torch_dtype=torch.float16)
    transformer = Transformer3DModel.from_pretrained("./model/transformer", torch_dtype=torch.float16)
    scheduler = UniPCMultistepScheduler.from_pretrained("./model/scheduler")

    pipe = Minimax_Remover_Pipeline(transformer=transformer, vae=vae, scheduler=scheduler)
    pipe.to(device)

    sam2_checkpoint = "./SAM2-Video-Predictor/checkpoints/sam2_hiera_large.pt"
    config = "sam2_hiera_l.yaml"

    model = build_sam2(config, sam2_checkpoint, device=device)
    model.image_size = 1024
    image_predictor = SAM2ImagePredictor(sam_model=model)

    return pipe, image_predictor

def get_image_info(image_pil, state):
    state["input_points"] = []
    state["scaled_points"] = []
    state["input_labels"] = []
    
    image_np = np.array(image_pil)

    if image_np.shape[0] > image_np.shape[1]:
        W_ = W
        H_ = int(W_ * image_np.shape[0] / image_np.shape[1])
    else:
        H_ = H
        W_ = int(H_ * image_np.shape[1] / image_np.shape[0])

    image_np = cv2.resize(image_np, (W_, H_))
    state["origin_image"] = image_np
    state["mask"] = None
    state["painted_image"] = None
    return Image.fromarray(image_np)

def segment_frame(evt: gr.SelectData, label, state):
    if state["origin_image"] is None:
        return None
    x, y = evt.index
    new_point = [x, y]
    label_value = 1 if label == "Positive" else 0

    state["input_points"].append(new_point)
    state["input_labels"].append(label_value)
    height, width = state["origin_image"].shape[0:2]
    scaled_points = []
    for pt in state["input_points"]:
        sx = pt[0]
        sy = pt[1]
        scaled_points.append([sx, sy])

    state["scaled_points"] = scaled_points

    image_predictor.set_image(state["origin_image"])
    mask, _, _ = image_predictor.predict(
        point_coords=np.array(state["scaled_points"]),
        point_labels=np.array(state["input_labels"]),
        multimask_output=False,
    )

    mask = np.squeeze(mask)
    mask = cv2.resize(mask, (width, height))
    mask = mask[:,:,None]

    color = np.array(COLOR_PALETTE[int(time.time()) % len(COLOR_PALETTE)], dtype=np.float32) / 255.0
    color = color[None, None, :]
    org_image = state["origin_image"].astype(np.float32) / 255.0
    painted_image = (1 - mask * 0.5) * org_image + mask * 0.5 * color
    painted_image = np.uint8(np.clip(painted_image * 255, 0, 255))
    state["painted_image"] = painted_image
    state["mask"] = mask[:,:,0]

    for i in range(len(state["input_points"])):
        point = state["input_points"][i]
        if state["input_labels"][i] == 0:
            cv2.circle(painted_image, tuple(point), radius=5, color=(0, 0, 255), thickness=-1)
        else:
            cv2.circle(painted_image, tuple(point), radius=5, color=(255, 0, 0), thickness=-1)

    return Image.fromarray(painted_image)

def clear_clicks(state):
    state["input_points"] = []
    state["input_labels"] = []
    state["scaled_points"] = []
    state["mask"] = None
    state["painted_image"] = None
    return Image.fromarray(state["origin_image"]) if state["origin_image"] is not None else None

def preprocess_for_removal(image, mask):
    if image.shape[0] > image.shape[1]:
        img_resized = cv2.resize(image, (480, 832), interpolation=cv2.INTER_LINEAR)
    else:
        img_resized = cv2.resize(image, (832, 480), interpolation=cv2.INTER_LINEAR)
    img_resized = img_resized.astype(np.float32) / 127.5 - 1.0  # [-1, 1]

    if mask.shape[0] > mask.shape[1]:
        msk_resized = cv2.resize(mask, (480, 832), interpolation=cv2.INTER_NEAREST)
    else:
        msk_resized = cv2.resize(mask, (832, 480), interpolation=cv2.INTER_NEAREST)
    msk_resized = msk_resized.astype(np.float32)
    msk_resized = (msk_resized > 0.5).astype(np.float32)
    
    return torch.from_numpy(img_resized).half().to(device), torch.from_numpy(msk_resized).half().to(device)

def inference_and_return_image(dilation_iterations, num_inference_steps, state=None):
    if state["origin_image"] is None or state["mask"] is None:
        return None
    image = state["origin_image"]
    mask = state["mask"]

    img_tensor, mask_tensor = preprocess_for_removal(image, mask)
    img_tensor = img_tensor.unsqueeze(0)
    mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(-1)

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
                num_frames=1,
                height=height,
                width=width,
                num_inference_steps=int(num_inference_steps),
                generator=torch.Generator(device=device).manual_seed(random_seed),
                iterations=int(dilation_iterations)
        ).frames[0]

        out = np.uint8(out * 255)
        
    return Image.fromarray(out[0])

pipe, image_predictor = get_pipe_and_predictor()

with gr.Blocks() as demo:
    state = gr.State({
        "origin_image": None,
        "mask": None,
        "painted_image": None,
        "input_points": [],
        "scaled_points": [],
        "input_labels": [],
    })
    gr.Markdown("<h1><center>Minimax-Remover: Image Object Removal</center></h1>")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Image", type="pil")
            get_info_btn = gr.Button("Load Image")
            
            point_prompt = gr.Radio(["Positive", "Negative"], label="Click Type", value="Positive")
            clear_btn = gr.Button("Clear All Clicks")

            dilation_slider = gr.Slider(minimum=1, maximum=20, value=6, step=1, label="Mask Dilation")
            inference_steps_slider = gr.Slider(minimum=1, maximum=100, value=20, step=1, label="Num Inference Steps")

            remove_btn = gr.Button("Remove Object")

        with gr.Column():
            image_output_segmentation = gr.Image(label="Segmentation", interactive=True)
            image_output_removed = gr.Image(label="Object Removed")

    get_info_btn.click(get_image_info, inputs=[image_input, state], outputs=image_output_segmentation)
    image_output_segmentation.select(fn=segment_frame, inputs=[point_prompt, state], outputs=image_output_segmentation)
    clear_btn.click(clear_clicks, inputs=state, outputs=image_output_segmentation)
    remove_btn.click(
        inference_and_return_image,
        inputs=[dilation_slider, inference_steps_slider, state],
        outputs=image_output_removed
    )

demo.launch(server_name="0.0.0.0", server_port=8000)
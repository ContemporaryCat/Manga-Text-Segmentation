import gradio as gr
import torch
import cv2
import numpy as np
import segmentation_models_pytorch as smp
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time

MODEL_PATH = "model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODER = "tu-efficientnetv2_rw_m"

def convert_batchnorm_to_groupnorm(module):
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = 8
            if num_channels < num_groups or num_channels % num_groups != 0:
                for i in range(min(num_channels, 8), 1, -1):
                    if num_channels % i == 0:
                        num_groups = i
                        break
                else:
                    num_groups = 1
            
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
        else:
            convert_batchnorm_to_groupnorm(child)

model = None

def load_model():
    global model
    if model is not None: return model
    
    print(f"Loading model from {MODEL_PATH}...")
    
    model = smp.UnetPlusPlus(
        encoder_name=ENCODER, 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=1,
        activation=None,
        decoder_attention_type='scse' 
    )
    
    convert_batchnorm_to_groupnorm(model.decoder)
    
    try:
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"WARNING: Model file not found at {MODEL_PATH}. Proceeding for UI demo.")

    model.to(DEVICE)
    model.eval()
    return model

def run_inference(image, tta_hflip, tta_vflip, progress=gr.Progress()):
    if image is None: return None
    
    progress(0, desc="Initializing...")
    
    net = load_model()
    
    h, w = image.shape[:2]
    pad_h = (32 - h % 32) % 32
    pad_w = (32 - w % 32) % 32
    
    transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    augmented = transform(image=image)
    tensor = augmented['image'].unsqueeze(0).to(DEVICE)
    
    if pad_h > 0 or pad_w > 0:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode='constant', value=0)
    
    steps = 1 + int(tta_hflip) + int(tta_vflip)
    current_step = 0
    
    accumulated_probs = None
    
    with torch.no_grad():
        progress(0.1, desc="Scanning page...")
        if DEVICE == 'cuda':
            with torch.amp.autocast('cuda'):
                logits = net(tensor)
                probs = logits.sigmoid()
        else:
            logits = net(tensor)
            probs = logits.sigmoid()
        
        accumulated_probs = probs
        current_step += 1

        if tta_hflip:
            progress(current_step / steps, desc="Augmenting (Horizontal)...")
            tensor_flip = torch.flip(tensor, [3]) # Flip width
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits_flip = net(tensor_flip)
                    probs_flip = logits_flip.sigmoid()
            else:
                logits_flip = net(tensor_flip)
                probs_flip = logits_flip.sigmoid()
            
            accumulated_probs += torch.flip(probs_flip, [3])
            current_step += 1
            
        if tta_vflip:
            progress(current_step / steps, desc="Augmenting (Vertical)...")
            tensor_flip = torch.flip(tensor, [2]) # Flip height
            if DEVICE == 'cuda':
                with torch.amp.autocast('cuda'):
                    logits_flip = net(tensor_flip)
                    probs_flip = logits_flip.sigmoid()
            else:
                logits_flip = net(tensor_flip)
                probs_flip = logits_flip.sigmoid()
            
            accumulated_probs += torch.flip(probs_flip, [2])
            current_step += 1

    final_probs = accumulated_probs / steps
    prob_map = final_probs[0, 0, :h, :w].cpu().numpy()
    
    progress(1.0, desc="Done")
    return prob_map

def process_outputs(image, prob_map, threshold, alpha, padding_iter, fill_holes, close_gaps_kernel):
    if image is None or prob_map is None: 
        return None, None, None
    
    if image.shape[:2] != prob_map.shape:
        return None, None, None

    binary_mask = (prob_map > threshold).astype(np.uint8) * 255
    
    if close_gaps_kernel > 0:
        k_size = int(close_gaps_kernel)
        if k_size % 2 == 0: k_size += 1
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel_morph)

    if fill_holes:
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(binary_mask, contours, -1, 255, -1)

    overlay = image.copy()
    red_layer = np.zeros_like(image)
    red_layer[:, :, 0] = 255 
    
    mask_bool = binary_mask == 255
    overlay[mask_bool] = cv2.addWeighted(image[mask_bool], 1-alpha, red_layer[mask_bool], alpha, 0)

    cleaned = image.copy()
    
    if padding_iter > 0:
        kernel_pad = np.ones((3,3), np.uint8)
        whiteout_mask = cv2.dilate(binary_mask, kernel_pad, iterations=int(padding_iter))
    else:
        whiteout_mask = binary_mask

    cleaned[whiteout_mask == 255] = [255, 255, 255]

    return overlay, cleaned, binary_mask

def clear_state():
    return None

manga_css = """
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif:ital,wght@0,400;0,700;1,400&display=swap');

body {
    background-color: #f0f0f0;
    background-image: radial-gradient(#cfcfcf 1px, transparent 1px);
    background-size: 20px 20px;
}
.gradio-container { font-family: 'Noto Serif', serif !important; }

h1, h2, h3, span, div { font-family: 'Noto Serif', serif !important; }
h1 { font-weight: 700; }

.block-container {
    background: white;
    border: 3px solid black !important;
    box-shadow: 8px 8px 0px rgba(0,0,0,1) !important;
    border-radius: 0px !important;
    margin-bottom: 20px;
}

button.primary {
    background-color: #000 !important;
    color: #fff !important;
    border: 2px solid black !important;
    font-family: 'Noto Serif', serif !important;
    font-weight: bold;
    font-size: 1.2rem !important;
    transition: all 0.2s ease;
    box-shadow: 4px 4px 0px #555 !important;
}
button.primary:hover { transform: translate(-2px, -2px); box-shadow: 6px 6px 0px #555 !important; }
button.primary:active { transform: translate(2px, 2px); box-shadow: 0px 0px 0px #555 !important; }

.image-container { border: 2px solid #000 !important; }
.tab-nav button.selected { border-bottom: 3px solid #000 !important; font-weight: bold; }

.progress-level-inner {
    background-color: #333 !important;
    background-image: repeating-linear-gradient(
        45deg,
        #000,
        #000 10px,
        #333 10px,
        #333 20px
    ) !important;
    border: 2px solid black !important;
    box-shadow: 2px 2px 0px rgba(0,0,0,0.5);
}

.progress-text {
    font-family: 'Noto Serif', serif !important;
    color: black !important;
    font-weight: bold;
}
"""

theme = gr.themes.Monochrome(
    primary_hue="neutral",
    radius_size=gr.themes.sizes.radius_none,
    font=[gr.themes.GoogleFont("Noto Serif"), "serif"],
).set(
    body_background_fill="#f0f0f0",
    block_background_fill="#ffffff",
    block_border_width="3px",
    block_border_color="#000000",
    input_border_width="2px",
    input_border_color="#000000",
)

with gr.Blocks(theme=theme, css=manga_css, title="Manga Cleaner") as app:
    
    cached_probs = gr.State(value=None)

    with gr.Row(elem_classes="header-row"):
        with gr.Column():
            gr.Markdown("# Manga Text Cleaner")
    
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown(
                    "<h3 style='color: white; background-color: black; padding: 8px; text-align: center; margin-bottom: 12px;'>Upload Page</h3>"
                )
                input_img = gr.Image(
                    label="Raw Scan", 
                    type="numpy", 
                    height=500,
                    elem_classes="image-container"
                )
            
            with gr.Accordion("Inference Settings", open=False):
                with gr.Row():
                    tta_h = gr.Checkbox(label="Horizontal flip TTA", value=False)
                    tta_v = gr.Checkbox(label="Vertical flip TTA", value=False)

            run_btn = gr.Button("Clean Page", variant="primary", size="lg")

            with gr.Accordion("Cleaning Settings", open=False):
                gr.Markdown("**Detection controls**")
                thresh_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.5, step=0.05, 
                    label="Detection sensitivity"
                )
                
                gr.Markdown("**Cleaning controls**")
                with gr.Row():
                    fill_holes_cb = gr.Checkbox(label="Fill enclosed holes", value=False)
                
                gap_slider = gr.Slider(
                    minimum=0, maximum=10, value=0, step=1,
                    label="Gap closing"
                )

                padding_slider = gr.Slider(
                    minimum=0, maximum=10, value=2, step=1, 
                    label="White-out expansion"
                )
                
                alpha_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.4, 
                    label="Overlay opacity",
                    visible=True
                )

        with gr.Column(scale=2):
            gr.Markdown("### Results")
            with gr.Tabs():
                with gr.TabItem("Detection Overlay"):
                    output_overlay = gr.Image(label="Overlay", type="numpy", height=700, show_label=False)
                
                with gr.TabItem("Binary Mask"):
                    output_mask = gr.Image(label="Mask", type="numpy", image_mode="L", height=700, show_label=False)

                with gr.TabItem("Cleaned Page"):
                    output_clean = gr.Image(label="Result", type="numpy", height=700, show_label=False)

    run_btn.click(
        fn=run_inference,
        inputs=[input_img, tta_h, tta_v],
        outputs=[cached_probs]
    ).then(
        fn=process_outputs,
        inputs=[input_img, cached_probs, thresh_slider, alpha_slider, padding_slider, fill_holes_cb, gap_slider],
        outputs=[output_overlay, output_clean, output_mask]
    )

    sliders = [thresh_slider, alpha_slider, padding_slider, fill_holes_cb, gap_slider]
    for slider in sliders:
        slider.change(
            fn=process_outputs,
            inputs=[input_img, cached_probs, thresh_slider, alpha_slider, padding_slider, fill_holes_cb, gap_slider],
            outputs=[output_overlay, output_clean, output_mask]
        )

    input_img.change(
        fn=clear_state,
        outputs=[cached_probs]
    )

if __name__ == "__main__":
    app.launch(inbrowser=True, share=False)
"""
Single-image zero-shot segmentation with DINOv3 + DINO-Text head.

Example:
  python dinotxt_seg_infer.py \
    --image E:/Python/pythonProjecttest/demo161/data/images/1_tile_r000_c000.jpg \
    --class-names "tree,soil,irrigation pipe" \
    --vision-weights E:/Python/pythonProjecttest/demo161/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
    --text-weights E:/Python/pythonProjecttest/demo161/dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth \
    --out-mask dinotxt_pred_idx.png \
    --out-overlay dinotxt_pred_overlay.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from PIL import Image

# ---- repo path / import model ----
REPO_DIR = Path(__file__).resolve().parent.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l  # noqa: E402


PROMPT_TEMPLATES: Sequence[str] = (
    "a bad photo of a {0}.",
    "a photo of many {0}.",
    "a sculpture of a {0}.",
    "a photo of the hard to see {0}.",
    "a low resolution photo of the {0}.",
    "a rendering of a {0}.",
    "graffiti of a {0}.",
    "a bad photo of the {0}.",
    "a cropped photo of the {0}.",
    "a tattoo of a {0}.",
    "the embroidered {0}.",
    "a photo of a hard to see {0}.",
    "a bright photo of a {0}.",
    "a photo of a clean {0}.",
    "a photo of a dirty {0}.",
    "a dark photo of the {0}.",
    "a drawing of a {0}.",
    "a photo of my {0}.",
    "the plastic {0}.",
    "a photo of the cool {0}.",
    "a close-up photo of a {0}.",
    "a black and white photo of the {0}.",
    "a painting of the {0}.",
    "a painting of a {0}.",
    "a pixelated photo of the {0}.",
    "a sculpture of the {0}.",
    "a bright photo of the {0}.",
    "a cropped photo of a {0}.",
    "a plastic {0}.",
    "a photo of the dirty {0}.",
    "a jpeg corrupted photo of a {0}.",
    "a blurry photo of the {0}.",
    "a photo of the {0}.",
    "a good photo of the {0}.",
    "a rendering of the {0}.",
    "a {0} in a video game.",
    "a photo of one {0}.",
    "a doodle of a {0}.",
    "a close-up photo of the {0}.",
    "a photo of a {0}.",
    "the origami {0}.",
    "the {0} in a video game.",
    "a sketch of a {0}.",
    "a doodle of the {0}.",
    "a origami {0}.",
    "a low resolution photo of a {0}.",
    "the toy {0}.",
    "a rendition of the {0}.",
    "a photo of the clean {0}.",
    "a photo of a large {0}.",
    "a rendition of a {0}.",
    "a photo of a nice {0}.",
    "a photo of a weird {0}.",
    "a blurry photo of a {0}.",
    "a cartoon {0}.",
    "art of a {0}.",
    "a sketch of the {0}.",
    "a embroidered {0}.",
    "a pixelated photo of a {0}.",
    "itap of the {0}.",
    "a jpeg corrupted photo of the {0}.",
    "a good photo of a {0}.",
    "a plushie {0}.",
    "a photo of the nice {0}.",
    "a photo of the small {0}.",
    "a photo of the weird {0}.",
    "the cartoon {0}.",
    "art of the {0}.",
    "a drawing of the {0}.",
    "a photo of the large {0}.",
    "a black and white photo of a {0}.",
    "the plushie {0}.",
    "a dark photo of a {0}.",
    "itap of a {0}.",
    "graffiti of the {0}.",
    "a toy {0}.",
    "itap of my {0}.",
    "a photo of a cool {0}.",
    "a photo of a small {0}.",
    "a tattoo of the {0}.",
)


class ShortSideResize(torch.nn.Module):
    def __init__(self, size: int, interpolation: TVT.InterpolationMode) -> None:
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = TVTF.get_dimensions(img)
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            new_w = self.size
            new_h = int(self.size * h / w)
            return TVTF.resize(img, [new_h, new_w], self.interpolation)
        new_h = self.size
        new_w = int(self.size * w / h)
        return TVTF.resize(img, [new_h, new_w], self.interpolation)


def encode_image(model, img: torch.Tensor):
    """Return patch tokens (blocks_feats) only, no backbone feats."""
    B, _, H, W = img.shape
    P = model.visual_model.backbone.patch_size
    new_H = int(np.ceil(H / P) * P)
    new_W = int(np.ceil(W / P) * P)
    if (H, W) != (new_H, new_W):
        img = F.interpolate(img, size=(new_H, new_W), mode="bicubic", align_corners=False)
    _, _, h_i, w_i = img.shape
    _, _, patch_tokens = model.visual_model.get_class_and_patch_tokens(img)
    blocks_feats = patch_tokens.reshape(B, h_i // P, w_i // P, -1).contiguous()
    return blocks_feats  # [B, h, w, D]


def predict_whole(model, img: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
    """Whole-image inference, returns [C, H, W] (low-res)."""
    _, H, W = img.shape
    blocks_feats = encode_image(model, img.unsqueeze(0))  # [1, h, w, D]
    _, h, w, _ = blocks_feats.shape
    blocks_feats = blocks_feats.squeeze(0)  # [h, w, D]
    blocks_feats = F.normalize(blocks_feats, p=2, dim=-1)
    cos = torch.einsum("cd,hwd->chw", text_features, blocks_feats)  # [C, h, w]
    return cos


def predict_slide(
    model,
    img: torch.Tensor,
    text_features: torch.Tensor,
    side: int,
    stride: int,
) -> torch.Tensor:
    """Sliding-window inference, returns [C, H, W] at image resolution."""
    _, H, W = img.shape
    C, _ = text_features.shape
    device = img.device
    probs = torch.zeros([C, H, W], device=device)
    counts = torch.zeros([H, W], device=device)
    h_grids = max(H - side + stride - 1, 0) // stride + 1
    w_grids = max(W - side + stride - 1, 0) // stride + 1
    for i in range(h_grids):
        for j in range(w_grids):
            y1 = i * stride
            x1 = j * stride
            y2 = min(y1 + side, H)
            x2 = min(x1 + side, W)
            y1 = max(y2 - side, 0)
            x1 = max(x2 - side, 0)

            img_win = img[:, y1:y2, x1:x2]
            cos = predict_whole(model, img_win, text_features)  # [C, h, w]
            cos = F.interpolate(
                cos.unsqueeze(0),
                size=img_win.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)  # [C, H_win, W_win]
            probs[:, y1:y2, x1:x2] += cos.softmax(dim=0)
            counts[y1:y2, x1:x2] += 1
    probs /= counts
    return probs


def parse_args():
    p = argparse.ArgumentParser(description="DINOv3 + DINO-Text zero-shot segmentation (single image)")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--class-names", required=True, help='Comma separated class names, e.g. "tree,soil,pipe"')
    p.add_argument("--vision-weights", required=True, help="Path to ViT-L vision backbone weights")
    p.add_argument("--text-weights", required=True, help="Path to dinotxt head + text encoder weights")
    p.add_argument("--bpe-path", default=None, help="Optional local BPE vocab path (bpe_simple_vocab_16e6.txt.gz)")
    p.add_argument("--out-mask", default="dinotxt_pred_idx.png", help="Output path for index mask PNG")
    p.add_argument("--out-overlay", default="dinotxt_pred_overlay.png", help="Output path for overlay PNG")
    p.add_argument("--resize", type=int, default=512, help="Short side resize before inference")
    p.add_argument("--mode", choices=["whole", "slide"], default="slide", help="Inference mode")
    p.add_argument("--side", type=int, default=384, help="Sliding window side (for slide mode)")
    p.add_argument("--stride", type=int, default=192, help="Sliding window stride (for slide mode)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cuda or cpu")
    return p.parse_args()


def main():
    args = parse_args()

    # Load model
    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        pretrained=True,
        weights=args.text_weights,
        backbone_weights=args.vision_weights,
        bpe_path_or_url=args.bpe_path,
    )
    device = torch.device(args.device)
    model.to(device, non_blocking=True).eval()
    tokenizer = tokenizer.tokenize

    # Transform
    transform = TVT.Compose(
        [
            ShortSideResize(args.resize, TVT.InterpolationMode.BICUBIC),
            TVT.ToTensor(),
            TVT.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Load image
    img_pil = Image.open(args.image).convert("RGB")
    H0, W0 = img_pil.height, img_pil.width
    img = transform(img_pil)  # [3, H, W]
    print(f"Image loaded: {args.image}, tensor shape: {img.shape}, orig HW=({H0},{W0})")

    # Text features
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]
    text_feats = []
    for cname in class_names:
        text = [tmpl.format(cname) for tmpl in PROMPT_TEMPLATES]
        tokens = tokenizer(text).to(device, non_blocking=True)
        feats = model.encode_text(tokens)          # [num_prompts, 2D]
        feats = feats[:, feats.shape[1] // 2 :]    # keep second half
        feats = F.normalize(feats, p=2, dim=-1)
        feats = feats.mean(dim=0)
        feats = F.normalize(feats, p=2, dim=-1)
        text_feats.append(feats)
    text_feats = torch.stack(text_feats)  # [C, D]
    print(f"text_feats: {text_feats.shape}, classes: {class_names}")

    img_tensor = img.to(device, non_blocking=True)
    if args.mode == "whole":
        pred = predict_whole(model, img_tensor, text_feats)  # [C, h, w]
    else:
        pred = predict_slide(model, img_tensor, text_feats, args.side, args.stride)  # [C, H, W]

    # Upsample and argmax
    pred = F.interpolate(pred.unsqueeze(0), size=(H0, W0), mode="bilinear", align_corners=False)
    pred = pred.squeeze(0).argmax(dim=0).cpu().numpy()  # [H0, W0]
    print(f"Pred mask shape: {pred.shape}")

    # Save mask
    import imageio

    imageio.imwrite(args.out_mask, pred.astype(np.uint8))
    print(f"Saved mask: {args.out_mask}")

    # Save overlay
    overlay = np.array(img_pil).astype(np.float32)
    rng_colors = np.array(
        [
            [0, 153, 0],
            [128, 77, 26],
            [0, 0, 255],
            [255, 0, 0],
            [255, 165, 0],
        ],
        dtype=np.float32,
    )
    colors = rng_colors[: len(class_names)]
    alpha = 0.45
    for idx, color in enumerate(colors):
        mask = pred == idx
        if not np.any(mask):
            continue
        overlay[mask] = overlay[mask] * (1 - alpha) + color * alpha
    Image.fromarray(overlay.astype(np.uint8)).save(args.out_overlay)
    print(f"Saved overlay: {args.out_overlay}")


if __name__ == "__main__":
    main()


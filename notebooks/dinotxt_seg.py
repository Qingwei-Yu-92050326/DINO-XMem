# -*- coding: utf-8 -*-
"""
DINOv3 + DINOtxt 零样本语义分割（单张图像 + 自定义文本类别）

功能特点：
1. 支持 whole-image 推理（显存友好）
2. 支持 sliding-window 推理（适合超大分辨率）
3. 文本侧使用 Prompt Ensembling（少模板，低显存）
4. 输出类别索引掩码 + 叠加可视化
"""

import dataclasses
import math
import os
import sys
import warnings
from typing import Callable, List, Tuple

import lovely_tensors
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F
import torchvision.transforms as TVT
import torchvision.transforms.functional as TVTF
from omegaconf import OmegaConf
from torch import Tensor, nn

# ============================================================
# 0. 路径配置（需要你自己修改的部分）
# ============================================================

# DINOv3 源码路径
DINOv3_REPO_DIR = r"E:\Python\pythonProjecttest\demo161\dinov3-main"

# DINOtxt 视觉头 + 文本编码器权重
WEIGHTS = r"E:\Python\pythonProjecttest\demo161\dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth"

# DINOv3 ViT-L backbone 预训练权重
BACKBONE_WEIGHTS = r"E:\Python\pythonProjecttest\demo161\dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"

# 待推理图像路径
IMAGE_PATH = r"E:\Python\pythonProjecttest\demo161\data\images\1_tile_r000_c000.jpg"

# 自定义文本类别（可任意修改）
CLASS_NAMES = ["tree", "crop", "green round plant"]

# Torch 权重缓存目录（可选）
os.environ["TORCH_HOME"] = r"E:\Python\pythonProjecttest\demo161"

# ============================================================
# 1. 配置类（集中管理推理参数）
# ============================================================

@dataclasses.dataclass
class Configuration:
    """
    推理相关配置
    """
    # 推理模式：
    #   "whole"：整图一次前向（推荐，显存占用最低）
    #   "slide"：滑窗推理（更稳健，但慢）
    mode: str = "whole"

    # 输入图像短边 resize 到该尺寸
    resize: int = 384

    # 以下参数仅在 slide 模式下使用
    side: int = 256       # 滑窗尺寸
    stride: int = 192     # 滑窗步长

    # 文本 prompt 相关
    prompt_templates_use: int = 4   # 使用前 K 个模板
    prompt_batch: int = 2           # 文本编码 batch size


# ============================================================
# 2. 图像与特征处理工具函数
# ============================================================

# ImageNet 标准归一化
NORMALIZE_IMAGENET = TVT.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


class ShortSideResize(nn.Module):
    """
    将图像短边 resize 到指定尺寸，保持纵横比
    """
    def __init__(self, size: int, interpolation: TVT.InterpolationMode):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, img: Tensor) -> Tensor:
        _, h, w = TVTF.get_dimensions(img)
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return img
        if w < h:
            new_w = self.size
            new_h = int(self.size * h / w)
        else:
            new_h = self.size
            new_w = int(self.size * w / h)
        return TVTF.resize(img, [new_h, new_w], self.interpolation)


def encode_image(model, img: Tensor) -> Tuple[Tensor, Tensor]:
    """
    从 DINOv3 中提取 patch-level 特征

    输入：
        img: [B, 3, H, W]

    输出：
        backbone_patches: 预留（此处不用）
        blocks_patches: [B, h, w, D]
    """
    B, _, H, W = img.shape
    P = model.visual_model.backbone.patch_size

    # 对齐到 patch size 的整数倍
    new_H = math.ceil(H / P) * P
    new_W = math.ceil(W / P) * P
    if (H, W) != (new_H, new_W):
        img = F.interpolate(img, (new_H, new_W), mode="bicubic", align_corners=False)

    backbone_patches = None
    _, _, patch_tokens = model.visual_model.get_class_and_patch_tokens(img)

    h = new_H // P
    w = new_W // P
    blocks_patches = patch_tokens.reshape(B, h, w, -1).contiguous()
    return backbone_patches, blocks_patches


def predict_whole(model, img: Tensor, text_features: Tensor) -> Tensor:
    """
    整图推理（patch grid 分辨率）

    输入：
        img: [3, H, W]
        text_features: [C, D]

    输出：
        cosine logits: [C, h, w]
    """
    _, blocks = encode_image(model, img.unsqueeze(0))
    blocks = F.normalize(blocks.squeeze(0), dim=-1)
    logits = torch.einsum("cd,hwd->chw", text_features, blocks)
    return logits


def predict_slide(model, img: Tensor, text_features: Tensor, side: int, stride: int) -> Tensor:
    """
    滑窗推理（在图像分辨率累积预测）
    """
    _, H, W = img.shape
    C = text_features.shape[0]

    probs = torch.zeros((C, H, W), device=img.device)
    counts = torch.zeros((H, W), device=img.device)

    h_grids = max(H - side + stride - 1, 0) // stride + 1
    w_grids = max(W - side + stride - 1, 0) // stride + 1

    for i in range(h_grids):
        for j in range(w_grids):
            y1, x1 = i * stride, j * stride
            y2, x2 = min(y1 + side, H), min(x1 + side, W)
            y1, x1 = max(y2 - side, 0), max(x2 - side, 0)

            window = img[:, y1:y2, x1:x2]
            logits = predict_whole(model, window, text_features)
            logits = F.interpolate(
                logits.unsqueeze(0),
                size=window.shape[1:],
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

            probs[:, y1:y2, x1:x2] += logits.softmax(dim=0)
            counts[y1:y2, x1:x2] += 1

    return probs / counts.clamp_min(1)


# ============================================================
# 3. 文本 Prompt 模板
# ============================================================

PROMPT_TEMPLATES = (
    "a photo of a {0}.",
    "a bad photo of a {0}.",
    "a photo of many {0}.",
    "a photo of the hard to see {0}.",
    "a drawing of a {0}.",
    "a close-up photo of a {0}.",
    "a cropped photo of the {0}.",
    "a bright photo of a {0}.",
    "a blurry photo of a {0}.",
    "a black and white photo of the {0}.",
)


def build_text_features(
    model,
    tokenizer_fn,
    class_names: List[str],
    templates: Tuple[str, ...],
    cfg: Configuration,
    device: torch.device,
) -> torch.Tensor:
    """
    构建文本特征（Prompt Ensembling）

    返回：
        text_features: [C, D]，L2 归一化
    """
    all_feats = []

    for cname in class_names:
        prompts = [t.format(cname) for t in templates[: cfg.prompt_templates_use]]
        feats_chunks = []

        for i in range(0, len(prompts), cfg.prompt_batch):
            tokens = tokenizer_fn(prompts[i : i + cfg.prompt_batch]).to(device)
            with torch.inference_mode(), torch.cuda.amp.autocast(False):
                feats = model.encode_text(tokens)
                feats = feats[:, feats.shape[1] // 2 :]
                feats = F.normalize(feats, dim=-1)
            feats_chunks.append(feats.cpu())

        feats = torch.cat(feats_chunks, dim=0).mean(dim=0)
        all_feats.append(F.normalize(feats, dim=0))

    return torch.stack(all_feats).to(device)


# ============================================================
# 4. 主函数
# ============================================================

def main():
    lovely_tensors.monkey_patch()
    warnings.filterwarnings("ignore", message="xFormers")

    cfg = OmegaConf.to_object(OmegaConf.structured(Configuration))
    print(OmegaConf.to_yaml(cfg))

    # 加载模型
    sys.path.append(DINOv3_REPO_DIR)
    from dinov3.hub.dinotxt import dinov3_vitl16_dinotxt_tet1280d20h24l

    model, tokenizer = dinov3_vitl16_dinotxt_tet1280d20h24l(
        pretrained=True,
        weights=WEIGHTS,
        backbone_weights=BACKBONE_WEIGHTS,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 图像预处理
    transform = TVT.Compose([
        ShortSideResize(cfg.resize, TVT.InterpolationMode.BICUBIC),
        TVT.ToTensor(),
        NORMALIZE_IMAGENET,
    ])

    img_pil = PIL.Image.open(IMAGE_PATH).convert("RGB")
    H0, W0 = img_pil.height, img_pil.width
    img = transform(img_pil).to(device)

    # 文本特征
    text_features = build_text_features(
        model,
        tokenizer.tokenize,
        CLASS_NAMES,
        PROMPT_TEMPLATES,
        cfg,
        device,
    )

    # 推理
    with torch.inference_mode(), torch.cuda.amp.autocast():
        if cfg.mode == "whole":
            logits = predict_whole(model, img, text_features)
            logits = F.interpolate(logits.unsqueeze(0), (H0, W0), mode="bilinear", align_corners=False)
            pred = logits.squeeze(0).argmax(0).cpu().numpy()
        else:
            probs = predict_slide(model, img, text_features, cfg.side, cfg.stride)
            pred = probs.argmax(0).cpu().numpy()

    # 可视化
    import matplotlib.pyplot as plt
    from matplotlib import colors

    cmap = colors.ListedColormap(np.random.rand(len(CLASS_NAMES), 3))
    plt.imshow(img_pil)
    plt.imshow(pred, cmap=cmap, alpha=0.4)
    plt.axis("off")
    plt.show()

    # 保存结果
    import imageio
    imageio.imwrite("dinotxt_pred_idx.png", pred.astype(np.uint8))
    print("Saved: dinotxt_pred_idx.png")


if __name__ == "__main__":
    main()

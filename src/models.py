"""Model loading and encoding helpers for the fashion retrieval pipeline.

The functions in this module mirror the working Colab notebook:

- OpenCLIP ViT-B/32 with LAION2B weights for image and text embeddings.
- BLIP base image captioning for auxiliary caption generation.
- L2-normalized embeddings so FAISS inner product behaves like cosine search.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import open_clip
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor


DEFAULT_OPENCLIP_MODEL = "ViT-B-32"
DEFAULT_OPENCLIP_PRETRAINED = "laion2b_s34b_b79k"
DEFAULT_BLIP_MODEL = "Salesforce/blip-image-captioning-base"


@dataclass(frozen=True)
class OpenCLIPBundle:
    """Container for the OpenCLIP model, tokenizer, preprocessing, and device."""

    model: torch.nn.Module
    preprocess: Any
    tokenizer: Any
    device: str


@dataclass(frozen=True)
class BLIPBundle:
    """Container for the BLIP captioning model, processor, and device."""

    model: BlipForConditionalGeneration
    processor: BlipProcessor
    device: str


def get_device() -> str:
    """Return the same CUDA-first device choice used by the notebook."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def load_openclip(
    model_name: str = DEFAULT_OPENCLIP_MODEL,
    pretrained: str = DEFAULT_OPENCLIP_PRETRAINED,
    device: str | None = None,
) -> OpenCLIPBundle:
    """Load OpenCLIP and its tokenizer.

    Args:
        model_name: OpenCLIP architecture name. The notebook uses ``ViT-B-32``.
        pretrained: OpenCLIP pretrained weights identifier.
        device: Optional target device. Defaults to CUDA when available.

    Returns:
        An :class:`OpenCLIPBundle` ready for image and text encoding.
    """

    resolved_device = device or get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    model = model.to(resolved_device)
    model.eval()

    return OpenCLIPBundle(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        device=resolved_device,
    )


def load_blip(
    model_id: str = DEFAULT_BLIP_MODEL,
    device: str | None = None,
) -> BLIPBundle:
    """Load the BLIP image captioning model used by the notebook."""

    resolved_device = device or get_device()
    processor = BlipProcessor.from_pretrained(model_id)
    model = BlipForConditionalGeneration.from_pretrained(model_id).to(resolved_device)
    model.eval()

    return BLIPBundle(model=model, processor=processor, device=resolved_device)


def encode_image(
    image_path: str | Path,
    clip: OpenCLIPBundle,
) -> np.ndarray:
    """Encode one image as a normalized OpenCLIP vector."""

    image = (
        clip.preprocess(Image.open(image_path).convert("RGB"))
        .unsqueeze(0)
        .to(clip.device)
    )

    with torch.no_grad():
        feature = clip.model.encode_image(image)
        feature /= feature.norm(dim=-1, keepdim=True)

    return feature.cpu().numpy()[0].astype("float32")


def encode_text(
    texts: str | Sequence[str],
    clip: OpenCLIPBundle,
) -> np.ndarray:
    """Encode one or more text prompts as normalized OpenCLIP vectors."""

    normalized_texts = [texts] if isinstance(texts, str) else list(texts)
    tokens = clip.tokenizer(normalized_texts).to(clip.device)

    with torch.no_grad():
        features = clip.model.encode_text(tokens)
        features /= features.norm(dim=-1, keepdim=True)

    return features.cpu().numpy().astype("float32")


def generate_caption(
    image_path: str | Path,
    blip: BLIPBundle,
    max_new_tokens: int = 20,
) -> str:
    """Generate one BLIP caption for an image."""

    image = Image.open(image_path).convert("RGB")
    inputs = blip.processor(images=image, return_tensors="pt").to(blip.device)

    with torch.no_grad():
        output = blip.model.generate(**inputs, max_new_tokens=max_new_tokens)

    return blip.processor.decode(output[0], skip_special_tokens=True)

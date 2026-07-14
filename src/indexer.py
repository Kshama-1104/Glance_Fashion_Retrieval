"""Embedding and FAISS indexing helpers.

This module extracts the reusable indexing pieces from the notebook while
preserving the retrieval design:

- OpenCLIP image embeddings are normalized to 512-dimensional ``float32``.
- FAISS ``IndexFlatIP`` performs inner-product search over normalized vectors.
- BLIP captions are embedded with the same OpenCLIP text encoder.
- Hybrid embeddings use the notebook's ``0.7 * image + 0.3 * caption`` rule.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import faiss
import numpy as np
from tqdm import tqdm

from .models import OpenCLIPBundle, encode_image, encode_text


def build_image_embeddings(
    image_paths: Iterable[str | Path],
    clip: OpenCLIPBundle,
    show_progress: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Encode images with OpenCLIP.

    Returns both embeddings and the image paths that were successfully encoded.
    This keeps FAISS ids aligned with image paths if an unreadable image is
    skipped.
    """

    paths = [str(path) for path in image_paths]
    iterator = tqdm(paths) if show_progress else paths
    embeddings: list[np.ndarray] = []
    valid_paths: list[str] = []

    for image_path in iterator:
        try:
            embeddings.append(encode_image(image_path, clip))
            valid_paths.append(image_path)
        except Exception:
            continue

    return np.array(embeddings).astype("float32"), valid_paths


def build_caption_embeddings(
    captions: Iterable[str],
    clip: OpenCLIPBundle,
    show_progress: bool = True,
) -> np.ndarray:
    """Encode BLIP captions with the OpenCLIP text encoder."""

    caption_list = list(captions)
    iterator = tqdm(caption_list) if show_progress else caption_list
    embeddings: list[np.ndarray] = []

    for caption in iterator:
        embedding = encode_text([caption], clip)[0]
        embeddings.append(embedding)

    return np.array(embeddings).astype("float32")


def create_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Create a FAISS inner-product index from normalized embeddings."""

    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    return index


def build_hybrid_embeddings(
    image_embeddings: np.ndarray,
    caption_embeddings: np.ndarray,
    image_weight: float = 0.7,
    caption_weight: float = 0.3,
) -> np.ndarray:
    """Combine image and caption embeddings using the notebook weighting."""

    if image_embeddings.shape != caption_embeddings.shape:
        raise ValueError("image and caption embeddings must have matching shapes")

    hybrid_embeddings = image_weight * image_embeddings + caption_weight * caption_embeddings
    hybrid_embeddings /= np.linalg.norm(hybrid_embeddings, axis=1, keepdims=True)
    return hybrid_embeddings.astype("float32")


def save_faiss_index(index: faiss.Index, path: str | Path) -> None:
    """Persist a FAISS index to disk."""

    faiss.write_index(index, str(path))


def load_faiss_index(path: str | Path) -> faiss.Index:
    """Load a FAISS index from disk."""

    return faiss.read_index(str(path))

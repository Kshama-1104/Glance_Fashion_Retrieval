"""Search and visualization helpers for text-to-fashion retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import faiss
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .models import OpenCLIPBundle, encode_text


@dataclass(frozen=True)
class RetrievalResult:
    """Search output aligned with FAISS ids and source image paths."""

    query: str
    scores: np.ndarray
    ids: np.ndarray
    image_paths: list[str]


def search(
    query: str,
    index: faiss.Index,
    clip: OpenCLIPBundle,
    image_paths: Sequence[str | Path],
    top_k: int = 5,
) -> RetrievalResult:
    """Search a FAISS index with an OpenCLIP text query."""

    query_embedding = encode_text([query], clip)
    scores, ids = index.search(query_embedding.astype("float32"), top_k)
    paths = [str(image_paths[idx]) for idx in ids[0]]

    return RetrievalResult(
        query=query,
        scores=scores[0],
        ids=ids[0],
        image_paths=paths,
    )


def plot_results(
    result: RetrievalResult,
    figsize: tuple[int, int] = (15, 5),
) -> None:
    """Plot retrieved images in the same style as the notebook."""

    top_k = len(result.ids)
    plt.figure(figsize=figsize)

    for position, image_path in enumerate(result.image_paths):
        plt.subplot(1, top_k, position + 1)
        plt.imshow(Image.open(image_path))
        plt.title(f"{result.scores[position]:.2f}")
        plt.axis("off")

    plt.show()


def search_and_plot(
    query: str,
    index: faiss.Index,
    clip: OpenCLIPBundle,
    image_paths: Sequence[str | Path],
    top_k: int = 5,
) -> RetrievalResult:
    """Run retrieval and display the result grid."""

    result = search(
        query=query,
        index=index,
        clip=clip,
        image_paths=image_paths,
        top_k=top_k,
    )
    plot_results(result)
    return result

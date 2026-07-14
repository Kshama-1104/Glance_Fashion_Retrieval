"""Utility functions for data discovery and artifact persistence."""

from __future__ import annotations

import glob
import pickle
import random
from pathlib import Path
from typing import Any, Iterable


def list_image_paths(root: str | Path, pattern: str = "**/*.jpg") -> list[str]:
    """Recursively list image paths using the notebook's glob pattern."""

    search_pattern = str(Path(root) / pattern)
    return glob.glob(search_pattern, recursive=True)


def sample_paths(
    image_paths: Iterable[str | Path],
    sample_size: int = 1000,
    seed: int = 42,
) -> list[str]:
    """Take the same deterministic random sample used in the notebook."""

    paths = [str(path) for path in image_paths]
    random.seed(seed)

    if sample_size >= len(paths):
        return paths

    return random.sample(paths, sample_size)


def save_pickle(value: Any, path: str | Path) -> None:
    """Save a Python object with pickle."""

    with open(path, "wb") as file:
        pickle.dump(value, file)


def load_pickle(path: str | Path) -> Any:
    """Load a Python object from pickle."""

    with open(path, "rb") as file:
        return pickle.load(file)


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not already exist."""

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory

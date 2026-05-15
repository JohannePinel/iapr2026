#!/usr/bin/env python3
"""
io_utils.py
===========
File-system I/O helpers for the preprocessing pipeline.

Kept separate from ``preprocess_utils`` (algorithms) and ``main`` (config +
orchestration) so the core image-processing functions stay decoupled from
disk concerns: anything that reads from or writes to the file system lives
here.
"""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image


def load_images_from(
    directory: str,
    pattern: str = "*.jpg",
) -> Dict[str, np.ndarray]:
    """Load every image matching ``pattern`` in ``directory``.

    Args:
        directory: Directory to scan (non-recursive).
        pattern: A glob pattern relative to ``directory``.

    Returns:
        Mapping ``{filename_stem: rgb_image}`` where ``rgb_image`` is a
        ``uint8`` array of shape ``(H, W, 3)``. Sorted by stem for
        reproducibility.

    Raises:
        FileNotFoundError: If ``directory`` does not exist or matches
            zero files.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"image directory not found: {directory}")

    paths = sorted(glob.glob(os.path.join(directory, pattern)))
    if not paths:
        raise FileNotFoundError(
            f"no files matching {pattern!r} in {directory}"
        )

    images: Dict[str, np.ndarray] = {}
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0]
        images[stem] = np.asarray(Image.open(path).convert("RGB"))
    print(f"loaded {len(images)} image(s) from {directory}")
    return images


def save_images_to(
    output_dir: str,
    filename: str,
    symbol_candidates: List[Dict[str, Any]],
    fmt: str = "png",
) -> List[str]:
    """Persist symbol-candidate tiles to ``output_dir/filename/``.

    Each candidate's ``content`` array is saved as
    ``<output_dir>/<filename>/tile_<filename>_<index>.<fmt>``.

    Args:
        output_dir: Root output directory. Created if missing.
        filename: Base name for this image (typically the source filename
            stem); a same-named sub-directory will be created underneath
            ``output_dir`` to group tiles.
        symbol_candidates: List returned by ``tile_selection`` -- each item
            must carry ``index`` and ``content``.
        fmt: Image format / extension (``png`` recommended for masks).

    Returns:
        List of absolute paths written.
    """
    target_dir = os.path.join(output_dir, filename)
    os.makedirs(target_dir, exist_ok=True)

    written: List[str] = []
    for candidate in symbol_candidates:
        idx  = candidate["index"]
        crop = candidate["content"]
        out_path = os.path.join(
            target_dir, f"tile_{filename}_{idx:05d}.{fmt}"
        )
        Image.fromarray(crop).save(out_path)
        written.append(out_path)

    print(f"saved {len(written)} candidate tile(s) to {target_dir}/")
    return written
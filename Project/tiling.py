#!/usr/bin/env python3
"""
tiling.py
=========
Background-masked sliding-window tiling for UNO card images.

Pipeline
--------
1. Build a foreground mask that filters out white-ish background; foreground
   ("patterns") is set to max white intensity (255), background to 0.
2. Slide a square convolution window across the masked image.
3. Discard "empty" tiles: a tile whose average whiteness is below
   `whiteness_keep_frac` of the max value (255) is not saved.
4. Save kept tiles into a `tiles/` subfolder, and emit three visualizations:
   the tiled original, a montage of 20 sample tiles, and their positions.
"""

import os
import random

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mask_cleanup import clean_mask


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------
def mask_white_background(img_rgb, sat_max, val_min):
    """
    Build a single-channel foreground mask by filtering out white-ish background.

    A pixel counts as white-ish background when, in HSV space, it has
    low saturation (S < sat_max) AND high value (V > val_min). Every other
    pixel is treated as foreground / pattern and set to max white intensity.

    Returns a uint8 array: 255 = foreground/pattern, 0 = background.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]
    is_background = (s < sat_max) & (v > val_min)
    mask = np.where(is_background, 0, 255).astype(np.uint8)
    return mask


# --------------------------------------------------------------------------
# Tile geometry
# --------------------------------------------------------------------------
def compute_tile_positions(width, height, tile_size, stride):
    """
    Return (tiles, n_cols, n_rows) where tiles is a list of (x, y, w, h).

    The final tile on each axis is snapped flush to the image edge so the
    whole frame is covered even when (dimension - tile_size) is not an exact
    multiple of stride.
    """
    def axis_starts(extent):
        if extent <= tile_size:
            return [0]
        starts = list(range(0, extent - tile_size + 1, stride))
        if starts[-1] != extent - tile_size:
            starts.append(extent - tile_size)  # snap last tile to the edge
        return starts

    xs = axis_starts(width)
    ys = axis_starts(height)
    w = min(tile_size, width)
    h = min(tile_size, height)
    tiles = [(x, y, w, h) for y in ys for x in xs]
    return tiles, len(xs), len(ys)


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def tile_image(image_path, tile_size=75, overlap=1.0 / 3.0,
               sat_max=40, val_min=180, whiteness_keep_frac=0.10,
               n_samples=20, seed=42, outdir=None, save_source="original"):
    """
    Run the masked sliding-window tiling pipeline on a single image.

    Args:
        image_path:          path to input image
        tile_size:           square window size in px
        overlap:             fractional window overlap in [0, 1)
        sat_max:             HSV saturation below this -> candidate background
        val_min:             HSV value above this -> candidate background
        whiteness_keep_frac: keep a tile if mean(mask tile) >= frac * 255
        n_samples:           number of sample tiles to visualize
        seed:                RNG seed for reproducible sampling
        outdir:              output directory (created if missing)
        save_source:         "original" saves the RGB crop (recommended for
                             downstream identification); "masked" saves the
                             binary-mask crop instead.

    Returns:
        dict with output paths, tile lists, and kept/discarded counts.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    if save_source not in ("original", "masked"):
        raise ValueError("save_source must be 'original' or 'masked'")
    stride = max(1, round(tile_size * (1.0 - overlap)))

    # ---- load image ----
    img = Image.open(image_path).convert("RGB")
    img_rgb = np.asarray(img)
    height, width = img_rgb.shape[:2]

    # ---- 1. mask out white-ish background ----
    mask = mask_white_background(img_rgb, sat_max, val_min)
    mask = clean_mask(mask)

    # ---- 2. tile geometry over the masked image ----
    tiles, n_cols, n_rows = compute_tile_positions(width, height, tile_size, stride)
    n_tiles = len(tiles)

    # ---- output locations ----
    stem = os.path.splitext(os.path.basename(image_path))[0]
    outdir = outdir or os.path.dirname(os.path.abspath(image_path))
    tiles_dir = os.path.join(outdir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    # ---- 3. traverse tiles, apply discard condition, save kept tiles ----
    keep_threshold = whiteness_keep_frac * 255.0
    kept, discarded = [], []
    for idx, (x, y, w, h) in enumerate(tiles):
        mask_tile = mask[y:y + h, x:x + w]
        avg_whiteness = float(mask_tile.mean())
        if avg_whiteness < keep_threshold:
            discarded.append(idx)
            continue
        kept.append(idx)
        # naming convention: tile_<imagename>_<index>
        fname = f"tile_{stem}_{idx:05d}.png"
        if save_source == "original":
            crop = img_rgb[y:y + h, x:x + w]
        else:  # "masked"
            crop = mask_tile
        Image.fromarray(crop).save(os.path.join(tiles_dir, fname))

    n_kept, n_discarded = len(kept), len(discarded)

    print(f"image            : {image_path}")
    print(f"dimensions       : {width} x {height} px")
    print(f"tile size        : {tile_size} x {tile_size} px")
    print(f"overlap / stride : {overlap:.4f} -> {stride} px")
    print(f"grid             : {n_cols} cols x {n_rows} rows = {n_tiles} tiles")
    print(f"mask thresholds  : S < {sat_max}, V > {val_min} flagged as background")
    print(f"discard threshold: mean whiteness < {keep_threshold:.1f} (= {whiteness_keep_frac:.0%} of 255)")
    print(f"kept / discarded : {n_kept} kept, {n_discarded} discarded")
    print(f"tiles saved to   : {tiles_dir}/  (source: {save_source})")

    # ---- output 0: the masking result ----
    mask_path = os.path.join(outdir, f"{stem}_mask.png")
    Image.fromarray(mask).save(mask_path)

    # ---- output 1: original image with the tiling ----
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=150)
    ax.imshow(img_rgb)
    for idx in discarded:
        x, y, w, h = tiles[idx]
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                       edgecolor="red", linewidth=0.2, alpha=0.15))
    for idx in kept:
        x, y, w, h = tiles[idx]
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                       edgecolor="lime", linewidth=0.5, alpha=0.9))
    ax.set_title(f"Tiling: {n_kept} kept (lime) / {n_discarded} discarded (red) "
                 f"- {tile_size}px window, stride {stride}px")
    ax.axis("off")
    grid_path = os.path.join(outdir, f"{stem}_tiling.png")
    fig.savefig(grid_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    # ---- pick reproducible samples from the KEPT tiles ----
    rng = random.Random(seed)
    k = min(n_samples, n_kept)
    sample_idx = sorted(rng.sample(kept, k)) if k > 0 else []

    # ---- output 2: montage of sample tiles (thresholded / masked) ----
    cols = 5
    rows = max(1, (k + cols - 1) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.8, rows * 1.8))
    axes = np.array(axes).reshape(-1)
    for ax in axes:
        ax.axis("off")
    for slot, idx in enumerate(sample_idx):
        x, y, w, h = tiles[idx]
        # thresholded view: the binary mask crop, not the RGB crop
        axes[slot].imshow(mask[y:y + h, x:x + w], cmap="gray", vmin=0, vmax=255)
        axes[slot].set_title(f"#{idx}\n(x={x}, y={y})", fontsize=7)
    fig.suptitle(f"{k} sample tiles (thresholded, from {n_kept} kept)")
    fig.tight_layout()
    samples_path = os.path.join(outdir, f"{stem}_samples.png")
    fig.savefig(samples_path, dpi=150)
    plt.close(fig)

    # ---- output 3: sample tile positions on the original ----
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=150)
    ax.imshow(img_rgb)
    for idx in kept:  # faint context of all kept tiles
        x, y, w, h = tiles[idx]
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                       edgecolor="lime", linewidth=0.2, alpha=0.3))
    for idx in sample_idx:
        x, y, w, h = tiles[idx]
        ax.add_patch(patches.Rectangle((x, y), w, h, fill=False,
                                       edgecolor="red", linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, str(idx), color="red", fontsize=7,
                ha="center", va="center", weight="bold")
    ax.set_title(f"Positions of {k} sample tiles")
    ax.axis("off")
    map_path = os.path.join(outdir, f"{stem}_sample_map.png")
    fig.savefig(map_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    print("\nwrote:")
    print(f"  {mask_path}")
    print(f"  {grid_path}")
    print(f"  {samples_path}")
    print(f"  {map_path}")
    print(f"  {tiles_dir}/  ({n_kept} tile images)")

    return {
        "mask": mask,
        "mask_path": mask_path,
        "grid_path": grid_path,
        "samples_path": samples_path,
        "map_path": map_path,
        "tiles_dir": tiles_dir,
        "n_tiles": n_tiles,
        "n_kept": n_kept,
        "n_discarded": n_discarded,
        "kept": kept,
        "discarded": discarded,
        "tiles": tiles,
        "stride": stride,
        "width": width,
        "height": height,
    }
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

from mask_cleanup import clean_mask, sobel_edges
from descriptors import plot_tile_descriptors


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
               sat_max=40, val_min=180, edge_keep_frac=0.03, edge_thresh=40,
               n_samples=20, seed=42, outdir=None, save_source="sobel",
               fd_length=11):
    """
    Run the masked sliding-window tiling pipeline on a single image.

    Mask chain:  white-background mask  ->  clean_mask (speckle + contour
    smoothing)  ->  sobel_edges (edges only).

    Both the empty-tile discard test and the Fourier-descriptor contour are
    computed on the SOBEL-EDGE tiles: a tile is kept only if a sufficient
    fraction of its Sobel tile is edge pixels. This drops tiles that are
    plain black in the Sobel map -- i.e. uniform card interior or pure
    background -- which the old cleaned-mask test let through.

    Args:
        image_path:     path to input image
        tile_size:      square window size in px
        overlap:        fractional window overlap in [0, 1)
        sat_max:        HSV saturation below this -> candidate background
        val_min:        HSV value above this -> candidate background
        edge_thresh:    Sobel intensity above which a pixel counts as an edge
        edge_keep_frac: keep a tile if (fraction of edge pixels in its Sobel
                        tile) >= edge_keep_frac
        n_samples:      number of sample tiles to visualize
        seed:           RNG seed for reproducible sampling
        outdir:         output directory (created if missing)
        save_source:    which crop to save per tile -- "sobel" (Sobel edge
                        map, default), "clean", "masked" or "original".
        fd_length:      n_samples for the padded Fourier descriptors.

    Returns:
        dict with output paths, tile lists, kept/discarded counts, and the
        complex Fourier descriptor array for the sampled tiles.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")
    if save_source not in ("sobel", "clean", "masked", "original"):
        raise ValueError("save_source must be 'sobel', 'clean', 'masked' or 'original'")
    stride = max(1, round(tile_size * (1.0 - overlap)))

    # ---- load image ----
    img = Image.open(image_path).convert("RGB")
    img_rgb = np.asarray(img)
    height, width = img_rgb.shape[:2]

    # ---- 1. mask chain: white-bg mask -> cleaned mask -> Sobel edge map ----
    mask_raw = mask_white_background(img_rgb, sat_max, val_min)
    mask_clean = clean_mask(mask_raw)               # speckle removal + smoothing
    mask_edges = sobel_edges(mask_clean)            # edges only ("sobeled" mask)

    # ---- 2. tile geometry ----
    tiles, n_cols, n_rows = compute_tile_positions(width, height, tile_size, stride)
    n_tiles = len(tiles)

    # ---- output locations ----
    stem = os.path.splitext(os.path.basename(image_path))[0]
    outdir = outdir or os.path.dirname(os.path.abspath(image_path))
    tiles_dir = os.path.join(outdir, "tiles")
    os.makedirs(tiles_dir, exist_ok=True)

    # ---- 3. traverse tiles, apply discard condition, save kept tiles ----
    # Discard is decided on the SOBEL-EDGE tile: keep a tile only if a
    # sufficient fraction of it is edge pixels. A tile that is uniform card
    # interior (or pure background) has a near-black Sobel tile and is
    # dropped -- which is what the cleaned-mask "mean whiteness" test missed.
    crop_source = {"sobel": mask_edges, "clean": mask_clean,
                   "masked": mask_raw, "original": img_rgb}[save_source]
    kept, discarded = [], []
    for idx, (x, y, w, h) in enumerate(tiles):
        edge_tile = mask_edges[y:y + h, x:x + w]
        edge_frac = float((edge_tile > edge_thresh).mean())
        if edge_frac < edge_keep_frac:
            discarded.append(idx)
            continue
        kept.append(idx)
        # naming convention: tile_<imagename>_<index>
        fname = f"tile_{stem}_{idx:05d}.png"
        crop = crop_source[y:y + h, x:x + w]
        Image.fromarray(crop).save(os.path.join(tiles_dir, fname))

    n_kept, n_discarded = len(kept), len(discarded)

    print(f"image            : {image_path}")
    print(f"dimensions       : {width} x {height} px")
    print(f"tile size        : {tile_size} x {tile_size} px")
    print(f"overlap / stride : {overlap:.4f} -> {stride} px")
    print(f"grid             : {n_cols} cols x {n_rows} rows = {n_tiles} tiles")
    print(f"mask thresholds  : S < {sat_max}, V > {val_min} flagged as background")
    print(f"mask chain       : raw -> clean_mask -> sobel_edges")
    print(f"discard rule     : edge-pixel fraction (Sobel > {edge_thresh}) < {edge_keep_frac:.0%}")
    print(f"kept / discarded : {n_kept} kept, {n_discarded} discarded, "
          f"ratio {n_kept / max(1, n_kept + n_discarded):.3f}")
    print(f"tiles saved to   : {tiles_dir}/  (source: {save_source})")

    # ---- output 0: the mask chain (raw, cleaned, Sobel edges) ----
    mask_raw_path = os.path.join(outdir, f"{stem}_mask_raw.png")
    mask_clean_path = os.path.join(outdir, f"{stem}_mask_clean.png")
    mask_path = os.path.join(outdir, f"{stem}_mask.png")  # the Sobel edge map
    Image.fromarray(mask_raw).save(mask_raw_path)
    Image.fromarray(mask_clean).save(mask_clean_path)
    Image.fromarray(mask_edges).save(mask_path)

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

    # ---- output 2: padded Fourier descriptors for the sampled tiles ----
    # contour + descriptor are taken from the SOBEL-edge tiles (binarised at
    # edge_thresh), consistent with the keep test and the saved tiles.
    samples_path = os.path.join(outdir, f"{stem}_descriptors.png")
    descriptors = plot_tile_descriptors(
        img_rgb, tiles, sample_idx, mask_edges,
        n_samples=fd_length, edge_thresh=edge_thresh, out_path=samples_path,
    )

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
    print(f"  {mask_raw_path}")
    print(f"  {mask_clean_path}")
    print(f"  {mask_path}  (Sobel edge map)")
    print(f"  {grid_path}")
    print(f"  {samples_path}  (Fourier descriptors)")
    print(f"  {map_path}")
    print(f"  {tiles_dir}/  ({n_kept} tile images, source: {save_source})")

    return {
        "mask_raw": mask_raw,
        "mask_clean": mask_clean,
        "mask_edges": mask_edges,
        "mask_raw_path": mask_raw_path,
        "mask_clean_path": mask_clean_path,
        "mask_path": mask_path,
        "grid_path": grid_path,
        "samples_path": samples_path,
        "map_path": map_path,
        "tiles_dir": tiles_dir,
        "descriptors": descriptors,
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
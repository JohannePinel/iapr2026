#!/usr/bin/env python3
"""
mask_cleanup.py
===============
Noise reduction + contour smoothing for the binary masks from tiling.py.

WHAT THE PROVIDED TILES ACTUALLY SHOW
-------------------------------------
Running the HSV threshold on the 8 sample tiles gives masks that are already
fairly clean in the connected-components sense (1-3 blobs, almost no isolated
specks). The visible "noise" is not salt-and-pepper inside the mask -- it is
the RAGGED CONTOUR BOUNDARY between the card colour and the white print,
caused by per-pixel thresholding of a worn / textured print.

So the cleanup is split into two concerns:

  A. Speckle removal (handles genuinely noisy tiles elsewhere in the set)
       1. median filter        - edge-preserving salt-and-pepper removal
       2. morphological OPEN    - removes leftover thin salt protrusions
       3. morphological CLOSE   - fills pepper pinholes, bridges hairline gaps
       4. area filter           - drops sub-threshold connected components

  B. Contour smoothing (the visible win on these particular tiles)
       5. Gaussian blur + re-threshold at 127  - rounds jagged edges and
          small boundary protrusions symmetrically, giving a contour that
          findContours / approxPolyDP can follow cleanly.

KERNEL SIZE WARNING: tiles are 75x75 and a digit stroke is only ~8-12 px wide.
Keep kernels at 3 (5 at most). A bigger kernel "cleans" the mask by eating the
strokes that identify the card.
"""

import glob
import os

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Masking step (mirrors tiling.py; in production import it from there)
# --------------------------------------------------------------------------
def mask_white_background(img_rgb, sat_max=40, val_min=180):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    is_background = (s < sat_max) & (v > val_min)
    return np.where(is_background, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# A. speckle removal + B. contour smoothing
# --------------------------------------------------------------------------
def clean_mask(mask, median_ksize=3, open_ksize=3, close_ksize=3,
               min_blob_area=20, smooth_ksize=5):
    """
    Clean a binary mask: remove speckle, then smooth the contour.

    Args:
        mask:          uint8 array, 0 / 255
        median_ksize:  median blur window (odd >=3; 0 to skip)
        open_ksize:    ellipse kernel for OPEN  (0 to skip)
        close_ksize:   ellipse kernel for CLOSE (0 to skip)
        min_blob_area: connected components smaller than this are dropped
                       (0 to skip)
        smooth_ksize:  Gaussian-blur-then-rethreshold window for contour
                       smoothing (odd; 0 to skip)

    Returns:
        uint8 array, 0 / 255.
    """
    m = mask.copy()

    # --- A. speckle removal ---
    if median_ksize and median_ksize >= 3:
        m = cv2.medianBlur(m, median_ksize)

    if open_ksize:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ksize, open_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)

    if close_ksize:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)

    if min_blob_area:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        cleaned = np.zeros_like(m)
        for i in range(1, n):  # 0 is background
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                cleaned[labels == i] = 255
        m = cleaned

    # --- B. contour smoothing: blur a binary mask, re-threshold at the midpoint.
    # This rounds jagged edges and small protrusions symmetrically -- cheaper
    # and more isotropic than a large morphological close.
    if smooth_ksize:
        k = smooth_ksize | 1  # force odd
        m = cv2.GaussianBlur(m, (k, k), 0)
        m = np.where(m >= 127, 255, 0).astype(np.uint8)

    return m


def largest_contours(mask, min_area=20):
    """External contours above an area floor, largest first."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) >= min_area]
    return sorted(cnts, key=cv2.contourArea, reverse=True)


def sobel_edges(mask, ksize=3):
    """
    Sobel gradient magnitude of a (binary) mask -> an "edges only" image.

    On a clean binary mask the Sobel response is bright exactly on the
    region boundaries and ~0 everywhere else, so the result is the edge
    map of the mask. Output is uint8, normalised to 0..255.

    Args:
        mask:  uint8 array (expected 0 / 255, but any single-channel works)
        ksize: Sobel kernel size (3 is appropriate for 75 px tiles)

    Returns:
        uint8 array, same shape as input, 0..255.
    """
    f = mask.astype(np.float32)
    gx = cv2.Sobel(f, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(f, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = cv2.magnitude(gx, gy)
    peak = float(mag.max())
    if peak > 0:
        mag = mag / peak * 255.0
    return mag.astype(np.uint8)


# --------------------------------------------------------------------------
# Demo: stage-by-stage on the uploaded tiles
# --------------------------------------------------------------------------
def main():
    tile_paths = sorted(glob.glob("/mnt/user-data/uploads/tile_*.png"))
    outdir = "/home/claude/cleanup_demo"
    os.makedirs(outdir, exist_ok=True)

    rows = len(tile_paths)
    fig, axes = plt.subplots(rows, 5, figsize=(5 * 1.9, rows * 1.9))
    col_titles = ["RGB tile", "raw mask", "A: speckle removed",
                  "B: + contour smoothed", "final contour on RGB"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=8)

    for r, path in enumerate(tile_paths):
        rgb = np.asarray(Image.open(path).convert("RGB"))
        raw = mask_white_background(rgb)
        speckle = clean_mask(raw, median_ksize=3, open_ksize=3,
                             close_ksize=3, min_blob_area=20, smooth_ksize=0)
        smoothed = clean_mask(raw, median_ksize=3, open_ksize=3,
                              close_ksize=3, min_blob_area=20, smooth_ksize=5)

        overlay = rgb.copy()
        cv2.drawContours(overlay, largest_contours(smoothed), -1, (255, 0, 255), 2)

        panels = [rgb, raw, speckle, smoothed, overlay]
        for c, img in enumerate(panels):
            ax = axes[r, c]
            if c in (1, 2, 3):
                ax.imshow(img, cmap="gray", vmin=0, vmax=255)
            else:
                ax.imshow(img)
            ax.axis("off")

    fig.suptitle("Mask cleanup: speckle removal + contour smoothing", fontsize=12)
    fig.tight_layout()
    out = os.path.join(outdir, "mask_cleanup_comparison.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
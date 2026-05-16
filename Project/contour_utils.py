"""
contour_utils.py
================

From-scratch contour primitives used by the tile noise-removal pipeline:

* connected-component labeling (iterative flood fill),
* size- and edge-based contour filtering,
* label-map colorization for visualisation.

All functions are pure (no I/O, no global config), so this module is safe to
import from anywhere in the codebase. The deliberately hand-written labeling
avoids `cv2.findContours` / `cv2.connectedComponents` / `scipy.ndimage.label`.
"""

from collections import deque
import colorsys


import numpy as np

__all__ = [
    "label_contours",
    "remove_small_contours",
    "remove_edge_contours",
    "make_palette",
    "colorize",
]


# ---------------------------------------------------------------------------
# Connected-component labeling
# ---------------------------------------------------------------------------
def label_contours(mask: np.ndarray, connectivity: int = 8,
                   min_size: int = 3) -> np.ndarray:
    """Connected-component labeling via iterative (BFS) flood fill.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean array -- True where a contour pixel sits.
    connectivity : int
        8 (default) or 4 -- neighbourhood used to link pixels. 8-connectivity
        keeps thin diagonal strokes intact.
    min_size : int
        Components with fewer pixels than this are discarded as noise.

    Returns
    -------
    np.ndarray
        int32 label map: 0 = background, 1..N = the N distinct contours
        (label k = the k-th component found in raster order).
    """
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=np.int32)

    if connectivity == 8:
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1),
                   (0, 1), (1, -1), (1, 0), (1, 1)]
    elif connectivity == 4:
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        raise ValueError("connectivity must be 4 or 8")

    next_label = 0
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or labels[r, c] != 0:
                continue                       # background or already labeled

            next_label += 1                    # start a fresh component
            component = []
            queue = deque([(r, c)])
            labels[r, c] = next_label

            while queue:                       # breadth-first flood fill
                y, x = queue.popleft()
                component.append((y, x))
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < H and 0 <= nx < W
                            and mask[ny, nx] and labels[ny, nx] == 0):
                        labels[ny, nx] = next_label
                        queue.append((ny, nx))

            if len(component) < min_size:      # drop tiny speck, reuse label
                for (y, x) in component:
                    labels[y, x] = 0
                next_label -= 1

    return labels


# ---------------------------------------------------------------------------
# Contour filtering
# ---------------------------------------------------------------------------
def remove_small_contours(labels: np.ndarray, min_pixels: int = 25) -> np.ndarray:
    """Erase every contour with fewer than `min_pixels` pixels.

    Survivors are renumbered 1..M contiguously.
    """
    out, new_label = np.zeros_like(labels), 0
    for k in range(1, int(labels.max()) + 1):
        comp = (labels == k)
        if comp.sum() >= min_pixels:           # pixel count == np.sum of the mask
            new_label += 1
            out[comp] = new_label
    return out


def remove_edge_contours(labels: np.ndarray, margin: int = 5) -> np.ndarray:
    """Erase every contour with a pixel less than `margin` px from any edge.

    A contour clipped by the tile boundary cannot be assessed from this tile
    alone (it may continue into a neighbouring tile), so it is dropped.
    Survivors are renumbered 1..M contiguously.
    """
    H, W = labels.shape
    border = np.zeros((H, W), dtype=bool)      # band within `margin` px of an edge
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    edge_labels = set(np.unique(labels[border])) - {0}

    out, new_label = np.zeros_like(labels), 0
    for k in range(1, int(labels.max()) + 1):
        if k in edge_labels:
            continue                           # contour touches edge -> drop
        comp = (labels == k)
        if comp.any():
            new_label += 1
            out[comp] = new_label
    return out


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------
def make_palette(n: int, seed: int = 7) -> list:
    """Return `n` visually-distinct RGB colors via evenly-spaced HSV hues."""
    rng = np.random.RandomState(seed)
    palette = []
    for i in range(n):
        hue = (i / max(n, 1)) % 1.0            # spread hues around the wheel
        sat = 0.70 + 0.30 * rng.rand()
        val = 0.85 + 0.15 * rng.rand()
        palette.append(tuple(int(255 * v) for v in colorsys.hsv_to_rgb(hue, sat, val)))
    return palette


def colorize(labels: np.ndarray, background: tuple = (0, 0, 0)) -> np.ndarray:
    """Render an int label map as an RGB image, one color per contour."""
    n = int(labels.max())
    palette = make_palette(n)
    out = np.zeros((*labels.shape, 3), dtype=np.uint8)
    out[:] = background
    for k in range(1, n + 1):
        out[labels == k] = palette[k - 1]
    return out

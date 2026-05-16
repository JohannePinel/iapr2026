"""
contour_utils.py
================

From-scratch contour primitives and the high-level noise-removal pass for the
tile pipeline:

* connected-component labeling (iterative flood fill),
* size- and edge-based contour filtering,
* ``remove_noise_contours`` -- a tile-aware wrapper that cleans a whole tile
  collection in one call. It understands the result dict produced by
  ``preprocess_utils.tile_image`` directly, so it drops straight into
  ``preprocess_image`` between ``tile_image`` and ``tile_selection``,
* label-map colorization for visualisation.

All functions are pure (no I/O, no global state), so this module is safe to
import from anywhere in the codebase. The deliberately hand-written labeling
avoids ``cv2.findContours`` / ``cv2.connectedComponents`` / ``scipy.ndimage.label``.

Module layout
-------------
* Connected-component labeling : :func:`label_contours`
* Contour filtering            : :func:`remove_small_contours`,
                                  :func:`remove_edge_contours`
* Single-tile cleanup          : :func:`clean_tile`
* Tile-collection cleanup      : :func:`remove_noise_contours`
* Visualisation                : :func:`make_palette`, :func:`colorize`
"""

from collections import deque
import colorsys

import numpy as np

__all__ = [
    "label_contours",
    "remove_small_contours",
    "remove_edge_contours",
    "clean_tile",
    "remove_noise_contours",
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
# Single-tile cleanup
# ---------------------------------------------------------------------------
def clean_tile(tile: np.ndarray, min_pixels: int = 100, edge_margin: int = 2,
               fg_threshold=None) -> np.ndarray:
    """Run the noise-removal pipeline on a single 2D tile.

    Stages: ``threshold/binarize -> label -> drop small contours ->
    drop edge contours``.

    Parameters
    ----------
    tile : np.ndarray
        A single 2D tile (e.g. one crop from ``tile_image``).
    min_pixels : int
        Contours with fewer than this many pixels are dropped.
    edge_margin : int
        Contours with a pixel within this many px of a tile edge are dropped.
    fg_threshold : int or None
        ``None`` -> any non-zero pixel is foreground (binary tiles).
        ``int``  -> binarize the tile with ``tile > fg_threshold``.

    Returns
    -------
    np.ndarray
        A mask the same shape and dtype as `tile`, with only the surviving
        contour pixels set (to the tile's foreground value) and the rest 0.
    """
    if fg_threshold is None:
        mask = tile.astype(bool)               # any non-zero pixel is foreground
    else:
        mask = tile > fg_threshold             # threshold a grayscale tile

    labels = label_contours(mask)
    labels = remove_small_contours(labels, min_pixels)
    labels = remove_edge_contours(labels, edge_margin)

    fg_value = tile.max()                      # preserve the input's foreground value
    out = np.zeros_like(tile)
    out[labels > 0] = fg_value
    return out


# ---------------------------------------------------------------------------
# Tile-collection cleanup  (tile-aware)
# ---------------------------------------------------------------------------
def _is_tile_image_result(obj) -> bool:
    """True when `obj` is a result dict produced by ``tile_image``.

    Such a dict carries an ``"items"`` list whose entries are per-tile dicts,
    each holding the tile pixels under a ``"content"`` key (alongside geometry
    metadata such as ``index, x, y, w, h``). This is the structure
    :func:`remove_noise_contours` cleans in place -- it is distinct from a
    plain ``{key: tile}`` mapping, which is handled separately.
    """
    if not (isinstance(obj, dict) and isinstance(obj.get("items"), list)):
        return False
    return all(isinstance(it, dict) and "content" in it
               for it in obj["items"])


def _clean_tile_image_result(result: dict, min_pixels: int, edge_margin: int,
                             fg_threshold) -> dict:
    """Clean every tile in a ``tile_image`` result, preserving its structure.

    Only each item's ``"content"`` array is replaced (by its cleaned mask);
    all geometry/metadata is passed through unchanged -- both per-item keys
    (``index, x, y, w, h``) and top-level keys (``source_image, tile_size,
    stride, n_cols, n_rows``). A shallow copy is returned; the input dict and
    its items are not mutated.
    """
    cleaned_items = [
        {**item,
         "content": clean_tile(np.asarray(item["content"]),
                               min_pixels, edge_margin, fg_threshold)}
        for item in result["items"]
    ]
    return {**result, "items": cleaned_items}


def remove_noise_contours(tiles, min_pixels: int = 100, edge_margin: int = 2,
                          fg_threshold=None):
    """Remove noise contours from a tile collection produced by ``tile_image``.

    For every tile this drops (a) contours smaller than `min_pixels` pixels and
    (b) contours touching the tile frame (any pixel within `edge_margin` px of
    an edge), then returns the cleaned tiles in the **same container type and
    shape** as the input -- so it is a drop-in transform::

        tiles = tile_image(edges, **tile_kw)
        tiles = remove_noise_contours(tiles, **noise_kw)   # <-- inserted here
        candidates = tile_selection(tiles, ...)

    Accepted containers
    -------------------
    * ``tile_image`` result dict          -> same dict, with every
      ``items[i]["content"]`` cleaned and all other metadata preserved.
    * plain dict ``{key: tile}``          -> dict with the same keys
    * list / tuple of tiles (or of grids) -> same sequence type, recursively
    * np.ndarray, 2D ``(H, W)``           -> a single tile
    * np.ndarray, 3D/4D ``(..., H, W)``   -> a stack/grid; the last two axes
                                             are treated as the tile

    The ``tile_image`` result dict is detected by structure (an ``"items"``
    list of per-tile dicts) and handled first, so passing the output of
    ``tile_image`` straight in works as expected -- its scalar metadata
    (``tile_size``, ``stride``, ...) is never mistaken for tile data.

    Foreground convention
    ---------------------
    ``tile_image`` runs after ``keep_edges``. ``keep_edges`` emits a Sobel
    *magnitude* map scaled to ``[0, 255]`` -- not a strict 0/255 binary -- so
    for the contour step to ignore weak gradient speckle, pass an integer
    `fg_threshold` (typically the same value ``tile_selection`` uses for
    ``edge_thresh``). With the default ``fg_threshold=None`` every non-zero
    pixel counts as foreground, which suits genuinely binary tiles.

    Parameters
    ----------
    tiles : dict | list | tuple | np.ndarray
        The tile collection to clean.
    min_pixels : int
        Small-contour dropout threshold (pixel count).
    edge_margin : int
        Frame-dropout margin in pixels.
    fg_threshold : int or None
        None -> non-zero pixels are foreground (binary tiles, the default).
        int  -> binarize each tile with ``tile > fg_threshold``.

    Returns
    -------
    Same type as `tiles`, with each tile replaced by its cleaned binary mask.

    Raises
    ------
    TypeError
        If `tiles` is not one of the accepted container types.
    """
    # tile_image result -> clean each item's `content`, keep all metadata.
    # Checked before the generic-dict case because such a result *is* a dict.
    if _is_tile_image_result(tiles):
        return _clean_tile_image_result(tiles, min_pixels, edge_margin,
                                        fg_threshold)

    # plain {key: tile} mapping -> clean each value, keep keys
    if isinstance(tiles, dict):
        return {k: remove_noise_contours(v, min_pixels, edge_margin, fg_threshold)
                for k, v in tiles.items()}

    # list / tuple (possibly nested, e.g. a grid of rows) -> recurse
    if isinstance(tiles, (list, tuple)):
        return type(tiles)(
            remove_noise_contours(t, min_pixels, edge_margin, fg_threshold)
            for t in tiles
        )

    # numpy array
    arr = np.asarray(tiles)
    if arr.ndim == 2:                          # a single tile
        return clean_tile(arr, min_pixels, edge_margin, fg_threshold)
    if arr.ndim >= 3:                          # stack/grid: last two axes = tile
        out = np.empty_like(arr)
        for idx in np.ndindex(arr.shape[:-2]):
            out[idx] = clean_tile(arr[idx], min_pixels, edge_margin, fg_threshold)
        return out

    raise TypeError(
        f"Unsupported tile container: {type(tiles).__name__} with ndim={arr.ndim}. "
        "Expected a tile_image result dict, a plain dict, a list/tuple, "
        "or a 2D/3D/4D numpy array."
    )


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
#!/usr/bin/env python3
"""
preprocess_utils.py
===================
Core image-processing utilities for the UNO-card symbol-candidate pipeline.

Pipeline (per image)
--------------------
    RGB image
        -> apply_threshold       : HSV-based white-background mask
        -> apply_morphology      : speckle removal + contour smoothing
        -> keep_edges            : Sobel edge map
        -> tile_image            : sliding-window crops
        -> remove_noise_contours : per-tile contour cleanup -- drop tiny and
                                   frame-clipped contours  (contour_utils)
        -> tile_selection        : keep tiles with enough edge content
                                   ("symbol candidates")

Every high-level step accepts ``show_plots: bool``. When True, the step emits
a matplotlib diagnostic figure; when False it runs silently. Algorithms are
decoupled from plotting -- the plotting helpers are kept private (``_plot_*``).

Stage parameters reach the per-image chain through ``preprocess_image``'s
``**overrides``. Routing of those kwargs to each stage is driven by a single
table, :data:`_STAGE_PARAMS`, so adding or moving a parameter is a one-line
change.
"""

from __future__ import annotations

import os
import random
from typing import Any, Dict, Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from contour_utils import remove_noise_contours


# ============================================================================
# Module-level defaults
# ============================================================================
# These are sane defaults; main.py is expected to override them via kwargs.
# They are kept here so the utility functions are runnable in isolation
# (e.g. from a notebook) without main.py.

# --- thresholding (HSV white-background filter) ---
DEFAULT_SAT_MAX = 40        # HSV saturation below this -> candidate background
DEFAULT_VAL_MIN = 180       # HSV value above this      -> candidate background

# --- morphology / mask cleanup ---
DEFAULT_MEDIAN_KSIZE  = 3   # median-blur window      (odd >=3; 0 to skip)
DEFAULT_OPEN_KSIZE    = 3   # MORPH_OPEN kernel       (0 to skip)
DEFAULT_CLOSE_KSIZE   = 3   # MORPH_CLOSE kernel      (0 to skip)
DEFAULT_MIN_BLOB_AREA = 20  # drop CCs smaller than this (0 to skip)
DEFAULT_SMOOTH_KSIZE  = 5   # Gaussian-then-rethreshold contour smoothing
                            # (0 to skip)

# --- edges ---
DEFAULT_SOBEL_KSIZE = 3

# --- tiling geometry ---
DEFAULT_TILE_SIZE = 99
DEFAULT_OVERLAP   = 2.0 / 3.0

# --- tile selection ---
DEFAULT_EDGE_THRESH    = 20      # pixel intensity above which a Sobel pixel
                                 # counts as an edge
DEFAULT_EDGE_KEEP_FRAC = 0.03    # keep a tile if its edge-pixel fraction
                                 # is at least this much

# --- noise removal (per-tile contour cleanup) ---
# The numeric defaults for `min_pixels` / `edge_margin` live with the
# algorithm in ``contour_utils.remove_noise_contours`` (single source of
# truth -- not duplicated here). `fg_threshold` is special: it decides which
# tile pixels count as "edge", so by default ``preprocess_image`` ties it to
# DEFAULT_EDGE_THRESH to stay consistent with tile_selection.

# --- diagnostic panel sampling ---
N_KEPT_PREVIEW      = 15
N_DISCARDED_PREVIEW = 5
DEFAULT_SEED        = 42


# ============================================================================
# 1. THRESHOLDING
# ============================================================================
def apply_threshold(
    image: np.ndarray,
    sat_max: int = DEFAULT_SAT_MAX,
    val_min: int = DEFAULT_VAL_MIN,
    show_plots: bool = True,
) -> np.ndarray:
    """Binarize an RGB image by masking out its white-ish background.

    A pixel is classified as background when, in HSV space, its saturation is
    below ``sat_max`` AND its value is above ``val_min``. Background pixels
    are set to 0; everything else is set to 255 (foreground / pattern).

    Args:
        image: Input RGB image, ``uint8`` array of shape ``(H, W, 3)``.
        sat_max: HSV saturation threshold below which a pixel is a background
            candidate.
        val_min: HSV value threshold above which a pixel is a background
            candidate.
        show_plots: If True, display a side-by-side comparison of the input
            image and the resulting binary mask.

    Returns:
        ``uint8`` binary mask of shape ``(H, W)`` with values in {0, 255}.

    Side effects:
        Renders a matplotlib figure when ``show_plots`` is True.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    saturation, value = hsv[:, :, 1], hsv[:, :, 2]
    is_background = (saturation < sat_max) & (value > val_min)
    mask = np.where(is_background, 0, 255).astype(np.uint8)

    if show_plots:
        _plot_before_after(
            image, mask,
            title_before="Original (RGB)",
            title_after=f"After threshold (S<{sat_max}, V>{val_min})",
            cmap_after="gray",
            sup_title="apply_threshold",
        )
    return mask


# ============================================================================
# 2. MORPHOLOGY
# ============================================================================
def apply_morphology(
    image: np.ndarray,
    median_ksize: int = DEFAULT_MEDIAN_KSIZE,
    open_ksize: int = DEFAULT_OPEN_KSIZE,
    close_ksize: int = DEFAULT_CLOSE_KSIZE,
    min_blob_area: int = DEFAULT_MIN_BLOB_AREA,
    smooth_ksize: int = DEFAULT_SMOOTH_KSIZE,
    show_plots: bool = True,
) -> np.ndarray:
    """Clean a binary mask: speckle removal followed by contour smoothing.

    Stages applied in order; any stage is skipped when its kernel/area
    argument is 0:

        1. Median filter     -- edge-preserving salt-and-pepper removal.
        2. MORPH_OPEN        -- removes thin salt protrusions.
        3. MORPH_CLOSE       -- fills pepper pinholes, bridges hairline gaps.
        4. Connected-component area filter -- drops sub-threshold blobs.
        5. Gaussian blur + re-threshold at 127 -- isotropic contour smoothing
           (rounds jagged edges symmetrically; cheaper than a large CLOSE).

    Note:
        Stage 5 is strictly speaking not "morphology" in the textbook sense,
        but it is part of the same mask-cleanup concern in this pipeline.

    Kernel-size warning:
        Tiles are ~75 px and a digit stroke is only ~8-12 px wide. Keep
        kernels at 3 (5 at most). A bigger kernel "cleans" the mask by
        eating the strokes that identify the card.

    Args:
        image: Binary mask, ``uint8`` array with values in {0, 255}.
        median_ksize: Median-blur window (odd >= 3; set 0 to skip).
        open_ksize: Ellipse kernel for OPEN (set 0 to skip).
        close_ksize: Ellipse kernel for CLOSE (set 0 to skip).
        min_blob_area: Connected components smaller than this (in pixels) are
            dropped (set 0 to skip).
        smooth_ksize: Gaussian-blur-then-rethreshold window for contour
            smoothing (odd; set 0 to skip).
        show_plots: If True, show a side-by-side before/after comparison.

    Returns:
        Cleaned ``uint8`` binary mask, values in {0, 255}, same shape as input.

    Side effects:
        Renders a matplotlib figure when ``show_plots`` is True.
    """
    cleaned = image.copy()

    # --- 1. median blur ---
    if median_ksize and median_ksize >= 3:
        cleaned = cv2.medianBlur(cleaned, median_ksize)

    # --- 2. opening (removes thin protrusions) ---
    if open_ksize:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

    # --- 3. closing (fills pinholes / bridges gaps) ---
    if close_ksize:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (close_ksize, close_ksize)
        )
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # --- 4. small connected-component filter ---
    if min_blob_area:
        n_cc, labels, stats, _ = cv2.connectedComponentsWithStats(
            cleaned, connectivity=8
        )
        filtered = np.zeros_like(cleaned)
        for i in range(1, n_cc):  # 0 is background
            if stats[i, cv2.CC_STAT_AREA] >= min_blob_area:
                filtered[labels == i] = 255
        cleaned = filtered

    # --- 5. contour smoothing (Gaussian -> re-threshold @ 127) ---
    if smooth_ksize:
        k = smooth_ksize | 1  # force odd
        cleaned = cv2.GaussianBlur(cleaned, (k, k), 0)
        cleaned = np.where(cleaned >= 127, 255, 0).astype(np.uint8)

    if show_plots:
        _plot_before_after(
            image, cleaned,
            title_before="Before morphology (raw mask)",
            title_after="After morphology (cleaned + smoothed)",
            cmap_before="gray", cmap_after="gray",
            sup_title="apply_morphology",
        )
    return cleaned


# ============================================================================
# 3. EDGES
# ============================================================================
def keep_edges(
    image: np.ndarray,
    ksize: int = DEFAULT_SOBEL_KSIZE,
    show_plots: bool = True,
) -> np.ndarray:
    """Convert a (binary) mask into an edge-only image via Sobel gradients.

    On a clean binary mask the Sobel response peaks exactly on the region
    boundaries and is ~0 in solid interiors, so the gradient magnitude IS
    the boundary map of the mask. The result is normalised to ``[0, 255]``.

    Args:
        image: Single-channel image, typically a ``uint8`` binary mask, but
            any single-channel image is accepted.
        ksize: Sobel kernel size; 3 is appropriate for 75 px tiles.
        show_plots: If True, display a side-by-side comparison of the input
            mask and its edge map.

    Returns:
        ``uint8`` edge map of the same shape as input, scaled to ``[0, 255]``.

    Side effects:
        Renders a matplotlib figure when ``show_plots`` is True.
    """
    src = image.astype(np.float32)
    gx = cv2.Sobel(src, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(src, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(gx, gy)
    peak = float(magnitude.max())
    if peak > 0:
        magnitude = magnitude / peak * 255.0
    edges = magnitude.astype(np.uint8)

    if show_plots:
        _plot_before_after(
            image, edges,
            title_before="Before Sobel (cleaned mask)",
            title_after="After Sobel (edges only)",
            cmap_before="gray", cmap_after="gray",
            sup_title="keep_edges",
        )
    return edges


# ============================================================================
# 4. TILING (geometry + crops, no filtering)
# ============================================================================
def tile_image(
    image: np.ndarray,
    tile_size: int = DEFAULT_TILE_SIZE,
    overlap: float = DEFAULT_OVERLAP,
) -> Dict[str, Any]:
    """Slide a square window across ``image`` and return all crops.

    The final tile on each axis is snapped flush to the image edge so the
    full frame is covered even when ``(dim - tile_size)`` is not an exact
    multiple of the stride. No filtering is performed here; selection is
    the job of :func:`tile_selection`.

    Args:
        image: Single- or multi-channel image to tile (typically the Sobel
            edge map produced by :func:`keep_edges`).
        tile_size: Square window size in pixels.
        overlap: Fractional window overlap in ``[0, 1)``. Stride is
            ``round(tile_size * (1 - overlap))``.

    Returns:
        A dict carrying everything :func:`tile_selection` needs::

            {
                "source_image": np.ndarray,         # the image that was tiled
                "tile_size":    int,
                "stride":       int,
                "n_cols":       int,
                "n_rows":       int,
                "items": [
                    {"index": int, "x": int, "y": int,
                     "w": int, "h": int, "content": np.ndarray},
                    ...
                ],
            }

        This is exactly the structure :func:`remove_noise_contours` accepts,
        so it can be cleaned in place before being handed to
        :func:`tile_selection`.

    Raises:
        ValueError: If ``overlap`` is outside ``[0, 1)``.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0, 1)")

    stride = max(1, round(tile_size * (1.0 - overlap)))
    height, width = image.shape[:2]

    def _axis_starts(extent: int) -> List[int]:
        if extent <= tile_size:
            return [0]
        starts = list(range(0, extent - tile_size + 1, stride))
        if starts[-1] != extent - tile_size:
            starts.append(extent - tile_size)  # snap last tile to the edge
        return starts

    xs = _axis_starts(width)
    ys = _axis_starts(height)
    w = min(tile_size, width)
    h = min(tile_size, height)

    items = []
    idx = 0
    for y in ys:
        for x in xs:
            items.append({
                "index": idx,
                "x": x, "y": y, "w": w, "h": h,
                "content": image[y:y + h, x:x + w].copy(),
            })
            idx += 1

    return {
        "source_image": image,
        "tile_size": tile_size,
        "stride": stride,
        "n_cols": len(xs),
        "n_rows": len(ys),
        "items": items,
    }


# ============================================================================
# 5. TILE SELECTION (filtering + diagnostics)
# ============================================================================
def tile_selection(
    tiles: Dict[str, Any],
    edge_thresh: int = DEFAULT_EDGE_THRESH,
    edge_keep_frac: float = DEFAULT_EDGE_KEEP_FRAC,
    seed: int = DEFAULT_SEED,
    show_plots: bool = True,
) -> List[Dict[str, Any]]:
    """Select symbol-candidate tiles based on edge-pixel content.

    A tile is kept when the fraction of pixels exceeding ``edge_thresh`` is
    at least ``edge_keep_frac``. The rationale: a tile that is uniform card
    interior (or pure background) has a near-black Sobel tile and a tiny
    edge fraction; symbol-bearing tiles have a much higher one.

    Statistics and the parameters used are always printed to the console,
    regardless of ``show_plots``.

    Args:
        tiles: The dict returned by :func:`tile_image` (optionally already
            passed through :func:`remove_noise_contours`).
        edge_thresh: Pixel intensity above which a Sobel pixel counts as an
            edge.
        edge_keep_frac: Minimum edge-pixel fraction for a tile to be kept,
            in ``[0, 1]``.
        seed: RNG seed for the (reproducible) sample drawn into the
            diagnostic panel.
        show_plots: If True, render the diagnostic panel containing the
            input image, sampled tile positions, and a grid of 15 kept +
            5 discarded sample tiles.

    Returns:
        ``symbol_candidates``: a list of kept-tile dicts, each carrying
        ``index, x, y, w, h, content`` plus the computed ``edge_fraction``.

    Side effects:
        Prints a parameter/statistics summary to stdout. Renders a
        matplotlib figure when ``show_plots`` is True.
    """
    items = tiles["items"]
    source = tiles["source_image"]

    kept: List[Dict[str, Any]] = []
    discarded: List[Dict[str, Any]] = []

    for t in items:
        edge_frac = float((t["content"] > edge_thresh).mean())
        t = {**t, "edge_fraction": edge_frac}
        if edge_frac >= edge_keep_frac:
            kept.append(t)
        else:
            discarded.append(t)

    n_total = len(items)
    n_kept, n_discarded = len(kept), len(discarded)
    keep_ratio = n_kept / max(1, n_total)

    # ---- console summary (always) ----
    print("tile_selection")
    print("-" * 50)
    print(f"  parameters    : edge_thresh={edge_thresh}, "
          f"edge_keep_frac={edge_keep_frac:.2%}")
    print(f"  geometry      : {tiles['n_cols']} cols x {tiles['n_rows']} rows "
          f"= {n_total} tiles "
          f"(tile {tiles['tile_size']}px, stride {tiles['stride']}px)")
    print(f"  result        : kept {n_kept} / discarded {n_discarded} "
          f"(keep ratio {keep_ratio:.3f})")
    if n_kept:
        ef = [t["edge_fraction"] for t in kept]
        print(f"  kept edge-frac: min={min(ef):.3f} "
              f"mean={sum(ef)/len(ef):.3f} max={max(ef):.3f}")
    print("-" * 50)

    if show_plots:
        _plot_tile_selection(
            source_image=source,
            kept=kept,
            discarded=discarded,
            tile_size=tiles["tile_size"],
            seed=seed,
            edge_thresh=edge_thresh,
            edge_keep_frac=edge_keep_frac,
        )

    return kept


# ============================================================================
# 6. PIPELINE ORCHESTRATOR
# ============================================================================
# Which override keys belong to which stage. This is the single source of
# truth for kwarg routing: `_split_overrides` reads it to fan a flat
# `**overrides` dict out into per-stage bundles, and it also defines the set
# of *accepted* override names (so typos can be rejected instead of silently
# ignored). To expose a new stage parameter, add its name here -- nothing
# else in the orchestrator needs to change.
_STAGE_PARAMS: Dict[str, List[str]] = {
    "threshold": ["sat_max", "val_min"],
    "morph":     ["median_ksize", "open_ksize", "close_ksize",
                  "min_blob_area", "smooth_ksize"],
    "edges":     ["ksize"],
    "tile":      ["tile_size", "overlap"],
    "noise":     ["min_pixels", "edge_margin", "fg_threshold"],
    "selection": ["edge_thresh", "edge_keep_frac", "seed"],
}


def _split_overrides(overrides: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Route a flat ``**overrides`` dict into per-stage kwarg bundles.

    Returns a dict ``{stage_name: {param: value, ...}}`` covering every stage
    in :data:`_STAGE_PARAMS` (bundles for stages with no overrides are empty).

    Raises:
        TypeError: If ``overrides`` contains a key that is not a recognised
            stage parameter -- this turns a silently-ignored config typo into
            an immediate, explanatory error.
    """
    known = {key for keys in _STAGE_PARAMS.values() for key in keys}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(
            f"preprocess_image got unexpected override(s): {sorted(unknown)}. "
            f"Accepted overrides: {sorted(known)}."
        )
    return {stage: _pick(overrides, keys)
            for stage, keys in _STAGE_PARAMS.items()}


def _preprocess_one(
    name: str,
    rgb: np.ndarray,
    stage_kw: Dict[str, Dict[str, Any]],
    show_plots: bool,
) -> List[Dict[str, Any]]:
    """Run the full stage chain on a single RGB image.

    This is the per-image worker behind :func:`preprocess_image`: the batch
    loop lives in the orchestrator, the actual stage sequence lives here, so
    the chain can be read (or reused / tested) on its own.

    Args:
        name: Identifier for the image, used only in console output.
        rgb: ``uint8`` RGB image of shape ``(H, W, 3)``.
        stage_kw: Per-stage kwarg bundles from :func:`_split_overrides`.
        show_plots: Propagated to every stage that renders a figure.

    Returns:
        The image's ``symbol_candidates`` list (see :func:`tile_selection`).
    """
    print(f"\n=== preprocess_image: {name} ===")
    mask    = apply_threshold(rgb,     show_plots=show_plots, **stage_kw["threshold"])
    cleaned = apply_morphology(mask,   show_plots=show_plots, **stage_kw["morph"])
    edges   = keep_edges(cleaned,      show_plots=show_plots, **stage_kw["edges"])
    tiles   = tile_image(edges,                               **stage_kw["tile"])
    tiles   = remove_noise_contours(tiles,                    **stage_kw["noise"])
    return tile_selection(tiles,       show_plots=show_plots, **stage_kw["selection"])


def preprocess_image(
    images: Dict[str, np.ndarray],
    show_plots: bool = True,
    **overrides: Any,
) -> Dict[str, List[Dict[str, Any]]]:
    """Run the full preprocessing pipeline on a batch of images.

    For each image, applies the chain:
    ``apply_threshold -> apply_morphology -> keep_edges -> tile_image
    -> remove_noise_contours -> tile_selection``. The ``show_plots`` flag is
    propagated to every stage.

    Note:
        ``remove_noise_contours`` cleans each tile's ``content`` in place but
        leaves ``tiles["source_image"]`` untouched -- the per-stage figure
        from :func:`tile_selection` therefore shows the original (un-cleaned)
        edge map as context, while the sampled-tile grid shows the cleaned
        tiles. The reported ``edge_fraction`` reflects the cleaned tiles.

    Args:
        images: Mapping of ``{filename_or_id: rgb_image}``. ``rgb_image``
            must be a ``uint8`` array of shape ``(H, W, 3)``.
        show_plots: If True, every stage of every image renders its
            diagnostic figure.
        **overrides: Optional keyword overrides for stage parameters. Any
            parameter listed in :data:`_STAGE_PARAMS` is accepted; passing
            an unrecognised name raises ``TypeError``. The noise-removal
            parameters are ``min_pixels``, ``edge_margin`` and
            ``fg_threshold`` (see ``contour_utils.remove_noise_contours``);
            when ``fg_threshold`` is not given it defaults to the selection
            stage's ``edge_thresh`` so both stages agree on what counts as
            an edge pixel.

    Returns:
        Mapping ``{filename: symbol_candidates}`` where ``symbol_candidates``
        is the list returned by :func:`tile_selection` for that image.

    Side effects:
        Renders matplotlib figures when ``show_plots`` is True. Always
        prints per-image statistics from :func:`tile_selection`.
    """
    stage_kw = _split_overrides(overrides)

    # Keep the noise step and tile_selection consistent: both decide what
    # counts as an "edge" pixel. Unless the caller overrode `fg_threshold`,
    # reuse the selection stage's `edge_thresh` for it.
    stage_kw["noise"].setdefault(
        "fg_threshold",
        stage_kw["selection"].get("edge_thresh", DEFAULT_EDGE_THRESH),
    )

    results: Dict[str, List[Dict[str, Any]]] = {}
    for name, rgb in images.items():
        results[name] = _preprocess_one(
            name, rgb, stage_kw=stage_kw, show_plots=show_plots,
        )
    return results


# ============================================================================
# Private helpers
# ============================================================================
def _pick(d: Dict[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    """Return the subset of ``d`` whose keys are in ``keys``."""
    return {k: d[k] for k in keys if k in d}


def _plot_before_after(
    before: np.ndarray,
    after: np.ndarray,
    title_before: str,
    title_after: str,
    cmap_before: str | None = None,
    cmap_after: str | None = None,
    sup_title: str | None = None,
) -> None:
    """Side-by-side ``before`` vs ``after`` comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(before, cmap=cmap_before, vmin=0,
                   vmax=255 if cmap_before == "gray" else None)
    axes[0].set_title(title_before)
    axes[0].axis("off")
    axes[1].imshow(after, cmap=cmap_after, vmin=0,
                   vmax=255 if cmap_after == "gray" else None)
    axes[1].set_title(title_after)
    axes[1].axis("off")
    if sup_title:
        fig.suptitle(sup_title, fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()


def _draw_kept_tiles_overlay(
    ax: "plt.Axes",
    source_image: np.ndarray,
    kept: List[Dict[str, Any]],
) -> None:
    """Draw *every* kept tile as a translucent filled box on the source image.

    Unlike the "sample positions" view (which outlines only 15 tiles), this
    shows the full set of kept tiles. Because the boxes are translucent and
    the tiling overlaps, regions covered by several kept tiles render
    brighter -- so the overlay doubles as a coverage heatmap: it shows not
    only *which* tiles were kept but *where they concentrate* on the image.

    Args:
        ax: Matplotlib Axes to draw onto.
        source_image: The image passed to :func:`tile_image` (Sobel edge map).
        kept: List of kept-tile dicts, each carrying ``x, y, w, h``.

    Side effects:
        Draws onto ``ax`` in place; does not create a figure or call show().
    """
    cmap = "gray" if source_image.ndim == 2 else None
    ax.imshow(source_image, cmap=cmap, vmin=0,
              vmax=255 if cmap == "gray" else None)
    for t in kept:
        ax.add_patch(patches.Rectangle(
            (t["x"], t["y"]), t["w"], t["h"],
            facecolor="lime", edgecolor="lime",
            alpha=0.12, linewidth=0.4))
    ax.set_title(
        f"All {len(kept)} kept tiles on the tile_image input "
        f"(brighter = more tile overlap)"
    )
    ax.axis("off")


def _plot_tile_selection(
    source_image: np.ndarray,
    kept: List[Dict[str, Any]],
    discarded: List[Dict[str, Any]],
    tile_size: int,
    seed: int,
    edge_thresh: int,
    edge_keep_frac: float,
) -> None:
    """Render the tile_selection diagnostic panel.

    Layout (top to bottom):
        Row 1 : (left) input image; (right) sampled tile positions.
        Row 2 : all kept tiles overlaid on the input image.
        Row 3 : 4 x 5 grid -- 15 kept (lime border) + 5 discarded (red).
    """
    rng = random.Random(seed)
    sample_kept      = _sample(kept,      N_KEPT_PREVIEW,      rng)
    sample_discarded = _sample(discarded, N_DISCARDED_PREVIEW, rng)

    fig = plt.figure(figsize=(14, 14))
    gs = fig.add_gridspec(
        nrows=3, ncols=1, height_ratios=[1.0, 1.0, 1.2], hspace=0.3
    )

    # ---- row 1: source + sample-position overlay ----
    top = gs[0].subgridspec(1, 2, wspace=0.05)
    ax_src = fig.add_subplot(top[0, 0])
    ax_ovr = fig.add_subplot(top[0, 1])

    cmap = "gray" if source_image.ndim == 2 else None
    ax_src.imshow(source_image, cmap=cmap, vmin=0,
                  vmax=255 if cmap == "gray" else None)
    ax_src.set_title("Input to tile_image (Sobel edge map)")
    ax_src.axis("off")

    ax_ovr.imshow(source_image, cmap=cmap, vmin=0,
                  vmax=255 if cmap == "gray" else None)
    for t in sample_kept:
        ax_ovr.add_patch(patches.Rectangle(
            (t["x"], t["y"]), t["w"], t["h"],
            fill=False, edgecolor="lime", linewidth=1.4))
    for t in sample_discarded:
        ax_ovr.add_patch(patches.Rectangle(
            (t["x"], t["y"]), t["w"], t["h"],
            fill=False, edgecolor="red",
            linewidth=1.4, linestyle="--"))
    ax_ovr.set_title(
        f"Sample tile positions  -  "
        f"{len(sample_kept)} kept (lime), {len(sample_discarded)} "
        f"discarded (red dashed)"
    )
    ax_ovr.axis("off")

    # ---- row 2: ALL kept tiles overlaid on the input image ----
    ax_all = fig.add_subplot(gs[1])
    _draw_kept_tiles_overlay(ax_all, source_image, kept)

    # ---- row 3: 4 x 5 tile grid (15 kept + 5 discarded) ----
    bottom = gs[2].subgridspec(4, 5, wspace=0.1, hspace=0.25)
    preview = [(t, True)  for t in sample_kept] \
            + [(t, False) for t in sample_discarded]
    for cell, (tile, is_kept) in enumerate(preview):
        r, c = divmod(cell, 5)
        ax = fig.add_subplot(bottom[r, c])
        ax.imshow(tile["content"], cmap="gray", vmin=0, vmax=255)
        ax.set_title(
            f"#{tile['index']}  ef={tile['edge_fraction']:.2f}",
            fontsize=8,
            color="green" if is_kept else "darkred",
        )
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("lime" if is_kept else "red")
            spine.set_linewidth(2.0)

    # Pad empty cells if we didn't have enough kept/discarded tiles.
    used = len(preview)
    for cell in range(used, N_KEPT_PREVIEW + N_DISCARDED_PREVIEW):
        r, c = divmod(cell, 5)
        ax = fig.add_subplot(bottom[r, c])
        ax.set_facecolor("#eeeeee")
        ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                color="#999", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(
        f"tile_selection  (thresh={edge_thresh}, "
        f"keep_frac>={edge_keep_frac:.0%})  -  "
        f"{len(kept)} kept / {len(discarded)} discarded",
        fontsize=13, fontweight="bold",
    )
    plt.show()


def _sample(
    population: List[Dict[str, Any]], k: int, rng: random.Random
) -> List[Dict[str, Any]]:
    """Reproducible sample of up to ``k`` items from ``population``."""
    if not population:
        return []
    return rng.sample(population, min(k, len(population)))
#!/usr/bin/env python3
"""
visualize.py — see the tiling produced by tile_image.py.

Two views:
  draw_grid(...)    overlays the tile grid on the original image so you can
                    see HOW it was cut.
  tile_montage(...) lays the actual tiles out with gutters so you can see the
                    individual chunks (optionally just a row/col sub-range,
                    since a full image is ~1800 tiles).

Both read the manifest.json that tile_image.py writes, so they stay in sync
with whatever mode/size you tiled with. Each returns a matplotlib Figure:
call .savefig(...) in a script, or plt.show() if you run locally.
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")  # script-safe backend; remove for interactive plt.show()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image


def load_manifest(tile_dir):
    """Load the manifest.json from a tile output folder."""
    with open(Path(tile_dir) / "manifest.json") as fh:
        return json.load(fh)


def draw_grid(source_image, tile_dir, every=1, linewidth=0.5):
    """
    Overlay the tile grid on the original image.

    source_image : path to the ORIGINAL image that was tiled
    tile_dir     : the folder tile_image.py wrote (contains manifest.json)
    every        : draw every Nth line (use >1 if lines are too dense to see)
    """
    manifest = load_manifest(tile_dir)
    img = Image.open(source_image)

    fig, ax = plt.subplots(figsize=(12, 12 * img.height / img.width))
    ax.imshow(img)

    # Vertical lines at unique tile x-starts; horizontal at unique y-starts.
    xs = sorted({t["x"] for t in manifest["tiles"]})
    ys = sorted({t["y"] for t in manifest["tiles"]})
    for x in xs[::every]:
        ax.axvline(x, color="red", linewidth=linewidth, alpha=0.7)
    for y in ys[::every]:
        ax.axhline(y, color="red", linewidth=linewidth, alpha=0.7)

    g = manifest["grid"]
    ax.set_title(f"{manifest['source']} — {g['cols']}x{g['rows']} "
                 f"= {manifest['tile_count']} tiles ({manifest['mode']})")
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)
    fig.tight_layout()
    return fig


def tile_montage(tile_dir, rows=None, cols=None, gutter=4, max_tiles=400):
    """
    Lay the actual saved tiles out in a grid with gutters between them.

    rows, cols : optional (start, end) tuples to show only a sub-range —
                 recommended, since a full image is hundreds of tiles.
    gutter     : pixels of padding drawn between tiles.
    max_tiles  : safety cap; raise if you really want a huge montage.
    """
    manifest = load_manifest(tile_dir)
    tile_dir = Path(tile_dir)

    r0, r1 = rows if rows else (0, manifest["grid"]["rows"])
    c0, c1 = cols if cols else (0, manifest["grid"]["cols"])
    sel = [t for t in manifest["tiles"]
           if r0 <= t["row"] < r1 and c0 <= t["col"] < c1]

    if not sel:
        raise ValueError("no tiles in that row/col range")
    if len(sel) > max_tiles:
        raise ValueError(f"{len(sel)} tiles exceeds max_tiles={max_tiles}; "
                         "narrow the rows/cols range or raise the cap")

    n_rows, n_cols = r1 - r0, c1 - c0
    cell = manifest["target_size"] + gutter
    canvas = Image.new("RGB",
                       (n_cols * cell + gutter, n_rows * cell + gutter),
                       (40, 40, 40))
    for t in sel:
        tile = Image.open(tile_dir / t["file"])
        x = (t["col"] - c0) * cell + gutter
        y = (t["row"] - r0) * cell + gutter
        canvas.paste(tile, (x, y))

    fig, ax = plt.subplots(figsize=(min(14, n_cols), min(14, n_rows)))
    ax.imshow(canvas)
    ax.set_title(f"tiles rows {r0}:{r1}, cols {c0}:{c1}  ({len(sel)} tiles)")
    ax.axis("off")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    # Example: tile an image, then render both views.
    from tile_image import tile_image

    SRC = "/mnt/user-data/uploads/L1000843.jpg"
    OUT = "tiles"

    manifest = tile_image(SRC, OUT, target=75, mode="adaptive",
                          edge="partial", fmt="png")
    tdir = Path(OUT) / Path(SRC).stem

    draw_grid(SRC, tdir, every=2).savefig("grid_overlay.png", dpi=120)
    tile_montage(tdir, rows=(12, 22), cols=(0, 14)).savefig(
        "tile_montage.png", dpi=120)
    print("wrote grid_overlay.png and tile_montage.png")

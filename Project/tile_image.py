#!/usr/bin/env python3
"""
tile_image.py — split an image into small (~75x75 px) chunks.

Two modes:
  adaptive (default) : pick a grid so every tile is as close to the target
                       size as possible. Tile sizes vary by +/-1 px so the
                       grid covers the image EXACTLY -- no padding, no ragged
                       sliver tiles at the edges.
  fixed              : every tile is exactly --size px. The right/bottom edge
                       is handled by --edge: 'partial' keeps undersized tiles,
                       'pad' pads them to full size, 'drop' discards them.

Each run also writes manifest.json describing the grid and the pixel bounding
box of every tile in the original image, so tiles can be mapped back to
image coordinates (or reassembled) later in a pipeline.

Usage:
    python tile_image.py input.jpg
    python tile_image.py input.jpg -o tiles/ --size 75
    python tile_image.py input.jpg --mode fixed --edge pad
    python tile_image.py *.jpg -o tiles/        # batch
"""

import argparse
import json
import os
import sys
from pathlib import Path

from PIL import Image


def _split_lengths(total, target):
    """
    Split `total` pixels into n contiguous segments whose lengths are all
    within 1 px of each other and as close to `target` as possible.

    Returns a list of integer lengths that sum exactly to `total`.
    """
    # Choose the segment count that gets us closest to the target size.
    n = max(1, round(total / target))
    base, remainder = divmod(total, n)
    # `remainder` segments get one extra pixel; the rest get `base`.
    return [base + 1 if i < remainder else base for i in range(n)]


def _offsets(lengths):
    """Convert a list of segment lengths into (start, length) pairs."""
    out = []
    pos = 0
    for L in lengths:
        out.append((pos, L))
        pos += L
    return out


def compute_grid(width, height, target, mode, edge):
    """
    Return two lists of (start, length) pairs: one for the x axis, one for y.
    """
    if mode == "adaptive":
        xs = _offsets(_split_lengths(width, target))
        ys = _offsets(_split_lengths(height, target))
        return xs, ys

    # fixed mode
    xs, ys = [], []
    for axis_total, store in ((width, xs), (height, ys)):
        pos = 0
        while pos < axis_total:
            length = min(target, axis_total - pos)
            if length < target and edge == "drop":
                break
            store.append((pos, length))
            pos += target
    return xs, ys


def tile_image(path, out_root, target, mode, edge, fmt):
    path = Path(path)
    img = Image.open(path)
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    width, height = img.size

    xs, ys = compute_grid(width, height, target, mode, edge)

    out_dir = Path(out_root) / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    n_rows, n_cols = len(ys), len(xs)
    row_pad = len(str(n_rows - 1))
    col_pad = len(str(n_cols - 1))

    tiles = []
    for r, (y0, th) in enumerate(ys):
        for c, (x0, tw) in enumerate(xs):
            crop = img.crop((x0, y0, x0 + tw, y0 + th))

            if mode == "fixed" and edge == "pad" and (tw < target or th < target):
                fill = 0 if img.mode == "L" else (0,) * len(img.getbands())
                padded = Image.new(img.mode, (target, target), fill)
                padded.paste(crop, (0, 0))
                crop = padded

            name = f"tile_r{r:0{row_pad}d}_c{c:0{col_pad}d}.{fmt}"
            crop.save(out_dir / name)
            tiles.append({
                "file": name,
                "row": r,
                "col": c,
                "x": x0,
                "y": y0,
                "width": tw,
                "height": th,
            })

    manifest = {
        "source": path.name,
        "source_size": {"width": width, "height": height},
        "mode": mode,
        "target_size": target,
        "grid": {"rows": n_rows, "cols": n_cols},
        "tile_count": len(tiles),
        "tiles": tiles,
    }
    with open(out_dir / "manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"{path.name}: {len(tiles)} tiles ({n_cols}x{n_rows}) -> {out_dir}/")
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Split image(s) into ~75x75 px chunks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("images", nargs="+", help="input image file(s)")
    ap.add_argument("-o", "--out", default="tiles",
                    help="output root directory")
    ap.add_argument("-s", "--size", type=int, default=75,
                    help="target tile size in pixels")
    ap.add_argument("-m", "--mode", choices=["adaptive", "fixed"],
                    default="adaptive",
                    help="adaptive = even grid near target; "
                         "fixed = exact size with edge handling")
    ap.add_argument("-e", "--edge", choices=["partial", "pad", "drop"],
                    default="partial",
                    help="fixed mode only: how to handle edge tiles")
    ap.add_argument("-f", "--format", default="png",
                    help="output image format/extension")
    args = ap.parse_args(argv)

    if args.size < 1:
        ap.error("--size must be >= 1")

    for img_path in args.images:
        if not os.path.isfile(img_path):
            print(f"skip (not a file): {img_path}", file=sys.stderr)
            continue
        try:
            tile_image(img_path, args.out, args.size,
                       args.mode, args.edge, args.format)
        except Exception as exc:  # noqa: BLE001
            print(f"error on {img_path}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()

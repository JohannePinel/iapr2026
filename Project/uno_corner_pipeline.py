"""
UNO Card Corner-Symbol Detection Pipeline
==========================================

Classical-CV pipeline (numpy + scipy.ndimage + PIL) that:
  - segments UNO cards from either a plain white background or the tropical
    leaf-print background,
  - rectifies each card to a canonical landscape view via PCA,
  - locates the two corner identifiers on each card,
  - decomposes each identifier into a "pill" (the white capsule) and a
    "glyph" (the dark digit or symbol inside it),
  - reports pixel measurements and optionally emits an annotated composite.

No OpenCV, no learned models — every operation is a short numpy /
scipy.ndimage call. The script is meant to be readable end-to-end and
serve as a reference for a manual-implementation CV project.

Stages
------
  1. Foreground mask of card bodies (background-specific)
  2. Morphological cleanup (binary dilation + hole filling)
  3. Connected-component labeling and area / aspect filtering
  4. PCA orientation recovery + affine rectification
  5. Pill detection inside a corner quadrant
  6. Glyph detection inside the pill (local-adaptive threshold)
  7. Annotation / composite visualization

Run as a script
---------------
  python uno_corner_pipeline.py            # processes the three sample images
  python uno_corner_pipeline.py --help     # CLI options
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage


# =====================================================================
# Data containers
# =====================================================================

@dataclass
class BBox:
    """Bounding box of a detected region inside the rectified card image.

    Stored as (h, w, area, slice) to match scipy.ndimage's convention.
    """
    h: int
    w: int
    area: int
    slc: Tuple[slice, slice]   # (row_slice, col_slice) into the card image


@dataclass
class CornerDetection:
    pill: Optional[BBox]
    glyph: Optional[BBox]


@dataclass
class CardResult:
    component_id: int
    card_image: np.ndarray              # rectified RGB crop, landscape oriented
    long_len: float                     # length along PCA major axis (px)
    short_len: float                    # length along PCA minor axis (px)
    corners: dict                       # {'TL': CornerDetection, 'BR': CornerDetection}


# =====================================================================
# Stage 1 — Foreground masks
# =====================================================================

def fg_mask_white_bg(image: np.ndarray,
                     sat_thresh: int = 35,
                     bright_thresh: int = 200) -> np.ndarray:
    """Foreground mask for cards on a plain bright (white / pale grey) table.

    A pixel is foreground if it is either:
      - SATURATED   (max channel - min channel > sat_thresh), OR
      - DARK        (max channel < bright_thresh).

    The saturation branch picks up the colored card borders. The darkness
    branch picks up printed black ink and dark wild cards that have low
    saturation but are still distinguishable from the bright background.
    """
    r = image[..., 0].astype(np.int16)
    g = image[..., 1].astype(np.int16)
    b = image[..., 2].astype(np.int16)
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    saturation = maxc - minc
    return (saturation > sat_thresh) | (maxc < bright_thresh)


def fg_mask_tropical_bg(image: np.ndarray,
                        white_thresh: int = 215) -> np.ndarray:
    """Foreground mask for cards on the tropical-leaf print.

    The leaves are themselves saturated, so the white-bg mask is useless
    here. Exploit the fact that every UNO card has white regions (border
    and central oval) and the leaves do not: require ALL three RGB channels
    above `white_thresh`. The leaves always have at least one weak channel.

    Only marks the WHITE regions of each card; the morphological dilation
    step then closes the gap to the colored body.
    """
    return ((image[..., 0] > white_thresh) &
            (image[..., 1] > white_thresh) &
            (image[..., 2] > white_thresh))


# =====================================================================
# Stage 2 — Card-blob extraction
# =====================================================================

def extract_card_blobs(mask: np.ndarray,
                       dilate_iters: int,
                       min_area: int = 30_000,
                       max_area: int = 350_000
                       ) -> Tuple[np.ndarray, List[int]]:
    """Turn the noisy foreground mask into clean per-card blobs.

    Steps:
      1. Binary dilation bridges the colored border + the central white
         oval + printed ink into a single connected card region.
      2. Hole filling makes each card a solid blob.
      3. Label each blob with a unique integer.
      4. Filter by area:
           - too small: noise specks / glyph fragments
           - too large: stacks of overlapping cards merged into one blob.

    The dilation iteration count is the main scale-dependent knob:
    too few leaves cards fragmented, too many merges adjacent cards.
    Values used here: 15 on white bg, 8 on the tropical bg (the tropical
    mask is sparser to start with, hence the smaller value to avoid
    merging neighbors).

    Returns
    -------
    labels : np.ndarray
        Same shape as the input mask, integer-labeled.
    valid : list[int]
        Component IDs whose area passes the filter.
    """
    dilated = ndimage.binary_dilation(mask, iterations=dilate_iters)
    filled = ndimage.binary_fill_holes(dilated)
    labels, n = ndimage.label(filled)
    sizes = ndimage.sum(filled, labels, range(1, n + 1))
    valid = [ci for ci in range(1, n + 1)
             if min_area < sizes[ci - 1] < max_area]
    return labels, valid


# =====================================================================
# Stage 3 — PCA orientation and affine rectification
# =====================================================================

def card_geometry(component_mask: np.ndarray
                  ) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Recover card center, principal axes, and dimensions via PCA.

    Build a 2-D point cloud from the component's pixel coordinates, center
    it, and SVD it. The rows of the right singular matrix V^T are two
    orthonormal axes ordered by variance: row 0 = card's long edge
    direction, row 1 = short edge direction.

    Projecting the centered points onto each axis and taking (max - min)
    gives the true card dimensions independent of in-plane rotation.

    PCA does NOT determine the sign of each axis, so there is a 180-degree
    orientation ambiguity in the returned axes. The pipeline copes by
    searching BOTH diagonal corners for a pill: since UNO cards have a
    corner pill at each of those two locations, the sign ambiguity
    becomes irrelevant.

    Returns
    -------
    center   : ndarray, shape (2,), [x, y] in image coordinates
    Vt       : ndarray, shape (2, 2); row 0 = long-axis unit vec,
               row 1 = short-axis unit vec
    long_len, short_len : float, lengths along those axes (pixels)
    """
    ys, xs = np.where(component_mask)
    pts = np.column_stack([xs, ys]).astype(np.float64)
    center = pts.mean(axis=0)
    centered = pts - center
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt.T
    long_len = float(proj[:, 0].max() - proj[:, 0].min())
    short_len = float(proj[:, 1].max() - proj[:, 1].min())
    return center, Vt, long_len, short_len


def warp_oriented_crop(image: np.ndarray,
                       center: np.ndarray,
                       Vt: np.ndarray,
                       long_offset: float,
                       short_offset: float,
                       out_w: int,
                       out_h: int) -> np.ndarray:
    """Inverse-warp affine resample with bilinear interpolation.

    Produces an axis-aligned crop of size (out_h, out_w) whose center sits
    in the source image at
        center + long_offset * long_axis + short_offset * short_axis
    and whose +x direction follows the card's long axis,
    +y its short axis.

    For each output pixel (u, v), the source location is
        src = patch_center + (u - W/2) * long_axis + (v - H/2) * short_axis
    Bilinear-sampled from the original image with edge clamping.
    """
    long_axis, short_axis = Vt[0], Vt[1]
    patch_center = center + long_offset * long_axis + short_offset * short_axis

    us, vs = np.meshgrid(np.arange(out_w), np.arange(out_h))
    du = us - out_w / 2
    dv = vs - out_h / 2
    sx = patch_center[0] + du * long_axis[0] + dv * short_axis[0]
    sy = patch_center[1] + du * long_axis[1] + dv * short_axis[1]

    x0 = np.clip(sx.astype(int), 0, image.shape[1] - 2)
    y0 = np.clip(sy.astype(int), 0, image.shape[0] - 2)
    fx = sx - x0
    fy = sy - y0

    out = np.zeros((out_h, out_w, image.shape[2]), dtype=np.float64)
    for c in range(image.shape[2]):
        ch = image[..., c]
        v00 = ch[y0,     x0]
        v10 = ch[y0,     x0 + 1]
        v01 = ch[y0 + 1, x0]
        v11 = ch[y0 + 1, x0 + 1]
        out[..., c] = (v00 * (1 - fx) * (1 - fy) +
                       v10 *      fx  * (1 - fy) +
                       v01 * (1 - fx) *      fy +
                       v11 *      fx  *      fy)
    return np.clip(out, 0, 255).astype(np.uint8)


def rectify_card(image: np.ndarray,
                 component_mask: np.ndarray,
                 padding: int = 20
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """End-to-end rectification: blob mask in, landscape-oriented crop out.

    Returns the rectified crop plus the PCA geometry for downstream use.
    """
    center, Vt, L, S = card_geometry(component_mask)
    out_w = int(round(L)) + padding
    out_h = int(round(S)) + padding
    crop = warp_oriented_crop(image, center, Vt, 0.0, 0.0, out_w, out_h)
    return crop, center, Vt, L, S


# =====================================================================
# Stage 4 — Pill detection inside a corner quadrant
# =====================================================================

def find_corner_pill(card_image: np.ndarray,
                     corner: str = 'TL',
                     bright_thresh: int = 200,
                     area_range: Tuple[int, int] = (200, 4000),
                     max_aspect: float = 3.0) -> Optional[BBox]:
    """Find the white capsule in the requested corner of a rectified card.

    Searches only the top-left or bottom-right third of the canvas — the
    pill is always near a corner by design. Inside that quadrant:

      1. Threshold for bright pixels: grayscale > bright_thresh.
      2. Connected components.
      3. Keep the largest CC whose:
           - area is in `area_range` (rejects noise and the big central
             white oval that extends into the quadrant), and
           - bounding box is roughly oval (max side <= max_aspect * min side).

    `corner` is 'TL' or 'BR'. The returned BBox slice is expressed in the
    full card_image coordinate system, not the quadrant.
    """
    H, W = card_image.shape[:2]
    if corner == 'TL':
        ys, ye, xs, xe = 0, H // 3, 0, W // 3
    elif corner == 'BR':
        ys, ye, xs, xe = 2 * H // 3, H, 2 * W // 3, W
    else:
        raise ValueError("corner must be 'TL' or 'BR'")

    quad = card_image[ys:ye, xs:xe]
    bright = quad.mean(axis=2) > bright_thresh
    lab, n = ndimage.label(bright)
    slices = ndimage.find_objects(lab)
    sizes = ndimage.sum(bright, lab, range(1, n + 1))

    best: Optional[BBox] = None
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        area = float(sizes[i])
        if not (area_range[0] < area < area_range[1]):
            continue
        h = sl[0].stop - sl[0].start
        w = sl[1].stop - sl[1].start
        if max(h, w) > min(h, w) * max_aspect:
            continue
        if best is None or area > best.area:
            full_sl = (slice(ys + sl[0].start, ys + sl[0].stop),
                       slice(xs + sl[1].start, xs + sl[1].stop))
            best = BBox(h=h, w=w, area=int(area), slc=full_sl)
    return best


# =====================================================================
# Stage 5 — Glyph detection inside the pill
# =====================================================================

def find_glyph_in_pill(card_image: np.ndarray,
                       pill: BBox,
                       local_window: int = 51,
                       darkness_offset: float = 25.0,
                       area_range: Tuple[int, int] = (30, 1500)
                       ) -> Optional[BBox]:
    """Find the dark digit/symbol inside a pill via local-adaptive threshold.

    A box-filtered grayscale gives the local background brightness. Pixels
    darker than (local_bg - darkness_offset) are marked. The largest CC of
    valid area is the glyph.

    Local adaptation matters because lighting across a single card varies
    by tens of grey levels; a global dark threshold either misses glyphs
    on the bright side or merges shadows on the dark side.

    The returned BBox slice is in full card_image coordinates.
    """
    pr = card_image[pill.slc]
    gray = pr.mean(axis=2)
    local_bg = ndimage.uniform_filter(gray, size=local_window)
    dark = (local_bg - gray) > darkness_offset

    lab, n = ndimage.label(dark)
    slices = ndimage.find_objects(lab)
    sizes = ndimage.sum(dark, lab, range(1, n + 1))

    best: Optional[BBox] = None
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        area = float(sizes[i])
        if not (area_range[0] < area < area_range[1]):
            continue
        if best is None or area > best.area:
            h = sl[0].stop - sl[0].start
            w = sl[1].stop - sl[1].start
            full_sl = (
                slice(pill.slc[0].start + sl[0].start,
                      pill.slc[0].start + sl[0].stop),
                slice(pill.slc[1].start + sl[1].start,
                      pill.slc[1].start + sl[1].stop),
            )
            best = BBox(h=h, w=w, area=int(area), slc=full_sl)
    return best


# =====================================================================
# Stage 6 — Visualization
# =====================================================================

def annotate_card(card_image: np.ndarray,
                  pill: Optional[BBox] = None,
                  glyph: Optional[BBox] = None,
                  pill_color=(0, 255, 0),
                  glyph_color=(255, 0, 255)) -> Image.Image:
    """Return a PIL image with detected pill/glyph bounding boxes drawn."""
    out = Image.fromarray(card_image.copy())
    draw = ImageDraw.Draw(out)
    if pill is not None:
        rs, cs = pill.slc
        draw.rectangle([cs.start - 3, rs.start - 3,
                        cs.stop + 3,  rs.stop + 3],
                       outline=pill_color, width=4)
    if glyph is not None:
        rs, cs = glyph.slc
        draw.rectangle([cs.start - 2, rs.start - 2,
                        cs.stop + 2,  rs.stop + 2],
                       outline=glyph_color, width=3)
    return out


def crop_pill_with_margin(card_image: np.ndarray,
                          pill: BBox,
                          margin: int = 10) -> np.ndarray:
    """Return a small crop centered on the pill with a contextual margin."""
    rs, cs = pill.slc
    H, W = card_image.shape[:2]
    y0 = max(0, rs.start - margin); y1 = min(H, rs.stop + margin)
    x0 = max(0, cs.start - margin); x1 = min(W, cs.stop + margin)
    return card_image[y0:y1, x0:x1]


def _get_font(size: int, bold: bool = True):
    path_b = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
    path_r = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    try:
        return ImageFont.truetype(path_b if bold else path_r, size)
    except Exception:
        return ImageFont.load_default()


def build_composite(samples: List[dict], out_path: Path,
                    W_card: int = 480, W_corner: int = 180,
                    PAD: int = 14, ROW_H: int = 320) -> Image.Image:
    """Build a multi-row composite figure.

    Each sample is a dict with keys:
        label    : caption text
        card     : rectified card ndarray
        tl_crop  : ndarray pill crop for top-left corner (or None)
        br_crop  : ndarray pill crop for bottom-right corner (or None)
        pill_hw  : (h, w) pill bbox to report
        glyph_hw : (h, w) glyph bbox to report
    """
    fb = _get_font(22, bold=True)
    fs = _get_font(16, bold=False)
    fsb = _get_font(16, bold=True)

    W = PAD + W_card + PAD + W_corner + PAD + W_corner + PAD
    H = PAD + len(samples) * (ROW_H + PAD)
    canvas = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(canvas)

    y = PAD
    for s in samples:
        card = Image.fromarray(s['card'])
        cw, ch = card.size
        sc = min(W_card / cw, (ROW_H - 50) / ch)
        card_r = card.resize((int(cw * sc), int(ch * sc)), Image.LANCZOS)
        canvas.paste(card_r, (PAD, y + 30))
        draw.text((PAD, y + 4), s['label'], fill='black', font=fb)

        for which, crop_arr, x in [
            ('TL', s.get('tl_crop'), PAD + W_card + PAD),
            ('BR', s.get('br_crop'), PAD + W_card + PAD + W_corner + PAD),
        ]:
            if crop_arr is None:
                continue
            p = Image.fromarray(crop_arr)
            pw, ph = p.size
            sc2 = min(W_corner / pw, 110 / ph)
            p_r = p.resize((int(pw * sc2), int(ph * sc2)), Image.LANCZOS)
            canvas.paste(p_r, (x, y + 30))
            draw.text((x, y + 30 + p_r.size[1] + 6),
                      f'{which} corner', fill='black', font=fsb)
            if which == 'TL':
                draw.text(
                    (x, y + 30 + p_r.size[1] + 28),
                    f'pill ≈ {s["pill_hw"][0]}×{s["pill_hw"][1]} px\n'
                    f'glyph ≈ {s["glyph_hw"][0]}×{s["glyph_hw"][1]} px',
                    fill='black', font=fs)
        y += ROW_H + PAD

    canvas.save(out_path)
    return canvas


# =====================================================================
# Top-level pipeline
# =====================================================================

# Per-background hyperparameters that I tuned for the three sample images.
BG_PRESETS = {
    'white':    {'mask_fn': fg_mask_white_bg,    'dilate_iters': 15},
    'tropical': {'mask_fn': fg_mask_tropical_bg, 'dilate_iters':  8},
}


""" def process_image(image_path: Path,
                  bg_mode: str,
                  aspect_range: Tuple[float, float] = (1.35, 1.75)
                  ) -> Tuple[np.ndarray, List[CardResult]]:
    ""Full pipeline: load -> mask -> blobs -> rectify -> detect corners.

    bg_mode : 'white' or 'tropical'.

    aspect_range filters single-card blobs (UNO real aspect is 88/57 ≈ 1.54);
    blobs outside this range are likely overlapping stacks or fragments.
    ""
    if bg_mode not in BG_PRESETS:
        raise ValueError(f"bg_mode must be one of {list(BG_PRESETS)}")
    preset = BG_PRESETS[bg_mode]

    image = np.array(Image.open(image_path))
    mask = preset['mask_fn'](image)
    labels, valid = extract_card_blobs(mask, dilate_iters=preset['dilate_iters'])

    cards: List[CardResult] = []
    for ci in valid:
        sm = (labels == ci)
        center, Vt, L, S = card_geometry(sm)
        if not (aspect_range[0] < L / S < aspect_range[1]):
            continue
        card_img = warp_oriented_crop(image, center, Vt, 0.0, 0.0,
                                      int(round(L)) + 20, int(round(S)) + 20)
        corners = {}
        for which in ('TL', 'BR'):
            pill = find_corner_pill(card_img, which)
            glyph = find_glyph_in_pill(card_img, pill) if pill else None
            corners[which] = CornerDetection(pill=pill, glyph=glyph)
        cards.append(CardResult(
            component_id=ci,
            card_image=card_img,
            long_len=L, short_len=S,
            corners=corners,
        ))
    return image, cards """

import matplotlib.pyplot as plt # New Import

# ... [Previous imports and data containers remain unchanged] ...

def process_image(image_path: Path,
                  bg_mode: str,
                  aspect_range: Tuple[float, float] = (1.35, 1.75),
                  show_debug: bool = True # Added flag
                  ) -> Tuple[np.ndarray, List[CardResult]]:
    """Full pipeline with visualization of intermediate steps."""
    if bg_mode not in BG_PRESETS:
        raise ValueError(f"bg_mode must be one of {list(BG_PRESETS)}")
    preset = BG_PRESETS[bg_mode]

    image = np.array(Image.open(image_path))
    mask = preset['mask_fn'](image)
    labels, valid = extract_card_blobs(mask, dilate_iters=preset['dilate_iters'])

    # --- Visualization: Masks and Labels ---
    if show_debug:
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        ax[0].imshow(image)
        ax[0].set_title("Original Image")
        ax[1].imshow(mask, cmap='gray')
        ax[1].set_title(f"Foreground Mask ({bg_mode})")
        ax[2].imshow(labels, cmap='nipy_spectral')
        ax[2].set_title(f"Detected Blobs (Valid: {len(valid)})")
        for a in ax: a.axis('off')
        plt.tight_layout()
        plt.show()

    cards: List[CardResult] = []
    
    # Process each valid blob
    for i, ci in enumerate(valid):
        sm = (labels == ci)
        center, Vt, L, S = card_geometry(sm)
        
        # Aspect ratio filter
        if not (aspect_range[0] < L / S < aspect_range[1]):
            continue
            
        card_img = warp_oriented_crop(image, center, Vt, 0.0, 0.0,
                                      int(round(L)) + 20, int(round(S)) + 20)
        
        corners = {}
        for which in ('TL', 'BR'):
            pill = find_corner_pill(card_img, which)
            glyph = find_glyph_in_pill(card_img, pill) if pill else None
            corners[which] = CornerDetection(pill=pill, glyph=glyph)
        
        res = CardResult(
            component_id=ci,
            card_image=card_img,
            long_len=L, short_len=S,
            corners=corners,
        )
        cards.append(res)

        # --- Visualization: Per-Card Detection ---
        if show_debug:
            annotated = annotate_card(card_img, corners['TL'].pill, corners['TL'].glyph)
            
            plt.figure(figsize=(10, 4))
            plt.suptitle(f"Card {i+1} (ID: {ci}) - Rectified & Detected", fontsize=14)
            
            plt.subplot(1, 2, 1)
            plt.imshow(annotated)
            plt.title("Rectified View (TL Annotations)")
            plt.axis('off')
            
            # Show the zoomed in pill/glyph if they exist
            if corners['TL'].pill:
                plt.subplot(1, 2, 2)
                zoom = crop_pill_with_margin(card_img, corners['TL'].pill)
                plt.imshow(zoom)
                plt.title("TL Corner Zoom")
            
            plt.axis('off')
            plt.show()

    return image, cards

# =====================================================================
# Driver — reproduces the measurements and composite from this thread
# =====================================================================

# The three reference images and the segmentation mode each one needs.
DEFAULT_INPUTS = [
    ('Image 1', '/mnt/user-data/uploads/L1000842.jpg', 'white'),
    ('Image 2', '/mnt/user-data/uploads/L1000902.jpg', 'tropical'),
    ('Image 3', '/mnt/user-data/uploads/L1000904.jpg', 'tropical'),
]


def main():
    p = argparse.ArgumentParser(description='UNO corner-symbol pipeline.')
    p.add_argument('--out-dir', type=Path, default=Path('./output'))
    p.add_argument('--composite', type=Path, default=None,
                   help='If set, save a composite figure with one row per card.')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    for label, path, bg in DEFAULT_INPUTS:
        print(f'\n=== {label}  ({Path(path).name}, bg={bg}) ===')
        _, cards = process_image(Path(path), bg)
        for i, c in enumerate(cards, start=1):
            print(f'  card {i}: long={int(c.long_len)} short={int(c.short_len)} '
                  f'(component {c.component_id})')
            for which in ('TL', 'BR'):
                det = c.corners[which]
                p_str = (f'pill {det.pill.h}x{det.pill.w} area={det.pill.area}'
                         if det.pill else 'pill ---')
                g_str = (f'glyph {det.glyph.h}x{det.glyph.w} area={det.glyph.area}'
                         if det.glyph else 'glyph ---')
                print(f'    {which}: {p_str} | {g_str}')

            # Save annotated rectified card
            ann = annotate_card(c.card_image,
                                pill=c.corners['TL'].pill,
                                glyph=c.corners['TL'].glyph)
            ann.save(args.out_dir / f'{label.replace(" ", "")}_card{i}_annot.png')

            # Collect a composite row for clean cards (TL pill present)
            tl = c.corners['TL']; br = c.corners['BR']
            if tl.pill is None:
                continue
            all_samples.append({
                'label': f'{label} — card {i}  ({int(c.long_len)}×{int(c.short_len)} px)',
                'card': c.card_image,
                'tl_crop': crop_pill_with_margin(c.card_image, tl.pill),
                'br_crop': (crop_pill_with_margin(c.card_image, br.pill)
                            if br.pill else None),
                'pill_hw': (tl.pill.h, tl.pill.w),
                'glyph_hw': (tl.glyph.h, tl.glyph.w) if tl.glyph else (0, 0),
            })

    if args.composite and all_samples:
        build_composite(all_samples, args.composite)
        print(f'\nComposite saved to {args.composite}')


if __name__ == '__main__':
    main()

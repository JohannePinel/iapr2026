#!/usr/bin/env python3
"""
main.py
=======
Entry point for the UNO card tiling pipeline.
Configure the parameters below, then run:  python main.py
"""

# ---- imports ----
from tiling import tile_image

# ---- default window and convolution params ----
TILE_SIZE = 75            # square convolution window size, in px
OVERLAP   = 1.0 / 3.0     # fractional overlap -> stride = round(TILE_SIZE * (1 - OVERLAP))

# ---- default threshold params ----
SAT_MAX             = 40    # HSV saturation below this -> candidate white-ish background
VAL_MIN             = 180   # HSV value above this      -> candidate white-ish background
WHITENESS_KEEP_FRAC = 0.10  # keep a tile only if its mean whiteness >= 10% of 255

# ---- sampling / output params ----
N_SAMPLES   = 20          # how many kept tiles to show in the sample montage
SEED        = 42          # RNG seed for reproducible sampling
SAVE_SOURCE = "original"  # "original" = save RGB crops, "masked" = save mask crops

# ---- path to image ----
IMAGE_PATH = "/mnt/user-data/uploads/L1000775.jpg"
OUTDIR     = "/home/claude/tiling_output"   # set to None to write next to the image

# ---- call the tiling function on image ----
if __name__ == "__main__":
    results = tile_image(
        image_path=IMAGE_PATH,
        tile_size=TILE_SIZE,
        overlap=OVERLAP,
        sat_max=SAT_MAX,
        val_min=VAL_MIN,
        whiteness_keep_frac=WHITENESS_KEEP_FRAC,
        n_samples=N_SAMPLES,
        seed=SEED,
        outdir=OUTDIR,
        save_source=SAVE_SOURCE,
    )

    print(f"\nDone. Kept {results['n_kept']} / {results['n_tiles']} tiles "
          f"({results['n_discarded']} discarded as background).")

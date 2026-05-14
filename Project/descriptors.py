#!/usr/bin/env python3
"""
descriptors.py
==============
Padded Fourier descriptors for tile contours.

`compute_descriptor_padding` is the provided reference implementation, used
verbatim. The helpers around it extract an ordered contour from a binary
mask crop and visualise the descriptor for each sampled tile.

NOTE on the reference method: it truncates / zero-pads the contour *point
list* to a fixed length and then FFTs. That is what "padded Fourier
descriptor" means here. (The more common variant FFTs the whole contour and
keeps the K lowest-frequency coefficients - worth knowing, but this module
follows the snippet as given.)
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Provided reference snippet (used as-is)
# --------------------------------------------------------------------------
def compute_descriptor_padding(contours, n_samples: int = 11):
    """
    Compute Fourier descriptors of input contours.

    Args
    ----
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element
        is an array of 2d coordinates (K, 2) where K depends on the number of
        elements that form the contour.
    n_samples: int
        Number of samples to consider. If the contour length is higher, discard
        the remaining part. If it is shorter, add padding. The first element of
        the descriptor represents the continuous (DC) component.

    Return
    ------
    descriptors: np.ndarray complex (N, n_samples)
        Computed complex Fourier descriptors for the given input contours.
    """
    N = len(contours)
    descriptors = np.zeros((N, n_samples), dtype=np.complex128)

    for i in range(N):
        contour = contours[i]
        len_contour = len(contour)

        if len_contour >= n_samples:
            contour = contour[:n_samples]
        else:
            padding = np.zeros((n_samples - len_contour, 2))
            contour = np.concatenate((contour, padding), axis=0)

        descriptors[i] = np.fft.fft(contour[:, 0] + 1j * contour[:, 1])

    return descriptors


# --------------------------------------------------------------------------
# Contour extraction
# --------------------------------------------------------------------------
def largest_tile_contour(mask_bin):
    """
    Largest external contour of a binary mask crop, as an ordered (K, 2) array.

    `mask_bin` is expected to be a binary (0/255) crop -- either a filled
    region mask or a binarised Sobel-edge tile. findContours returns ordered
    boundaries; the largest by area is returned. Returns an empty (0, 2) array
    when the crop contains no contour.
    """
    cnts, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2), dtype=np.int32)
    return max(cnts, key=cv2.contourArea).reshape(-1, 2)


# --------------------------------------------------------------------------
# Visualisation
# --------------------------------------------------------------------------
def plot_tile_descriptors(rgb, tiles, sample_idx, mask_edges,
                          n_samples=11, edge_thresh=40, out_path=None):
    """
    For every sampled tile: show the RGB crop with its contour, the Sobel-edge
    crop (what gets saved as the tile), and a stem plot of |Fourier descriptor|.

    The contour is extracted from the SOBEL-EDGE tile (binarised at
    `edge_thresh`), not the cleaned region mask -- so the descriptor describes
    the same "sobeled" tile that is saved to disk and that the keep test sees.

    Returns:
        descriptors: complex array (len(sample_idx), n_samples)
    """
    # one ordered contour per sampled tile, taken from the binarised Sobel tile
    contours = []
    for idx in sample_idx:
        x, y, w, h = tiles[idx]
        edge_tile = mask_edges[y:y + h, x:x + w]
        edge_bin = (edge_tile > edge_thresh).astype(np.uint8) * 255
        contours.append(largest_tile_contour(edge_bin))

    descriptors = compute_descriptor_padding(contours, n_samples=n_samples)

    k = max(1, len(sample_idx))
    fig, axes = plt.subplots(k, 3, figsize=(3 * 2.4, k * 1.7))
    axes = np.atleast_2d(axes)
    axes[0, 0].set_title("RGB tile + Sobel contour", fontsize=8)
    axes[0, 1].set_title("Sobel edges (saved tile)", fontsize=8)
    axes[0, 2].set_title(f"|Fourier descriptor|  (n_samples={n_samples})", fontsize=8)

    for row, idx in enumerate(sample_idx):
        x, y, w, h = tiles[idx]
        cnt = contours[row]

        # col 0: RGB crop with contour overlaid
        rgb_crop = np.ascontiguousarray(rgb[y:y + h, x:x + w].copy())
        if len(cnt):
            cv2.polylines(rgb_crop, [cnt.reshape(-1, 1, 2)], True, (255, 0, 255), 1)
        axes[row, 0].imshow(rgb_crop)
        axes[row, 0].set_xticks([]); axes[row, 0].set_yticks([])
        axes[row, 0].set_ylabel(f"#{idx}", fontsize=7)

        # col 1: Sobel-edge crop (the tile that gets saved to disk)
        axes[row, 1].imshow(mask_edges[y:y + h, x:x + w], cmap="gray",
                            vmin=0, vmax=255)
        axes[row, 1].axis("off")

        # col 2: descriptor magnitude. Index 0 (DC / continuous component)
        # is drawn in a different colour since it encodes position, not shape.
        mag = np.abs(descriptors[row])
        markerline, stemlines, baseline = axes[row, 2].stem(range(n_samples), mag)
        plt.setp(markerline, markersize=3)
        if n_samples > 0:
            axes[row, 2].plot(0, mag[0], "o", color="red", markersize=4)
        axes[row, 2].set_xticks(range(0, n_samples, max(1, n_samples // 5)))
        axes[row, 2].tick_params(labelsize=6)

    fig.suptitle(f"Padded Fourier descriptors - {len(sample_idx)} sample tiles",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    if out_path:
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return descriptors
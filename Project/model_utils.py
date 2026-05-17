#!/usr/bin/env python3
"""
model_utils.py
==============
Dataset generation, augmentation and model-training code for the UNO-card
symbol-candidate classifier.

This module is the downstream counterpart to the preprocessing pipeline
(``preprocess_utils`` / ``contour_utils`` / ``io_utils``): preprocessing turns
raw card photos into *symbol-candidate tiles*; the functions here turn those
candidates into a labelled, on-disk **dataset** ready for a classifier, and
later will host the augmentation and training code that consumes it.

Module layout
-------------
* Dataset persistence   : :func:`save_dataset_to`
* Dataset inspection    : :func:`show_dataset`
* Augmentation          : (to be appended)
* Model training        : (to be appended)

Conventions matched against the existing project
-------------------------------------------------
* A ``results`` object is the mapping returned by
  ``preprocess_utils.preprocess_image`` -- ``{source_image_name: candidates}``,
  where ``candidates`` is a list of tile dicts each carrying ``index`` and
  ``content`` (a 2D ``uint8`` tile).
* Saved-tile filenames follow ``io_utils.save_images_to`` and the template
  CSV: ``tile_<source>_<index:05d>.<ext>`` (e.g. ``tile_L1000983_00020.png``).
* The per-folder label CSV uses the exact header of the template CSV
  (``image_name,label``).

The module is import-light: only the standard library, NumPy and Pillow are
imported at module scope. Matplotlib is imported lazily inside
:func:`show_dataset` so that importing ``save_dataset_to`` into the
preprocessing notebook pulls in no plotting stack.
"""

from __future__ import annotations

import csv
import glob
import os
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


__all__ = [
    "save_dataset_to",
    "show_dataset",
]


# ===========================================================================
# Shared constants
# ===========================================================================
#: The three dataset subfolders, keyed by the 6-char source-image prefix that
#: routes into them. ``L1000770`` -> prefix ``L10007`` -> ``L10007xx``.
_PREFIX_TO_SUBFOLDER: Dict[str, str] = {
    "L10007": "L10007xx",
    "L10008": "L10008xx",
    "L10009": "L10009xx",
}
_SUBFOLDERS: List[str] = list(_PREFIX_TO_SUBFOLDER.values())

#: Fallback CSV header used when the template CSV cannot be located.
_DEFAULT_HEADER: List[str] = ["image_name", "label"]

#: Name of the column that holds the tile filename, matched case-insensitively
#: in the template header.
_IMAGE_COLUMN: str = "image_name"


# ===========================================================================
# Dataset persistence
# ===========================================================================
def _read_template_header(template_csv: str, base_dir: str) -> List[str]:
    """Return the column header of the template CSV.

    Looks for ``template_csv`` as given, then relative to ``base_dir``. Falls
    back to :data:`_DEFAULT_HEADER` (``image_name,label``) when no template
    file is found, so the function still works in a bare checkout.
    """
    candidates = [template_csv, os.path.join(base_dir, template_csv)]
    for path in candidates:
        if os.path.isfile(path):
            with open(path, newline="") as fh:
                header = next(csv.reader(fh), None)
            if header:
                return [col.strip() for col in header]
    print(f"  note: template CSV {template_csv!r} not found; "
          f"using default header {_DEFAULT_HEADER}")
    return list(_DEFAULT_HEADER)


def _image_column_index(header: Sequence[str]) -> int:
    """Index of the filename column in ``header`` (``image_name``, else 0)."""
    for i, col in enumerate(header):
        if col.strip().lower() == _IMAGE_COLUMN:
            return i
    return 0


def _candidate_filename(source_name: str, index: int, ext: str) -> str:
    """Build a tile filename matching the project convention.

    ``tile_<source>_<index:05d>.<ext>`` -- identical to the names produced by
    ``io_utils.save_images_to`` and listed in the template CSV.
    """
    return f"tile_{source_name}_{int(index):05d}.{ext}"


def _route_subfolder(source_name: str) -> Optional[str]:
    """Map a source image name to its dataset subfolder by 6-char prefix.

    Returns ``None`` when the prefix is not one of the three expected ranges,
    so the caller can skip (rather than crash on) an unexpected image.
    """
    return _PREFIX_TO_SUBFOLDER.get(source_name[:6])


def _to_uint8_image(content: Any) -> Image.Image:
    """Convert a tile's ``content`` array into a saveable PIL image.

    Tiles are 2D ``uint8`` edge/mask crops; this coerces dtype defensively
    and returns an ``L`` (grayscale) or ``RGB`` PIL image.
    """
    arr = np.asarray(content)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def _write_subfolder_csv(subfolder_path: str, header: Sequence[str],
                         ext: str) -> Dict[str, int]:
    """(Re)write a subfolder's label CSV from the image files on disk.

    The CSV is rebuilt from the actual tile files present in ``subfolder_path``
    (sorted for reproducibility). Any non-empty label already recorded in a
    pre-existing CSV is **preserved** -- so re-running :func:`save_dataset_to`
    after some manual labelling neither duplicates rows nor wipes labels.

    Returns a small summary ``{"rows": n, "labelled": n_with_label}``.
    """
    csv_path = os.path.join(subfolder_path, "labels.csv")
    img_col = _image_column_index(header)

    # Preserve labels already assigned in a previous run.
    existing: Dict[str, List[str]] = {}
    if os.path.isfile(csv_path):
        with open(csv_path, newline="") as fh:
            reader = csv.reader(fh)
            old_header = next(reader, None)
            old_img_col = (_image_column_index(old_header)
                           if old_header else img_col)
            for row in reader:
                if len(row) > old_img_col and row[old_img_col]:
                    existing[row[old_img_col]] = row

    # Enumerate the tile files actually on disk.
    image_files = sorted(
        os.path.basename(p)
        for p in glob.glob(os.path.join(subfolder_path, f"*.{ext}"))
    )

    rows: List[List[str]] = []
    labelled = 0
    for name in image_files:
        row = [""] * len(header)
        row[img_col] = name
        old = existing.get(name)
        if old:  # carry over every previously-filled column
            for i in range(len(header)):
                val = old[i] if i < len(old) else ""
                if i != img_col and val:
                    row[i] = val
        if any(v for i, v in enumerate(row) if i != img_col):
            labelled += 1
        rows.append(row)

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)          # default dialect -> CRLF, as template
        writer.writerow(list(header))
        writer.writerows(rows)

    return {"rows": len(rows), "labelled": labelled}


def save_dataset_to(results: Dict[str, List[Dict[str, Any]]],
                    image_names: Optional[Iterable[str]] = None,
                    base_dir: str = ".",
                    dataset_name: str = "classifier_candidates_dataset",
                    template_csv: str = "Template labels.csv",
                    ext: str = "png") -> str:
    """Persist preprocessing candidates to disk as a labelled-dataset skeleton.

    Creates ``<base_dir>/<dataset_name>/`` with three subfolders
    (``L10007xx``, ``L10008xx``, ``L10009xx``), routes every candidate tile to
    a subfolder by its source image's 6-char prefix, saves each tile as an
    image, and writes one label CSV per subfolder using the template schema.

    The operation is **idempotent**: tile filenames are deterministic, so a
    re-run overwrites rather than duplicates; each CSV is rebuilt from the
    files on disk while preserving any labels filled in since the last run.

    Args:
        results: Mapping ``{source_image_name: candidates}`` exactly as
            returned by ``preprocess_utils.preprocess_image``. Each candidate
            is a dict carrying at least ``index`` and ``content``.
        image_names: Optional explicit set of source names to export. When
            ``None`` (default) every key of ``results`` is exported. Provided
            for parity with the task spec; with the dict-shaped ``results`` of
            this project it is normally left unset.
        base_dir: Directory the dataset folder is created in. Relative to the
            current working directory; no absolute paths are used.
        dataset_name: Name of the root dataset folder.
        template_csv: Path to the template CSV whose header is copied into
            every subfolder CSV. Falls back to ``image_name,label`` if absent.
        ext: Image file extension for saved tiles (``png`` recommended).

    Returns:
        The path to the created dataset root folder.
    """
    names = list(image_names) if image_names is not None else list(results.keys())

    dataset_root = os.path.join(base_dir, dataset_name)
    os.makedirs(dataset_root, exist_ok=True)
    for sub in _SUBFOLDERS:
        os.makedirs(os.path.join(dataset_root, sub), exist_ok=True)

    header = _read_template_header(template_csv, base_dir)

    saved_counts = {sub: 0 for sub in _SUBFOLDERS}
    skipped: List[str] = []

    for source_name in names:
        candidates = results.get(source_name)
        if candidates is None:
            print(f"  note: {source_name!r} not in results; skipped")
            continue

        subfolder = _route_subfolder(source_name)
        if subfolder is None:
            skipped.append(source_name)
            continue

        subfolder_path = os.path.join(dataset_root, subfolder)
        for cand in candidates:
            fname = _candidate_filename(source_name, cand["index"], ext)
            _to_uint8_image(cand["content"]).save(
                os.path.join(subfolder_path, fname))
            saved_counts[subfolder] += 1

    # ---- per-subfolder label CSVs ----
    csv_summary = {}
    for sub in _SUBFOLDERS:
        csv_summary[sub] = _write_subfolder_csv(
            os.path.join(dataset_root, sub), header, ext)

    # ---- console summary ----
    print("save_dataset_to")
    print("-" * 50)
    print(f"  dataset root  : {dataset_root}")
    print(f"  csv schema    : {','.join(header)}")
    for sub in _SUBFOLDERS:
        print(f"  {sub:<10s}: saved this run {saved_counts[sub]:>4d}  |  "
              f"csv rows {csv_summary[sub]['rows']:>4d}  "
              f"(labelled {csv_summary[sub]['labelled']})")
    total = sum(c["rows"] for c in csv_summary.values())
    print(f"  total on disk : {total} candidate tile(s)")
    if skipped:
        print(f"  WARNING       : {len(skipped)} image(s) had an unexpected "
              f"prefix and were skipped: {sorted(set(skipped))}")
    print("-" * 50)

    return dataset_root


# ===========================================================================
# Dataset inspection
# ===========================================================================
def _read_labels(subfolder_path: str) -> Dict[str, str]:
    """Return ``{image_name: label}`` for a subfolder, ``""`` when unlabelled."""
    csv_path = os.path.join(subfolder_path, "labels.csv")
    labels: Dict[str, str] = {}
    if not os.path.isfile(csv_path):
        return labels
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return labels
        img_col = _image_column_index(header)
        # the label column is the first non-image column, else the last column
        label_col = next((i for i in range(len(header)) if i != img_col),
                          len(header) - 1)
        for row in reader:
            if len(row) > img_col and row[img_col]:
                label = row[label_col] if len(row) > label_col else ""
                labels[row[img_col]] = label.strip()
    return labels


def show_dataset(n: int = 20, show_labels: bool = True,
                 dataset_name: str = "classifier_candidates_dataset",
                 base_dir: str = ".", ext: str = "png",
                 seed: Optional[int] = None) -> None:
    """Visually inspect the candidates dataset and print its statistics.

    Randomly samples candidates with an equal count drawn from each of the
    three subfolders (``n`` split three ways, remainder distributed to the
    first folders), shows them in a single matplotlib grid, and prints
    dataset-wide statistics.

    Args:
        n: Total number of candidates to display.
        show_labels: When True, each tile's label (read from its subfolder
            CSV) is shown beneath it; ``na`` is shown for an empty label.
        dataset_name: Name of the dataset root folder.
        base_dir: Directory containing the dataset folder.
        ext: Image file extension of the saved tiles.
        seed: Optional RNG seed for a reproducible sample.
    """
    import matplotlib.pyplot as plt          # lazy: keeps the module light

    dataset_root = os.path.join(base_dir, dataset_name)
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"dataset not found: {dataset_root}")

    rng = random.Random(seed)

    # ---- per-subfolder file lists + labels ----
    per_folder_files: Dict[str, List[str]] = {}
    per_folder_labels: Dict[str, Dict[str, str]] = {}
    for sub in _SUBFOLDERS:
        sub_path = os.path.join(dataset_root, sub)
        files = sorted(
            os.path.basename(p)
            for p in glob.glob(os.path.join(sub_path, f"*.{ext}"))
        )
        per_folder_files[sub] = files
        per_folder_labels[sub] = _read_labels(sub_path) if show_labels else {}

    # ---- equal split of n across the 3 subfolders ----
    base, rem = divmod(max(0, n), len(_SUBFOLDERS))
    quota = {sub: base + (1 if i < rem else 0)
             for i, sub in enumerate(_SUBFOLDERS)}

    sampled: List[Dict[str, str]] = []   # {path, subfolder, name, label}
    for sub in _SUBFOLDERS:
        files = per_folder_files[sub]
        k = min(quota[sub], len(files))
        for name in rng.sample(files, k):
            sampled.append({
                "path": os.path.join(dataset_root, sub, name),
                "subfolder": sub,
                "name": name,
                "label": per_folder_labels[sub].get(name, "") or "na",
            })
    rng.shuffle(sampled)

    # ---- statistics over the WHOLE dataset ----
    total = sum(len(f) for f in per_folder_files.values())
    class_dist: Dict[str, int] = {}
    for sub in _SUBFOLDERS:
        labels = _read_labels(os.path.join(dataset_root, sub))
        for name in per_folder_files[sub]:
            label = labels.get(name, "").strip() or "na"
            class_dist[label] = class_dist.get(label, 0) + 1

    print("show_dataset")
    print("-" * 50)
    print(f"  dataset root      : {dataset_root}")
    print(f"  total candidates  : {total}")
    for sub in _SUBFOLDERS:
        print(f"    {sub:<10s}: {len(per_folder_files[sub])}")
    print("  class distribution:")
    for label in sorted(class_dist):
        print(f"    {label:<12s}: {class_dist[label]}")
    print(f"  showing           : {len(sampled)} sampled tile(s)")
    print("-" * 50)

    # ---- matplotlib grid panel ----
    if not sampled:
        print("  (nothing to display -- dataset is empty)")
        return

    n_show = len(sampled)
    n_cols = min(5, n_show)
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.4 * n_cols, 2.7 * n_rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, item in zip(axes, sampled):
        ax.imshow(np.asarray(Image.open(item["path"])),
                  cmap="gray", vmin=0, vmax=255)
        title = item["subfolder"]
        if show_labels:
            title += f"\n{item['label']}"
        ax.set_title(title, fontsize=8)
        ax.axis("off")
    for ax in axes[n_show:]:
        ax.axis("off")

    fig.suptitle(f"{dataset_name} -- {n_show} sampled candidate(s)",
                 fontsize=11)
    fig.tight_layout()
    plt.show()


# ===========================================================================
# Augmentation        (to be appended)
# ===========================================================================


# ===========================================================================
# Model training      (to be appended)
# ===========================================================================

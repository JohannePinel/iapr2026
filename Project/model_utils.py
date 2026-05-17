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
import shutil
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from PIL import Image


__all__ = [
    "save_dataset_to",
    "show_dataset",
    "prepare_training_dataset",
    "data_augmentation",
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
# Training dataset preparation
# ===========================================================================
def _index_image_files(data_path: str) -> Dict[str, str]:
    """Build a ``{filename: full_path}`` index of every image under ``data_path``.

    Searches ``data_path`` recursively, so it works whether the images sit
    directly in the folder or are split across subfolders (e.g. the
    ``L10007xx`` / ``L10008xx`` / ``L10009xx`` layout of
    ``classifier_candidates_dataset``). On a duplicate filename the first hit
    (sorted order) wins.
    """
    index: Dict[str, str] = {}
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for path in sorted(glob.glob(os.path.join(data_path, "**", "*"),
                                 recursive=True)):
        if os.path.isfile(path) and os.path.splitext(path)[1].lower() in exts:
            index.setdefault(os.path.basename(path), path)
    return index


def _read_label_csv(csv_path: str) -> List[Dict[str, str]]:
    """Read an ``image_name,label`` CSV into a list of ``{name, label}`` dicts."""
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            return []
        img_col = _image_column_index(header)
        label_col = next((i for i in range(len(header)) if i != img_col),
                         len(header) - 1)
        rows: List[Dict[str, str]] = []
        for row in reader:
            if len(row) > img_col and row[img_col]:
                label = row[label_col] if len(row) > label_col else ""
                rows.append({"name": row[img_col], "label": label.strip()})
    return rows


def _resolve_target_counts(by_label: Dict[str, List[str]],
                           class_distribution: Optional[Dict[str, int]]
                           ) -> Dict[str, int]:
    """Decide how many images to draw per class.

    * ``class_distribution is None`` -> **even**: every class contributes the
      same number of images, equal to the size of the smallest class (a fully
      balanced set, no oversampling).
    * ``class_distribution`` given -> a ``{label: count}`` request; only the
      listed labels are included, and any count above what is available is
      capped (with a warning).
    """
    if class_distribution is None:
        per_class = min(len(v) for v in by_label.values()) if by_label else 0
        return {label: per_class for label in by_label}

    targets: Dict[str, int] = {}
    for label, requested in class_distribution.items():
        available = len(by_label.get(label, []))
        if available == 0:
            print(f"  WARNING: requested class {label!r} has no images; skipped")
            continue
        if requested > available:
            print(f"  WARNING: class {label!r} requested {requested} "
                  f"but only {available} available; capped at {available}")
        targets[label] = min(int(requested), available)
    return targets


def prepare_training_dataset(csv_path: str,
                             data_path: str,
                             output_path: str,
                             class_distribution: Optional[Dict[str, int]] = None,
                             show_data: bool = True,
                             exclude_classes: Sequence[str] = ("discard",),
                             seed: Optional[int] = None) -> str:
    """Assemble a labelled ``training_set`` with a chosen class distribution.

    Reads a labelled CSV, locates each image under ``data_path``, samples them
    according to ``class_distribution`` (even by default), and copies the
    selection into ``<output_path>/training_set/<label>/`` -- the per-class
    folder layout consumed by Keras ``image_dataset_from_directory`` / PyTorch
    ``ImageFolder``. A ``training_labels.csv`` manifest is written alongside.

    The ``discard`` class (and any other name in ``exclude_classes``), as well
    as rows with an empty label, are ignored -- they never enter the training
    set.

    ``training_set`` is rebuilt from scratch on every call (it is a derived,
    regenerable artifact), so the operation is idempotent and a re-run with a
    different distribution leaves no stale files behind.

    Args:
        csv_path: CSV with ``image_name,label`` columns.
        data_path: Directory searched (recursively) for the image files named
            in the CSV.
        output_path: Directory in which the ``training_set`` subfolder is
            created.
        class_distribution: ``{label: count}`` mapping for a custom mix, or
            ``None`` (default) for an even split -- every class contributes as
            many images as the smallest class has.
        show_data: When True, display 20 sampled training images in a grid,
            each captioned with its label.
        exclude_classes: Class names to drop entirely (default ``("discard",)``).
        seed: Optional RNG seed for a reproducible sample.

    Returns:
        The path to the created ``training_set`` directory.
    """
    rng = random.Random(seed)
    excluded = {c.strip().lower() for c in exclude_classes}

    # ---- read labels, group by class (dropping excluded / unlabelled) ----
    rows = _read_label_csv(csv_path)
    by_label: Dict[str, List[str]] = {}
    n_excluded = n_unlabelled = 0
    for row in rows:
        label = row["label"]
        if not label:
            n_unlabelled += 1
            continue
        if label.lower() in excluded:
            n_excluded += 1
            continue
        by_label.setdefault(label, []).append(row["name"])

    if not by_label:
        raise ValueError("no usable labelled rows after excluding "
                          f"{sorted(excluded)} and unlabelled images")

    # ---- locate the image files on disk ----
    file_index = _index_image_files(data_path)

    # ---- decide per-class counts, then sample ----
    targets = _resolve_target_counts(by_label, class_distribution)

    selection: List[Dict[str, str]] = []   # {name, label, src}
    missing: List[str] = []
    for label, count in targets.items():
        names = [n for n in by_label.get(label, []) if n in file_index]
        missing += [n for n in by_label.get(label, []) if n not in file_index]
        k = min(count, len(names))
        for name in rng.sample(names, k):
            selection.append({"name": name, "label": label,
                              "src": file_index[name]})

    # ---- (re)build the training_set folder tree ----
    training_root = os.path.join(output_path, "training_set")
    if os.path.isdir(training_root):
        shutil.rmtree(training_root)
    os.makedirs(training_root, exist_ok=True)

    manifest: List[List[str]] = []
    for item in selection:
        class_dir = os.path.join(training_root, item["label"])
        os.makedirs(class_dir, exist_ok=True)
        dst = os.path.join(class_dir, item["name"])
        shutil.copy2(item["src"], dst)
        manifest.append([os.path.join(item["label"], item["name"]),
                         item["label"]])

    manifest.sort()
    with open(os.path.join(training_root, "training_labels.csv"),
              "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_name", "label"])
        writer.writerows(manifest)

    # ---- console summary ----
    final_dist: Dict[str, int] = {}
    for item in selection:
        final_dist[item["label"]] = final_dist.get(item["label"], 0) + 1

    mode = "even" if class_distribution is None else "custom"
    print("prepare_training_dataset")
    print("-" * 50)
    print(f"  source csv     : {csv_path}")
    print(f"  training root  : {training_root}")
    print(f"  distribution   : {mode}")
    print(f"  excluded class : {sorted(excluded)}  "
          f"({n_excluded} image(s) dropped)")
    if n_unlabelled:
        print(f"  unlabelled     : {n_unlabelled} image(s) dropped")
    print("  class counts   :")
    for label in sorted(final_dist):
        print(f"    {label:<16s}: {final_dist[label]}")
    print(f"  total selected : {len(selection)} image(s) "
          f"across {len(final_dist)} class(es)")
    if missing:
        print(f"  WARNING        : {len(missing)} CSV image(s) not found "
              f"under {data_path!r} and skipped")
    print("-" * 50)

    # ---- optional 20-image preview grid ----
    if show_data and selection:
        import matplotlib.pyplot as plt        # lazy: keeps the module light

        sample = rng.sample(selection, min(20, len(selection)))
        n_cols = min(5, len(sample))
        n_rows = (len(sample) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(2.4 * n_cols, 2.7 * n_rows))
        axes = np.atleast_1d(axes).ravel()
        for ax, item in zip(axes, sample):
            ax.imshow(np.asarray(Image.open(item["src"])),
                      cmap="gray", vmin=0, vmax=255)
            ax.set_title(item["label"], fontsize=8)
            ax.axis("off")
        for ax in axes[len(sample):]:
            ax.axis("off")
        fig.suptitle(f"training_set -- {len(sample)} sampled image(s)",
                     fontsize=11)
        fig.tight_layout()
        plt.show()

    return training_root


# ===========================================================================
# Data augmentation
# ===========================================================================
# Each augmentation step is a small pure function with the uniform signature
#   step(img, rng, np_rng, demo) -> PIL.Image
# `rng`/`np_rng` are the seeded RNGs; `demo=True` forces a visible, deterministic
# variant for the `show_result` preview (instead of rolling the random branch).
# `data_augmentation` chains the enabled steps to top each class up to `count`.

def _fill_color(img: Image.Image):
    """Background fill (black) matching the image's mode (scalar or tuple)."""
    if img.mode in ("L", "I", "F"):
        return 0
    return tuple([0] * len(img.getbands()))


def _scale_contour(img: Image.Image, factor: float) -> Image.Image:
    """Scale the tile's content by `factor`, keeping the canvas size fixed.

    ``factor > 1`` zooms in (the enlarged image is centre-cropped back to the
    original size); ``factor < 1`` zooms out (the shrunk image is centred on a
    black canvas). The contour stays centred either way.
    """
    w, h = img.size
    nw, nh = max(1, round(w * factor)), max(1, round(h * factor))
    resized = img.resize((nw, nh), Image.BILINEAR)
    if factor >= 1.0:
        left, top = (nw - w) // 2, (nh - h) // 2
        return resized.crop((left, top, left + w, top + h))
    canvas = Image.new(img.mode, (w, h), _fill_color(img))
    canvas.paste(resized, ((w - nw) // 2, (h - nh) // 2))
    return canvas


def _translate_contour(img: Image.Image, dx: float, dy: float) -> Image.Image:
    """Shift the tile content by ``(dx, dy)`` px; vacated area filled black."""
    return img.transform(img.size, Image.AFFINE, (1, 0, -dx, 0, 1, -dy),
                         resample=Image.BILINEAR, fillcolor=_fill_color(img))


def _jitter_and_noise(img: Image.Image, np_rng) -> Image.Image:
    """Apply mild brightness/contrast jitter plus speckle + salt-and-pepper noise."""
    arr = np.asarray(img).astype(np.float32)
    arr *= np_rng.uniform(0.85, 1.15)                        # brightness
    mean = arr.mean()
    arr = (arr - mean) * np_rng.uniform(0.85, 1.15) + mean   # contrast
    arr += np_rng.normal(0.0, 6.0, arr.shape)                # gaussian speckle
    sp = np_rng.random(arr.shape[:2])                        # salt & pepper
    arr[sp < 0.005] = 0.0
    arr[sp > 0.995] = 255.0
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode=img.mode)


# ---- individual steps -----------------------------------------------------
def _step_rotate180(img, rng, np_rng, demo):
    """Rotate 180deg with probability 1/2 (demo: always rotate)."""
    if demo or rng.random() < 0.5:
        return img.rotate(180, resample=Image.BILINEAR, fillcolor=_fill_color(img))
    return img


def _step_rotate90(img, rng, np_rng, demo):
    """Rotate +/-90deg: 1/3 clockwise, 1/3 counter-clockwise, 1/3 unchanged."""
    r = 0.0 if demo else rng.random()                        # demo -> clockwise
    if r < 1.0 / 3.0:
        return img.rotate(-90, resample=Image.BILINEAR, fillcolor=_fill_color(img))
    if r < 2.0 / 3.0:
        return img.rotate(90, resample=Image.BILINEAR, fillcolor=_fill_color(img))
    return img


def _step_rotate10(img, rng, np_rng, demo):
    """Rotate +/-10deg: 1/3 clockwise, 1/3 counter-clockwise, 1/3 unchanged."""
    r = 0.0 if demo else rng.random()                        # demo -> clockwise
    if r < 1.0 / 3.0:
        return img.rotate(-10, resample=Image.BILINEAR, fillcolor=_fill_color(img))
    if r < 2.0 / 3.0:
        return img.rotate(10, resample=Image.BILINEAR, fillcolor=_fill_color(img))
    return img


def _step_scale(img, rng, np_rng, demo):
    """Scale the contour by a uniform factor in ``[0.9, 1.1]`` (demo: 1.1)."""
    factor = 1.1 if demo else rng.uniform(0.9, 1.1)
    return _scale_contour(img, factor)


def _step_translate(img, rng, np_rng, demo):
    """Translate by a 0-5 px magnitude on each axis, random sign (demo: +5,+5)."""
    if demo:
        dx, dy = 5.0, 5.0
    else:
        dx = rng.choice((-1.0, 1.0)) * rng.uniform(0.0, 5.0)
        dy = rng.choice((-1.0, 1.0)) * rng.uniform(0.0, 5.0)
    return _translate_contour(img, dx, dy)


def _step_jitter(img, rng, np_rng, demo):
    """Apply brightness/contrast jitter plus speckle and salt-and-pepper noise."""
    return _jitter_and_noise(img, np_rng)


#: Ordered augmentation pipeline -- (toggle keyword, human label, step function).
_AUG_STEPS = [
    ("rotate_180deg", "rotate 180deg",   _step_rotate180),
    ("rotate_90deg",  "rotate 90deg",    _step_rotate90),
    ("rotate_10deg",  "rotate 10deg",    _step_rotate10),
    ("scale",         "scale 0.9-1.1",   _step_scale),
    ("translate",     "translate 0-5px", _step_translate),
    ("jitter_noise",  "jitter + noise",  _step_jitter),
]


def _list_class_images(class_dir: str, ext: str):
    """Return ``(originals, augmented)`` filename lists for one class folder.

    Augmented files are recognised by the ``aug_`` filename prefix, so a
    re-run can discard and regenerate them without ever consuming a previously
    augmented image as if it were an original.
    """
    files = sorted(os.path.basename(p)
                   for p in glob.glob(os.path.join(class_dir, f"*.{ext}")))
    originals = [f for f in files if not f.startswith("aug_")]
    augmented = [f for f in files if f.startswith("aug_")]
    return originals, augmented


def _resolve_augmented_paths(training_dir: str, output_dir: Optional[str]):
    """Decide where the augmented copy of the training set should live.

    The original ``training_dir`` is never written to -- augmentation always
    runs on a fresh copy. Returns ``(copy_src, augmented_root, aug_class_root)``:

    * ``copy_src``       -- the directory copied verbatim into the destination.
    * ``augmented_root`` -- the ``augmented_train_set`` folder itself.
    * ``aug_class_root`` -- the per-class root inside it (where augmentation runs
      and the manifest is written).

    Default layout (``output_dir is None``): the *parent* of ``training_dir``
    -- i.e. the ``train_set`` folder produced by :func:`prepare_training_dataset`
    -- is copied wholesale into an ``augmented_train_set`` folder created **next
    to that ``train_set`` folder**. So::

        <root>/train_set/training_set/<class>/      (original, untouched)
        <root>/augmented_train_set/training_set/<class>/   (copy + augmented)

    When ``training_dir`` has no parent (e.g. the bare default ``"training_set"``)
    it is mirrored directly into a sibling ``augmented_train_set``. An explicit
    ``output_dir`` overrides the ``augmented_root`` location entirely.
    """
    training_dir = os.path.normpath(training_dir)
    train_set_folder = os.path.dirname(training_dir)      # the 'train_set' folder

    if train_set_folder and train_set_folder not in (".", os.sep):
        copy_src = train_set_folder
        default_root = os.path.join(os.path.dirname(train_set_folder) or ".",
                                    "augmented_train_set")
        augmented_root = os.path.normpath(output_dir) if output_dir else default_root
        aug_class_root = os.path.join(augmented_root,
                                      os.path.basename(training_dir))
    else:                                                 # no usable parent
        copy_src = training_dir
        default_root = os.path.join(".", "augmented_train_set")
        augmented_root = os.path.normpath(output_dir) if output_dir else default_root
        aug_class_root = augmented_root

    src_abs, root_abs = os.path.abspath(copy_src), os.path.abspath(augmented_root)
    if root_abs == src_abs or root_abs == os.path.abspath(training_dir):
        raise ValueError("augmented destination must differ from the source; "
                          "pass a distinct output_dir")
    if root_abs.startswith(src_abs + os.sep):
        raise ValueError("augmented destination must not sit inside the source "
                          "training set; pass a distinct output_dir")
    return copy_src, augmented_root, aug_class_root


def data_augmentation(training_dir: str = "training_set",
                      count: int = 200,
                      rotate_180deg: bool = True,
                      rotate_90deg: bool = True,
                      rotate_10deg: bool = True,
                      scale: bool = True,
                      translate: bool = True,
                      jitter_noise: bool = True,
                      show_result: bool = True,
                      seed: Optional[int] = None,
                      ext: str = "png",
                      output_dir: Optional[str] = None) -> str:
    """Augment a copy of a ``training_set`` up to a minimum image count.

    The original training set is **never modified**. Its parent ``train_set``
    folder is copied wholesale into a new ``augmented_train_set`` folder created
    next to it, and augmentation runs on that copy::

        <root>/train_set/training_set/<class>/            (original, untouched)
        <root>/augmented_train_set/training_set/<class>/   (copy + augmented)

    For every class subfolder of the copy, augmented images are generated by
    chaining the enabled transform steps until the class holds at least
    ``count`` images (copied originals + augmented combined). Augmented files
    carry an ``aug_`` prefix. The whole ``augmented_train_set`` folder is rebuilt
    from scratch on each call, so the operation is idempotent.

    Transform steps (each gated by its own keyword, all default ``True``):

    * ``rotate_180deg`` -- rotate 180deg with probability 1/2.
    * ``rotate_90deg``  -- 1/3 clockwise, 1/3 counter-clockwise, 1/3 unchanged.
    * ``rotate_10deg``  -- 1/3 clockwise, 1/3 counter-clockwise, 1/3 unchanged.
    * ``scale``         -- scale the contour by a uniform factor in [0.9, 1.1].
    * ``translate``     -- shift 0-5 px (random sign) on each axis.
    * ``jitter_noise``  -- brightness/contrast jitter + speckle / salt-pepper.

    Args:
        training_dir: Path to the original ``training_set`` folder (per-class
            subfolders), as produced by :func:`prepare_training_dataset`. Read
            only -- it is copied, not modified.
        count: Minimum number of images per class after augmentation.
        rotate_180deg, rotate_90deg, rotate_10deg, scale, translate,
        jitter_noise: Per-step on/off toggles.
        show_result: When True, display a preview with 2 before/after example
            pairs for each enabled step.
        seed: Optional RNG seed for reproducible augmentation.
        ext: Image file extension of the tiles.
        output_dir: Optional explicit path for the ``augmented_train_set``
            folder. When ``None`` (default) it is created next to the source
            ``train_set`` folder.

    Returns:
        The path to the augmented per-class root (the copied + augmented
        ``training_set`` inside ``augmented_train_set``).

    Note:
        ``rotate_180deg`` and ``rotate_90deg`` are label-changing for the digit
        classes (a quarter-turned digit is invalid). Disable them for
        digit-bearing runs, or augment digit and symbol classes separately.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if not os.path.isdir(training_dir):
        raise FileNotFoundError(f"training set not found: {training_dir}")

    # ---- copy the original training set into augmented_train_set ----
    copy_src, augmented_root, aug_class_root = _resolve_augmented_paths(
        training_dir, output_dir)
    if os.path.exists(augmented_root):
        shutil.rmtree(augmented_root)            # fresh copy -> idempotent
    shutil.copytree(copy_src, augmented_root)
    if not os.path.isdir(aug_class_root):
        raise FileNotFoundError(
            f"expected class folders at {aug_class_root} after copy")

    toggles = {
        "rotate_180deg": rotate_180deg, "rotate_90deg": rotate_90deg,
        "rotate_10deg": rotate_10deg, "scale": scale,
        "translate": translate, "jitter_noise": jitter_noise,
    }
    enabled = [(key, label, fn) for key, label, fn in _AUG_STEPS if toggles[key]]

    class_dirs = sorted(
        d for d in os.listdir(aug_class_root)
        if os.path.isdir(os.path.join(aug_class_root, d))
    )

    summary: List[Dict[str, Any]] = []
    original_pool: List[str] = []          # paths used for the preview panel

    for cls in class_dirs:
        class_path = os.path.join(aug_class_root, cls)
        originals, old_augmented = _list_class_images(class_path, ext)

        # drop any stale augmented files carried over in the copy
        for f in old_augmented:
            os.remove(os.path.join(class_path, f))

        original_pool += [os.path.join(class_path, f) for f in originals]

        n_orig = len(originals)
        need = max(0, count - n_orig)
        generated = 0

        if n_orig == 0:
            print(f"  WARNING: class {cls!r} has no original images; skipped")
        elif need == 0:
            pass                            # already at/above target
        elif not enabled:
            print(f"  WARNING: all transforms disabled; class {cls!r} not "
                  f"augmented ({n_orig}/{count})")
        else:
            for i in range(need):
                src = rng.choice(originals)
                img = Image.open(os.path.join(class_path, src))
                for _key, _label, fn in enabled:
                    img = fn(img, rng, np_rng, demo=False)
                out_name = f"aug_{cls}_{i:05d}.{ext}"
                img.save(os.path.join(class_path, out_name))
                generated += 1

        summary.append({"class": cls, "before": n_orig,
                         "after": n_orig + generated, "generated": generated})

    # ---- rebuild the manifest CSV (inside the augmented copy) ----
    manifest: List[List[str]] = []
    for cls in class_dirs:
        class_path = os.path.join(aug_class_root, cls)
        for f in sorted(os.path.basename(p)
                        for p in glob.glob(os.path.join(class_path, f"*.{ext}"))):
            manifest.append([f"{cls}/{f}", cls])
    with open(os.path.join(aug_class_root, "training_labels.csv"),
              "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image_name", "label"])
        writer.writerows(sorted(manifest))

    # ---- console summary ----
    print("data_augmentation")
    print("-" * 50)
    print(f"  source set     : {training_dir}  (untouched)")
    print(f"  augmented set  : {aug_class_root}")
    print(f"  target count   : {count} image(s) per class")
    print(f"  enabled steps  : {[k for k, _, _ in enabled] or 'none'}")
    print("  class          before  ->  after   (+generated)")
    for s in summary:
        print(f"    {s['class']:<14s}{s['before']:>5d}   -> {s['after']:>5d}   "
              f"(+{s['generated']})")
    total_gen = sum(s["generated"] for s in summary)
    total_after = sum(s["after"] for s in summary)
    print(f"  generated      : {total_gen} augmented image(s)")
    print(f"  total on disk  : {total_after} image(s) across "
          f"{len(class_dirs)} class(es)")
    print("-" * 50)

    # ---- per-step before/after preview ----
    if show_result and enabled and original_pool:
        import matplotlib.pyplot as plt        # lazy: keeps the module light

        n_steps = len(enabled)
        fig, axes = plt.subplots(n_steps, 4,
                                 figsize=(4 * 2.4, n_steps * 2.7))
        axes = np.atleast_2d(axes)
        for row, (_key, label, fn) in enumerate(enabled):
            for ex in range(2):
                src = rng.choice(original_pool)
                before = Image.open(src)
                after = fn(before.copy(), rng, np_rng, demo=True)
                axes[row, 2 * ex].imshow(np.asarray(before),
                                         cmap="gray", vmin=0, vmax=255)
                axes[row, 2 * ex].set_title(f"{label}\nbefore", fontsize=8)
                axes[row, 2 * ex + 1].imshow(np.asarray(after),
                                             cmap="gray", vmin=0, vmax=255)
                axes[row, 2 * ex + 1].set_title(f"{label}\nafter", fontsize=8)
            for col in range(4):
                axes[row, col].axis("off")
        fig.suptitle("data_augmentation -- per-step preview "
                     "(2 before/after examples each)", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        plt.show()

    return aug_class_root


# ===========================================================================
# Model training      (to be appended)
# ===========================================================================
#!/usr/bin/env python3
# Collect the PanoCaps images from their source datasets into data/PanoCaps/images/.
# Sources: ADE20K (Zhou et al., CVPR 2017), COCONut (Deng et al., CVPR 2024) and
#          VIPSeg (Miao et al., CVPR 2022); set DATASET_PATHS to your local copies.
# VIPSeg: point DATASET_PATHS at the original frames (VIPSeg/imgs); the PanoCaps frames are
# resized to 720p here, exactly as VIPSeg's change2_720p.py does, so that script is not needed.
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

from PIL import Image


DATASET_PATHS = {
    "ADE20K": {
        "train": Path("ADEChallengeData2016/images/training"),
        "val": Path("ADEChallengeData2016/images/validation"),
    },
    "COCONut": {
        "train": Path("train2017"),
        "val": Path("val2017"),
    },
    "VIPSeg": {
        "train": Path("VIPSeg/imgs"),
        "val": Path("VIPSeg/imgs"),
    },
}


def ensure_output_dirs(output_dir: Path) -> Tuple[Path, Path, Path]:
    images_dir = output_dir / "images"
    train_dir = images_dir / "train"
    test_val_dir = images_dir / "test_val"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_val_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, train_dir, test_val_dir


def iter_mask_json_files(annotation_dir: Path) -> Iterable[Path]:
    return annotation_dir.rglob("*_mask.json")


def split_for_annotation_file(mask_json_path: Path) -> str:
    name = mask_json_path.name.lower()
    if name == "train_mask.json":
        return "train"
    if name in {"test_mask.json", "val_mask.json"}:
        return "test_val"
    return "train"


def source_split_for_annotation_file(mask_json_path: Path) -> str:
    name = mask_json_path.name.lower()
    if name == "train_mask.json":
        return "train"
    if name in {"test_mask.json", "val_mask.json"}:
        return "val"
    return "train"


def load_images_list(mask_json_path: Path) -> list[Dict[str, Any]]:
    with mask_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    images = data.get("images")
    if not isinstance(images, list):
        raise ValueError(f"{mask_json_path} missing 'images' list.")
    return images


def vipseg_relpath_from_filename(file_name: str) -> Path:
    p = Path(file_name)
    stem = p.stem
    ext = p.suffix
    if "_" not in stem:
        raise ValueError(f"Bad VIPSeg file_name: {file_name}")
    video_id, frame = stem.rsplit("_", 1)
    if not video_id or not frame:
        raise ValueError(f"Bad VIPSeg file_name: {file_name}")
    return Path(video_id) / f"{frame}{ext}"


def get_source_image_path(dataset: str, file_name: str, source_split: str) -> Path:
    if dataset not in DATASET_PATHS:
        raise KeyError(f"Unknown dataset '{dataset}'")
    if source_split not in DATASET_PATHS[dataset]:
        raise KeyError(f"Dataset '{dataset}' missing split '{source_split}'")

    root = DATASET_PATHS[dataset][source_split]

    if dataset == "VIPSeg":
        return root / vipseg_relpath_from_filename(file_name)

    return root / Path(file_name).name


def safe_copy(src: Path, dst: Path, overwrite: bool) -> bool:
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def safe_resize_720p(src: Path, dst: Path, overwrite: bool) -> bool:
    # Same resize and save as VIPSeg's change2_720p.py.
    if dst.exists() and not overwrite:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    img = Image.open(src)
    w, h = img.size
    img = img.resize((int(720 * w / h), 720), Image.BILINEAR)
    img.save(dst)
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--annotation-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _, train_dir, test_val_dir = ensure_output_dirs(args.output_dir)

    json_files = list(iter_mask_json_files(args.annotation_dir))
    if not json_files:
        print(f"No *_mask.json files found in {args.annotation_dir}")
        return 0

    copied = skipped = missing = invalid = path_errors = 0

    for mask_json in json_files:
        dest_split = split_for_annotation_file(mask_json)
        source_split = source_split_for_annotation_file(mask_json)
        dest_dir = train_dir if dest_split == "train" else test_val_dir

        try:
            images_list = load_images_list(mask_json)
        except Exception as e:
            print(f"[ERROR] {mask_json}: {e}")
            invalid += 1
            continue

        for item in images_list:
            if not isinstance(item, dict):
                invalid += 1
                continue

            file_name = item.get("file_name")
            dataset = item.get("data_source")
            if not isinstance(file_name, str) or not file_name.strip():
                invalid += 1
                continue
            if not isinstance(dataset, str) or not dataset.strip():
                invalid += 1
                continue

            file_name = file_name.strip()
            dataset = dataset.strip()

            try:
                src = get_source_image_path(dataset, file_name, source_split)
            except (KeyError, ValueError) as e:
                print(f"[PATH] {mask_json.name}: {e}")
                path_errors += 1
                continue

            dst = dest_dir / Path(file_name).name

            if args.dry_run:
                print(f"{src} -> {dst}")
                continue

            if not src.exists():
                print(f"[MISSING] {src} ({mask_json.name}, dataset={dataset})")
                missing += 1
                continue

            write = safe_resize_720p if dataset == "VIPSeg" else safe_copy
            if write(src, dst, overwrite=args.overwrite):
                copied += 1
            else:
                skipped += 1

    print(
        f"Done. json={len(json_files)} copied={copied} skipped={skipped} "
        f"missing={missing} invalid={invalid} path_errors={path_errors}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

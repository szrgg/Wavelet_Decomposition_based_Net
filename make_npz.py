

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _index_files(directory):
    """Return a filename-stem to path mapping and reject duplicate stems."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory does not exist: {directory}")

    indexed = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        if path.stem in indexed:
            raise ValueError(
                f"Duplicate filename stem '{path.stem}' in {directory}: "
                f"{indexed[path.stem].name} and {path.name}"
            )
        indexed[path.stem] = path

    if not indexed:
        raise ValueError(f"No supported image files found in: {directory}")
    return indexed


def convert_to_npz(
    image_dir,
    label_dir,
    output_dir,
    image_size=(400, 400),
    label_threshold=0,
    overwrite=False,
):

    images = _index_files(image_dir)
    labels = _index_files(label_dir)

    missing_labels = sorted(images.keys() - labels.keys())
    missing_images = sorted(labels.keys() - images.keys())
    if missing_labels or missing_images:
        messages = []
        if missing_labels:
            messages.append(
                "images without matching masks: " + ", ".join(missing_labels[:10])
            )
        if missing_images:
            messages.append(
                "masks without matching images: " + ", ".join(missing_images[:10])
            )
        raise ValueError("Filename mismatch; " + "; ".join(messages))

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    height, width = image_size

    converted = 0
    for name in sorted(images):
        output_path = output_dir / f"{name}.npz"
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {output_path}. Use --overwrite to replace it."
            )

        with Image.open(images[name]) as image_file:
            image = image_file.convert("RGB").resize(
                (width, height), Image.Resampling.BILINEAR
            )
            image_array = np.asarray(image, dtype=np.uint8)

        with Image.open(labels[name]) as label_file:
            label = label_file.convert("L").resize(
                (width, height), Image.Resampling.NEAREST
            )
            label_array = (
                np.asarray(label, dtype=np.uint8) > label_threshold
            ).astype(np.uint8)

        np.savez_compressed(output_path, image=image_array, label=label_array)
        converted += 1
        print(f"[{converted}/{len(images)}] Saved {output_path.name}")

    return converted


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert paired images and segmentation masks to NPZ files."
    )
    parser.add_argument("--image-dir", required=True, help="Input image directory.")
    parser.add_argument("--label-dir", required=True, help="Input mask directory.")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output directory shared by all training, validation and test samples.",
    )
    parser.add_argument("--height", type=int, default=400)
    parser.add_argument("--width", type=int, default=400)
    parser.add_argument(
        "--label-threshold",
        type=int,
        default=0,
        help="Mask values greater than this value are converted to class 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace NPZ files that already exist in the output directory.",
    )
    return parser


def main(args=None):
    args = build_parser().parse_args(args)
    if args.height <= 0 or args.width <= 0:
        raise ValueError("--height and --width must be positive integers.")
    if not 0 <= args.label_threshold <= 255:
        raise ValueError("--label-threshold must be between 0 and 255.")

    count = convert_to_npz(
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        image_size=(args.height, args.width),
        label_threshold=args.label_threshold,
        overwrite=args.overwrite,
    )
    print(f"Completed: {count} paired samples converted.")


if __name__ == "__main__":
    main()

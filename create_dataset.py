from pathlib import Path

import cv2
import yaml
import numpy as np
import SimpleITK as sitk
from PIL import Image

CONFIG_PATH = Path("dataset_9.yaml")

VALID_SPLITS = {"train", "val", "test"}

def mask_to_yolo(mask_slice, label_path):
    """Converts binary mask to YOLO segmentation polygon and saves it to label_path"""

    binary = (mask_slice > 0).astype(np.uint8)

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    height, width = binary.shape
    lines = []

    for contour in contours:
        epsilon = 0.002 * cv2.arcLength(contour, True)

        polygon = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)

        if len(polygon) < 3:
            continue

        coords = []

        for x, y in polygon:
            coords.extend([x / width, y / height])

        lines.append("0 " + " ".join(f"{v:.6f}" for v in coords))

    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(lines), encoding="utf-8")

def main():
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["folder"])

    for case in config["cases"]:
        name = case["name"]
        split = case["split"]

        if split not in VALID_SPLITS:
            raise ValueError(
                f"{name}: invalid split '{split}'. "
                f"Allowed splits: {VALID_SPLITS}"
            )

        image_path = Path(case["image"])
        mask_path = Path(case["mask"])

        mri = sitk.GetArrayFromImage(sitk.ReadImage(str(image_path)))

        mask = sitk.GetArrayFromImage(sitk.ReadImage(str(mask_path)))

        if mri.shape != mask.shape:
            raise ValueError(
                f"{name}: MRI and mask have different dimensions."
            )

        image_dir = output_dir / "images" / split
        label_dir = output_dir / "labels" / split

        mask_dir = output_dir / "masks" / name

        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        # Normalize MRI 0-255
        mri = mri.astype(np.float32)
        low, high = np.percentile(mri, [1, 99.5])

        if high > low:
            mri = np.clip(mri, low, high)
            mri = (mri - low) / (high - low)
            mri = np.round(mri * 255).astype(np.uint8)
        else:
            mri = np.zeros(mri.shape, dtype=np.uint8)

        # Binary mask
        mask = (mask > 0).astype(np.uint8) * 255

        for i in range(mri.shape[0]):
            filename = f"{name}_slice_{i:03d}.png"
            label_path = label_dir / f"{name}_slice_{i:03d}.txt"

            Image.fromarray(mri[i]).save(image_dir / filename)

            Image.fromarray(mask[i]).save(mask_dir / filename)

            mask_to_yolo(mask[i], label_path)

        # Automatické vytvoření konfigurace pro Ultralytics YOLO
        data_yaml = {
            "path": str(output_dir.resolve()).replace("\\", "/"),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {
                0: "hypothalamus",
            },
        }

        with (output_dir / "data.yaml").open(
            "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(
                data_yaml,
                f,
                sort_keys=False,
                allow_unicode=True,
            )

    print(f"Done! Dataset: {output_dir.resolve()}")
    print(f"YOLO configuration: {(output_dir / 'data.yaml').resolve()}")

if __name__ == "__main__":
    main()
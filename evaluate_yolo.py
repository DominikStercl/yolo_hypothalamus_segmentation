
from pathlib import Path

import numpy as np
import nrrd
import SimpleITK as sitk
from PIL import Image
from ultralytics import YOLO


def dice_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    denominator = pred.sum() + gt.sum()

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def iou_score(pred, gt):
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()

    if union == 0:
        return 1.0

    return intersection / union


def main():
    model_path = Path(
        "runs/segment/runs/hypothalamus_9mri/weights/best.pt"
    )

    test_dir = Path("yolo_dataset/images/test")
    gt_dir = Path("yolo_dataset/masks/mri03")

    # ORIGINAL MRI VOLUME
    mri_path = Path(
        "T2 for segmentation/sub-003_ses-1_acq-RARE_T2w.nii"
    )

    output_dir = Path("runs/nrrd_mask_test")
    pred_dir = output_dir / "pred_masks"
    overlay_dir = output_dir / "overlays"

    pred_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not test_dir.exists() or not gt_dir.exists():
        raise FileNotFoundError("Test or ground-truth folder not found.")

    if not mri_path.exists():
        raise FileNotFoundError(
            f"Original MRI not found: {mri_path}"
        )

    mri = sitk.ReadImage(str(mri_path))
    mri_array = sitk.GetArrayFromImage(mri)

    print("Original MRI array shape:", mri_array.shape)
    print("MRI spacing:", mri.GetSpacing())

    model = YOLO(str(model_path))
    images = sorted(test_dir.glob("*.png"))

    if not images:
        raise RuntimeError(f"No test PNG files in {test_dir}")

    all_pred = []
    all_gt = []
    slice_dice = []
    slice_iou = []

    print(f"Model: {model_path}")
    print(f"Test slices: {len(images)}")

    for image_path in images:
        gt_path = gt_dir / image_path.name

        if not gt_path.exists():
            raise FileNotFoundError(
                f"Reference mask not found: {gt_path}"
            )

        gt = np.array(
            Image.open(gt_path).convert("L")
        ) > 0

        result = model.predict(
            source=str(image_path),
            imgsz=256,
            device="cpu",
            conf=0.25,
            retina_masks=True,
            verbose=False,
        )[0]

        pred = np.zeros(gt.shape, dtype=bool)

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

            for mask in masks:
                resized = Image.fromarray(
                    (mask * 255).astype(np.uint8)
                ).resize(
                    (gt.shape[1], gt.shape[0]),
                    Image.Resampling.NEAREST,
                )

                pred |= np.array(resized) > 127

        if pred.shape != gt.shape:
            raise ValueError(
                f"Mask dimensions differ: {pred.shape} vs {gt.shape}"
            )

        dice = dice_score(pred, gt)
        iou = iou_score(pred, gt)

        all_pred.append(pred)
        all_gt.append(gt)

        if gt.any():
            slice_dice.append(dice)
            slice_iou.append(iou)

        Image.fromarray(
            pred.astype(np.uint8) * 255
        ).save(pred_dir / image_path.name)

        image = np.array(
            Image.open(image_path).convert("RGB")
        )

        overlay = image.copy()
        overlay[gt] = (
            0.5 * overlay[gt] + 0.5 * np.array([0, 255, 0])
        ).astype(np.uint8)
        overlay[pred] = (
            0.5 * overlay[pred] + 0.5 * np.array([255, 0, 0])
        ).astype(np.uint8)

        Image.fromarray(overlay).save(
            overlay_dir / image_path.name
        )

        print(
            f"{image_path.name}: "
            f"Dice={dice:.4f}, IoU={iou:.4f}"
        )

    pred_volume = np.stack(all_pred)
    gt_volume = np.stack(all_gt)

    if pred_volume.shape != mri_array.shape:
        raise ValueError(
            "Prediction volume does not match original MRI shape!\n"
            f"Prediction: {pred_volume.shape}\n"
            f"MRI:        {mri_array.shape}\n"
            "Check slice count, slice ordering and image dimensions."
        )

    global_dice = dice_score(pred_volume, gt_volume)
    global_iou = iou_score(pred_volume, gt_volume)

    print("\n--- TEST MRI mri03 ---")
    print(f"Dice (all pixels): {global_dice:.4f}")
    print(f"IoU  (all pixels): {global_iou:.4f}")

    if slice_dice:
        print(
            "Mean Dice (slices with reference mask): "
            f"{np.mean(slice_dice):.4f}"
        )
        print(
            "Mean IoU (slices with reference mask): "
            f"{np.mean(slice_iou):.4f}"
        )

    pred_nrrd = pred_volume.astype(np.uint8)

    pred_image = sitk.GetImageFromArray(pred_nrrd)
    pred_image.CopyInformation(mri)

    nrrd_path = output_dir / "mri03_yolo_prediction.nrrd"
    sitk.WriteImage(pred_image, str(nrrd_path))

    print(f"\nPredicted NRRD: {nrrd_path}")
    print(f"Predicted PNG masks: {pred_dir}")
    print(f"Overlays: {overlay_dir}")


if __name__ == "__main__":
    main()

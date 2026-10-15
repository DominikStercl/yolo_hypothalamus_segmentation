from ultralytics import YOLO


def main():
    model = YOLO("yolo26n.pt")

    model.train(
        data="yolo_dataset/data.yaml",
        epochs=30,
        imgsz=256,
        batch=2,
        device="cpu",
        workers=0,
        project="runs",
        name="hypothalamus_yolo",
        seed=42,
    )


if __name__ == "__main__":
    main()
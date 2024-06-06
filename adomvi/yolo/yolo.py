from ultralytics import YOLO
from pathlib import Path


def train(
    base_model: str, data: Path, epochs: int = 60, imgsz: int = 640, batch: int = 16
):
    model = YOLO(base_model)  # load a pretrained model
    model.train(data=data, epochs=epochs, imgsz=imgsz, batch=batch)


def predict(model_path: Path, source: Path):
    model = YOLO(model_path)

    # Run inference on the source
    results = model(source, stream=False)
    return results

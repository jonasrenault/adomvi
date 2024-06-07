from pathlib import Path

from ultralytics import YOLO


def train(
    base_model: str | Path,
    data: Path,
    epochs: int = 60,
    imgsz: int = 640,
    batch: int = 16,
    device: list[int] | str | None = None,
    **kwargs,
):
    """
    Train a Yolo base model with data in the data directory.

    Args:
        base_model (str | Path): base model name
        data (Path): training data directory
        epochs (int, optional): epochs. Defaults to 60.
        imgsz (int, optional): image size. Defaults to 640.
        batch (int, optional): batch size. Defaults to 16.
        device (list[int] | str | None, optional): device to use. Defaults to None.
    """
    model = YOLO(base_model)  # load a pretrained model
    model.train(
        data=data, epochs=epochs, imgsz=imgsz, batch=batch, device=device, **kwargs
    )


def predict(
    model_path: str | Path,
    source: str | Path,
    save_txt: bool = True,
    save_conf: bool = True,
):
    """
    Run prediction on a source using the given model

    Args:
        model_path (str | Path): the model to use for prediction
        source (str | Path): the source of data to predict
        save_txt (bool, optional): save detection results in a txt file. Defaults to True.
        save_conf (bool, optional): save confidence score for each detection.
            Defaults to True.

    Returns:
        _type_: _description_
    """
    model = YOLO(model_path)

    # Run inference on the source
    results = model.predict(source, stream=False, save_txt=save_txt, save_conf=save_conf)
    return results

# Automated Detection of Military Vehicles from Video Input (ADOMVI)

<div align="center">
  <img src="resources/video_tracking.gif" width="640"/>
</div>

## Introduction

This repository contains notebooks and resources used to train a state-of-the-art military vehicle tracker. Its main focus is on building a dataset of relevant images and annotations to fine-tune pre-trained object detection models, namely a [Yolov8](https://github.com/ultralytics) model. The [yolo-tracking](https://github.com/mikel-brostrom/yolo_tracking) library is used to provide the multi-object tracker algorithm.

## Contents

- The [adomvi](./adomvi/) directory contains jupyter notebooks to create a dataset of military vehicles, use this dataset to finetune a Yolov8 model for object detection, and to run object tracking on video inputs.
- The [resources](./resources/) directory contains video samples for vehicle detection task.

## Installation

You can install the project locally using [poetry](https://python-poetry.org/) with

```console
poetry install
```

## Run the notebooks

You can run the notebooks from this project in Google Colab to benefit from GPU acceleration:

<ul>
    <li>Train a YOLOv8 model with a custom dataset: <a target="_blank" href="https://colab.research.google.com/github/jonasrenault/adomvi/blob/main/adomvi/TankDetectionYolov8Train.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></li>
    <li>Run tracking using the trained model on a sample video: <a target="_blank" href="https://colab.research.google.com/github/jonasrenault/adomvi/blob/main/adomvi/TankTracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></li>
</ul>

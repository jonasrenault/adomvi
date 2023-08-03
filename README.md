# Automated Detection of Military Vehicles from Video Input (ADOMVI)

<div align="center">
  <img src="resources/video_tracking.gif" width="640"/>
</div>

## Introduction

This repository contains notebooks and resources used to train a state-of-the-art military vehicle tracker. Its main focus is on building a dataset of relevant images and annotations to fine-tune pre-trained object detection models, namely a [Yolov8](https://github.com/ultralytics) model. The [yolo-tracking](https://github.com/mikel-brostrom/yolo_tracking) library is used to provide the multi-object tracker algorithm.

After a first pass at training such a model from images available from object detection datasets, we explore other options to improve the performance of our model. We use scraping tools to collect more images of military vehicles from Google images. We also extend the classes to be able to discriminate between different types of vehicles: **Armoured Fighting Vehicle (AFV)**, **Armoured Personnel Carrier (APC)**, **Military Engineering Vehicle (MEV)** and **Light armoured vehicle (LAV)**. We provide a sample annotated dataset to test performance improvement from extending our training data.

We also explore using [diffusion models](https://huggingface.co/docs/diffusers/using-diffusers/conditional_image_generation) and the [dreambooth](https://huggingface.co/docs/diffusers/training/dreambooth) method to generate new training images in different scenes and conditions.

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
    <li>01 - Train a YOLOv8 model with a custom dataset: <a target="_blank" href="https://colab.research.google.com/github/jonasrenault/adomvi/blob/main/adomvi/01_TankDetectionYolov8Train.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></li>
    <li>02 - Run tracking using the trained model on a sample video: <a target="_blank" href="https://colab.research.google.com/github/jonasrenault/adomvi/blob/main/adomvi/02_TankTracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></li>
    <li>03 - Scrape images from google to extend the training dataset: <a target="_blank" href="https://colab.research.google.com/github/jonasrenault/adomvi/blob/main/adomvi/02_TankTracking.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></li>
    <li>04 - Fine tune Dreambooth to generate images of a tank: <a target="_blank" href="https://colab.research.google.com/github/jonasrenault/adomvi/blob/main/adomvi/04_DreamboothFineTuning.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a></li>
</ul>


## Tracking of military vehicles with multi-class object detection model

Some sample results of tracking different types of military vehicles (AFV, APC, MEV, LAV) using a finetuned yolov8-large model.

<div align="center">
  <img src="resources/apc.gif" width="640"/>
  <img src="resources/mev.gif" width="640"/>
  <img src="resources/lav.gif" width="640"/>
</div>

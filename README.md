# Automated Detection of Military Vehicles from Video Input (ADOMVI)

## Datasets

A list of image / video datasets used to train a military vehicle detection algorithm

- [The Search 2](https://figshare.com/articles/dataset/The_Search_2_dataset/1041463) dataset consists of 44 high-resolution digital color images of different complex natural scenes. Each scene (image) contains a single military vehicle that serves as a search target.
- [Sensor Data Management System (SDMS)](https://www.sdms.afrl.af.mil/index.php?collection=video_sample_set_1). The video sample sets #1 and #2 contain video clips from the DARPA VIVID Data Collection 1 collected at Eglin, AFB December 2nd and 3rd 2003. The clips are taken from three sensors: EO Daylight TV(DLTV), EO DLTV Spotter, and IR. The scenes contain military and civilian vehicles in many settings. All clips are approximately 60 seconds each and are in AVI format.
- Military vehicles datasets from Kaggle:
    * [Military Tanks](https://www.kaggle.com/datasets/antoreepjana/military-tanks-dataset-images)
    * [Military Vehicles](https://www.kaggle.com/datasets/amanrajbose/millitary-vechiles)
    * [Normal vs Military Vehicles](https://www.kaggle.com/datasets/amanrajbose/normal-vs-military-vehicles)
    * [War Tank Images](https://www.kaggle.com/datasets/icanerdogan/war-tank-images-dataset)
- [Open Images V7](https://storage.googleapis.com/openimages/web/index.html). Open Images is a dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, visual relationships, and localized narratives. It contains a total of 16M bounding boxes for 600 object classes on 1.9M images.
- [ImageNet](https://image-net.org/download-images). ImageNet, also known as ILSVRC 2012, is an image dataset organized according to the WordNet hierarchy. Each meaningful concept in WordNet, possibly described by multiple words or word phrases, is called a “synonym set” or “synset”. There are more than 100,000 synsets in WordNet, majority of them are nouns (80,000+). ImageNet provides on average 1,000 images to illustrate each synset.
- [Army Recognition](https://armyrecognition.com/vehicules_blindes_artillerie_armoured_france/caesar_sherpa_5_nexter_systems_obusier_automoteur_roues_artillerie_fiche_technique_description_fr.html) Database of world army equipment and recognition information.

## Projects & papers

- [Identify Military Vehicles in Satellite Imagery with TensorFlow](https://python.plainenglish.io/identifying-military-vehicles-in-satellite-imagery-with-tensorflow-96015634129d)
- [Military Vehicles Tracking](https://github.com/Lin-Sinorodin/Military_Vehicles_Tracking)
- [Automated Military Vehicle Detection From Low-Altitude Aerial Images](https://dll.seecs.nust.edu.pk/wp-content/uploads/2020/06/Automated-Military-Vehicle-Detection-from-Low-Altitude-Aerial-Images.pdf)
- [Military Vehicles CNN](https://www.kaggle.com/code/mpwolke/military-vehicles-cnn)
- [Fine-tune YOLOv8 models for custom use cases with the help of FiftyOne](https://docs.voxel51.com/tutorials/yolov8.html)

## Classes

- Vehicle or Non-Vehicle
- Vehicle: Military or Non-Military
- Vehicle - Military:
    * MILITARY ARMOURED
    * HEAVY EXPANDED MOBILITY TACTICAL TRUCK (HEMTT)
    * MILITARY TRUCK
    * HIGH MOBILITY MULTI-PURPOSE WHEELED VEHICLE
    * MILITARY CAR
    * MILITARY AMBULANCE
    * Self-propelled artillery
    * Self-propelled anti-aircraft
    * Amphibious armored scout car
    * Armored personnel carrier
    * Military cargo truck
    * Main battle tank

# Overhead Person Detection Dataset

![alt](https://raw.githubusercontent.com/bencejdanko/prepare-overhead-person-detection/refs/heads/main/grid_5x5.png)

## Hugging Face Dataset

In total, these are 13,469 images and annotations for overhead head-count detection in confined spaces.

This dataset has been processed and uploaded to Hugging Face:
- **Repository**: [bdanko/overhead-person-detection](https://huggingface.co/datasets/bdanko/overhead-person-detection)
- **Monochrome**: 1 channel (Grayscale)
- **Resolution**: 192x192 with letterbox padding
- **Single Class**: Person (Single Class Uniformity)

## Overview

We source data from Roboflow, an end-to-end computer vision platform that includes dataset hosting and annotation. We saved copies from 5 separately hosted datasets:

* people-detection-overhead-v2-6npnq (689 images)
* lift-overhead-detection-jvi8g (4890 images)
* overhead-eeifj-65fc8 (5079 images)
* overhead-91wif-ohhjg (2304 images)
* top-down-people-mmue8-ogvbu (507 images)

All data was exported using the YOLOv8 format, a structured annotation format consisting of a text file (.txt) for every image in the dataset. In total, these are 13,469 images and annotations for object detection. The raw data is stored in a [public Google Drive folder](https://drive.google.com/drive/folders/16jwyGjDevlD8W9jckSS1KESkTRr-YMGy?usp=sharing).

## Data Normalization

Our normalization techniques include:

- monochrome conversion
- 192x192 scaling with letterbox padding
- ensure 1 class (person) uniformity across the dataset

We decided on monochrome representation (1 channel instead of 3 RGB channels) to decreases data set size and noise, as well as to highlight intensity-based features rather then color. It was found experimentally that 192x192 scaling offered the required performance to reach our KPI  of 15+ FPS. We decided on letterbox padding due to the variety of dimensions from our data. Letterbox preserves the scale of human proportions.

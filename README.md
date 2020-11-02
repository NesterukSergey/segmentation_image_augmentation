# Segmentation images augmentation
This repository provides tools for augmentation images accompanied by segmentation masks. 

![Sample](https://github.com/NesterukSergey/segmentation_image_augmentation/blob/master/data/examples/simple_mpa_example.jpg)

![Sample](https://github.com/NesterukSergey/segmentation_image_augmentation/blob/master/data/examples/mpa_example.jpg)

It provides easy ways to:
* Modify target image and segmentation mask simultaneously
* Create boolean, multi-object, multi-part, semantic and class masks simultaneously
* Find bounding boxes
* Add background
* Move, flip and rotate object
* Add noise (salt, pepper, Gauss)
* Smooth image
* Apply perspective transform


## Measured time
* ~ 0.05s per scene for 777x565x3 final scene; 6 input objects; 1 type of output masks
* ~ 0.7s per scene for 777x565x3 final scene; 6 input objects; 4 types of output masks + bounding boxes


## Usage
You can find examples at [Demos.ipynb](https://github.com/NesterukSergey/segmentation_image_augmentation/blob/master/Demos.ipynb)


## Dataset

For plant images sample [IPPN](http://www.plant-phenotyping.org/datasets) dataset is used.

See more about the dataset at:

Massimo Minervini, Andreas Fischbach, Hanno Scharr, Sotirios A. Tsaftaris, Finely-grained annotated datasets for image-based plant phenotyping, Pattern  Recognition Letters (2015), doi: 10.1016/j.patrec.2015.10.013

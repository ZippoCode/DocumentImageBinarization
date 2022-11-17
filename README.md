## Thesis

## Description:

### Original: contains training image
- DIBCO: contains the testing dataset which comprises handwritten images. Each folder contains the image of the respective year. Each folder contains respectevely: 10 - 10 - 16 - 14 - 16 - 10 - 10 - 20 - 10 - 20 images. Totally they are 136 images
- [RealNoisyOffice](https://archive.ics.uci.edu/ml/datasets/NoisyOffice#)
	* real_noisy_images_grayscale: 72 grayscale images of scanned 'noisy' images.
	* real_noisy_images_grayscale_doubleresolution: idem, double resolution.
	* simulated_noisy_images_grayscale: 72 grayscale images of scanned 'simulated noisy' images for training, validation and test.


### Ground Truth:
- DIBCO: contains the associated ground truth
- RealNoisyOffice
	* clean_images_grayscale_doubleresolution: Grayscale ground truth of the images with double resolution.
	* clean_images_grayscale: Grayscale ground truth of the images with smoothing on the borders (normal resolution).
	* clean_images_binary: Binary ground truth of the images (normal resolution).


## Directory Structrure

```
dataset
│   README.md   
│
└───original
│   │
│   └───DIBCO
│   │    └─── from 2009 to 2019
│   │
│   └───RealNoisyOffice
│   │    └───real_noisy_image_grayscale
│   │    └───real_noisy_images_grayscale_doubleresolution
│   │    └───simulated_noisy_images_grayscale
│
└───ground_truth
│   │
│   └───DIBCO
│   │    └─── from 2009 to 2019
│   │
│   └───RealNoisyOffice
│   │    └───clean_images_grayscale
│   │    └───clean_images_grayscale_doubleresolution
│   │    └───clean_images_binaryscale

```



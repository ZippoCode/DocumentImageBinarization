# Binarization

## Description
This project focuses on the use of the Fast Fourier Convolution technique to enhance the quality of digitized documents that exhibit noise and disturbances caused by both natural factors such as ink oxidation, stains, and scratches, as well as digital factors like lighting and image compression. The primary objective is to optimize the readability and analysis of digitized documents through image binarization. This is a crucial phase in Document Analysis as it allows for the separation of text from the background, simplifying subsequent processing steps such as optical character recognition and text segmentation.


## Results
![Origin Image](https://github.com/ZippoCode/DocumentImageBinarization/blob/main/images/1.png)
![Binarization Image](https://github.com/ZippoCode/DocumentImageBinarization/blob/main/images/1_bin.png)
![Origin Image](https://github.com/ZippoCode/DocumentImageBinarization/blob/main/images/9.png)
![Binarization Image](https://github.com/ZippoCode/DocumentImageBinarization/blob/main/images/9_bin.png)

## Environment setup

Clone the repo `git clone https://github.com/ZippoCode/DocumentImageBinarization.git`

Python virtualenv:
`
python3 -m venv .venv/binarization
source .venv/binarization/bin/activate
pip install torch torchvision

cd DocumentImageBinarization
pip install -r requirements.txt
`

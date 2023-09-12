# Binarization

## Description
Document Image Binarization focuses on the use of the **Fast Fourier Convolution** technique to enhance the quality of
digitized documents that exhibit noise and disturbances caused by both natural factors such as ink oxidation, stains,
and scratches, as well as digital factors like lighting and image compression. The primary objective is to optimize 
the readability and analysis of digitized documents through image binarization. 


## Results
Image 1
<p float="left" align="center">
  <img src="https://github.com/ZippoCode/DocumentImageBinarization/blob/main/images/1.png" alt="drawing" width="50%"/> 
  <img src="https://github.com/ZippoCode/DocumentImageBinarization/blob/main/images/1_bin.png" alt="drawing" width="50%"/>
</p>

Image 2
<p float="left" align="center">
  <img src="https://github.com/ZippoCode/DocumentImageBinarization/blob/main/images/9.png" alt="drawing" width="50%"/> 
  <img src="https://github.com/ZippoCode/DocumentImageBinarization/blob/main/images/9_bin.png" alt="drawing" width="50%"/>
</p>

## Environment setup

Clone the repo `git clone https://github.com/ZippoCode/DocumentImageBinarization.git`

Python virtualenv:
```
python3 -m venv .venv/binarization
source .venv/binarization/bin/activate
pip install torch torchvision

cd DocumentImageBinarization
pip install -r requirements.txt
```

### Download pre-trained models
Start by downloading pre-trained models from this [link](https://drive.google.com/file/d/1zj_QGlWJlS0KWvwH5qhl4c_fF1PLBmsO/view?usp=drive_link), and then run:
```
unzip lama_checkpoints.zip
```

## Run
```
source .venv/binarization/bin/activate
python3 main.py --image=<path>
```

## Reference repository
* [LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions](https://github.com/advimman/lama)

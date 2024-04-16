# Brahma-GAN: AI-Powered Low-Dose CT Enhancement

## Project Description

This project utilizes deep learning models to enhance the quality of low-dose medical images to the quality of high-dose images. This is particularly important in medical imaging, like CT scans, where reducing the dose of radiation is beneficial for patient safety but maintaining image quality is crucial for accurate diagnosis. We use state-of-the-art convolutional neural networks (CNNs) that learn from pairs of low and high-dose images to predict high-dose quality images from new low-dose images.

![model](https://github.com/pallav-d3/Frost_Hack/assets/34905952/d58f2ba4-7f45-4cd5-9767-2237feaa80c8)



## Features

- **Image Enhancement**: Converts low-dose medical images to high-dose quality.
- **Deep Learning Model**: Uses advanced CNN architectures optimized for image translation tasks.
- **Evaluation Metrics**: Includes scripts to evaluate model performance using metrics such as PSNR, SSIM, and MSE.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8 or higher
- Pip package manager
- Access to a GPU for training and inference (recommended)

## Run command
- 

To install the required packages, follow these steps:

```bash
git clone https://github.com/your-repo/low-to-high-dose.git
cd low-to-high-dose
pip install -r requirements.txt

```
## Citations : 
[Mayo Grand challenge](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h/file/858370564530)
[VQGAN](https://github.com/dome272/VQGAN-pytorch)
[CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

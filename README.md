 ## Image SuperResolution(ISR) using Random Forest Regressor(RFSR) and Support Vector Regressor(SVR) Machine learning Models:-

## 1. Problem Statement
<div align="justify">Image super-resolution (ISR) focuses on enhancing low-resolution (LR) images into high-resolution (HR) counterparts. This study explores three distinct machine learning approaches: Residual learning with Random Forest (RFSR), Channel-wise super-resolution using Support Vector Machines (SVMs), and Patch-based Random Forest regression. These methods aim to address the challenge of reconstructing fine details like edges and textures in LR images while evaluating performance using metrics such as Structural Similarity Index (SSIM) and Mean Absolute Error (MAE).</div>

<p float="left">
 <img src="isr_images/isr01.jpg" alt="Image" width="600" />
  
</p>


## 1.1. Dataset Description:
<div align="justify">The dataset consists of paired 50 low-resolution (LR) and 50 high-resolution (HR) images. LR images are smaller in size and lack fine details, while HR images serve as the target for reconstruction. Each image is divided into patches for training and prediction, with LR patches flattened as input features and HR center pixels used as targets.</div>

## 1.2. EDA:
<p float="left">
 <img src="isr_images/isr02.jpg" alt="Image" width="600" />
  
</p>

Figure 1. (Distributions of images - Denoising)

## 2. Methodologies:

### 2.1 Random Forest Based Super-Resolution (RFSR) on Y-channel:
<div align="justify"> This methodology applies residual learning to reconstruct the difference between bicubic-upscaled LR images and their HR counterparts:<br>

1. Preprocessing: Convert images to the YCbCr colorspace, focusing on the Y-channel for structural details.<br>
2. Patch Extraction: Overlapping LR and HR patches are paired, and bicubic interpolation is used to match dimensions.<br>
3. Model Training: A Random Forest Regressor predicts residual patches for the Y-channel.<br>
4. Output:Combine predicted residuals with bicubic-upscaled patches to generate HR images.<br></div>

<p float="left">
 <img src="isr_images/isr03.jpg" alt="Image" width="600" />
  
</p>


### 2.2 Patch-Based ISR Using Support Vector Machines (SVMs):
<div align="justify">This approach focuses on reconstructing HR images channel-wise using SVM models:<br>
Steps:<br>
 
1. Extract and normalize LR and HR patches.<br>
2. Train SVM models for individual RGB channels.<br>
3. Predict HR patches using the trained models.<br>
4. Combine channel-wise predictions to reconstruct HR images.<br>

Advantages: Allows flexibility in patch size and improves channel-specific reconstructions.<br> </div>

<p float="left">
 <img src="isr_images/isr04.jpg" alt="Image" width="600" />
  
</p>


### 2.3 Patch-Based Regression with Random Forests:
<div align="justify"> A patch-based learning approach where overlapping patches are used to train Random Forest regressors for each color channel (R, G, B):<br>

1. Extract LR and HR patch pairs (Pi = (Li, Hi))<br>
2. Train regressors to predict HR patches from LR inputs.<br>
3. Reconstruct HR images by combining predicted patches.<br></div>

## 3. Models
### 3.1 Support Vector Machines (SVM)
<div align="justify"> Utilizes kernels to handle complex, high-dimensional data.
1. Predicts pixel values for each RGB channel independently.</div>

###3.2 Random Forest Regressor
<div align="justify"> 1. Trained on LR-HR patch pairs.
2. Handles patch-based mapping for each RGB channel.</div>
   
### 3.3 Random Forest Super-Resolution (RFSR)
<div align="justify"> 1. Uses a residual learning approach to predict the missing fine details (edges, textures).
2. Targets only the Y-channel (structural details) in the YCbCr color space.Employs Random Forest Regressors trained on residual patches for enhancement.</div>





## Models to implment:
1. Forest Based SISR (have code)
   1. [paper](papers/random_forests.pdf)
   2. [code](https://github.com/jshermeyer/RFSR)
2. [optional] Local Regression Based SISR (need to code; will be easy)
   1. [paper](papers/local_regression.pdf)


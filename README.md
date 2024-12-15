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
<div align="justify"> Utilizes kernels to handle complex, high-dimensional data.<br>
1. Predicts pixel values for each RGB channel independently.<br></div>

### 3.2 Random Forest Regressor
<div align="justify"> 1. Trained on LR-HR patch pairs.<br>
2. Handles patch-based mapping for each RGB channel.<br> </div>
   
### 3.3 Random Forest Super-Resolution (RFSR)
<div align="justify"> 1. Uses a residual learning approach to predict the missing fine details (edges, textures).<br>
2. Targets only the Y-channel (structural details) in the YCbCr color space.Employs Random Forest Regressors trained on residual patches for enhancement.<br> </div>

## 4. Analysis
### 4.1 Metrics:<br>
<div align="justify"> SSIM for structural quality, MAE for pixel error.<br></div>

### 4.2 Performance:<br>
<div align="justify"> RFSR excels in edges, SVMs in color precision, and Random Forest regression in consistency.<br></div>

<p float="left">
 <img src="isr_images/isr05.jpg" alt="Image" width="600" />
  
</p>

### 4.3 Challenges:<br>
<div align="justify"> Minor artifacts from patch reconstruction; occasional color mismatches in YCbCr conversion. Periodic diagonal streaks appear on the predicted HR patches due to crude pixel features.<br></div>

## 6. Conclusion and Future Scope:
<div align="justify"> This study highlights the efficacy of machine learning approaches in ISR, outperforming traditional bicubic interpolation in both visual quality and numerical metrics. The patch-based methods (RFSR and Random Forest regression) exhibited high structural integrity, while SVMs excelled in color consistency.</div>


<p float="left">
 <img src="isr_images/isr06.jpg" alt="Image" width="600" />
  
</p>


## 7. References
<div align="justify">
1. Ni, Karl S. and Truong Q. Nguyen. “Image Super resolution Using Support Vector Regression.” IEEE Transactions on Image Processing 16 (2007): 1596-1610.
2. L. An and B. Bhanu, "Improved image super-resolution by Support Vector Regression," The 2011 International Joint Conference on Neural Networks, San Jose, CA, USA, 2011, pp. 696-700, doi: 10.1109/IJCNN.2011.6033289.
3. Jianchao Yang, J. Wright, T. Huang and Yi Ma, "Image super-resolution as sparse representation of raw image patches," 2008 IEEE Conference on Computer Vision and Pattern Recognition, Anchorage, AK, USA, 2008, pp. 1-8, doi: 10.1109/CVPR.2008.4587647.
4. Ni, Karl S. and Truong Q. Nguyen. “Image Superresolution Using Support Vector Regression.” IEEE Transactions on Image Processing 16 (2007): 1596-1610.
5. Jianchao Yang, Student Member, IEEE, John Wright, Member, IEEE, Thomas S. Huang, Fellow, IEEE, and Yi Ma, Senior Member, IEEE.
6. H. Li, K.-M. Lam, and M. Wang, "Image Super-resolution via Feature-augmented Random Forest," Pattern Recognition
Letters, vol. 108, pp. 31–37, 2018, doi: 10.1016/j.patrec.2018.01.014 </div>



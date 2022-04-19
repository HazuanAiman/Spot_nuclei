# Spot Cell Nuclei using Semantic Segmentation with U-Net
## Problem Statement
The aim of the project is to detect a cell nuclei based on the images set. The cell nuclei are vary in shapes and sizes thus, semantic segmentation is used to perform this project. The model is trained using the [Data Science Bowl 2018](https://www.kaggle.com/c/data-science-bowl-2018) dataset.
## Methodology
#### IDE and Library
This project is made using Spyder as the main IDE. The main libraries used in this project are Tensorflow, Numpy, Matplotlib, OpenCV and Scikit-learn.
#### Model Pipeline
The model architecture used for this project is U-Net. The model consists of two components, the downward stack and upward stack. The downward stack serves as feature extractor while the upward stack helps to produce pixel-wise output. Figure below shows the structure of the model.

<p align="center">
<img width="900" height="900" src="https://github.com/HazuanAiman/Spot_nuclei/blob/main/images/model%20pipeline.PNG">

The model is trained with a batch size of 16 and 100 epochs. Early stopping is applied in the training and it triggers at epochs 26. This is to prevent the model from overfitting. The training accuracy achieved is 97% and the validation accuracy is 96%. Figure below shows the graphs of the training process. 
<p align="center">
  <img src="https://github.com/HazuanAiman/Spot_nuclei/blob/main/images/epoch%20accuracy.PNG">
<p align="center">
<img src="https://github.com/HazuanAiman/Spot_nuclei/blob/main/images/epoch%20loss.PNG">

## Results
The model is trained using the train dataset and evaluated using the test dataset. The test result are as show below.
<p align="center">
<img src="https://github.com/HazuanAiman/Spot_nuclei/blob/main/images/result.PNG">

Some predictions are also made with the model using some of the test data. The actual output masks and prediction masks are shown in figures below.
<p align="center">
<img src="https://github.com/HazuanAiman/Spot_nuclei/blob/main/images/predicted%20result.png">
<img src="https://github.com/HazuanAiman/Spot_nuclei/blob/main/images/predicted%20result1.png">
<img src="https://github.com/HazuanAiman/Spot_nuclei/blob/main/images/predicted%20result2.png">
<p>
Overall, the model is capable of segmenting the cell neuclei with an excellent accuracy.

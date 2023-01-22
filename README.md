# PointCloudCompression
 
This project aims to provide an implementation of Point Cloud Compression (PCC) methods for segmented point clouds. 


Developed together with Saverio Cavasin (@SvrCvs)
## Task 

<p align="center">
  <img src="readme_imgs/task.png" width="800" title="Task">
</p>

## Dataset
The dataset is the [SELMA](https://scanlab.dei.unipd.it/selma-dataset/) dataset, which is made by data collected by of 3 LIDARs located on a vehicle in a urban scenario. 
The location of the sensors is the following:

<p align="center">
  <img src="imgs/sensors.png" width="700" title="Dataset">
</p>

An example of the dataset is shown in the figure below:

<p align="center">
  <img src="readme_imgs/pc.png" width="700" title="Dataset">
</p>


The point clouds are segmented in different clsses, widely discussed [here](https://scanlab.dei.unipd.it/selma-dataset/).

## Strategies 
### 1) **DRACO** [2] 
is a compression library for 3D geometric meshes and point clouds. It is based on the Google Draco library, which is a general-purpose 3D geometry compression library.

<p align="center">
  <img src="readme_imgs/draco.png" width="600" title="Dataset">
</p>

### 2) **DBScan** 
is a density-based clustering algorithm. It is a popular algorithm for clustering in a spatial context. The algorithm groups together points that are closely packed together (points with many nearby neighbors), marking as outliers points that lie alone in low-density regions (whose nearest neighbors are too far away).

<p align="center">
  <img src="readme_imgs/dbscan.png" width="600" title="Dataset">
</p>

### 3) **Convolutional Autoencoder**
trained to learn a representation (encoding) of a sample, with the lower possible loss in the reconstruction of the input data. 
The network is composed by an encoder and a decoder. The encoder compresses the input data into a lower dimensional space, while the decoder reconstructs the input data from the compressed representation. The architecture of the network is the following:

<p align="center">
  <img src="readme_imgs/net.png" width="300" title="Dataset">
</p>


The training is done on 400 Point clouds  of each class, and the test is done on 100 point clouds of each class. The results are the following:

<p align="center">
  <img src="readme_imgs/train.png" width="800" title="Dataset">
</p>

Some examples of the reconstruction are the following:

<p align="center">
  <img src="readme_imgs/rec.png" width="800" title="Dataset">
</p>


### 4) **2D Projection** [3][4]
Finally, we propose to change the coordinate system of the point cloud from cartesian to spherical and then project the point cloud on a 2D plane, whith a specific grid size. Some examples are the following:

<p align="center">
  <img src="readme_imgs/img.png" width="600" title="Dataset">
</p>


### References

<p align="center">
  <img src="readme_imgs/ref.png" width="500" title="Dataset">
</p>



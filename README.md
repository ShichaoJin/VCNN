# VCNN
## This responsitory is about our paper "Separating the structural components of maize for field phenotyping using terrestrial lidar data and deep convolutional neural networks", which is early accessiable at https://ieeexplore.ieee.org/document/8931235 The code is for learning use only, and any commercial use is not allowed.

## Basic information about the code
The code was written by me in July 2017 in windows using Pytorch deep learnng toolkit. The pytorch version is 0.1.12, python version is Anaconda Python 3.6.6.
It ran well in windows10 with NVIDIA GPU support. 
If you cannot configure the pytorch environment, you can download my previous enironment at: https://pan.baidu.com/s/1g-NKGyUka42_mvw-4-Zy4A   Extracting Password：wpw8

## Demo data and instructions
https://github.com/ShichaoJin/VCNN/tree/master/data

https://github.com/ShichaoJin/VCNN/blob/master/data/data.md

## Pretrained model and instructions
https://github.com/ShichaoJin/VCNN/blob/master/model/model.md

## Code instructions step by step
In the script directory, you wil see code for training, 001_train_vcnn.py
In the data directorty, a demo training data is provided. 
In the model directory, the pretrained model used in our paper is presented.


Step 1: Run the 001_train_vcnn.py
![traindemo](https://github.com/ShichaoJin/VCNN/blob/master/IMG/train_demo.png)






Step 2: Testing dataset and code can be modified from the training scripy. By the way, the algorithm will be intergrated into Greenvalley International LiDAR360 software, which will enable a convenient application with GUI support.

Finally, the code will be updated into the latest version in the further. If you have any questions, please post it in the ISSUEs.



## If you use any part of code, please cite the following paper，
Jin S, Su Y, Gao S, Wu F, Ma Q, Xu K, Ma Q, Hu T, Liu J, Pang S, Guan H, Zhang J, Guo Q. 2019. 
Separating the structural components of maize for field phenotyping using terrestrial lidar data and deep convolutional neural networks. 
IEEE Transactions on Geoscience and Remote Sensing, accepted: 1-15

## More related references
[1] S. Jin, Y. Su, S. Gao et al., “Separating the structural components of maize for field phenotyping using terrestrial lidar data and deep convolutional neural networks,” IEEE Transactions on Geoscience and Remote Sensing, accepted, 2019.
[2] S. Jin, Y. Su, S. Gao et al., “Deep Learning: Individual Maize Segmentation From Terrestrial Lidar Data Using Faster R-CNN and Regional Growth Algorithms,” Frontiers in Plant Science, vol. 9, pp. 866-875, 2018.
[3] S. Jin, Y. Su, F. Wu et al., “Stem-Leaf Segmentation and Phenotypic Trait Extraction of Individual Maize Using Terrestrial LiDAR Data,” IEEE Transactions on Geoscience and Remote Sensing, vol. 57, no. 3, pp. 1336-1346, 2018.
[4] Q. Guo, F. Wu, S. Pang et al., “Crop 3D—a LiDAR based platform for 3D high-throughput crop phenotyping,” Science China Life Sciences, pp. 1-12, 2017.
[5] Su, Y., Wu, F., Ao, Z., Jin, S., Qin, F., Liu, B., Pang, S., Liu, L., Guo, Q., 2019. Evaluating maize phenotype dynamics under drought stress using terrestrial lidar. Plant methods 15, 11-26.
[6] 郭庆华, 杨维才, 吴芳芳, 庞树鑫, 金时超, 陈凡, 王秀杰, 2018. 高通量作物表型监测: 育种和精准农业发展的加速器. 中国科学院院刊 33, 940-946

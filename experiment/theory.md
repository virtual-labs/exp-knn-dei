### Theory

The k-Nearest Neighbours (KNN) algorithm is a supervised machine learning algorithm used for classification and regression. It is a non-parametric method, as it does not assume any prior distribution of the data. In KNN, classification is performed based on the similarity between data samples, which is commonly measured using distance metrics such as Euclidean distance.

#### 1. Algorithm Overview

The nearest neighbour classification concept was introduced by Fix and Hodges and later extended by Cover and Hart. In this method, a test sample is assigned a class label by identifying the k nearest training samples in the feature space and applying majority voting. The value of k represents the number of nearest neighbours considered for classification.

#### 2. Lazy Learning

KNN does not involve an explicit training phase. Instead, it stores the entire training dataset and performs computation during the testing phase. Hence, it is referred to as a **lazy learning** or **instance-based learning** algorithm. The choice of k significantly affects classifier performance. Smaller values of k may result in overfitting, while larger values of k may lead to underfitting.

#### 3. Feature Scaling

Since KNN is based on distance calculations, feature scaling is important to ensure that all features contribute equally. Without proper scaling, features with larger numerical ranges may dominate the distance computation.

#### 4. Merits of k-Nearest Neighbours (KNN)

- **Simple and easy to understand**
- **Does not require an explicit training phase**
- **Makes no assumption about data distribution**
- **Works well for small and well-separated datasets**
- **Can be used for both classification and regression**

#### 5. Demerits of k-Nearest Neighbours (KNN)

- **High computational cost during prediction**
- **Requires large memory to store training data**
- **Highly sensitive to feature scaling**
- **Performance degrades in high-dimensional data**
- **Sensitive to noise and outliers**

The figure below illustrates how K-NN assigns a class to a new data point by considering the majority class among its nearest neighbours.

![KNN Classification](images/knn_new_data_assign.png)
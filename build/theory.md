<style>
.formula-block {
    text-align: center;
    margin: 18px 0;
}

.formula-text {
    display: inline-block;
    font-family: "Cambria Math", "Times New Roman", "Georgia", serif;
    font-size: 1.15em;
    line-height: 1.4;
}

.figure-block {
    text-align: center;
    margin: 18px 0;
}

.figure-block img {
    max-height: 320px;
    width: auto;
}

.figure-caption {
    color: #64748b;
    font-size: 0.92rem;
    margin-top: 8px;
    font-style: italic;
}
</style>

#### 1. Overview

The k-Nearest Neighbours (KNN) algorithm is a supervised machine learning algorithm commonly used for both classification and regression tasks. It is considered a non-parametric method because it does not assume any predefined distribution for the data. Instead of learning a mathematical model during training, KNN makes predictions by comparing new data points with existing training samples based on similarity.

This similarity is usually measured using distance metrics such as Euclidean distance in feature space. The concept of nearest neighbour classification was introduced by Fix and Hodges and later extended by Cover and Hart.

When a new test sample is given, KNN identifies the k nearest training samples and assigns a class label by majority voting among those neighbours. The value of k plays an important role in model behaviour and prediction quality.

Unlike many machine learning algorithms, KNN does not involve an explicit training phase. It stores the full training set and computes neighbours only at prediction time, so it is called a lazy learning or instance-based learning algorithm.

Since KNN relies on distance calculations, feature scaling is important; otherwise, features with larger ranges may dominate distance values and degrade model performance.

#### 2. Working of KNN

The KNN workflow is:

1. Choose the number of neighbours k.
2. Compute distance from the new sample to all training samples.
3. Sort distances and select the k nearest neighbours.
4. Count class labels among those neighbours.
5. Assign the majority class to the new sample.

Thus, a prediction is based directly on proximity and local voting.

#### 3. Determining the Value of k

Choosing k is critical:

- If k is too small, the model becomes sensitive to noise and may overfit.
- If k is too large, distant neighbours influence prediction and may cause underfitting.

In practice, several k values are tested, and the value with best validation performance is selected.

#### 4. Distance Metrics in KNN

#### 4.1 Euclidean Distance (L2 norm)

Euclidean distance is the most common metric for KNN. It is the straight-line distance between two points.

<div class="formula-block">
    <span class="formula-text">
        <i>d</i> = &radic;( &sum;<sub><i>i</i>=1</sub><sup><i>n</i></sup> (<i>x</i><sub>i</sub> - <i>y</i><sub>i</sub>)<sup>2</sup> )
    </span>
</div>

Where:
- x<sub>i</sub>: value of the i-th feature of point X
- y<sub>i</sub>: value of the i-th feature of point Y
- n: total number of features

For 2D:

<div class="formula-block">
    <span class="formula-text">
        <i>d</i> = &radic;((<i>x</i><sub>1</sub>-<i>y</i><sub>1</sub>)<sup>2</sup> + (<i>x</i><sub>2</sub>-<i>y</i><sub>2</sub>)<sup>2</sup>)
    </span>
</div>

#### 4.2 Manhattan Distance (L1 norm)

Manhattan distance sums absolute coordinate differences:

<div class="formula-block">
    <span class="formula-text">
        <i>d</i> = &sum;<sub><i>i</i>=1</sub><sup><i>n</i></sup> |<i>x</i><sub>i</sub>-<i>y</i><sub>i</sub>|
    </span>
</div>

For 2D:

<div class="formula-block">
    <span class="formula-text">
        <i>d</i> = |<i>x</i><sub>1</sub>-<i>y</i><sub>1</sub>| + |<i>x</i><sub>2</sub>-<i>y</i><sub>2</sub>|
    </span>
</div>

It is often more robust to outliers and useful in high-dimensional settings where axis-wise differences are meaningful.

#### 5. Non-Parametric Nature of KNN

KNN is called non-parametric because it does not fit a fixed functional form and does not estimate model parameters from assumptions about the data distribution. Predictions are made directly from stored training instances.

#### 6. Hyperparameter Selection (Choosing k)

The value of k is a hyperparameter controlling the number of neighbours used in classification.

- Small k (for example, k=1): low bias, high variance, noise sensitive.
- Large k: smoother boundaries, higher bias, possible underfitting.

Therefore, k should be selected to balance bias and variance, typically using cross-validation.

<div class="figure-block">
<img src="images/fig1_knn.png" alt="Illustration of KNN classification showing reassignment of a new data point">
<p class="figure-caption">Figure 1: Illustration of KNN classification showing reassignment of a new data point based on nearest neighbours.</p>
</div>

The figure shows how a new sample is assigned to a class after checking the majority class among nearby points.

#### 7. Algorithm

Step 1: Store all training data  
Step 2: Choose value of k  
Step 3: For a new sample, compute distance to all training samples  
Step 4: Sort distances in ascending order  
Step 5: Select k nearest neighbours  
Step 6: For classification, use majority voting among the k neighbours  

<div class="formula-block">
    <span class="formula-text">
        <i>y&#770;</i> = mode({<i>y</i><sub>(1)</sub>, <i>y</i><sub>(2)</sub>, ..., <i>y</i><sub>(<i>k</i>)</sub>})
    </span>
</div>

Step 7: For regression, average the k neighbour target values:

<div class="formula-block">
    <span class="formula-text">
        <i>y&#770;</i> = (1 / <i>k</i>) &sum;<sub><i>j</i>=1</sub><sup><i>k</i></sup> <i>y</i><sub>(<i>j</i>)</sub>
    </span>
</div>

#### 8. Practical Considerations in KNN

#### 8.1 Handling Ties
If tied votes occur:
- Reduce k by 1 and re-vote, or
- Choose class of the nearest sample among tied classes, or
- Randomly choose among tied classes.

#### 8.2 Feature Scaling (Critical)
KNN depends on distance; larger-scale features dominate without scaling.

Use either:
- Min-Max scaling:
    <div class="formula-block">
        <span class="formula-text">
            <i>x</i><sub>scaled</sub> = (<i>x</i> - <i>x</i><sub>min</sub>) / (<i>x</i><sub>max</sub> - <i>x</i><sub>min</sub>)
        </span>
    </div>
- Z-score normalization:
    <div class="formula-block">
        <span class="formula-text">
            <i>z</i> = (<i>x</i> - &mu;) / &sigma;
        </span>
    </div>

#### 8.3 Choosing Optimal k via Cross-Validation
1. Split into training and validation data.
2. Try multiple k values (1, 3, 5, 7, ...).
3. Evaluate validation accuracy for each k.
4. Select k with best validation performance.

#### 9. Merits and Demerits of KNN

| Merits of KNN | Demerits of KNN |
|---|---|
| Simple and easy to understand | High computational cost during prediction |
| No explicit training phase | High memory usage for training data storage |
| No distribution assumptions | Sensitive to feature scaling |
| Works well for small, well-separated datasets | Performance degrades in high dimensions |
| Supports classification and regression | Sensitive to noise and outliers |

#### 10. Applications

KNN is used in:
- Image recognition
- Recommendation systems
- Medical diagnosis support
- Pattern recognition (such as handwriting and speech)
- Credit scoring and fraud detection


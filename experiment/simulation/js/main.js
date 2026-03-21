
// ==========================================
// KNN EXPERIMENT STEPS
// ==========================================

const STEPS_KNN = [
    // ========== STEP 1: IMPORTING LIBRARIES (Cells 1-2) ==========
    {
        title: "Importing Libraries",
        blocks: [
            {
                // Cell 1
                comment: "Import analysis and plotting libraries",
                code: `# Import Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
print("Imported Analysis and Plotting libraries")`,
                output: `<div class="output-success">Imported Analysis and Plotting libraries</div>`
            },
            {
                // Cell 2
                comment: "Import scikit-learn modules",
                code: `from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)
print("Imported Sklearn Modules")`,
                output: `<div class="output-success">Imported Sklearn Modules</div>`
            }
        ]
    },
    // ========== STEP 2: LOADING DATASET (Cells 3-4) ==========
    {
        title: "Loading Dataset",
        blocks: [
            {
                // Cell 3
                comment: "Loading and Reading Data",
                code: `# Loading and Reading Data 

iris = load_iris()
X = iris.data
y = iris.target
print("Extracted Feature and Target as numpy Arrays")`,
                output: `<div class="output-success">Extracted Feature and Target as numpy Arrays</div>`
            },
            {
                // Cell 4
                comment: "Creating DataFrame with feature and target names",
                code: `# extracting feature names
names = iris.feature_names

# extracting target names
target_names = iris.target_names

# making a dataframe with column names
iris_df = pd.DataFrame(X, columns=names)
iris_df["label"] = [target_names[i] for i in y]
iris_df.head()`,
                output: `<div class="table-wrapper">
<table border="1" class="dataframe data-table">
  <thead>
    <tr style="text-align: right;">
      <th></th><th>sepal length (cm)</th><th>sepal width (cm)</th><th>petal length (cm)</th><th>petal width (cm)</th><th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr><th>0</th><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>setosa</td></tr>
    <tr><th>1</th><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td><td>setosa</td></tr>
    <tr><th>2</th><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td><td>setosa</td></tr>
    <tr><th>3</th><td>4.6</td><td>3.1</td><td>1.5</td><td>0.2</td><td>setosa</td></tr>
    <tr><th>4</th><td>5.0</td><td>3.6</td><td>1.4</td><td>0.2</td><td>setosa</td></tr>
  </tbody>
</table>
</div>`
            }
        ]
    },
    // ========== STEP 3: DATA ANALYSIS (Cells 5-10) ==========
    {
        title: "Data Analysis",
        blocks: [
            {
                // Cell 5
                comment: "Overview of the dataset",
                code: `# Data Analysis

# Overview
iris_df.info()`,
                output: `<div class="output-text">
&lt;class 'pandas.core.frame.DataFrame'&gt;<br>
RangeIndex: 150 entries, 0 to 149<br>
Data columns (total 5 columns):<br>
 #   Column             Non-Null Count  Dtype  <br>
---  ------             --------------  -----  <br>
 0   sepal length (cm)  150 non-null    float64<br>
 1   sepal width (cm)   150 non-null    float64<br>
 2   petal length (cm)  150 non-null    float64<br>
 3   petal width (cm)   150 non-null    float64<br>
 4   label              150 non-null    object <br>
dtypes: float64(4), object(1)<br>
memory usage: 6.0+ KB
</div>`
            },
            {
                // Cell 6
                comment: "Checking target distribution",
                code: `# Target counts
iris_df["label"].value_counts()`,
                output: `<div class="output-text">
label<br>
setosa        50<br>
versicolor    50<br>
virginica     50<br>
Name: count, dtype: int64
</div>`
            },
            {
                // Cell 7
                comment: "Statistical summary of the dataset",
                code: `# Dataset description
iris_df.describe()`,
                output: `<div class="table-wrapper">
<table border="1" class="dataframe data-table">
  <thead>
    <tr style="text-align: right;">
      <th></th><th>sepal length (cm)</th><th>sepal width (cm)</th><th>petal length (cm)</th><th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr><th>count</th><td>150.000000</td><td>150.000000</td><td>150.000000</td><td>150.000000</td></tr>
    <tr><th>mean</th><td>5.843333</td><td>3.057333</td><td>3.758000</td><td>1.199333</td></tr>
    <tr><th>std</th><td>0.828066</td><td>0.435866</td><td>1.765298</td><td>0.762238</td></tr>
    <tr><th>min</th><td>4.300000</td><td>2.000000</td><td>1.000000</td><td>0.100000</td></tr>
    <tr><th>25%</th><td>5.100000</td><td>2.800000</td><td>1.600000</td><td>0.300000</td></tr>
    <tr><th>50%</th><td>5.800000</td><td>3.000000</td><td>4.350000</td><td>1.300000</td></tr>
    <tr><th>75%</th><td>6.400000</td><td>3.300000</td><td>5.100000</td><td>1.800000</td></tr>
    <tr><th>max</th><td>7.900000</td><td>4.400000</td><td>6.900000</td><td>2.500000</td></tr>
  </tbody>
</table>
</div>`
            },
            {
                // Cell 8
                comment: "Finding any null values in dataset",
                code: `# Finding any null values
iris_df.isnull().sum()`,
                output: `<div class="output-text">
sepal length (cm)    0<br>
sepal width (cm)     0<br>
petal length (cm)    0<br>
petal width (cm)     0<br>
label                0<br>
dtype: int64
</div>`
            },
            {
                // Cell 9
                comment: "Plotting histogram distribution of data points with labelled classes",
                code: `# Plotting a histogram distribution of data points with labelled classes

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.ravel(), names):
    sns.histplot(
    data=iris_df, x=col, hue="label",
    kde=True, element="step", ax=ax
    )
    ax.set_xlabel(col, fontweight='bold')
    ax.set_ylabel("Count", fontweight='bold')
plt.suptitle("Feature Distributions by Class", y=1.02, fontweight='bold')
plt.tight_layout()
plt.show()`,
                output: `<img src="images/Feature_Distribution_by_Class.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            },
            {
                // Cell 10
                comment: "Plotting correlation between features",
                code: `# Plotting correlation between features

plt.figure(figsize=(10, 8))
ax = sns.heatmap(
    iris_df.iloc[:, :4].corr(),
    annot=True,
    cmap="crest",
    fmt=".2f",
    linewidths=0.7,
    # Force annotation text to be black
    annot_kws={"color": "black"}
    )
plt.title("Feature Correlation Heatmap", fontweight='bold')
plt.show()`,
                output: `<img src="images/Feature_Correlation_Heatmap.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            }
        ]
    },
    // ========== STEP 4: DATA PREPROCESSING (Cells 11-17) ==========
    {
        title: "Data Preprocessing",
        blocks: [
            {
                // Cell 11
                comment: "Data Encoding - Adding target column",
                code: `# Data Preprocessing

# Data Encoding
# we already have encoded targets from sklearn datasets
iris_df["target"] = y
iris_df.head()`,
                output: `<div class="table-wrapper">
<table border="1" class="dataframe data-table">
  <thead>
    <tr style="text-align: right;">
      <th></th><th>sepal length (cm)</th><th>sepal width (cm)</th><th>petal length (cm)</th><th>petal width (cm)</th><th>label</th><th>target</th>
    </tr>
  </thead>
  <tbody>
    <tr><th>0</th><td>5.1</td><td>3.5</td><td>1.4</td><td>0.2</td><td>setosa</td><td>0</td></tr>
    <tr><th>1</th><td>4.9</td><td>3.0</td><td>1.4</td><td>0.2</td><td>setosa</td><td>0</td></tr>
    <tr><th>2</th><td>4.7</td><td>3.2</td><td>1.3</td><td>0.2</td><td>setosa</td><td>0</td></tr>
    <tr><th>3</th><td>4.6</td><td>3.1</td><td>1.5</td><td>0.2</td><td>setosa</td><td>0</td></tr>
    <tr><th>4</th><td>5.0</td><td>3.6</td><td>1.4</td><td>0.2</td><td>setosa</td><td>0</td></tr>
  </tbody>
</table>
</div>`
            },
            {
                // Cell 12
                comment: "Train/Test Split with stratification",
                code: `X = iris_df.iloc[:,:4].values
y = iris_df["target"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, 
    # Stratification is important to balance classes between split
    stratify=y,
    random_state=42
)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)`,
                output: `<div class="output-text">(105, 4) (45, 4) (105,) (45,)</div>`
            },
            {
                // Cell 13
                comment: "Initialize Standard Scaler",
                code: `# Scaling the values

scaler = StandardScaler()
scaler`,
                output: `<div class="output-text">StandardScaler()</div>`
            },
            {
                // Cell 14
                comment: "Apply scaling to training and test data",
                code: `X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("Values scaled with Standard Scaler")`,
                output: `<div class="output-success">Values scaled with Standard Scaler</div>`
            },
            {
                // Cell 15
                comment: "View original training data (before scaling)",
                code: `X_train[:5]`,
                output: `<div class="output-text">array([[5.1, 2.5, 3. , 1.1],<br>
       [6.2, 2.2, 4.5, 1.5],<br>
       [5.1, 3.8, 1.5, 0.3],<br>
       [6.8, 3.2, 5.9, 2.3],<br>
       [5.7, 2.8, 4.1, 1.3]])</div>`
            },
            {
                // Cell 16
                comment: "View scaled training data (after scaling)",
                code: `X_train_s[:5]`,
                output: `<div class="output-text">array([[-0.90045861, -1.22024754, -0.4419858 , -0.13661044],<br>
       [ 0.38036614, -1.87955796,  0.40282929,  0.38029394],<br>
       [-0.90045861,  1.63676428, -1.2868009 , -1.17041921],<br>
       [ 1.07899781,  0.31814344,  1.19132338,  1.41410271],<br>
       [-0.20182693, -0.56093712,  0.17754527,  0.12184175]])</div>`
            },
            {
                // Cell 17
                comment: "Binarize labels for ROC-AUC metric calculation",
                code: `# Binarizing encoded labels for ROC-AUC metric calculation
y_test_bin = label_binarize(y_test, classes=[0,1,2])
y_test_bin[:5]`,
                output: `<div class="output-text">array([[0, 0, 1],<br>
       [0, 1, 0],<br>
       [0, 0, 1],<br>
       [0, 1, 0],<br>
       [0, 0, 1]])</div>`
            }
        ]
    },
    // ========== STEP 5: MODEL TRAINING (Cell 18) ==========
    {
        title: "Model Training",
        blocks: [
            {
                // Cell 18
                comment: "Initialize and fit KNN model with k=10",
                code: `# Model Training

# k = 10 (Heuristic Rule)
model = KNeighborsClassifier(n_neighbors=10)
model.fit(X_train_s, y_train)`,
                output: `<div class="output-text">KNeighborsClassifier(n_neighbors=10)</div>`
            }
        ]
    },
    // ========== STEP 6: MODEL EVALUATION (Cells 19-22) ==========
    {
        title: "Model Evaluation",
        blocks: [
            {
                // Cell 19
                comment: "Generate predictions on test data",
                code: `# Model Evaluation

y_pred = model.predict(X_test_s)
y_pred`,
                output: `<div class="output-text">array([2, 1, 1, 1, 2, 2, 1, 1, 0, 2, 0, 0, 2, 2, 0, 2, 1, 0, 0, 0, 1, 0,<br>
       1, 2, 1, 1, 1, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 1, 2,<br>
       1])</div>`
            },
            {
                // Cell 20
                comment: "Get prediction probabilities",
                code: `y_prob = model.predict_proba(X_test_s)
y_prob[:5]`,
                output: `<div class="output-text">array([[0. , 0.1, 0.9],<br>
       [0. , 0.9, 0.1],<br>
       [0. , 0.7, 0.3],<br>
       [0. , 0.7, 0.3],<br>
       [0. , 0.4, 0.6]])</div>`
            },
            {
                // Cell 21
                comment: "Generate classification report",
                code: `print("\\nClassification Report (k=10):")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    digits=4
))`,
                output: `<div class="output-text" style="white-space: pre-wrap; font-family: monospace;">
Classification Report (k=10):
              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        15
  versicolor     0.8333    1.0000    0.9091        15
   virginica     1.0000    0.8000    0.8889        15

    accuracy                         0.9333        45
   macro avg     0.9444    0.9333    0.9327        45
weighted avg     0.9444    0.9333    0.9327        45
</div>`
            },
            {
                // Cell 22
                comment: "Visualize confusion matrix heatmap",
                code: `plt.figure(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens",
    xticklabels=target_names, yticklabels=target_names
    )
plt.xlabel("Predicted", fontweight='bold')
plt.ylabel("True", fontweight='bold')
plt.title("Confusion Matrix Heatmap (KNN, k=10)", fontweight='bold')
plt.tight_layout()
plt.show()`,
                output: `<img src="images/Confusion_Matrix_Heatmap_KNN_k_10.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            }
        ]
    },
    // ========== STEP 7: MODEL SIMULATION (Cell 23) ==========
    {
        title: "Model Simulation",
        blocks: [
            {
                // Cell 23
                comment: "Final Metrics - Accuracy, Precision, Recall, F1 and ROC-AUC",
                code: `# Final Metrics

# Accuracy will be same as F1 score as dataset is perfectly balanced and stratified
print(f'''Accuracy   : {accuracy_score(y_test, y_pred)}
Precision  : {precision_score(y_test, y_pred, average="macro")}
Recall     : {recall_score(y_test, y_pred, average="macro")}
F1         : {f1_score(y_test, y_pred, average="macro")}
ROC        : {roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")}
''')`,
                output: `<div class="output-text" style="white-space: pre-wrap; font-family: monospace;">Accuracy   : 0.9333333333333333
Precision  : 0.9444444444444445
Recall     : 0.9333333333333332
F1         : 0.9326599326599326
ROC        : 0.9948148148148149
</div>`
            }
        ]
    }
];

// ==========================================
// GLOBAL STATE
// ==========================================
let hasCompletedOnce = sessionStorage.getItem('knn_completed') === 'true';

let STEPS = [];
let EXPERIMENT_STATE = {
    stepIndex: 0,
    subStepIndex: 0,
    stepsStatus: []
};

let currentConfig = {
    metric: 'euclidean'
};

const EUCLIDEAN_STEPS = {
    training: {
        title: "Model Training",
        blocks: [{
            comment: "Initialize and fit KNN model with Euclidean distance (k=10)",
            code: `# Model Training

# k = 10 (Heuristic Rule)
model = KNeighborsClassifier(n_neighbors=10, metric='euclidean')
model.fit(X_train_s, y_train)
print("Model trained using Euclidean distance with k=10")`,
            output: `<div class="output-success">Model trained using Euclidean distance with k=10<br>KNeighborsClassifier(metric='euclidean', n_neighbors=10)</div>`
        }]
    },
    evaluation: {
        title: "Model Evaluation",
        blocks: [
            {
                comment: "Generate predictions on test data",
                code: `# Model Evaluation

y_pred = model.predict(X_test_s)
y_pred`,
                output: `<div class="output-text">array([2, 1, 1, 1, 2, 2, 1, 1, 0, 2, 0, 0, 2, 2, 0, 2, 1, 0, 0, 0, 1, 0,<br>
       1, 2, 1, 1, 1, 1, 1, 0, 2, 2, 1, 0, 2, 0, 0, 0, 0, 1, 1, 0, 1, 2,<br>
       1])</div>`
            },
            {
                comment: "Get prediction probabilities",
                code: `y_prob = model.predict_proba(X_test_s)
y_prob[:5]`,
                output: `<div class="output-text">array([[0. , 0.1, 0.9],<br>
       [0. , 0.9, 0.1],<br>
       [0. , 0.7, 0.3],<br>
       [0. , 0.7, 0.3],<br>
       [0. , 0.4, 0.6]])</div>`
            },
            {
                comment: "Generate classification report",
                code: `print("\\nClassification Report (Euclidean, k=10):")
print(classification_report(
    y_test,
    y_pred,
    target_names=target_names,
    digits=4
))`,
                output: `<div class="output-text"><pre style="font-family: monospace; margin: 0;">
Classification Report (Euclidean, k=10):
              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        15
  versicolor     0.8333    1.0000    0.9091        15
   virginica     1.0000    0.8000    0.8889        15

    accuracy                         0.9333        45
   macro avg     0.9444    0.9333    0.9327        45
weighted avg     0.9444    0.9333    0.9327        45
</pre></div>`
            },
            {
                comment: "Visualize confusion matrix heatmap",
                code: `plt.figure(figsize=(7, 5))
sns.heatmap(
    confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Greens",
    xticklabels=target_names, yticklabels=target_names
    )
plt.xlabel("Predicted", fontweight='bold')
plt.ylabel("True", fontweight='bold')
plt.title("Confusion Matrix Heatmap (KNN, Euclidean, k=10)", fontweight='bold')
plt.tight_layout()
plt.show()`,
                output: `<img src="images/Confusion_Matrix_Heatmap_KNN_k_10.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            }
        ]
    },
    simulation: {
        title: "Model Simulation",
        blocks: [
            {
                comment: "Final Metrics - Accuracy, Precision, Recall, F1 and ROC-AUC (Euclidean)",
                code: `# Final Metrics

# Accuracy will be same as F1 score as dataset is perfectly balanced and stratified
print(f'''Accuracy   : {accuracy_score(y_test, y_pred)}
Precision  : {precision_score(y_test, y_pred, average="macro")}
Recall     : {recall_score(y_test, y_pred, average="macro")}
F1         : {f1_score(y_test, y_pred, average="macro")}
ROC        : {roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="macro")}
''')`,
                output: `<div class="output-text" style="white-space: pre-wrap; font-family: monospace;">Accuracy   : 0.9333333333333333
Precision  : 0.9444444444444445
Recall     : 0.9333333333333332
F1         : 0.9326599326599326
ROC        : 0.9948148148148149
</div>`
            }
        ]
    }
};

// ==========================================
// INITIALIZATION
// ==========================================

document.addEventListener('DOMContentLoaded', () => {
    init();
});

// DOM Elements
let stepsContainer, codeDisplay, outputDisplay, runBtn, bottomPane;

function init() {
    stepsContainer = document.getElementById('stepsContainer');
    codeDisplay = document.getElementById('codeDisplay');
    outputDisplay = document.getElementById('outputDisplay');
    bottomPane = document.querySelector('.bottom-pane');
    runBtn = document.getElementById('runBtn');

    // Load all 7 steps from STEPS_KNN (23 cells mapped across 7 categories)
    STEPS = STEPS_KNN.map(step => ({ ...step }));
    STEPS[4] = { ...EUCLIDEAN_STEPS.training };
    STEPS[5] = { ...EUCLIDEAN_STEPS.evaluation };
    STEPS[6] = { ...EUCLIDEAN_STEPS.simulation };

    // Initialize State
    EXPERIMENT_STATE.stepIndex = 0;
    EXPERIMENT_STATE.subStepIndex = 0;
    EXPERIMENT_STATE.stepsStatus = STEPS.map((_, i) => ({ unlocked: i === 0, completed: false, partial: false }));

    renderSidebar();
    loadStep(0);
}

// Global exposure
window.selectDistanceMetric = selectDistanceMetric;
window.runStep = runStep;
window.nextSubStep = nextSubStep;
window.restartExperiment = restartExperiment;
window.showMetricSelector = showMetricSelector;

// Show Metric Selector
function showMetricSelector() {
    loadStep(4);
}

function selectDistanceMetric(metric) {
    currentConfig.metric = 'euclidean';
    STEPS[4] = { ...EUCLIDEAN_STEPS.training };
    STEPS[5] = { ...EUCLIDEAN_STEPS.evaluation };
    STEPS[6] = { ...EUCLIDEAN_STEPS.simulation };
    EXPERIMENT_STATE.stepsStatus[4].unlocked = true;
    renderSidebar();
    loadStep(4);
}

// Render Sidebar
function renderSidebar() {
    stepsContainer.innerHTML = '';

    STEPS.forEach((step, index) => {
        const status = EXPERIMENT_STATE.stepsStatus[index];
        const btn = document.createElement('button');
        btn.classList.add('step-btn');
        
        let label = `${index + 1}. ${step.title}`;
        if (status.completed) label = `✓ ${step.title}`;
        btn.textContent = label;

        if (status.unlocked) {
            if (status.completed) btn.classList.add('completed');
            else if (status.partial) btn.classList.add('in-progress');
            
            btn.disabled = false;
            btn.style.cursor = 'pointer';
            if (index === EXPERIMENT_STATE.stepIndex) btn.classList.add('active');
            btn.onclick = () => loadStep(index);
        } else {
            btn.classList.add('disabled');
            btn.disabled = true;
        }
        stepsContainer.appendChild(btn);
    });

    // Sidebar footer buttons
    const restartBtn = document.createElement('button');
    restartBtn.classList.add('step-btn', 'restart-btn');
    restartBtn.innerText = "Restart Experiment";
    restartBtn.style.textAlign = 'center';
    restartBtn.style.marginTop = "auto";
    restartBtn.style.backgroundColor = "#333";
    restartBtn.style.color = "white";
    restartBtn.onclick = restartExperiment;
    stepsContainer.appendChild(restartBtn);

    const downloadBtn = document.createElement('button');
    downloadBtn.classList.add('step-btn', 'download-btn');
    downloadBtn.style.textAlign = 'center';
    downloadBtn.style.marginTop = "10px";
    downloadBtn.style.marginBottom = "20px";
    
    // Check if all steps are completed (or were completed before a restart)
    const allStepsCompleted = hasCompletedOnce || (
        Array.isArray(EXPERIMENT_STATE.stepsStatus) &&
        EXPERIMENT_STATE.stepsStatus.length === STEPS.length &&
        EXPERIMENT_STATE.stepsStatus.every(status => status.completed)
    );
    
    if (allStepsCompleted) {
        downloadBtn.style.backgroundColor = "#F57C2A";
        downloadBtn.style.color = "white";
        downloadBtn.style.cursor = "pointer";
        downloadBtn.disabled = false;
        downloadBtn.onclick = downloadTrainingAsPDF;
    } else {
        downloadBtn.style.backgroundColor = "#f5f5f5";
        downloadBtn.style.color = "#9e9e9e";
        downloadBtn.style.cursor = "default";
        downloadBtn.style.border = "1px solid #e0e0e0";
        downloadBtn.disabled = false;
        downloadBtn.title = "Need to run the Experiment to download the pdf.";
        downloadBtn.onclick = function () {
            alert("Need to run the Experiment to download the pdf.");
        };
    }
    
    downloadBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"
            stroke-linejoin="round" style="margin-right:8px; vertical-align: middle;">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
            <polyline points="7 10 12 15 17 10"></polyline>
            <line x1="12" y1="15" x2="12" y2="3"></line>
        </svg>
        Download Experiment
    `;
    stepsContainer.appendChild(downloadBtn);
}

function loadStep(index) {
    EXPERIMENT_STATE.stepIndex = index;
    EXPERIMENT_STATE.subStepIndex = 0;
    renderSidebar();
    updateUI();
}

function restartExperiment() {
    EXPERIMENT_STATE.stepIndex = 0;
    EXPERIMENT_STATE.subStepIndex = 0;
    EXPERIMENT_STATE.stepsStatus = STEPS.map(() => ({ unlocked: false, completed: false, partial: false }));
    EXPERIMENT_STATE.stepsStatus[0].unlocked = true;

    bottomPane.classList.remove('active-output', 'completed-output');
    bottomPane.style.display = '';
    bottomPane.innerHTML = '<div class="output-content" id="outputDisplay"><div class="placeholder-text">Click the Run button to execute...</div></div>';
    outputDisplay = document.getElementById('outputDisplay');

    init();
}

function updateUI() {
    const step = STEPS[EXPERIMENT_STATE.stepIndex];
    if (!step || !step.blocks || step.blocks.length === 0) return;
    
    const block = step.blocks[EXPERIMENT_STATE.subStepIndex];
    if (!block) return;

    // Header Comment Logic (DT Style)
    let headerComment = "";
    let displayCode = block.code || "";
    
    if (block && block.comment) {
        headerComment = block.comment;
    } else if (block && block.code) {
        const commentMatch = block.code.match(/#\s*([^<\n\r]*)/);
        if (commentMatch) {
            headerComment = commentMatch[1].trim();
        }
    }
    // Clean code of comments
    displayCode = displayCode.split('\n').filter(line => !line.trim().startsWith('#')).join('\n').trim();

    const codeHeaderBar = document.getElementById('codeHeaderBar');
    if (codeHeaderBar) {
        if (headerComment) {
            codeHeaderBar.innerText = "# " + headerComment;
            codeHeaderBar.style.display = 'block';
        } else {
            codeHeaderBar.style.display = 'none';
        }
    }

    codeDisplay.innerHTML = highlightCode(displayCode);

    // Reset Output Pane
    bottomPane.classList.remove('active-output', 'completed-output');
    bottomPane.style.display = '';
    bottomPane.style.flexDirection = 'column';
    bottomPane.style.justifyContent = 'flex-start';
    bottomPane.style.alignItems = 'stretch';
    
    if (outputDisplay) {
        outputDisplay.innerHTML = '<div class="placeholder-text">Click the Run button to execute...</div>';
    }

    // Run Button
    runBtn.style.display = 'flex';
    runBtn.classList.remove('completed', 'arrow-mode');
    runBtn.style.backgroundColor = '#F57C2A';
    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>';
    runBtn.disabled = false;
    runBtn.onclick = runStep;
}

function runStep() {
    const step = STEPS[EXPERIMENT_STATE.stepIndex];
    const block = step.blocks[EXPERIMENT_STATE.subStepIndex];

    outputDisplay.innerHTML = '<div class="loading-spinner">Running code...</div>';
    runBtn.disabled = true;

    setTimeout(() => {
        outputDisplay.innerHTML = block.output;
        bottomPane.classList.add('active-output');

        runBtn.classList.add('completed');
        runBtn.style.backgroundColor = '#A6CE63';
        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"></polyline></svg>';

        EXPERIMENT_STATE.stepsStatus[EXPERIMENT_STATE.stepIndex].partial = true;
        renderSidebar();

        const hasNextBlock = EXPERIMENT_STATE.subStepIndex < step.blocks.length - 1;

        if (hasNextBlock) {
             setTimeout(() => {
                runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                runBtn.classList.remove('completed');
                runBtn.classList.add('arrow-mode');
                runBtn.style.backgroundColor = '#5FA8E4';
                runBtn.disabled = false;
                runBtn.onclick = nextSubStep;
             }, 500);
        } else {
            EXPERIMENT_STATE.stepsStatus[EXPERIMENT_STATE.stepIndex].completed = true;
            renderSidebar();

            if (EXPERIMENT_STATE.stepIndex < STEPS.length - 1) {
                EXPERIMENT_STATE.stepsStatus[EXPERIMENT_STATE.stepIndex + 1].unlocked = true;
                renderSidebar();
                
                // After Data Preprocessing (step 3), continue directly to Model Training
                if (EXPERIMENT_STATE.stepIndex === 3) {
                    setTimeout(() => {
                        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                        runBtn.classList.remove('completed');
                        runBtn.classList.add('arrow-mode');
                        runBtn.style.backgroundColor = '#5FA8E4';
                        runBtn.disabled = false;
                        runBtn.onclick = function() { loadStep(EXPERIMENT_STATE.stepIndex + 1); };
                    }, 500);
                } else {
                    setTimeout(() => {
                        runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                        runBtn.classList.remove('completed');
                        runBtn.classList.add('arrow-mode');
                        runBtn.style.backgroundColor = '#5FA8E4';
                        runBtn.disabled = false;
                        runBtn.onclick = function() { loadStep(EXPERIMENT_STATE.stepIndex + 1); };
                    }, 500);
                }
            } else {
                 setTimeout(() => {
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.classList.remove('completed');
                    runBtn.classList.add('arrow-mode');
                    runBtn.style.backgroundColor = '#5FA8E4';
                    runBtn.disabled = false;
                    runBtn.onclick = showCompletionMessage;
                 }, 500);
            }
        }
    }, 800);
}

function nextSubStep() {
    EXPERIMENT_STATE.subStepIndex++;
    updateUI();
}

function showCompletionMessage() {
    hasCompletedOnce = true;
    sessionStorage.setItem('knn_completed', 'true');
    const outputDisplay = document.getElementById('outputDisplay');
    const runBtn = document.getElementById('runBtn');

    outputDisplay.innerHTML = `
        <style>
            @keyframes clap {
                0%, 100% { transform: rotate(-15deg) scale(1); }
                50% { transform: rotate(15deg) scale(1.1); }
            }
            .clapping-hands {
                display: inline-block;
                font-size: 2.5rem;
                animation: clap 0.5s ease-in-out infinite;
                margin: 0 5px;
            }
        </style>
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 50vh; text-align: center; gap: 20px;">
            <div style="margin-bottom: 10px;">
                <span class="clapping-hands">👏</span>
                <span class="clapping-hands" style="animation-delay: 0.15s;">👏</span>
                <span class="clapping-hands" style="animation-delay: 0.3s;">👏</span>
            </div>
            <div>
                <h2 style="color: #3d8b8b; font-family: 'Courier New', monospace; font-size: 2rem; font-weight: bold; margin-bottom: 10px;">
                    Congratulations!
                </h2>
                <p style="color: #555; font-size: 1.1rem; font-family: 'Courier New', monospace; max-width: 600px;">You have successfully completed the K-Nearest Neighbours (KNN) experiment. You now understand how KNN performs classification, how the choice of k affects model performance.</p>
            </div>

            <button onclick="if(window.KNNAnimation){KNNAnimation.show()}else{alert('Animation not loaded')}" style="
                background: #1e293b;
                color: white;
                border: none;
                padding: 16px 32px;
                border-radius: 12px;
                font-weight: 700;
                font-size: 1rem;
                cursor: pointer;
                display: flex;
                align-items: center;
                gap: 12px;
                transition: all 0.2s;
                box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
            " onmouseover="this.style.background='#334155'; this.style.transform='translateY(-2px)'" 
               onmouseout="this.style.background='#1e293b'; this.style.transform='translateY(0)'">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg>
                Enter Interactive Animation
            </button>
        </div>
    `;
    runBtn.disabled = true;
    runBtn.style.backgroundColor = '#ccc';
    runBtn.style.cursor = 'default';
}

function highlightCode(code) {
  if (!code) return '';
  
  // First, protect strings from keyword replacement
  const stringPlaceholders = [];
  let protectedCode = code.replace(/(["'])(?:(?!\1)[^\\]|\\.)*\1/g, (match) => {
    stringPlaceholders.push(match);
    return `__STRING_${stringPlaceholders.length - 1}__`;
  });
  
  // Apply keyword highlighting (only on non-string parts)
  protectedCode = protectedCode
    .replace(/\bimport\b/g, '<span class="kw">import</span>')
    .replace(/\bfrom\b/g, '<span class="kw">from</span>')
    .replace(/\bas\b/g, '<span class="kw">as</span>')
    .replace(/\bprint\b/g, '<span class="func">print</span>')
    .replace(/#.*$/gm, match => `<span class="comment">${match}</span>`);
  
  // Restore strings with string highlighting
  protectedCode = protectedCode.replace(/__STRING_(\d+)__/g, (_, idx) => {
    return `<span class="string">${stringPlaceholders[parseInt(idx)]}</span>`;
  });
  
  return protectedCode;
}

function downloadTrainingAsPDF() {
    // Redirect to the PDF file for download
    window.open('assets/Exp-4.pdf', '_blank');
}

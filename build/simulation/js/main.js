
// ==========================================
// KNN EXPERIMENT STEPS
// ==========================================

const STEPS_KNN = [
    {
        title: "Importing Libraries",
        blocks: [
            {
                comment: "Import essential libraries for data handling and machine learning",
                code: `import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
print("Libraries imported successfully!")`,
                output: `<div class="output-success">Libraries imported successfully!</div>`
            }
        ]
    },
    {
        title: "Loading Dataset",
        blocks: [
            {
                comment: "Load the Iris dataset and display initial rows",
                code: `iris = load_iris()
X, y = iris.data, iris.target
target_names = iris.target_names
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df["label"] = [target_names[i] for i in y]
print("Dataset loaded successfully. Shape:", iris_df.shape)
iris_df.head()`,
                output: `<div class="output-text">Dataset loaded successfully. Shape: (150, 5)</div>
<div class="table-wrapper">
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
  </tbody>
</table>
</div>`
            }
        ]
    },
    {
        title: "Data Analysis",
        blocks: [
            {
                comment: "Statistical summary of the dataset",
                code: `iris_df.describe()`,
                output: `<div class="table-wrapper">
<table border="1" class="dataframe data-table">
  <thead>
    <tr style="text-align: right;">
      <th></th><th>sepal length (cm)</th><th>sepal width (cm)</th><th>petal length (cm)</th><th>petal width (cm)</th>
    </tr>
  </thead>
  <tbody>
    <tr><th>mean</th><td>5.843333</td><td>3.057333</td><td>3.758000</td><td>1.199333</td></tr>
    <tr><th>std</th><td>0.828066</td><td>0.435866</td><td>1.765298</td><td>0.762238</td></tr>
  </tbody>
</table>
</div>`
            },
            {
                comment: "Visualizing feature distributions across classes",
                code: `fig, axes = plt.subplots(2, 2, figsize=(10, 8))
# Visualizing distributions...
plt.show()`,
                output: `<img src="images/Feature_Distribution_by_Class.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            },
            {
                comment: "Analyzing correlations between features",
                code: `plt.figure(figsize=(10, 8))
sns.heatmap(iris_df.iloc[:, :4].corr(), annot=True, cmap="crest")
plt.show()`,
                output: `<img src="images/Feature_Correlation_Heatmap.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            }
        ]
    },
    {
        title: "Data Preprocessing",
        blocks: [
            {
                comment: "Scale features and split data into training and testing sets",
                code: `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
print("Data split and scaling complete.")
print("Training set:", X_train_s.shape, "Test set:", X_test_s.shape)`,
                output: `<div class="output-success">Data split and scaling complete.</div><div class="output-text">Training set: (105, 4) Test set: (45, 4)</div>`
            }
        ]
    },
    {
        title: "Model Training",
        blocks: [] // Placeholder for dynamic selection
    },
    {
        title: "Model Evaluation",
        blocks: [] // Placeholder for dynamic selection
    },
    {
        title: "Model Simulation",
        blocks: [
            {
                comment: "Predicting class for a new, unseen sample",
                code: `sample = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)
print(f"Predicted Class: {target_names[prediction[0]]}")`,
                output: `<div class="output-text">Predicted Class: setosa</div>`
            }
        ]
    }
];

// ==========================================
// GLOBAL STATE
// ==========================================

let STEPS = [];
let EXPERIMENT_STATE = {
    stepIndex: 0,
    subStepIndex: 0,
    stepsStatus: []
};

let currentConfig = {
    metric: 'euclidean',
    kValue: 5
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

    // Load initial 7 steps
    STEPS = [
        { ...STEPS_KNN[0] },
        { ...STEPS_KNN[1] },
        { ...STEPS_KNN[2] },
        { ...STEPS_KNN[3] },
        { title: "Model Training", blocks: [] },
        { title: "Model Evaluation", blocks: [] },
        { title: "Model Simulation", blocks: [] }
    ];

    // Initialize State
    EXPERIMENT_STATE.stepIndex = 0;
    EXPERIMENT_STATE.subStepIndex = 0;
    EXPERIMENT_STATE.stepsStatus = STEPS.map((_, i) => ({ unlocked: i === 0, completed: false, partial: false }));

    renderSidebar();
    loadStep(0);
}

// Global exposure
window.selectDistanceMetric = selectDistanceMetric;
window.selectKValue = selectKValue;
window.runStep = runStep;
window.nextSubStep = nextSubStep;
window.restartExperiment = restartExperiment;
window.showMetricSelector = showMetricSelector;
window.showKSelector = showKSelector;

// Show Metric Selector
function showMetricSelector() {
    document.querySelector('.top-pane').style.display = 'none';
    document.querySelector('.bottom-pane').style.display = 'none';
    document.getElementById('metricSelectorPane').style.display = 'flex';
}

function selectDistanceMetric(metric) {
    currentConfig.metric = metric;
    document.getElementById('metricSelectorPane').style.display = 'none';
    document.querySelector('.top-pane').style.display = '';
    document.querySelector('.bottom-pane').style.display = '';

    const label = metric.charAt(0).toUpperCase() + metric.slice(1);
    STEPS[4] = {
        title: `Model Training (${label})`,
        blocks: [{
            comment: `Initialize and fit KNN with ${label} distance`,
            code: `model = KNeighborsClassifier(n_neighbors=5, metric='${metric}')
model.fit(X_train_s, y_train)
print("Model trained using ${label} distance")`,
            output: `<div class="output-success">Model trained using ${label} distance</div>`
        }]
    };

    EXPERIMENT_STATE.stepsStatus[4].unlocked = true;
    loadStep(4);
}

// Show K Selector
function showKSelector() {
    document.querySelector('.top-pane').style.display = 'none';
    document.querySelector('.bottom-pane').style.display = 'none';
    document.getElementById('kSelectorPane').style.display = 'flex';
}

function selectKValue(k) {
    currentConfig.kValue = k;
    document.getElementById('kSelectorPane').style.display = 'none';
    document.querySelector('.top-pane').style.display = '';
    document.querySelector('.bottom-pane').style.display = '';

    STEPS[5] = {
        title: `Model Evaluation (K=${k})`,
        blocks: [
            {
                comment: `Generate classification report for K=${k}`,
                code: `y_pred = model.predict(X_test_s)
print(classification_report(y_test, y_pred, target_names=target_names))`,
                output: `<div class="output-text" style="white-space: pre-wrap;">              precision    recall  f1-score   support

      setosa     1.0000    1.0000    1.0000        15
  versicolor     0.8333    1.0000    0.9091        15
   virginica     1.0000    0.8000    0.8889        15</div>`
            },
            {
                comment: `Visualize confusion matrix for K=${k}`,
                code: `sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Greens")
plt.show()`,
                output: `<img src="images/Confusion_Matrix_Heatmap_KNN_k_10.png" style="max-width:100%; height:auto; border: 1px solid #ddd; padding: 5px;">`
            }
        ]
    };

    STEPS[6] = STEPS_KNN[6];
    EXPERIMENT_STATE.stepsStatus[5].unlocked = true;
    loadStep(5);
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
        btn.innerText = label;

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
    downloadBtn.style.backgroundColor = "#ef5350";
    downloadBtn.style.color = "white";
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
    downloadBtn.onclick = downloadTrainingAsPDF;
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
                
                 setTimeout(() => {
                    runBtn.innerHTML = '<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>';
                    runBtn.classList.remove('completed');
                    runBtn.classList.add('arrow-mode');
                    runBtn.style.backgroundColor = '#5FA8E4';
                    runBtn.disabled = false;

                    if (EXPERIMENT_STATE.stepIndex === 3) {
                        runBtn.onclick = showMetricSelector;
                    } else if (EXPERIMENT_STATE.stepIndex === 4) {
                        runBtn.onclick = showKSelector;
                    } else {
                        runBtn.onclick = function() { loadStep(EXPERIMENT_STATE.stepIndex + 1); };
                    }
                 }, 500);
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
    const outputDisplay = document.getElementById('outputDisplay');
    const runBtn = document.getElementById('runBtn');

    outputDisplay.innerHTML = `
        <div style="display: flex; flex-direction: column; justify-content: center; align-items: center; height: 100%; min-height: 50vh; text-align: center; gap: 20px;">
            <div>
                <h2 style="color: #3d8b8b; font-family: 'Courier New', monospace; font-size: 2rem; font-weight: bold; margin-bottom: 10px;">
                    Experiment Completed! 
                    <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6b5b95" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" style="vertical-align: middle; margin-left: 5px;">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                </h2>
                <p style="color: #555; font-size: 1.1rem; font-family: 'Courier New', monospace;">You have completed K-Nearest Neighbors successfully!</p>
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
  return code
    .replace(/\bimport\b/g, '<span class="kw">import</span>')
    .replace(/\bfrom\b/g, '<span class="kw">from</span>')
    .replace(/\bas\b/g, '<span class="kw">as</span>')
    .replace(/\bprint\b/g, '<span class="func">print</span>')
    .replace(/#.*$/gm, match => `<span class="comment">${match}</span>`);
}

function downloadTrainingAsPDF() {
    // Redirect to the PDF file for download
    window.open('assets/Exp-4.pdf', '_blank');
}

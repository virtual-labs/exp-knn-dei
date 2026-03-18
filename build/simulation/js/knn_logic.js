/**
 * KNN Visualizer - Vanilla JS Port
 * Ported from React App.jsx and Sidebar.jsx
 * Uses global window.CanvasRenderer from knn_canvas.js
 */

const CanvasRenderer = window.CanvasRenderer;

// ==================== STATE ====================
const STATE = {
    plotData: null,
    currentK: 6,
    currentSampleIdx: 0,
    sampleSizes: [],
    stepIncrement: 5,
    animationSpeed: 750,
    isAutoRunning: false,
    featureWindowOpen: false,
    zoomPanel: null,
    blurTrain: true,
    blurTest: false,
    selectedTestIdx: null,
    autoIntervalId: null,
    globalBounds: { x_min: -4.0, x_max: 4.0, y_min: -1.5, y_max: 1.5 },
    
    // Dynamic Config
    distanceMetric: 'euclidean', // 'euclidean' or 'manhattan'
    useScaling: false,
    
    // Client-computed results
    clientResults: null,
    
    cachedDimensions: {
        boundary: null,
        metrics: null,
        roc: null
    }
};

// ==================== DOM REFERENCES ====================
const DOM = {};

// ==================== UTILITY FUNCTIONS ====================

function generateSampleSizes(k, maxN, increment) {
    const sizes = [];
    // Always start at K as requested
    for (let n = k; n <= maxN; n += increment) {
        sizes.push(n);
    }
    if (sizes[sizes.length - 1] !== maxN) {
        sizes.push(maxN);
    }
    return sizes;
}

/**
 * Perform all KNN computations client-side based on current configuration
 * Updates STATE.clientResults
 */
function computeClientSideKNN() {
    if (!STATE.plotData) return;

    const engine = window.KNNEngine;
    const iris = engine.IRIS_RAW;
    const distFn = STATE.distanceMetric === 'euclidean' ? engine.euclideanDistance : engine.manhattanDistance;
    const totalTrainIndices = iris.trainIndices;
    const testIndices = iris.testIndices;
    const sampleSize = STATE.sampleSizes[STATE.currentSampleIdx];

    // 1. Prepare current train set (first N samples from the shuffle)
    const currentTrainIndices = totalTrainIndices.slice(0, sampleSize);
    let trainX = currentTrainIndices.map(idx => iris.features[idx]);
    let trainY = currentTrainIndices.map(idx => iris.targets[idx]);
    let testX = testIndices.map(idx => iris.features[idx]);
    let testY = testIndices.map(idx => iris.targets[idx]);

    // 2. Apply Scaling if enabled
    if (STATE.useScaling) {
        const scaled = engine.standardScale(trainX, testX);
        trainX = scaled.scaledTrain;
        testX = scaled.scaledTest;
    }

    // 3. Compute current predictions and metrics
    const result = engine.runKNN(trainX, trainY, testX, testY, STATE.currentK, distFn);
    
    // 4. Compute ROC curves
    const roc = engine.computeROCCurve(trainX, trainY, testX, testY, STATE.currentK, distFn);

    // 5. Compute Metrics History for the chart (Efficiently)
    const history = {
        samples: [],
        accuracy: [],
        precision: [],
        recall: [],
        f1: [],
        roc_auc: []
    };

    STATE.sampleSizes.forEach(n => {
        const nIndices = totalTrainIndices.slice(0, n);
        let nTrainX = nIndices.map(idx => iris.features[idx]);
        let nTrainY = nIndices.map(idx => iris.targets[idx]);
        let nTestX = testIndices.map(idx => iris.features[idx]);
        if (STATE.useScaling) {
            const nScaled = engine.standardScale(nTrainX, nTestX);
            nTrainX = nScaled.scaledTrain;
            nTestX = nScaled.scaledTest;
        }
        const nRes = engine.runKNN(nTrainX, nTrainY, nTestX, testY, STATE.currentK, distFn);
        
        // For the chart, we also need ROC-AUC
        const nRoc = engine.computeROCCurve(nTrainX, nTrainY, nTestX, testY, STATE.currentK, distFn);
        const avgAuc = Object.values(nRoc).reduce((acc, c) => acc + c.auc, 0) / 3;

        history.samples.push(n);
        history.accuracy.push(nRes.metrics.accuracy);
        history.precision.push(nRes.metrics.precision);
        history.recall.push(nRes.metrics.recall);
        history.f1.push(nRes.metrics.f1);
        history.roc_auc.push(avgAuc);
    });

    // 6. Map to local results object
    STATE.clientResults = {
        predictions: result.predictions,
        metrics: result.metrics,
        roc: roc,
        metrics_history: history,
        currentTrainIndices: currentTrainIndices
    };
}

async function loadPlotData() {
    try {
        console.log('[KNN] Fetching plot data...');
        const response = await fetch('assets/plot_data.json');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        console.log('[KNN] Data loaded successfully');
        
        // --- Pre-processing: Extract training coordinates into root for easy lookup ---
        // We take the largest precomputed split (70/30, K=6, Samples=105) 
        // because it contains the PCA and feature coordinates for ALL 105 training samples.
        let fullSplit = null;
        try {
            if (data.splits && data.splits['70_30'] && data.splits['70_30']['6']) {
                fullSplit = data.splits['70_30']['6']['105'];
            }
        } catch (e) {
            console.error('[KNN] Error searching for full split:', e);
        }
        
        if (fullSplit) {
            data.train_points_pca = fullSplit.pca.train;
            data.train_points_features = fullSplit.features.train;
            console.log('[KNN] Extracted training coordinates');
        } else {
            console.warn('[KNN] Could not find full training split in plot_data.json. Dynamic lookup may fail.');
            // Fallback: try to find any split and use its train points if they are large enough
            data.train_points_pca = [];
            data.train_points_features = {};
        }
        
        STATE.plotData = data;
        return data;
    } catch (error) {
        console.error('[KNN] Error loading plot data:', error);
        throw error;
    }
}

// ==================== RENDERING ====================

function renderAllCanvases() {
    if (!STATE.plotData) return;

    // Compute everything dynamic first
    computeClientSideKNN();

    const sampleSize = STATE.sampleSizes[STATE.currentSampleIdx];
    const results = STATE.clientResults;
    if (!results) return;

    // Use precomputed coordinates from plotData but map them to our dynamic labels
    const mappedPCA = {
        train: STATE.clientResults.currentTrainIndices.map((idx, i) => {
            // Use split-relative index i to look up coordinates from the pre-processed map
            return STATE.plotData.train_points_pca[i];
        }),
        predictions: results.predictions
    };

    // Render PCA boundary
    if (DOM.boundaryCanvas) {
        const canvas = DOM.boundaryCanvas;
        CanvasRenderer.renderDecisionBoundary(canvas, mappedPCA, {
            width: STATE.cachedDimensions.boundary?.width || canvas.parentElement.clientWidth,
            height: STATE.cachedDimensions.boundary?.height || canvas.parentElement.clientHeight,
            testPoints: STATE.plotData.test_points_pca,
            classNames: STATE.plotData.metadata.class_names,
            blurTrain: STATE.blurTrain,
            blurTest: STATE.blurTest,
            selectedTestIdx: STATE.selectedTestIdx,
            k: STATE.currentK,
            resolution: STATE.plotData.metadata.resolution,
            distanceMetric: STATE.distanceMetric
        });
    }

    // Render metrics evolution
    if (DOM.metricsCanvas) {
        CanvasRenderer.renderMetricsEvolution(DOM.metricsCanvas, results.metrics_history, sampleSize, {
            width: STATE.cachedDimensions.metrics?.width || DOM.metricsCanvas.parentElement.clientWidth,
            height: STATE.cachedDimensions.metrics?.height || DOM.metricsCanvas.parentElement.clientHeight
        });
    }

    // Render ROC curve
    if (DOM.rocCanvas) {
        CanvasRenderer.renderROCCurve(DOM.rocCanvas, results.roc, {
            width: STATE.cachedDimensions.roc?.width || DOM.rocCanvas.parentElement.clientWidth,
            height: STATE.cachedDimensions.roc?.height || DOM.rocCanvas.parentElement.clientHeight,
            classNames: STATE.plotData.metadata.class_names
        });
    }

    updateUIText();
    if (STATE.zoomPanel) renderZoomCanvas();
    if (STATE.featureWindowOpen) renderFeatureCanvases();
}

function updateToggleLabels() {
    const dContainer = document.querySelector('#distanceToggle')?.closest('.control-group');
    if (dContainer) {
        const left = dContainer.querySelector('.toggle-left-label');
        const right = dContainer.querySelector('.toggle-right-label');
        const active = STATE.distanceMetric === 'euclidean' ? left : right;
        dContainer.querySelectorAll('.toggle-label').forEach(l => l.classList.remove('active'));
        active.classList.add('active');
    }

    const sContainer = document.querySelector('#scalingToggle')?.closest('.control-group');
    if (sContainer) {
        const left = sContainer.querySelector('.toggle-left-label');
        const right = sContainer.querySelector('.toggle-right-label');
        const active = !STATE.useScaling ? left : right;
        sContainer.querySelectorAll('.toggle-label').forEach(l => l.classList.remove('active'));
        active.classList.add('active');
    }
}

function updateUIText() {
    const sampleSize = STATE.sampleSizes[STATE.currentSampleIdx] || 0;
    const trainTotal = STATE.plotData?.metadata?.train_total || 105;
    const sampleData = STATE.plotData?.splits['70_30']?.[String(STATE.currentK)]?.[String(sampleSize)];

    // Update sample value display
    if (DOM.sampleValue) DOM.sampleValue.textContent = sampleSize;
    if (DOM.trainTotal) DOM.trainTotal.textContent = trainTotal;

    // Update slider
    if (DOM.sampleSlider) {
        DOM.sampleSlider.max = STATE.sampleSizes.length - 1;
        DOM.sampleSlider.value = STATE.currentSampleIdx;
    }

    // Update metrics
    if (STATE.clientResults && STATE.clientResults.metrics) {
        const metrics = STATE.clientResults.metrics;
        const auc = Object.values(STATE.clientResults.roc).reduce((sum, r) => sum + r.auc, 0) / 3;

        if (DOM.metricAcc) DOM.metricAcc.textContent = metrics.accuracy.toFixed(4);
        if (DOM.metricPrec) DOM.metricPrec.textContent = metrics.precision.toFixed(4);
        if (DOM.metricRec) DOM.metricRec.textContent = metrics.recall.toFixed(4);
        if (DOM.metricF1) DOM.metricF1.textContent = metrics.f1.toFixed(4);
        if (DOM.metricAUC) DOM.metricAUC.textContent = auc.toFixed(4);
    }

    // Update K Slider/Value
    if (DOM.kSlider) DOM.kSlider.value = STATE.currentK;
    if (DOM.kValueDisplay) DOM.kValueDisplay.textContent = STATE.currentK;

    // Update Sample Slider
    if (DOM.sampleSlider) {
        // Sample count is from K to 105
        DOM.sampleSlider.min = 0;
        DOM.sampleSlider.max = STATE.sampleSizes.length - 1;
        DOM.sampleSlider.value = STATE.currentSampleIdx;
    }

    // Update toggle labels and active states
    updateToggleLabels();

    // Update speed display - show multiplier based on actual animation speed
    if (DOM.speedValue) {
        const speedMultiplier = (1000 / STATE.animationSpeed).toFixed(1);
        DOM.speedValue.textContent = `${speedMultiplier}x`;
    }

    // Update K buttons
    document.querySelectorAll('.k-btn').forEach(btn => {
        const k = parseInt(btn.dataset.k);
        btn.classList.toggle('active', k === STATE.currentK);
    });

    // Update auto-run button
    if (DOM.autoRunBtn) {
        DOM.autoRunBtn.textContent = STATE.isAutoRunning ? '⏸ PAUSE' : '▶ PLAY';
        DOM.autoRunBtn.classList.toggle('active', STATE.isAutoRunning);
    }

    // Update blur toggles - active means points are VISIBLE (not blurred)
    if (DOM.blurTrainBtn) DOM.blurTrainBtn.classList.toggle('active', !STATE.blurTrain);
    if (DOM.blurTestBtn) DOM.blurTestBtn.classList.toggle('active', !STATE.blurTest);

    // Also update feature window toggles
    const featureBlurTrainBtn = document.getElementById('featureBlurTrainBtn');
    const featureBlurTestBtn = document.getElementById('featureBlurTestBtn');
    if (featureBlurTrainBtn) featureBlurTrainBtn.classList.toggle('active', !STATE.blurTrain);
    if (featureBlurTestBtn) featureBlurTestBtn.classList.toggle('active', !STATE.blurTest);
}

// ==================== EVENT HANDLERS ====================

function handleKChange(newK) {
    if (STATE.currentK === newK || !STATE.plotData) return;
    STATE.currentK = newK;
    STATE.sampleSizes = generateSampleSizes(newK, STATE.plotData.metadata.train_total, STATE.stepIncrement);
    STATE.currentSampleIdx = 0;
    STATE.selectedTestIdx = null;
    renderAllCanvases();
}

function handleDistanceToggle(e) {
    if (e && e.target) {
        STATE.distanceMetric = e.target.value;
    } else {
        const checked = document.querySelector('input[name="distanceToggle"]:checked');
        STATE.distanceMetric = checked ? checked.value : 'euclidean';
    }
    renderAllCanvases();
}

function handleScalingToggle(e) {
    if (e && e.target) {
        STATE.useScaling = e.target.value === 'standard';
    } else {
        const checked = document.querySelector('input[name="scalingToggle"]:checked');
        STATE.useScaling = checked ? checked.value === 'standard' : false;
    }
    renderAllCanvases();
}

function handleSampleChange(newIdx) {
    STATE.currentSampleIdx = parseInt(newIdx);
    renderAllCanvases();
}

function handleIncSamples() {
    if (STATE.currentSampleIdx < STATE.sampleSizes.length - 1) {
        handleSampleChange(STATE.currentSampleIdx + 1);
    }
}

function handleDecSamples() {
    if (STATE.currentSampleIdx > 0) {
        handleSampleChange(STATE.currentSampleIdx - 1);
    }
}

function toggleAutoRun() {
    if (STATE.isAutoRunning) {
        clearInterval(STATE.autoIntervalId);
        STATE.isAutoRunning = false;
    } else {
        STATE.isAutoRunning = true;
        STATE.autoIntervalId = setInterval(() => {
            STATE.currentSampleIdx = (STATE.currentSampleIdx + 1) % STATE.sampleSizes.length;
            renderAllCanvases();
        }, STATE.animationSpeed);
    }
    updateUIText();
}

function handleSpeedChange(newSpeed) {
    // Invert the slider value: slider goes 100-2000, but we want right=fast (low ms)
    // So: animationSpeed = 2100 - sliderValue
    STATE.animationSpeed = 2100 - parseInt(newSpeed);
    
    // Restart interval if auto-running
    if (STATE.isAutoRunning) {
        clearInterval(STATE.autoIntervalId);
        STATE.autoIntervalId = setInterval(() => {
            STATE.currentSampleIdx = (STATE.currentSampleIdx + 1) % STATE.sampleSizes.length;
            renderAllCanvases();
        }, STATE.animationSpeed);
    }
    
    updateUIText();
}

function handleBlurTrainToggle() {
    STATE.blurTrain = !STATE.blurTrain;
    renderAllCanvases();
}

function handleBlurTestToggle() {
    STATE.blurTest = !STATE.blurTest;
    renderAllCanvases();
}

function handlePCACanvasClick(e) {
    e.stopPropagation();
    if (!DOM.boundaryCanvas || !STATE.plotData) return;

    const testPoints = STATE.plotData.test_points_pca;
    const clickedIdx = CanvasRenderer.findClickedTestPoint(DOM.boundaryCanvas, e.clientX, e.clientY, testPoints);

    if (clickedIdx !== null) {
        STATE.selectedTestIdx = STATE.selectedTestIdx === clickedIdx ? null : clickedIdx;
    } else {
        STATE.selectedTestIdx = null;
    }

    renderAllCanvases();
}

function handleFeatureWindowToggle() {
    STATE.featureWindowOpen = !STATE.featureWindowOpen;
    if (DOM.featureWindow) {
        DOM.featureWindow.style.display = STATE.featureWindowOpen ? 'flex' : 'none';
    }
    
    // Render feature canvases if opening
    if (STATE.featureWindowOpen) {
        renderFeatureCanvases();
    }
}

function handleZoomOpen(panelType) {
    STATE.zoomPanel = panelType;
    if (DOM.zoomModal) {
        DOM.zoomModal.style.display = 'flex';
        setTimeout(() => renderZoomCanvas(), 50);
    }
}

function handleZoomClose() {
    STATE.zoomPanel = null;
    if (DOM.zoomModal) {
        DOM.zoomModal.style.display = 'none';
    }
}

function handleBackToExperiment() {
    // Stop auto-run if active
    if (STATE.isAutoRunning) {
        clearInterval(STATE.autoIntervalId);
        STATE.isAutoRunning = false;
    }
    
    // Navigate back to mode selection instead of hiding entirely
    if (window.handleBackToModeSelection) {
        window.handleBackToModeSelection();
    } else {
        const container = document.getElementById('knnAnimationContainer');
        if (container) container.style.display = 'none';
    }
}

// ==================== FEATURE CANVASES ====================

function renderFeatureCanvases() {
    if (!STATE.plotData || !STATE.clientResults) return;

    const featurePairs = [
        ['f0', 'f1', 'Sepal Length vs Sepal Width'],
        ['f0', 'f2', 'Sepal Length vs Petal Length'],
        ['f0', 'f3', 'Sepal Length vs Petal Width'],
        ['f1', 'f2', 'Sepal Width vs Petal Length'],
        ['f1', 'f3', 'Sepal Width vs Petal Width'],
        ['f2', 'f3', 'Petal Length vs Petal Width']
    ];

    const results = STATE.clientResults;

    const labelMap = {
        'f0': 'Sepal Length',
        'f1': 'Sepal Width',
        'f2': 'Petal Length',
        'f3': 'Petal Width'
    };

    featurePairs.forEach(([f1, f2, title]) => {
        const key = `${f1}_${f2}`;
        const canvas = document.getElementById(`featureCanvas_${key}`);
        if (!canvas) return;

        // Use the dynamic train indices and predictions
        const mappedFeatureData = {
            train: results.currentTrainIndices.map((idx, i) => {
                // Look up coordinates in our pre-processed global feature map
                return STATE.plotData.train_points_features[key][i] || [0,0];
            }),
            predictions: results.predictions
        };

        CanvasRenderer.renderDecisionBoundary(canvas, mappedFeatureData, {
            width: 350,
            height: 280,
            testPoints: STATE.plotData.test_points_features[key],
            classNames: STATE.plotData.metadata.class_names,
            blurTrain: STATE.blurTrain,
            blurTest: STATE.blurTest,
            selectedTestIdx: STATE.selectedTestIdx,
            k: STATE.currentK,
            resolution: STATE.plotData.metadata.resolution,
            showBoundary: false,
            isFeaturePair: true,
            xAxisLabel: labelMap[f1],
            yAxisLabel: labelMap[f2],
            distanceMetric: STATE.distanceMetric
        });
    });
}

function renderZoomCanvas() {
    const width = 1200;
    const height = 900;
    
    DOM.zoomCanvas.style.width = '100%';
    DOM.zoomCanvas.style.height = '100%';

    // If it's a decision boundary plot (PCA or Feature Pair), we don't need sampleData
    if (STATE.zoomPanel === 'pca' || STATE.zoomPanel.startsWith('feature_')) {
        if (STATE.zoomPanel === 'pca') {
            const mappedPCA = {
                train: STATE.clientResults.currentTrainIndices.map((idx, i) => STATE.plotData.train_points_pca[i]),
                predictions: STATE.clientResults.predictions
            };
            CanvasRenderer.renderDecisionBoundary(DOM.zoomCanvas, mappedPCA, {
                width, height,
                testPoints: STATE.plotData.test_points_pca,
                classNames: STATE.plotData.metadata.class_names,
                blurTrain: STATE.blurTrain,
                blurTest: STATE.blurTest,
                selectedTestIdx: STATE.selectedTestIdx,
                k: STATE.currentK,
                showBoundary: false,
                resolution: STATE.plotData.metadata.resolution,
                distanceMetric: STATE.distanceMetric
            });
        } else if (STATE.zoomPanel.startsWith('feature_')) {
            const key = STATE.zoomPanel.replace('feature_', '');
            const [f1, f2] = key.split('_');
            const labelMap = {
                'f0': 'Sepal Length', 'f1': 'Sepal Width',
                'f2': 'Petal Length', 'f3': 'Petal Width'
            };
            const mappedFeatureData = {
                train: STATE.clientResults.currentTrainIndices.map((idx, i) => {
                    return STATE.plotData.train_points_features[key][i] || [0,0];
                }),
                predictions: STATE.clientResults.predictions
            };
            
            CanvasRenderer.renderDecisionBoundary(DOM.zoomCanvas, mappedFeatureData, {
                width: 1100, height: 800,
                testPoints: STATE.plotData.test_points_features[key],
                classNames: STATE.plotData.metadata.class_names,
                blurTrain: STATE.blurTrain,
                blurTest: STATE.blurTest,
                selectedTestIdx: STATE.selectedTestIdx,
                k: STATE.currentK,
                resolution: STATE.plotData.metadata.resolution,
                showBoundary: false,
                isFeaturePair: true,
                xAxisLabel: labelMap[f1],
                yAxisLabel: labelMap[f2],
                distanceMetric: STATE.distanceMetric
            });
        }
    } else {
        // For non-boundary plots (metrics, roc), we NEED sampleData
        const sampleSize = STATE.sampleSizes[STATE.currentSampleIdx];
        const sampleData = STATE.plotData.splits['70_30']?.[String(STATE.currentK)]?.[String(sampleSize)];
        if (!sampleData) return;

        if (STATE.zoomPanel === 'metrics') {
            CanvasRenderer.renderMetricsEvolution(DOM.zoomCanvas, STATE.clientResults.metrics_history, sampleSize, { width, height });
        } else if (STATE.zoomPanel === 'roc') {
            CanvasRenderer.renderROCCurve(DOM.zoomCanvas, STATE.clientResults.roc, { width, height, classNames: STATE.plotData.metadata.class_names });
        }
    }
}

// ==================== INITIALIZATION ====================

async function initKNNVisualizer() {
    // Get DOM references
    DOM.boundaryCanvas = document.getElementById('boundaryCanvas');
    DOM.metricsCanvas = document.getElementById('metricsCanvas');
    DOM.rocCanvas = document.getElementById('rocCanvas');
    DOM.kSlider = document.getElementById('kSlider');
    DOM.kValueDisplay = document.getElementById('kValueDisplay');
    DOM.decKBtn = document.getElementById('decKBtn');
    DOM.incKBtn = document.getElementById('incKBtn');

    DOM.zoomCanvas = document.getElementById('zoomCanvas');
    DOM.sampleValue = document.getElementById('sampleValue');
    DOM.trainTotal = document.getElementById('trainTotal');
    DOM.sampleSlider = document.getElementById('sampleSlider');
    DOM.metricAcc = document.getElementById('metricAcc');
    DOM.metricPrec = document.getElementById('metricPrec');
    DOM.metricRec = document.getElementById('metricRec');
    DOM.metricF1 = document.getElementById('metricF1');
    DOM.metricAUC = document.getElementById('metricAUC');
    DOM.speedValue = document.getElementById('speedValue');
    DOM.autoRunBtn = document.getElementById('autoRunBtn');
    DOM.speedSlider = document.getElementById('speedSlider');
    DOM.blurTrainBtn = document.getElementById('blurTrainBtn');
    DOM.blurTestBtn = document.getElementById('blurTestBtn');
    DOM.featureWindow = document.getElementById('featureWindow');
    DOM.zoomModal = document.getElementById('zoomModal');
    DOM.loadingScreen = document.getElementById('knnLoadingScreen');
    DOM.loadingProgress = document.getElementById('loadingProgress');
    DOM.loadingText = document.getElementById('loadingText');
    DOM.loadingPercent = document.getElementById('loadingPercent');

    // Show loading screen and start progress animation
    let progress = 0;
    let dataReady = false;

    const updateLoadingUI = () => {
        if (DOM.loadingProgress) DOM.loadingProgress.style.width = `${Math.min(100, progress)}%`;
        if (DOM.loadingPercent) DOM.loadingPercent.textContent = `${Math.floor(progress)}%`;
        if (DOM.loadingText) {
            DOM.loadingText.textContent = 
                progress < 30 ? 'Initializing core...' :
                progress < 60 ? 'Loading datasets...' :
                progress < 90 ? 'Computing neighbors...' : 'Ready!';
        }
    };

    const progressInterval = setInterval(() => {
        // 1. Initial fast phase
        if (progress < 60) {
            progress += Math.random() * 3 + 1;
        }
        // 2. Slow down phase
        else if (progress < 85) {
            progress += Math.random() * 0.5;
        }
        // 3. Waiting for data phase
        else if (!dataReady && progress < 98) {
            progress += 0.05;
        }
        // 4. Data ready - fast finish
        else if (dataReady) {
            progress += 4;
        }

        updateLoadingUI();

        if (progress >= 100) {
            clearInterval(progressInterval);
            // Hide loading screen after short delay
            setTimeout(() => {
                if (DOM.loadingScreen) {
                    DOM.loadingScreen.classList.add('hidden');
                }
            }, 200);
        }
    }, 30);

    // Load data
    try {
        const data = await loadPlotData();
        dataReady = true;  // Signal that data is ready
        
        if (!data || !data.metadata) {
            throw new Error('Invalid data format: metadata missing');
        }

        const k = data.metadata.k_values[0];
        const sizes = generateSampleSizes(k, data.metadata.train_total, 5);

        STATE.currentK = k;
        STATE.sampleSizes = sizes;
        STATE.currentSampleIdx = 0;

        // Attach event listeners
        attachEventListeners();

        // Initial render
        renderAllCanvases();
    } catch (error) {
        console.error('Failed to initialize KNN Visualizer:', error);
        clearInterval(progressInterval);
        if (DOM.loadingText) DOM.loadingText.textContent = 'Error: ' + error.message;
    }
}

function attachEventListeners() {
    // K Slider
    if (DOM.kSlider) {
        DOM.kSlider.addEventListener('input', (e) => handleKChange(parseInt(e.target.value)));
    }
    if (DOM.decKBtn) DOM.decKBtn.addEventListener('click', () => handleKChange(Math.max(4, STATE.currentK - 1)));
    if (DOM.incKBtn) DOM.incKBtn.addEventListener('click', () => handleKChange(Math.min(15, STATE.currentK + 1)));

    // Toggles
    document.querySelectorAll('input[name="distanceToggle"]').forEach(radio => {
        radio.addEventListener('change', handleDistanceToggle);
    });
    document.querySelectorAll('input[name="scalingToggle"]').forEach(radio => {
        radio.addEventListener('change', handleScalingToggle);
    });

    // K value buttons (Legacy support if any exist)
    document.querySelectorAll('.k-btn').forEach(btn => {
        btn.addEventListener('click', () => handleKChange(parseInt(btn.dataset.k)));
    });

    // Sample slider
    if (DOM.sampleSlider) {
        DOM.sampleSlider.addEventListener('input', (e) => handleSampleChange(e.target.value));
    }

    // Sample increment/decrement buttons
    const incBtn = document.getElementById('incSamplesBtn');
    const decBtn = document.getElementById('decSamplesBtn');
    if (incBtn) incBtn.addEventListener('click', handleIncSamples);
    if (decBtn) decBtn.addEventListener('click', handleDecSamples);

    // Auto-run button
    if (DOM.autoRunBtn) {
        DOM.autoRunBtn.addEventListener('click', toggleAutoRun);
    }

    // Speed slider
    if (DOM.speedSlider) {
        DOM.speedSlider.addEventListener('input', (e) => handleSpeedChange(e.target.value));
    }

    // Blur toggles
    if (DOM.blurTrainBtn) {
        DOM.blurTrainBtn.addEventListener('click', handleBlurTrainToggle);
    }
    if (DOM.blurTestBtn) {
        DOM.blurTestBtn.addEventListener('click', handleBlurTestToggle);
    }

    // PCA canvas click
    if (DOM.boundaryCanvas) {
        DOM.boundaryCanvas.addEventListener('click', handlePCACanvasClick);
    }

    // Feature window toggle
    const featureBtn = document.getElementById('featureBtn');
    if (featureBtn) {
        featureBtn.addEventListener('click', handleFeatureWindowToggle);
    }

    // Feature window close
    const featureCloseBtn = document.getElementById('featureCloseBtn');
    const featureMinBtn = document.getElementById('featureMinBtn');
    if (featureCloseBtn) featureCloseBtn.addEventListener('click', handleFeatureWindowToggle);
    if (featureMinBtn) featureMinBtn.addEventListener('click', handleFeatureWindowToggle);

    // Zoom buttons
    document.querySelectorAll('.zoom-btn').forEach(btn => {
        btn.addEventListener('click', () => handleZoomOpen(btn.dataset.panel));
    });

    document.querySelectorAll('.card-zoom-btn').forEach(btn => {
        btn.addEventListener('click', () => handleZoomOpen(btn.dataset.panel));
    });

    // Zoom modal close
    const zoomCloseBtn = document.querySelector('.zoom-close');
    if (zoomCloseBtn) zoomCloseBtn.addEventListener('click', handleZoomClose);
    if (DOM.zoomModal) {
        DOM.zoomModal.addEventListener('click', (e) => {
            if (e.target === DOM.zoomModal) handleZoomClose();
        });
    }

    // Zoom canvas click (for point selection)
    if (DOM.zoomCanvas) {
        DOM.zoomCanvas.addEventListener('click', (e) => {
            e.stopPropagation();
            if (!STATE.plotData || !STATE.zoomPanel) return;

            let testPoints;
            let isFeaturePanel = false;
            let featureKey = null;
            
            if (STATE.zoomPanel === 'pca') {
                testPoints = STATE.plotData.test_points_pca;
            } else if (STATE.zoomPanel.startsWith('feature_')) {
                isFeaturePanel = true;
                featureKey = STATE.zoomPanel.replace('feature_', '');
                testPoints = STATE.plotData.test_points_features[featureKey];
            } else {
                return;
            }

            const clickedIdx = CanvasRenderer.findClickedTestPoint(DOM.zoomCanvas, e.clientX, e.clientY, testPoints);
            
            if (clickedIdx !== null) {
                // Synchronize global selection from zoom modal
                STATE.selectedTestIdx = STATE.selectedTestIdx === clickedIdx ? null : clickedIdx;
            } else {
                STATE.selectedTestIdx = null;
            }
            renderZoomCanvas();
            renderAllCanvases();
        });
    }

    // Back button
    const backBtn = document.getElementById('knnBackBtn');
    if (backBtn) {
        backBtn.addEventListener('click', handleBackToExperiment);
    }

    // Feature blur toggles (in feature window)
    const featureBlurTrainBtn = document.getElementById('featureBlurTrainBtn');
    const featureBlurTestBtn = document.getElementById('featureBlurTestBtn');
    if (featureBlurTrainBtn) featureBlurTrainBtn.addEventListener('click', handleBlurTrainToggle);
    if (featureBlurTestBtn) featureBlurTestBtn.addEventListener('click', handleBlurTestToggle);

    // Feature canvas clicks - each card has its own selection state
    const featurePairs = ['f0_f1', 'f0_f2', 'f0_f3', 'f1_f2', 'f1_f3', 'f2_f3'];
    featurePairs.forEach(key => {
        const canvas = document.getElementById(`featureCanvas_${key}`);
        if (!canvas) return;

        canvas.addEventListener('click', (e) => {
            e.stopPropagation();
            if (!STATE.plotData) return;

            const testPoints = STATE.plotData.test_points_features[key];
            const clickedIdx = CanvasRenderer.findClickedTestPoint(canvas, e.clientX, e.clientY, testPoints);

            if (clickedIdx !== null) {
                // Synchronize global selection
                STATE.selectedTestIdx = STATE.selectedTestIdx === clickedIdx ? null : clickedIdx;
            } else {
                STATE.selectedTestIdx = null;
            }

            renderAllCanvases();
        });
    });

    // Info buttons - header info
    const headerInfoBtn = document.getElementById('headerInfoBtn');
    const headerInfoPopup = document.getElementById('headerInfoPopup');
    const headerInfoClose = document.getElementById('headerInfoClose');
    
    if (headerInfoBtn && headerInfoPopup) {
        headerInfoBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            headerInfoPopup.classList.toggle('active');
        });
    }
    
    if (headerInfoClose && headerInfoPopup) {
        headerInfoClose.addEventListener('click', (e) => {
            e.stopPropagation();
            headerInfoPopup.classList.remove('active');
        });
    }

    // Info buttons - panel info buttons
    document.querySelectorAll('.panel-info-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const infoType = btn.dataset.info;
            const popup = btn.nextElementSibling;
            
            // Close all other popups first
            document.querySelectorAll('.panel-info-popup.active').forEach(p => {
                if (p !== popup) p.classList.remove('active');
            });
            
            if (popup) {
                popup.classList.toggle('active');
            }
        });
    });

    // Close popups when clicking outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.info-btn') && !e.target.closest('.info-popup')) {
            document.querySelectorAll('.info-popup.active').forEach(p => {
                p.classList.remove('active');
            });
        }
    });
}

// Make init function globally available
window.initKNNVisualizer = initKNNVisualizer;
// ==================== GLOBAL API (matches RF_ANIMATOR pattern) ====================
window.KNNAnimation = {
    show: function() {
        const container = document.getElementById('knnAnimationContainer');
        if (!container) {
            console.error('[KNN_ANIMATION] Container not found');
            return;
        }
        
        console.log('[KNN_ANIMATION] Showing mode selection');
        container.style.display = 'flex';
        
        // Route through mode selection instead of directly showing playground
        if (window.showModeSelectionLoading) {
            window.showModeSelectionLoading();
        } else {
            // Fallback: direct playground init
            initKNNVisualizer();
        }
    },
    
    // Direct playground init (called from mode selection after choice)
    showPlaygroundDirect: function() {
        const container = document.getElementById('knnAnimationContainer');
        if (container) container.style.display = 'flex';
        initKNNVisualizer();
    },
    
    hide: function () {
        const container = document.getElementById('knnAnimationContainer');
        if (container) {
            container.style.display = 'none';
        }
    }
};

console.log('[KNN] Animation API loaded');

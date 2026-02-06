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
    animationSpeed: 750,  // 2100 - 1350 (initial slider value)
    isAutoRunning: false,
    featureWindowOpen: false,
    zoomPanel: null,
    blurTrain: true,
    blurTest: false,
    selectedTestIdx: null,
    featureSelectedIdx: {},  // Local selection state per feature card
    autoIntervalId: null,
    globalBounds: { x_min: -4.0, x_max: 4.0, y_min: -1.5, y_max: 1.5 },
    // Cached canvas dimensions to prevent jittering
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
    const sizes = [k];
    let start = Math.ceil(k / increment) * increment;
    for (let n = start; n <= maxN; n += increment) {
        if (n > k) sizes.push(n);
    }
    if (sizes[sizes.length - 1] !== maxN) {
        sizes.push(maxN);
    }
    return sizes;
}

async function loadPlotData() {
    try {
        const response = await fetch('./assets/plot_data.json');
        if (!response.ok) throw new Error(`HTTP error! status: ${response.status}`);
        const data = await response.json();
        STATE.plotData = data;
        return data;
    } catch (error) {
        console.error('Error loading plot data:', error);
        throw error;
    }
}

// ==================== RENDERING ====================

function renderAllCanvases() {
    if (!STATE.plotData) return;

    const sampleSize = STATE.sampleSizes[STATE.currentSampleIdx];
    if (!sampleSize) return;

    const sampleData = STATE.plotData.splits['70_30']?.[String(STATE.currentK)]?.[String(sampleSize)];
    if (!sampleData) return;

    // Render PCA boundary
    if (DOM.boundaryCanvas) {
        const canvas = DOM.boundaryCanvas;
        const wrapper = canvas.parentElement;
        
        // Cache dimensions on first render to prevent jittering
        if (!STATE.cachedDimensions.boundary) {
            STATE.cachedDimensions.boundary = {
                width: wrapper.clientWidth || 800,
                height: wrapper.clientHeight || 600
            };
        }
        const { width, height } = STATE.cachedDimensions.boundary;

        canvas.style.width = '100%';
        canvas.style.height = '100%';

        CanvasRenderer.renderDecisionBoundary(canvas, sampleData.pca, {
            width,
            height,
            testPoints: STATE.plotData.test_points_pca,
            classNames: STATE.plotData.metadata.class_names,
            blurTrain: STATE.blurTrain,
            blurTest: STATE.blurTest,
            selectedTestIdx: STATE.selectedTestIdx,
            k: STATE.currentK,
            showBoundary: false,
            resolution: STATE.plotData.metadata.resolution,
            bounds: STATE.globalBounds
        });
    }

    // Render metrics evolution
    if (DOM.metricsCanvas) {
        const canvas = DOM.metricsCanvas;
        const wrapper = canvas.parentElement;
        
        // Cache dimensions on first render to prevent jittering
        if (!STATE.cachedDimensions.metrics) {
            STATE.cachedDimensions.metrics = {
                width: wrapper.clientWidth || 500,
                height: wrapper.clientHeight || 300
            };
        }
        const { width, height } = STATE.cachedDimensions.metrics;

        canvas.style.width = '100%';
        canvas.style.height = '100%';

        const metricsHistory = STATE.plotData.splits['70_30'][String(STATE.currentK)].metrics_history;
        CanvasRenderer.renderMetricsEvolution(canvas, metricsHistory, sampleSize, {
            width,
            height
        });
    }

    // Render ROC curve
    if (DOM.rocCanvas) {
        const canvas = DOM.rocCanvas;
        const wrapper = canvas.parentElement;
        
        // Cache dimensions on first render to prevent jittering
        if (!STATE.cachedDimensions.roc) {
            STATE.cachedDimensions.roc = {
                width: wrapper.clientWidth || 500,
                height: wrapper.clientHeight || 300
            };
        }
        const { width, height } = STATE.cachedDimensions.roc;

        canvas.style.width = '100%';
        canvas.style.height = '100%';

        CanvasRenderer.renderROCCurve(canvas, sampleData.roc, {
            width,
            height,
            classNames: STATE.plotData.metadata.class_names
        });
    }

    // Update UI text
    updateUIText();

    // Also render zoom canvas if open (for auto-pilot animation)
    if (STATE.zoomPanel) {
        renderZoomCanvas();
    }

    // Also render feature canvases if window is open
    if (STATE.featureWindowOpen) {
        renderFeatureCanvases();
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
    if (sampleData && sampleData.metrics) {
        const metrics = sampleData.metrics;
        if (DOM.metricAcc) DOM.metricAcc.textContent = metrics.accuracy.toFixed(4);
        if (DOM.metricPrec) DOM.metricPrec.textContent = metrics.precision.toFixed(4);
        if (DOM.metricRec) DOM.metricRec.textContent = metrics.recall.toFixed(4);
        if (DOM.metricF1) DOM.metricF1.textContent = metrics.f1.toFixed(4);
        if (DOM.metricAUC) DOM.metricAUC.textContent = metrics.roc_auc.toFixed(4);
    }

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

    if (STATE.isAutoRunning) {
        clearInterval(STATE.autoIntervalId);
        STATE.isAutoRunning = false;
    }

    STATE.currentK = newK;
    STATE.sampleSizes = generateSampleSizes(newK, STATE.plotData.metadata.train_total, STATE.stepIncrement);
    STATE.currentSampleIdx = 0;
    STATE.selectedTestIdx = null;

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
    const container = document.getElementById('knnAnimationContainer');
    if (container) {
        container.style.display = 'none';
    }
    
    // Stop auto-run if active
    if (STATE.isAutoRunning) {
        clearInterval(STATE.autoIntervalId);
        STATE.isAutoRunning = false;
    }
}

// ==================== FEATURE CANVASES ====================

function renderFeatureCanvases() {
    if (!STATE.plotData) return;

    const featurePairs = [
        ['f0', 'f1', 'Sepal Length vs Sepal Width'],
        ['f0', 'f2', 'Sepal Length vs Petal Length'],
        ['f0', 'f3', 'Sepal Length vs Petal Width'],
        ['f1', 'f2', 'Sepal Width vs Petal Length'],
        ['f1', 'f3', 'Sepal Width vs Petal Width'],
        ['f2', 'f3', 'Petal Length vs Petal Width']
    ];

    const sampleSize = STATE.sampleSizes[STATE.currentSampleIdx];
    if (!sampleSize) return;

    const sampleData = STATE.plotData.splits['70_30']?.[String(STATE.currentK)]?.[String(sampleSize)];
    if (!sampleData) return;

    featurePairs.forEach(([f1, f2, _]) => {
        const key = `${f1}_${f2}`;
        const canvas = document.getElementById(`featureCanvas_${key}`);
        if (!canvas) return;

        const wrapper = canvas.parentElement;
        const width = wrapper.clientWidth || 300;
        const height = wrapper.clientHeight || 250;

        canvas.style.width = '100%';
        canvas.style.height = '100%';

        // Access using correct data structure: features.boundaries[key]
        const boundaryData = sampleData.features?.boundaries?.[key];
        const trainPoints = sampleData.features?.train?.[key] || [];
        if (!boundaryData) return;

        // Use local selection state for this specific feature card
        const localSelectedIdx = STATE.featureSelectedIdx[key] ?? null;

        CanvasRenderer.renderDecisionBoundary(canvas, { boundary: boundaryData, train: trainPoints }, {
            width: 350,
            height: 280,
            testPoints: STATE.plotData.test_points_features[key],
            classNames: STATE.plotData.metadata.class_names,
            blurTrain: STATE.blurTrain,
            blurTest: STATE.blurTest,
            selectedTestIdx: localSelectedIdx,
            k: STATE.currentK,
            bounds: STATE.plotData.metadata.feature_bounds[key],
            resolution: STATE.plotData.metadata.resolution,
            showBoundary: false
        });
    });
}

function renderZoomCanvas() {
    if (!STATE.zoomPanel || !DOM.zoomCanvas || !STATE.plotData) return;

    const sampleSize = STATE.sampleSizes[STATE.currentSampleIdx];
    if (!sampleSize) return;

    const sampleData = STATE.plotData.splits['70_30']?.[String(STATE.currentK)]?.[String(sampleSize)];
    if (!sampleData) return;

    const width = 1200;
    const height = 900;

    DOM.zoomCanvas.style.width = '100%';
    DOM.zoomCanvas.style.height = '100%';

    if (STATE.zoomPanel === 'pca') {
        CanvasRenderer.renderDecisionBoundary(DOM.zoomCanvas, sampleData.pca, {
            width,
            height,
            testPoints: STATE.plotData.test_points_pca,
            classNames: STATE.plotData.metadata.class_names,
            blurTrain: STATE.blurTrain,
            blurTest: STATE.blurTest,
            selectedTestIdx: STATE.selectedTestIdx,
            k: STATE.currentK,
            showBoundary: false,
            resolution: STATE.plotData.metadata.resolution,
            bounds: STATE.globalBounds
        });
    } else if (STATE.zoomPanel === 'metrics') {
        const metricsHistory = STATE.plotData.splits['70_30'][String(STATE.currentK)].metrics_history;
        CanvasRenderer.renderMetricsEvolution(DOM.zoomCanvas, metricsHistory, sampleSize, {
            width,
            height
        });
    } else if (STATE.zoomPanel === 'roc') {
        CanvasRenderer.renderROCCurve(DOM.zoomCanvas, sampleData.roc, {
            width,
            height,
            classNames: STATE.plotData.metadata.class_names
        });
    } else if (STATE.zoomPanel.startsWith('feature_')) {
        const key = STATE.zoomPanel.replace('feature_', '');
        const boundaryData = sampleData.features?.boundaries?.[key];
        const trainPoints = sampleData.features?.train?.[key] || [];
        if (!boundaryData) return;

        // Use the feature-specific selection index for feature panels
        const selectedIdx = STATE.featureSelectedIdx[key] !== undefined 
            ? STATE.featureSelectedIdx[key] 
            : STATE.selectedTestIdx;

        CanvasRenderer.renderDecisionBoundary(DOM.zoomCanvas, { boundary: boundaryData, train: trainPoints }, {
            width: 1100,
            height: 800,
            testPoints: STATE.plotData.test_points_features[key],
            classNames: STATE.plotData.metadata.class_names,
            blurTrain: STATE.blurTrain,
            blurTest: STATE.blurTest,
            selectedTestIdx: selectedIdx,
            k: STATE.currentK,
            bounds: STATE.plotData.metadata.feature_bounds[key],
            resolution: STATE.plotData.metadata.resolution,
            showBoundary: false
        });
    }
}

// ==================== INITIALIZATION ====================

async function initKNNVisualizer() {
    // Get DOM references
    DOM.boundaryCanvas = document.getElementById('boundaryCanvas');
    DOM.metricsCanvas = document.getElementById('metricsCanvas');
    DOM.rocCanvas = document.getElementById('rocCanvas');
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
        if (DOM.loadingText) DOM.loadingText.textContent = 'Error loading data!';
    }
}

function attachEventListeners() {
    // K value buttons
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
            
            if (isFeaturePanel) {
                // Update feature-specific selection for feature panels
                if (clickedIdx !== null) {
                    STATE.featureSelectedIdx[featureKey] = STATE.featureSelectedIdx[featureKey] === clickedIdx ? null : clickedIdx;
                } else {
                    STATE.featureSelectedIdx[featureKey] = null;
                }
                renderZoomCanvas();
                renderFeatureCanvases();
            } else {
                // Update global selection for PCA panel
                if (clickedIdx !== null) {
                    STATE.selectedTestIdx = STATE.selectedTestIdx === clickedIdx ? null : clickedIdx;
                } else {
                    STATE.selectedTestIdx = null;
                }
                renderZoomCanvas();
                renderAllCanvases();
            }
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
                // Toggle selection for THIS specific feature card only
                STATE.featureSelectedIdx[key] = STATE.featureSelectedIdx[key] === clickedIdx ? null : clickedIdx;
            } else {
                // Clear selection for THIS feature card
                STATE.featureSelectedIdx[key] = null;
            }

            renderFeatureCanvases();
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
        
        console.log('[KNN_ANIMATION] Showing visualizer');
        container.style.display = 'flex';
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

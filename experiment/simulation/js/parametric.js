/**
 * Parametric Analysis of KNN
 * Client-side KNN implementation supporting:
 * - Variable K (6, 8, 10)
 * - Variable sample size (30, 60, 90, 105)
 * - Distance metrics (Euclidean, Manhattan)
 * - Feature scaling (None, Standard)
 * - Feature pair selection for 2D visualization
 * - Interactive point clicking with distance-aware lines
 */

const FEATURE_NAMES = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'];
const CLASS_NAMES = ['Setosa', 'Versicolor', 'Virginica'];

const ParametricState = {
    plotData: null,
    currentK: null,
    currentSamples: null,
    currentDistance: null,
    currentScaling: null,
    featureX: 0,  // Index into FEATURE_NAMES
    featureY: 1,
    trained: false,
    lastMetrics: null,
    // Computed plot data (after training)
    trainPoints: null,  // [{x, y, c, features}]
    testPoints: null,   // [{x, y, c, features}]
    trainX: null,       // Full feature vectors for KNN
    trainY: null,
    testX: null,
    testY: null,
    selectedTestIdx: null,
    listenersAttached: false
};

// IRIS_RAW and core KNN functionality now provided via window.KNNEngine from js/knn_engine.js
const IRIS_RAW = window.KNNEngine.IRIS_RAW;
const euclideanDistance = window.KNNEngine.euclideanDistance;
const manhattanDistance = window.KNNEngine.manhattanDistance;
const standardScale = window.KNNEngine.standardScale;
const runKNN = window.KNNEngine.runKNN.bind(window.KNNEngine);
const computeMetrics = window.KNNEngine.computeMetrics.bind(window.KNNEngine);
const findKNearestForPoint = window.KNNEngine.findKNearestForPoint.bind(window.KNNEngine);

// ==================== TRAIN WITH CURRENT CONFIG ====================

function trainParametricModel() {
    const k = ParametricState.currentK;
    const sampleCount = ParametricState.currentSamples;
    const distMetric = ParametricState.currentDistance;
    const scaling = ParametricState.currentScaling;
    const fxIdx = ParametricState.featureX;
    const fyIdx = ParametricState.featureY;

    if (!k || !sampleCount || !distMetric || !scaling) {
        alert("Please select all configuration options (K Value, Sample Size, Distance Metric, and Feature Scaling) before training the model.");
        return false;
    }

    // Get train/test data
    const trainIdx = IRIS_RAW.trainIndices.slice(0, sampleCount);
    const testIdx = IRIS_RAW.testIndices;

    let trainX = trainIdx.map(i => [...IRIS_RAW.features[i]]);
    let trainY = trainIdx.map(i => IRIS_RAW.targets[i]);
    let testX = testIdx.map(i => [...IRIS_RAW.features[i]]);
    let testY = testIdx.map(i => IRIS_RAW.targets[i]);

    // Apply scaling if needed
    if (scaling === 'standard') {
        const scaled = standardScale(trainX, testX);
        trainX = scaled.scaledTrain;
        testX = scaled.scaledTest;
    }

    // Select distance function
    const distFn = distMetric === 'manhattan' ? manhattanDistance : euclideanDistance;

    // Run KNN
    const result = runKNN(trainX, trainY, testX, testY, k, distFn);

    // Store computed data for plotting
    ParametricState.trainX = trainX;
    ParametricState.trainY = trainY;
    ParametricState.testX = testX;
    ParametricState.testY = testY;

    // Build 2D plot points using selected features
    ParametricState.trainPoints = trainX.map((features, i) => ({
        x: features[fxIdx],
        y: features[fyIdx],
        c: trainY[i],
        features: features
    }));

    ParametricState.testPoints = testX.map((features, i) => ({
        x: features[fxIdx],
        y: features[fyIdx],
        c: testY[i],
        features: features
    }));

    ParametricState.lastMetrics = result.metrics;
    ParametricState.trained = true;
    ParametricState.selectedTestIdx = null;

    updateParametricMetricsUI();
    renderFeaturePlot();
    return true;
}

// ==================== CONFIG SUMMARY (LIVE UPDATE) ====================

function updateConfigSummary() {
    const el = (id) => document.getElementById(id);
    const configK = el('configK');
    const configSamples = el('configSamples');
    const configDistance = el('configDistance');
    const configScaling = el('configScaling');

    if (configK) configK.textContent = ParametricState.currentK || '--';
    if (configSamples) configSamples.textContent = ParametricState.currentSamples || '--';
    
    if (configDistance) {
        if (ParametricState.currentDistance === 'euclidean') configDistance.textContent = 'Euclidean';
        else if (ParametricState.currentDistance === 'manhattan') configDistance.textContent = 'Manhattan';
        else configDistance.textContent = '--';
    }
    
    if (configScaling) {
        if (ParametricState.currentScaling === 'standard') configScaling.textContent = 'Standard';
        else if (ParametricState.currentScaling === 'none') configScaling.textContent = 'None';
        else configScaling.textContent = '--';
    }
}

// ==================== METRICS UI ====================

function updateParametricMetricsUI() {
    const metrics = ParametricState.lastMetrics;
    if (!metrics) return;

    const noDataMsg = document.getElementById('paramNoDataMsg');
    const chart = document.getElementById('paramMetricsChart');
    if (noDataMsg) noDataMsg.style.display = 'none';
    if (chart) chart.style.display = 'flex';

    const metricPairs = [
        ['paramBarAccuracy', 'paramValAccuracy', metrics.accuracy],
        ['paramBarPrecision', 'paramValPrecision', metrics.precision],
        ['paramBarRecall', 'paramValRecall', metrics.recall],
        ['paramBarF1', 'paramValF1', metrics.f1]
    ];

    function getBarColor(val) {
        if (val >= 0.95) return 'linear-gradient(90deg, #4CAF50, #81C784)';
        if (val >= 0.85) return 'linear-gradient(90deg, #66BB6A, #A5D6A7)';
        if (val >= 0.70) return 'linear-gradient(90deg, #FFA726, #FFB74D)';
        return 'linear-gradient(90deg, #EF5350, #E57373)';
    }

    metricPairs.forEach(([barId, valId, value]) => {
        const bar = document.getElementById(barId);
        const val = document.getElementById(valId);
        if (bar) {
            setTimeout(() => {
                bar.style.width = `${value * 100}%`;
                bar.style.background = getBarColor(value);
            }, 50);
        }
        if (val) val.textContent = value.toFixed(4);
    });
}

// ==================== FEATURE SPACE PLOT RENDERER ====================

function renderFeaturePlot() {
    const canvas = document.getElementById('paramBoundaryCanvas');
    if (!canvas) return;

    const CR = window.CanvasRenderer;
    if (!CR) return;

    const trainPoints = ParametricState.trainPoints;
    const testPoints = ParametricState.testPoints;
    if (!trainPoints || !testPoints) return;

    const fxIdx = ParametricState.featureX;
    const fyIdx = ParametricState.featureY;
    const selectedIdx = ParametricState.selectedTestIdx;

    const wrapper = canvas.parentElement;
    const width = wrapper.clientWidth || 600;
    const height = wrapper.clientHeight || 450;

    canvas.style.width = '100%';
    canvas.style.height = '100%';

    const ctx = CR.setupHighDPICanvas(canvas, width, height);

    const padding = { top: 30, right: 30, bottom: 50, left: 55 };
    const plotWidth = width - padding.left - padding.right;
    const plotHeight = height - padding.top - padding.bottom;

    // Clear
    ctx.fillStyle = CR.COLORS.background;
    ctx.fillRect(0, 0, width, height);

    // Compute bounds from all points
    const allPoints = [...trainPoints, ...testPoints];
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    allPoints.forEach(p => {
        if (p.x < minX) minX = p.x;
        if (p.x > maxX) maxX = p.x;
        if (p.y < minY) minY = p.y;
        if (p.y > maxY) maxY = p.y;
    });
    const padX = (maxX - minX) * 0.1 || 0.5;
    const padY = (maxY - minY) * 0.1 || 0.5;
    minX -= padX; maxX += padX;
    minY -= padY; maxY += padY;

    const scaleX = (x) => padding.left + ((x - minX) / (maxX - minX)) * plotWidth;
    const scaleY = (y) => padding.top + plotHeight - ((y - minY) / (maxY - minY)) * plotHeight;

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 0.5;
    for (let i = 0; i <= 20; i++) {
        const x = padding.left + (plotWidth / 20) * i;
        ctx.beginPath(); ctx.moveTo(x, padding.top); ctx.lineTo(x, padding.top + plotHeight); ctx.stroke();
        const y = padding.top + (plotHeight / 20) * i;
        ctx.beginPath(); ctx.moveTo(padding.left, y); ctx.lineTo(padding.left + plotWidth, y); ctx.stroke();
    }

    // Axes
    ctx.strokeStyle = CR.COLORS.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    // Axis tick labels
    ctx.fillStyle = CR.COLORS.text;
    ctx.font = CR.FONT.tick;
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
        const val = minX + (maxX - minX) * (i / 5);
        const x = padding.left + (plotWidth / 5) * i;
        ctx.fillText(val.toFixed(1), x, padding.top + plotHeight + 18);
    }
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
        const val = minY + (maxY - minY) * (i / 5);
        const y = padding.top + plotHeight - (plotHeight / 5) * i;
        ctx.fillText(val.toFixed(1), padding.left - 8, y + 4);
    }

    // Axis labels
    ctx.font = CR.FONT.label;
    ctx.fillStyle = CR.COLORS.text;
    ctx.textAlign = 'center';
    ctx.fillText(FEATURE_NAMES[fxIdx], padding.left + plotWidth / 2, height - 8);
    ctx.save();
    ctx.translate(14, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText(FEATURE_NAMES[fyIdx], 0, 0);
    ctx.restore();

    // Clip to plot area
    ctx.save();
    ctx.beginPath();
    ctx.rect(padding.left, padding.top, plotWidth, plotHeight);
    ctx.clip();

    // === DRAW ORDER: Arrows -> Train -> Test -> Tooltip ===

    // 1. Draw KNN lines if a test point is selected
    let selectedNeighbors = null;
    let selectedPointCoords = null;
    if (selectedIdx !== null && selectedIdx !== undefined && testPoints[selectedIdx]) {
        const sp = testPoints[selectedIdx];
        const distFn = ParametricState.currentDistance === 'manhattan' ? manhattanDistance : euclideanDistance;
        selectedNeighbors = findKNearestForPoint(
            ParametricState.testX[selectedIdx],
            ParametricState.trainX,
            ParametricState.trainY,
            ParametricState.currentK,
            distFn
        );
        selectedPointCoords = { x: scaleX(sp.x), y: scaleY(sp.y) };

        const drawLineFn = ParametricState.currentDistance === 'manhattan' ? CR.drawManhattanLine : CR.drawArrow;
        selectedNeighbors.forEach(neighbor => {
            const tp = trainPoints[neighbor.idx];
            drawLineFn(
                ctx,
                selectedPointCoords.x,
                selectedPointCoords.y,
                scaleX(tp.x),
                scaleY(tp.y),
                CR.COLORS.classes[tp.c]
            );
        });
    }

    // Set of neighbor indices for highlighting
    const neighborIndices = new Set();
    if (selectedNeighbors) selectedNeighbors.forEach(n => neighborIndices.add(n.idx));

    // 2. Draw training points (circles)
    const isSelected = selectedIdx !== null;
    trainPoints.forEach((p, idx) => {
        const px = scaleX(p.x), py = scaleY(p.y);
        const isNeighbor = neighborIndices.has(idx);
        const alpha = isSelected && !isNeighbor ? 0.15 : 1;
        const size = isNeighbor ? 8 : 5;
        const baseColor = CR.COLORS.classes[p.c];

        ctx.save();
        ctx.globalAlpha = alpha;

        // Gradient fill
        const grad = ctx.createRadialGradient(px, py - size * 0.25, 0, px, py, size);
        grad.addColorStop(0, CR.lightenColor(baseColor, 12));
        grad.addColorStop(0.7, baseColor);
        grad.addColorStop(1, CR.darkenColor(baseColor, 8));

        ctx.beginPath();
        ctx.arc(px, py, size, 0, Math.PI * 2);
        ctx.fillStyle = grad;
        ctx.fill();

        if (isNeighbor) {
            ctx.strokeStyle = 'rgba(255,255,255,0.8)';
            ctx.lineWidth = 2;
            ctx.stroke();
            ctx.beginPath();
            ctx.arc(px, py, size + 3, 0, Math.PI * 2);
            ctx.strokeStyle = 'rgba(255,255,255,0.25)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        ctx.restore();
    });

    // 3. Draw test points (diamond shape)
    testPoints.forEach((p, idx) => {
        const px = scaleX(p.x), py = scaleY(p.y);
        const isThisSelected = selectedIdx === idx;
        const alpha = isSelected && !isThisSelected ? 0.2 : 1;
        const size = isThisSelected ? 9 : 6;
        const baseColor = CR.COLORS.classes[p.c];

        ctx.save();
        ctx.globalAlpha = alpha;

        // Draw diamond shape
        ctx.beginPath();
        ctx.moveTo(px, py - size);
        ctx.lineTo(px + size, py);
        ctx.lineTo(px, py + size);
        ctx.lineTo(px - size, py);
        ctx.closePath();

        const grad = ctx.createRadialGradient(px, py - size * 0.3, 0, px, py, size);
        grad.addColorStop(0, CR.lightenColor(baseColor, 15));
        grad.addColorStop(0.6, baseColor);
        grad.addColorStop(1, CR.darkenColor(baseColor, 10));
        ctx.fillStyle = grad;
        ctx.fill();

        // White border
        ctx.strokeStyle = isThisSelected ? 'rgba(255,255,255,0.9)' : 'rgba(255,255,255,0.5)';
        ctx.lineWidth = isThisSelected ? 2.5 : 1.5;
        ctx.stroke();

        if (isThisSelected) {
            ctx.beginPath();
            ctx.moveTo(px, py - size - 4);
            ctx.lineTo(px + size + 4, py);
            ctx.lineTo(px, py + size + 4);
            ctx.lineTo(px - size - 4, py);
            ctx.closePath();
            ctx.strokeStyle = 'rgba(255,255,255,0.3)';
            ctx.lineWidth = 2;
            ctx.stroke();
        }
        ctx.restore();
    });

    // Restore clip
    ctx.restore();

    // 4. Draw tooltip for selected point
    if (selectedNeighbors && selectedPointCoords) {
        const counts = [0, 0, 0];
        selectedNeighbors.forEach(n => counts[n.label]++);
        let maxCount = -1, predictedClass = -1;
        counts.forEach((count, cIdx) => {
            if (count > maxCount) { maxCount = count; predictedClass = cIdx; }
        });

        // Two-line tooltip: Actual and Predicted
        const sp = testPoints[selectedIdx];
        const actualName = CLASS_NAMES[sp.c];
        const predName = CLASS_NAMES[predictedClass];
        const isCorrect = predictedClass === sp.c;

        const tx = selectedPointCoords.x;
        const ty = selectedPointCoords.y - 15;

        ctx.save();
        ctx.font = 'bold 11px Consolas, monospace';
        const line1 = `Actual: ${actualName}`;
        const line2 = `Pred:   ${predName}`;
        const w1 = ctx.measureText(line1).width;
        const w2 = ctx.measureText(line2).width;
        const tw = Math.max(w1, w2) + 20;
        const th = 40;
        const bx = tx - tw / 2;
        const by = ty - th;

        ctx.fillStyle = 'rgba(15, 15, 20, 0.9)';
        ctx.strokeStyle = isCorrect ? CR.COLORS.classes[predictedClass] : '#ff6b6b';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.roundRect(bx, by, tw, th, 6);
        ctx.fill();
        ctx.stroke();

        ctx.fillStyle = '#b0bec5';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(line1, tx, by + 14);
        ctx.fillStyle = isCorrect ? CR.COLORS.classes[predictedClass] : '#ff6b6b';
        ctx.fillText(line2, tx, by + 28);
        ctx.restore();
    }

    // 5. Draw legend
    const legendX = padding.left + plotWidth - 145;
    const legendY = padding.top + 10;

    ctx.fillStyle = 'rgba(13, 13, 13, 0.85)';
    ctx.beginPath();
    ctx.roundRect(legendX - 8, legendY - 8, 152, 95, 6);
    ctx.fill();

    ctx.font = CR.FONT.legend;

    // Class entries
    CLASS_NAMES.forEach((name, i) => {
        const y = legendY + i * 18;
        // Color circle
        ctx.fillStyle = CR.COLORS.classes[i];
        ctx.beginPath();
        ctx.arc(legendX + 6, y + 2, 4, 0, Math.PI * 2);
        ctx.fill();
        // Label
        ctx.fillStyle = CR.COLORS.text;
        ctx.textAlign = 'left';
        ctx.fillText(name, legendX + 16, y + 6);
    });

    // Train/Test markers legend
    const markerY = legendY + 58;
    // Train = circle
    ctx.fillStyle = '#888';
    ctx.beginPath();
    ctx.arc(legendX + 6, markerY + 2, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#888';
    ctx.textAlign = 'left';
    ctx.fillText('Train', legendX + 16, markerY + 6);

    // Test = diamond
    ctx.fillStyle = '#888';
    ctx.beginPath();
    ctx.moveTo(legendX + 76, markerY - 2);
    ctx.lineTo(legendX + 80, markerY + 2);
    ctx.lineTo(legendX + 76, markerY + 6);
    ctx.lineTo(legendX + 72, markerY + 2);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#888';
    ctx.fillText('Test', legendX + 86, markerY + 6);

    // Store scale info for click detection
    canvas._paramScaleInfo = {
        width, height, padding, plotWidth, plotHeight,
        scaleX, scaleY,
        minX, maxX, minY, maxY
    };
}

// ==================== CANVAS CLICK HANDLER ====================

function handleParamCanvasClick(e) {
    if (!ParametricState.trained || !ParametricState.testPoints) return;

    const canvas = document.getElementById('paramBoundaryCanvas');
    if (!canvas || !canvas._paramScaleInfo) return;

    const rect = canvas.getBoundingClientRect();
    const info = canvas._paramScaleInfo;
    const scaleFactorX = info.width / rect.width;
    const scaleFactorY = info.height / rect.height;

    const clickX = (e.clientX - rect.left) * scaleFactorX;
    const clickY = (e.clientY - rect.top) * scaleFactorY;

    const clickRadius = 20;
    let closestIdx = null;
    let closestDist = Infinity;

    ParametricState.testPoints.forEach((p, idx) => {
        const px = info.scaleX(p.x);
        const py = info.scaleY(p.y);
        const dist = Math.sqrt((clickX - px) ** 2 + (clickY - py) ** 2);
        if (dist < clickRadius && dist < closestDist) {
            closestDist = dist;
            closestIdx = idx;
        }
    });

    // Toggle selection
    if (closestIdx === ParametricState.selectedTestIdx) {
        ParametricState.selectedTestIdx = null;
    } else {
        ParametricState.selectedTestIdx = closestIdx;
    }

    renderFeaturePlot();
}

// ==================== INFO BUTTON HANDLERS ====================

function attachInfoButtonListeners() {
    document.querySelectorAll('#parametricAnalysisScreen .param-info-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.stopPropagation();
            const popup = btn.querySelector('.param-info-popup');
            if (!popup) return;

            // Close all other popups
            document.querySelectorAll('#parametricAnalysisScreen .param-info-popup.visible').forEach(p => {
                if (p !== popup) p.classList.remove('visible');
            });

            popup.classList.toggle('visible');
        });
    });

    // Close popups on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('.param-info-btn')) {
            document.querySelectorAll('#parametricAnalysisScreen .param-info-popup.visible').forEach(p => {
                p.classList.remove('visible');
            });
        }
    });
}

// ==================== FEATURE SELECTOR HANDLERS ====================

function attachFeatureSelectorListeners() {
    const xSelect = document.getElementById('paramXAxis');
    const ySelect = document.getElementById('paramYAxis');
    if (!xSelect || !ySelect) return;

    function syncDropdowns() {
        const xVal = parseInt(xSelect.value);
        const yVal = parseInt(ySelect.value);

        // Re-enable all options
        [xSelect, ySelect].forEach(sel => {
            Array.from(sel.options).forEach(opt => { opt.disabled = false; });
        });

        // Disable selected value in the other dropdown
        Array.from(ySelect.options).forEach(opt => {
            if (parseInt(opt.value) === xVal) opt.disabled = true;
        });
        Array.from(xSelect.options).forEach(opt => {
            if (parseInt(opt.value) === yVal) opt.disabled = true;
        });
    }

    xSelect.addEventListener('change', () => {
        ParametricState.featureX = parseInt(xSelect.value);
        syncDropdowns();
        if (ParametricState.trained) {
            // Rebuild plot points with new feature pair
            rebuildPlotPoints();
            renderFeaturePlot();
        }
    });

    ySelect.addEventListener('change', () => {
        ParametricState.featureY = parseInt(ySelect.value);
        syncDropdowns();
        if (ParametricState.trained) {
            rebuildPlotPoints();
            renderFeaturePlot();
        }
    });

    syncDropdowns();
}

function rebuildPlotPoints() {
    const fxIdx = ParametricState.featureX;
    const fyIdx = ParametricState.featureY;

    if (ParametricState.trainX) {
        ParametricState.trainPoints = ParametricState.trainX.map((features, i) => ({
            x: features[fxIdx], y: features[fyIdx], c: ParametricState.trainY[i], features
        }));
    }
    if (ParametricState.testX) {
        ParametricState.testPoints = ParametricState.testX.map((features, i) => ({
            x: features[fxIdx], y: features[fyIdx], c: ParametricState.testY[i], features
        }));
    }
    ParametricState.selectedTestIdx = null;
}

// ==================== EVENT LISTENERS ====================

function attachParametricListeners() {
    if (ParametricState.listenersAttached) return;
    ParametricState.listenersAttached = true;

    // K value buttons
    document.querySelectorAll('#paramKButtons .k-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#paramKButtons .k-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            ParametricState.currentK = parseInt(btn.dataset.k);
            updateConfigSummary();
        });
    });

    // Sample buttons
    document.querySelectorAll('#paramSampleButtons .param-sample-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('#paramSampleButtons .param-sample-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            ParametricState.currentSamples = parseInt(btn.dataset.samples);
            updateConfigSummary();
        });
    });

    // Distance metric radios
    document.querySelectorAll('input[name="paramDistance"]').forEach(radio => {
        radio.addEventListener('change', () => {
            ParametricState.currentDistance = radio.value;
            updateConfigSummary();
        });
    });

    // Feature scaling radios
    document.querySelectorAll('input[name="paramScaling"]').forEach(radio => {
        radio.addEventListener('change', () => {
            ParametricState.currentScaling = radio.value;
            updateConfigSummary();
        });
    });

    // Train button
    const trainBtn = document.getElementById('paramTrainBtn');
    if (trainBtn) {
        trainBtn.addEventListener('click', () => {
            if (!ParametricState.currentK || !ParametricState.currentSamples || !ParametricState.currentDistance || !ParametricState.currentScaling) {
                alert("Please select all configuration options before training.");
                return;
            }
            trainBtn.classList.add('training');
            trainBtn.innerHTML = '<span class="train-spinner"></span> TRAINING...';
            setTimeout(() => {
                trainParametricModel();
                trainBtn.classList.remove('training');
                trainBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="5 3 19 12 5 21 5 3"></polygon></svg> TRAIN MODEL`;
            }, 600);
        });
    }

    // Canvas click
    const canvas = document.getElementById('paramBoundaryCanvas');
    if (canvas) {
        canvas.addEventListener('click', handleParamCanvasClick);
    }

    // Feature selectors
    attachFeatureSelectorListeners();

    // Info buttons
    attachInfoButtonListeners();
}

// ==================== INITIALIZATION ====================

function initParametricAnalysis() {
    attachParametricListeners();
    updateConfigSummary();

    // Show no-data message initially
    const noDataMsg = document.getElementById('paramNoDataMsg');
    const chart = document.getElementById('paramMetricsChart');
    if (noDataMsg) noDataMsg.style.display = 'block';
    if (chart) chart.style.display = 'none';
}

// ==================== GLOBAL API ====================

window.ParametricAnalysis = {
    show: function() {
        const screen = document.getElementById('parametricAnalysisScreen');
        if (screen) screen.style.display = 'flex';
        initParametricAnalysis();
    },
    hide: function() {
        const screen = document.getElementById('parametricAnalysisScreen');
        if (screen) screen.style.display = 'none';
    }
};

console.log('[PARAMETRIC] Parametric Analysis module loaded');

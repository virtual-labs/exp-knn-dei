/**
 * KNN Animation - Vanilla JS Port
 * Ported from React App.jsx and CanvasRenderer.js
 */

// ==========================================
// CANVAS RENDERER
// ==========================================

const CanvasRenderer = (function() {
    
    // Color palette for dark theme
    const COLORS = {
        class0: '#4fc3f7', // Light blue
        class1: '#ffb74d', // Light orange  
        class2: '#81c784', // Light green
        classes: ['#4fc3f7', '#ffb74d', '#81c784'],
        classesAlpha: ['rgba(79, 195, 247, 0.35)', 'rgba(255, 183, 77, 0.35)', 'rgba(129, 199, 132, 0.35)'],
        background: '#0d0d0d',
        grid: '#1a1a1a',
        axis: '#555555',
        text: '#e0e0e0',
        textMuted: '#888888',
        diagonal: '#555555',
        arrow: '#ff5252'
    };

    // Font settings with Consolas
    const FONT = {
        title: 'bold 14px Consolas, Monaco, monospace',
        label: '12px Consolas, Monaco, monospace',
        tick: '10px Consolas, Monaco, monospace',
        legend: '11px Consolas, Monaco, monospace'
    };

    // Grid divisions (finer grid)
    const GRID_DIVISIONS = 100;

    /**
     * Set up high-DPI canvas
     */
    function setupHighDPICanvas(canvas, width, height) {
        const dpr = window.devicePixelRatio || 1;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = width + 'px';
        canvas.style.height = height + 'px';
        const ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return ctx;
    }

    /**
     * Calculate distance between two points
     */
    function distance(x1, y1, x2, y2) {
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2);
    }

    /**
     * Find K nearest neighbors from training points to a given point
     */
    function findKNearestNeighbors(targetPoint, trainPoints, k) {
        const distances = trainPoints.map((p, idx) => ({
            point: p,
            idx,
            dist: distance(targetPoint.x, targetPoint.y, p.x, p.y)
        }));

        distances.sort((a, b) => a.dist - b.dist);
        return distances.slice(0, k);
    }

    /**
     * Draw arrow from source to target
     */
    function drawArrow(ctx, fromX, fromY, toX, toY, color = COLORS.arrow) {
        const headLength = 5;
        const angle = Math.atan2(toY - fromY, toX - fromX);

        ctx.strokeStyle = color;
        ctx.fillStyle = color;
        ctx.lineWidth = 0.8;
        ctx.globalAlpha = 0.8;

        // Draw line
        ctx.beginPath();
        ctx.moveTo(fromX, fromY);
        ctx.lineTo(toX, toY);
        ctx.stroke();

        // Draw arrowhead
        ctx.beginPath();
        ctx.moveTo(toX, toY);
        ctx.lineTo(
            toX - headLength * Math.cos(angle - Math.PI / 6),
            toY - headLength * Math.sin(angle - Math.PI / 6)
        );
        ctx.lineTo(
            toX - headLength * Math.cos(angle + Math.PI / 6),
            toY - headLength * Math.sin(angle + Math.PI / 6)
        );
        ctx.closePath();
        ctx.fill();

        ctx.globalAlpha = 1;
    }

    /**
     * Render decision boundary plot on canvas
     */
    function renderDecisionBoundary(canvas, data, options = {}) {
        const width = options.width || 500;
        const height = options.height || 400;
        const ctx = setupHighDPICanvas(canvas, width, height);

        const padding = { top: 0, right: 0, bottom: 0, left: 0 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Clear canvas with dark background
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, width, height);

        const boundary = data.boundary;
        const trainPoints = data.train || [];
        const testPoints = options.testPoints || [];
        const blurTrain = options.blurTrain || false;
        const blurTest = options.blurTest || false;
        const selectedTestIdx = options.selectedTestIdx;
        const k = options.k || 6;

        // Scale functions
        const scaleX = (x) => padding.left + ((x - boundary.x_min) / (boundary.x_max - boundary.x_min)) * plotWidth;
        const scaleY = (y) => padding.top + plotHeight - ((y - boundary.y_min) / (boundary.y_max - boundary.y_min)) * plotHeight;

        // Draw decision boundary (filled contour)
        const grid = boundary.grid;
        const gridSize = grid.length;
        const cellWidth = plotWidth / (gridSize - 1);
        const cellHeight = plotHeight / (gridSize - 1);

        for (let i = 0; i < gridSize; i++) {
            for (let j = 0; j < gridSize; j++) {
                const classIdx = grid[i][j];
                ctx.fillStyle = COLORS.classesAlpha[classIdx];
                const x = padding.left + j * cellWidth;
                const y = padding.top + (gridSize - 1 - i) * cellHeight;
                ctx.fillRect(x, y, cellWidth + 1, cellHeight + 1);
            }
        }

        // Draw finer grid lines (30 divisions, more visible)
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        ctx.lineWidth = 0.5;
        const GRID_DIVISIONS_FINE = 30;

        for (let i = 0; i <= GRID_DIVISIONS_FINE; i++) {
            const x = padding.left + (plotWidth / GRID_DIVISIONS_FINE) * i;
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, padding.top + plotHeight);
            ctx.stroke();
        }

        for (let i = 0; i <= GRID_DIVISIONS_FINE; i++) {
            const y = padding.top + (plotHeight / GRID_DIVISIONS_FINE) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(padding.left + plotWidth, y);
            ctx.stroke();
        }

        // Draw training points (larger, with black edge)
        const trainAlpha = blurTrain ? 0.15 : 1;
        const pointSize = Math.max(4, Math.min(7, 9 - trainPoints.length / 15));
        trainPoints.forEach(p => {
            ctx.beginPath();
            ctx.arc(scaleX(p.x), scaleY(p.y), pointSize, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.classes[p.c];
            ctx.globalAlpha = trainAlpha;
            ctx.fill();
            ctx.strokeStyle = '#000000';
            ctx.lineWidth = 1.5;
            ctx.stroke();
            ctx.globalAlpha = 1;
        });

        // Draw test points (smaller, with white edge)
        const testAlpha = blurTest ? 0.15 : 1;
        testPoints.forEach((p, idx) => {
            const isSelected = selectedTestIdx === idx;
            ctx.beginPath();
            ctx.arc(scaleX(p.x), scaleY(p.y), isSelected ? 8 : 5, 0, Math.PI * 2);
            ctx.fillStyle = COLORS.classes[p.c];
            ctx.globalAlpha = isSelected ? 1 : testAlpha;
            ctx.fill();
            ctx.strokeStyle = isSelected ? '#ff5252' : '#ffffff';
            ctx.lineWidth = isSelected ? 2.5 : 1.5;
            ctx.stroke();
            ctx.globalAlpha = 1;
        });

        // Draw KNN arrows if a test point is selected
        if (selectedTestIdx !== undefined && selectedTestIdx !== null && testPoints[selectedTestIdx]) {
            const selectedPoint = testPoints[selectedTestIdx];
            const neighbors = findKNearestNeighbors(selectedPoint, trainPoints, k);

            neighbors.forEach(neighbor => {
                drawArrow(
                    ctx,
                    scaleX(selectedPoint.x),
                    scaleY(selectedPoint.y),
                    scaleX(neighbor.point.x),
                    scaleY(neighbor.point.y),
                    COLORS.classes[neighbor.point.c]
                );
            });
        }

        // Store scale functions on canvas for click detection
        canvas._scaleInfo = {
            padding,
            plotWidth,
            plotHeight,
            boundary,
            scaleX,
            scaleY,
            inverseScaleX: (px) => boundary.x_min + ((px - padding.left) / plotWidth) * (boundary.x_max - boundary.x_min),
            inverseScaleY: (py) => boundary.y_max - ((py - padding.top) / plotHeight) * (boundary.y_max - boundary.y_min)
        };
    }

    /**
     * Find which test point was clicked (if any)
     */
    function findClickedTestPoint(canvas, clickX, clickY, testPoints) {
        const rect = canvas.getBoundingClientRect();
        const x = (clickX - rect.left);
        const y = (clickY - rect.top);

        const scaleInfo = canvas._scaleInfo;
        if (!scaleInfo) return null;

        const clickRadius = 12;

        for (let i = 0; i < testPoints.length; i++) {
            const p = testPoints[i];
            const px = scaleInfo.scaleX(p.x);
            const py = scaleInfo.scaleY(p.y);

            if (distance(x, y, px, py) < clickRadius) {
                return i;
            }
        }

        return null;
    }

    /**
     * Render ROC curve plot on canvas
     */
    function renderROCCurve(canvas, data, options = {}) {
        const width = options.width || 400;
        const height = options.height || 350;
        const ctx = setupHighDPICanvas(canvas, width, height);

        const padding = { top: 40, right: 20, bottom: 50, left: 55 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Clear canvas with dark background
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, width, height);

        // Scale functions (0 to 1)
        const scaleX = (x) => padding.left + x * plotWidth;
        const scaleY = (y) => padding.top + plotHeight - y * plotHeight;

        // Draw finer grid
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 0.3;

        for (let i = 0; i <= GRID_DIVISIONS; i++) {
            const x = padding.left + (plotWidth / GRID_DIVISIONS) * i;
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, padding.top + plotHeight);
            ctx.stroke();

            const y = padding.top + (plotHeight / GRID_DIVISIONS) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(padding.left + plotWidth, y);
            ctx.stroke();
        }

        // Draw diagonal reference line
        ctx.strokeStyle = COLORS.diagonal;
        ctx.lineWidth = 1;
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        ctx.moveTo(scaleX(0), scaleY(0));
        ctx.lineTo(scaleX(1), scaleY(1));
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw axes
        ctx.strokeStyle = COLORS.axis;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, padding.top + plotHeight);
        ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
        ctx.stroke();

        // Tick labels
        ctx.fillStyle = COLORS.text;
        ctx.font = FONT.tick;
        ctx.textAlign = 'center';

        for (let i = 0; i <= 5; i++) {
            const val = i / 5;
            const x = padding.left + (plotWidth / 5) * i;
            ctx.fillText(val.toFixed(1), x, padding.top + plotHeight + 18);
        }

        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const val = i / 5;
            const y = padding.top + plotHeight - (plotHeight / 5) * i;
            ctx.fillText(val.toFixed(1), padding.left - 8, y + 4);
        }

        // Draw ROC curves for each class
        const classNames = options.classNames || ['Setosa', 'Versicolor', 'Virginica'];

        for (let c = 0; c < 3; c++) {
            const rocData = data[String(c)];
            if (!rocData || !rocData.fpr || !rocData.tpr) continue;

            ctx.strokeStyle = COLORS.classes[c];
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < rocData.fpr.length; i++) {
                const x = scaleX(rocData.fpr[i]);
                const y = scaleY(rocData.tpr[i]);
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();
        }

        // Title
        ctx.fillStyle = COLORS.text;
        ctx.font = FONT.title;
        ctx.textAlign = 'center';
        ctx.fillText('ROC Curve (Test Set)', width / 2, 20);

        // Axis labels
        ctx.font = FONT.label;
        ctx.fillText('False Positive Rate', width / 2, height - 5);

        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('True Positive Rate', 0, 0);
        ctx.restore();

        // Legend (positioned to avoid overlap - bottom right inside plot)
        const legendX = padding.left + plotWidth - 130;
        const legendY = padding.top + plotHeight - 60;

        // Legend background
        ctx.fillStyle = 'rgba(13, 13, 13, 0.85)';
        ctx.fillRect(legendX - 5, legendY - 12, 135, classNames.length * 16 + 8);

        ctx.font = FONT.legend;
        classNames.forEach((name, i) => {
            const rocData = data[String(i)];
            const aucText = rocData ? ` (AUC=${rocData.auc.toFixed(2)})` : '';
            const y = legendY + i * 16;

            ctx.strokeStyle = COLORS.classes[i];
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(legendX, y);
            ctx.lineTo(legendX + 20, y);
            ctx.stroke();

            ctx.fillStyle = COLORS.text;
            ctx.textAlign = 'left';
            ctx.fillText(name + aucText, legendX + 25, y + 4);
        });
    }

    /**
     * Render metrics evolution chart on canvas
     */
    function renderMetricsEvolution(canvas, data, currentSample, options = {}) {
        const width = options.width || 450;
        const height = options.height || 350;
        const ctx = setupHighDPICanvas(canvas, width, height);

        const padding = { top: 40, right: 110, bottom: 50, left: 55 };
        const plotWidth = width - padding.left - padding.right;
        const plotHeight = height - padding.top - padding.bottom;

        // Clear canvas with dark background
        ctx.fillStyle = COLORS.background;
        ctx.fillRect(0, 0, width, height);

        const samples = data.samples;
        const xMin = Math.min(...samples);
        const xMax = Math.max(...samples);

        // Scale functions
        const scaleX = (x) => padding.left + ((x - xMin) / (xMax - xMin)) * plotWidth;
        const scaleY = (y) => padding.top + plotHeight - y * plotHeight;

        // Draw finer grid
        ctx.strokeStyle = COLORS.grid;
        ctx.lineWidth = 0.3;

        for (let i = 0; i <= GRID_DIVISIONS; i++) {
            const x = padding.left + (plotWidth / GRID_DIVISIONS) * i;
            ctx.beginPath();
            ctx.moveTo(x, padding.top);
            ctx.lineTo(x, padding.top + plotHeight);
            ctx.stroke();

            const y = padding.top + (plotHeight / GRID_DIVISIONS) * i;
            ctx.beginPath();
            ctx.moveTo(padding.left, y);
            ctx.lineTo(padding.left + plotWidth, y);
            ctx.stroke();
        }

        // Draw axes
        ctx.strokeStyle = COLORS.axis;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(padding.left, padding.top);
        ctx.lineTo(padding.left, padding.top + plotHeight);
        ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
        ctx.stroke();

        // Tick labels
        ctx.fillStyle = COLORS.text;
        ctx.font = FONT.tick;
        ctx.textAlign = 'center';

        for (let i = 0; i <= 5; i++) {
            const val = xMin + (xMax - xMin) * (i / 5);
            const x = padding.left + (plotWidth / 5) * i;
            ctx.fillText(Math.round(val), x, padding.top + plotHeight + 18);
        }

        ctx.textAlign = 'right';
        for (let i = 0; i <= 5; i++) {
            const val = i / 5;
            const y = padding.top + plotHeight - (plotHeight / 5) * i;
            ctx.fillText(val.toFixed(1), padding.left - 8, y + 4);
        }

        // Metrics to plot
        const metrics = [
            { key: 'accuracy', color: '#4fc3f7', name: 'Accuracy' },
            { key: 'precision', color: '#ffb74d', name: 'Precision' },
            { key: 'recall', color: '#81c784', name: 'Recall' },
            { key: 'f1', color: '#f48fb1', name: 'F1 Score' },
            { key: 'roc_auc', color: '#ce93d8', name: 'ROC-AUC' }
        ];

        // Draw lines for each metric
        metrics.forEach(metric => {
            const values = data[metric.key];
            if (!values) return;

            ctx.strokeStyle = metric.color;
            ctx.lineWidth = 2;
            ctx.beginPath();

            for (let i = 0; i < samples.length; i++) {
                const x = scaleX(samples[i]);
                const y = scaleY(values[i]);
                if (i === 0) {
                    ctx.moveTo(x, y);
                } else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.stroke();

            // Draw points
            for (let i = 0; i < samples.length; i++) {
                ctx.beginPath();
                ctx.arc(scaleX(samples[i]), scaleY(values[i]), 3, 0, Math.PI * 2);
                ctx.fillStyle = metric.color;
                ctx.fill();
            }
        });

        // Draw vertical line at current sample
        if (currentSample) {
            const currentX = scaleX(currentSample);
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(currentX, padding.top);
            ctx.lineTo(currentX, padding.top + plotHeight);
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Title
        ctx.fillStyle = COLORS.text;
        ctx.font = FONT.title;
        ctx.textAlign = 'center';
        ctx.fillText('Testing Metrics Evolution', width / 2, 20);

        // Axis labels
        ctx.font = FONT.label;
        ctx.fillText('Training Samples', width / 2, height - 5);

        ctx.save();
        ctx.translate(15, height / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Score', 0, 0);
        ctx.restore();

        // Legend (right side)
        const legendX = width - padding.right + 10;
        const legendY = padding.top + 20;

        ctx.font = FONT.legend;
        metrics.forEach((metric, i) => {
            const y = legendY + i * 20;

            ctx.strokeStyle = metric.color;
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(legendX, y);
            ctx.lineTo(legendX + 15, y);
            ctx.stroke();

            ctx.beginPath();
            ctx.arc(legendX + 7.5, y, 3, 0, Math.PI * 2);
            ctx.fillStyle = metric.color;
            ctx.fill();

            ctx.fillStyle = COLORS.text;
            ctx.textAlign = 'left';
            ctx.fillText(metric.name, legendX + 20, y + 4);
        });
    }

    return {
        renderDecisionBoundary,
        renderROCCurve,
        renderMetricsEvolution,
        findClickedTestPoint,
        findKNearestNeighbors,
        COLORS
    };

})();


// ==========================================
// KNN ANIMATION CONTROLLER
// ==========================================

const KNNAnimation = (function() {
    
    // State
    const state = {
        data: null,
        currentK: 6,
        currentSampleIdx: 0,
        sampleSizes: [],
        stepIncrement: 10,
        isAutoRunning: false,
        autoIntervalId: null,
        selectedTestIdx: null,
        blurTrain: false,
        blurTest: false,
        featureWindowOpen: false,
        zoomPanel: null,
        initialized: false
    };

    // DOM Elements cache
    const els = {};

    // Feature pairs for visualization
    const FEATURE_PAIRS = [
        [0, 1], [0, 2], [0, 3],
        [1, 2], [1, 3],
        [2, 3]
    ];

    /**
     * Generate sample sizes array
     */
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

    /**
     * Initialize the animation
     */
    function init() {
        // Prevent multiple initializations
        if (state.initialized) {
            return;
        }
        
        // Cache DOM elements
        els.container = document.getElementById('knnAnimationContainer');
        els.boundaryCanvas = document.getElementById('boundaryCanvas');
        els.rocCanvas = document.getElementById('rocCanvas');
        els.metricsCanvas = document.getElementById('metricsCanvas');
        els.zoomCanvas = document.getElementById('zoomCanvas');
        
        // Sidebar elements
        els.kButtons = document.querySelectorAll('.knn-animation .k-btn');
        els.sampleSlider = document.getElementById('sampleSlider');
        els.sampleValue = document.getElementById('sampleValue');
        els.trainTotal = document.getElementById('trainTotal');
        els.autoRunBtn = document.getElementById('autoRunBtn');
        els.featureBtn = document.getElementById('featureBtn');
        
        // Blur toggles
        els.blurTrainBtn = document.getElementById('blurTrainBtn');
        els.blurTestBtn = document.getElementById('blurTestBtn');
        
        // Metrics display
        els.metricAcc = document.getElementById('metricAcc');
        els.metricPrec = document.getElementById('metricPrec');
        els.metricRec = document.getElementById('metricRec');
        els.metricF1 = document.getElementById('metricF1');
        els.metricAUC = document.getElementById('metricAUC');
        
        // Modals
        els.featureWindow = document.getElementById('featureWindow');
        els.zoomModal = document.getElementById('zoomModal');
        
        // Back button
        els.backBtn = document.getElementById('knnBackBtn');

        // Load data
        if (typeof KNN_PLOT_DATA !== 'undefined') {
            state.data = KNN_PLOT_DATA;
            setupEventListeners();
            initializeState();
            state.initialized = true;
        } else {
            console.error('KNN_PLOT_DATA not found!');
        }
    }

    /**
     * Initialize state with first K value
     */
    function initializeState() {
        const k = state.data.metadata.k_values[0];
        state.currentK = k;
        state.sampleSizes = generateSampleSizes(k, state.data.metadata.train_total, state.stepIncrement);
        state.currentSampleIdx = 0;
        
        // Update slider
        if (els.sampleSlider) {
            els.sampleSlider.max = state.sampleSizes.length - 1;
            els.sampleSlider.value = 0;
        }
        if (els.trainTotal) {
            els.trainTotal.textContent = state.data.metadata.train_total;
        }
        
        updateKButtons();
        render();
    }

    /**
     * Setup all event listeners
     */
    function setupEventListeners() {
        // K buttons
        els.kButtons.forEach(btn => {
            btn.addEventListener('click', () => {
                const k = parseInt(btn.dataset.k);
                handleKChange(k);
            });
        });

        // Sample slider
        if (els.sampleSlider) {
            els.sampleSlider.addEventListener('input', (e) => {
                handleSampleChange(parseInt(e.target.value));
            });
        }

        // Slider control buttons
        const decBtn = document.getElementById('decSamplesBtn');
        const incBtn = document.getElementById('incSamplesBtn');
        if (decBtn) decBtn.addEventListener('click', () => handleSampleChange(state.currentSampleIdx - 1));
        if (incBtn) incBtn.addEventListener('click', () => handleSampleChange(state.currentSampleIdx + 1));

        // Auto-run button
        if (els.autoRunBtn) {
            els.autoRunBtn.addEventListener('click', toggleAutoRun);
        }

        // Feature button
        if (els.featureBtn) {
            els.featureBtn.addEventListener('click', () => {
                state.featureWindowOpen = !state.featureWindowOpen;
                toggleFeatureWindow();
            });
        }

        // Blur toggles
        if (els.blurTrainBtn) {
            els.blurTrainBtn.addEventListener('click', () => {
                state.blurTrain = !state.blurTrain;
                els.blurTrainBtn.classList.toggle('active', !state.blurTrain);
                render();
            });
        }
        if (els.blurTestBtn) {
            els.blurTestBtn.addEventListener('click', () => {
                state.blurTest = !state.blurTest;
                els.blurTestBtn.classList.toggle('active', !state.blurTest);
                render();
            });
        }

        // PCA Canvas click for test point selection
        if (els.boundaryCanvas) {
            els.boundaryCanvas.addEventListener('click', (e) => {
                e.stopPropagation();
                handleCanvasClick(e, els.boundaryCanvas, state.data.test_points_pca);
            });
        }

        // Zoom buttons
        document.querySelectorAll('.knn-animation .zoom-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const panel = btn.dataset.panel;
                openZoom(panel);
            });
        });

        // Zoom modal close
        if (els.zoomModal) {
            els.zoomModal.addEventListener('click', (e) => {
                if (e.target === els.zoomModal) closeZoom();
            });
            const closeBtn = els.zoomModal.querySelector('.zoom-close');
            if (closeBtn) closeBtn.addEventListener('click', closeZoom);
        }

        // Feature window controls
        const featureClose = document.getElementById('featureCloseBtn');
        const featureMin = document.getElementById('featureMinBtn');
        if (featureClose) featureClose.addEventListener('click', () => { state.featureWindowOpen = false; toggleFeatureWindow(); });
        if (featureMin) featureMin.addEventListener('click', () => els.featureWindow.classList.toggle('minimized'));

        // Back button
        if (els.backBtn) {
            els.backBtn.addEventListener('click', hide);
        }

        // Window resize
        window.addEventListener('resize', render);
    }

    /**
     * Handle K value change
     */
    function handleKChange(newK) {
        if (state.currentK === newK) return;

        if (state.isAutoRunning) {
            clearInterval(state.autoIntervalId);
            state.isAutoRunning = false;
            updateRunButton();
        }

        state.currentK = newK;
        state.sampleSizes = generateSampleSizes(newK, state.data.metadata.train_total, state.stepIncrement);
        state.currentSampleIdx = 0;
        state.selectedTestIdx = null;

        if (els.sampleSlider) {
            els.sampleSlider.max = state.sampleSizes.length - 1;
            els.sampleSlider.value = 0;
        }

        updateKButtons();
        render();
    }

    /**
     * Handle sample slider change
     */
    function handleSampleChange(idx) {
        if (idx < 0 || idx >= state.sampleSizes.length) return;
        
        if (state.isAutoRunning) {
            clearInterval(state.autoIntervalId);
            state.isAutoRunning = false;
            updateRunButton();
        }

        state.currentSampleIdx = idx;
        if (els.sampleSlider) els.sampleSlider.value = idx;
        render();
    }

    /**
     * Toggle auto-run
     */
    function toggleAutoRun() {
        if (state.isAutoRunning) {
            clearInterval(state.autoIntervalId);
            state.isAutoRunning = false;
        } else {
            state.isAutoRunning = true;
            state.autoIntervalId = setInterval(() => {
                let next = state.currentSampleIdx + 1;
                if (next >= state.sampleSizes.length) next = 0;
                state.currentSampleIdx = next;
                if (els.sampleSlider) els.sampleSlider.value = next;
                render();
            }, 300);
        }
        updateRunButton();
    }

    /**
     * Update run button text
     */
    function updateRunButton() {
        if (els.autoRunBtn) {
            els.autoRunBtn.classList.toggle('active', state.isAutoRunning);
            els.autoRunBtn.textContent = state.isAutoRunning ? '⏸ Pause' : '▶ Play';
        }
    }

    /**
     * Update K button active states
     */
    function updateKButtons() {
        els.kButtons.forEach(btn => {
            const k = parseInt(btn.dataset.k);
            btn.classList.toggle('active', k === state.currentK);
        });
    }

    /**
     * Handle canvas click for test point selection
     */
    function handleCanvasClick(e, canvas, testPoints) {
        const clickedIdx = CanvasRenderer.findClickedTestPoint(canvas, e.clientX, e.clientY, testPoints);
        
        if (clickedIdx !== null) {
            state.selectedTestIdx = state.selectedTestIdx === clickedIdx ? null : clickedIdx;
        } else {
            state.selectedTestIdx = null;
        }
        render();
    }

    /**
     * Toggle feature window visibility
     */
    function toggleFeatureWindow() {
        if (els.featureWindow) {
            els.featureWindow.style.display = state.featureWindowOpen ? 'flex' : 'none';
            if (state.featureWindowOpen) {
                renderFeaturePairs();
            }
        }
    }

    /**
     * Open zoom modal
     */
    function openZoom(panel) {
        state.zoomPanel = panel;
        if (els.zoomModal) {
            els.zoomModal.style.display = 'flex';
            renderZoom();
        }
    }

    /**
     * Close zoom modal
     */
    function closeZoom() {
        state.zoomPanel = null;
        if (els.zoomModal) {
            els.zoomModal.style.display = 'none';
        }
    }

    /**
     * Main render function
     */
    function render() {
        if (!state.data) return;

        const currentSampleSize = state.sampleSizes[state.currentSampleIdx];
        if (!currentSampleSize) return;

        // Update sample display
        if (els.sampleValue) els.sampleValue.textContent = currentSampleSize;

        // Get current sample data
        const sampleData = state.data.splits['70_30']?.[String(state.currentK)]?.[String(currentSampleSize)];
        if (!sampleData) return;

        // Update metrics
        if (sampleData.metrics) {
            if (els.metricAcc) els.metricAcc.textContent = sampleData.metrics.accuracy?.toFixed(4) || '-';
            if (els.metricPrec) els.metricPrec.textContent = sampleData.metrics.precision?.toFixed(4) || '-';
            if (els.metricRec) els.metricRec.textContent = sampleData.metrics.recall?.toFixed(4) || '-';
            if (els.metricF1) els.metricF1.textContent = sampleData.metrics.f1?.toFixed(4) || '-';
            if (els.metricAUC) els.metricAUC.textContent = sampleData.metrics.roc_auc?.toFixed(4) || '-';
        }

        // Render PCA boundary
        if (els.boundaryCanvas) {
            const wrapper = els.boundaryCanvas.parentElement;
            CanvasRenderer.renderDecisionBoundary(els.boundaryCanvas, sampleData.pca, {
                width: Math.max(400, wrapper.clientWidth - 10),
                height: Math.max(350, wrapper.clientHeight - 10),
                testPoints: state.data.test_points_pca,
                classNames: state.data.metadata.class_names,
                blurTrain: state.blurTrain,
                blurTest: state.blurTest,
                selectedTestIdx: state.selectedTestIdx,
                k: state.currentK
            });
        }

        // Render metrics evolution
        if (els.metricsCanvas) {
            const wrapper = els.metricsCanvas.parentElement;
            const metricsHistory = state.data.splits['70_30'][String(state.currentK)].metrics_history;
            CanvasRenderer.renderMetricsEvolution(els.metricsCanvas, metricsHistory, currentSampleSize, {
                width: Math.max(350, wrapper.clientWidth - 10),
                height: Math.max(250, wrapper.clientHeight - 10)
            });
        }

        // Render ROC curve
        if (els.rocCanvas) {
            const wrapper = els.rocCanvas.parentElement;
            CanvasRenderer.renderROCCurve(els.rocCanvas, sampleData.roc, {
                width: Math.max(350, wrapper.clientWidth - 10),
                height: Math.max(250, wrapper.clientHeight - 10),
                classNames: state.data.metadata.class_names
            });
        }

        // Update feature pairs if open
        if (state.featureWindowOpen) {
            renderFeaturePairs();
        }

        // Update zoom if open
        if (state.zoomPanel) {
            renderZoom();
        }
    }

    /**
     * Render feature pairs in modal
     */
    function renderFeaturePairs() {
        const currentSampleSize = state.sampleSizes[state.currentSampleIdx];
        const sampleData = state.data.splits['70_30']?.[String(state.currentK)]?.[String(currentSampleSize)];
        if (!sampleData) return;

        FEATURE_PAIRS.forEach(([f1, f2]) => {
            const key = `f${f1}_f${f2}`;
            const canvas = document.getElementById(`featureCanvas_${key}`);
            if (!canvas) return;

            const boundaryData = {
                boundary: sampleData.features.boundaries[key],
                train: sampleData.features.train[key]
            };
            const testPoints = state.data.test_points_features[key];

            CanvasRenderer.renderDecisionBoundary(canvas, boundaryData, {
                width: 350,
                height: 280,
                testPoints,
                classNames: state.data.metadata.class_names,
                blurTrain: state.blurTrain,
                blurTest: state.blurTest,
                selectedTestIdx: state.selectedTestIdx,
                k: state.currentK
            });
        });
    }

    /**
     * Render zoom modal content
     */
    function renderZoom() {
        if (!els.zoomCanvas || !state.zoomPanel) return;

        const currentSampleSize = state.sampleSizes[state.currentSampleIdx];
        const sampleData = state.data.splits['70_30']?.[String(state.currentK)]?.[String(currentSampleSize)];
        if (!sampleData) return;

        if (state.zoomPanel === 'pca') {
            CanvasRenderer.renderDecisionBoundary(els.zoomCanvas, sampleData.pca, {
                width: 800,
                height: 600,
                testPoints: state.data.test_points_pca,
                classNames: state.data.metadata.class_names,
                blurTrain: state.blurTrain,
                blurTest: state.blurTest,
                selectedTestIdx: state.selectedTestIdx,
                k: state.currentK
            });
        } else if (state.zoomPanel === 'metrics') {
            const metricsHistory = state.data.splits['70_30'][String(state.currentK)].metrics_history;
            CanvasRenderer.renderMetricsEvolution(els.zoomCanvas, metricsHistory, currentSampleSize, {
                width: 800,
                height: 550
            });
        } else if (state.zoomPanel === 'roc') {
            CanvasRenderer.renderROCCurve(els.zoomCanvas, sampleData.roc, {
                width: 700,
                height: 550,
                classNames: state.data.metadata.class_names
            });
        } else if (state.zoomPanel.startsWith('feature_')) {
            const featureKey = state.zoomPanel.replace('feature_', '');
            const boundaryData = {
                boundary: sampleData.features.boundaries[featureKey],
                train: sampleData.features.train[featureKey]
            };
            const testPoints = state.data.test_points_features[featureKey];
            CanvasRenderer.renderDecisionBoundary(els.zoomCanvas, boundaryData, {
                width: 700,
                height: 550,
                testPoints,
                classNames: state.data.metadata.class_names,
                blurTrain: state.blurTrain,
                blurTest: state.blurTest,
                selectedTestIdx: state.selectedTestIdx,
                k: state.currentK
            });
        }
    }

    /**
     * Show the animation container
     */
    function show() {
        // Hide training section
        const container = document.querySelector('.container');
        if (container) container.style.display = 'none';
        
        // Show animation fullscreen
        if (els.container) {
            els.container.style.display = 'flex';
            setTimeout(() => {
                init();
                render();
            }, 100);
        }
    }

    /**
     * Hide the animation container
     */
    function hide() {
        if (state.isAutoRunning) {
            clearInterval(state.autoIntervalId);
            state.isAutoRunning = false;
        }
        if (els.container) {
            els.container.style.display = 'none';
        }
        
        // Show training section again
        const container = document.querySelector('.container');
        if (container) container.style.display = '';
    }

    return {
        init,
        show,
        hide
    };

})();

// Expose to global scope
window.KNNAnimation = KNNAnimation;

// Initialize on DOM load
document.addEventListener('DOMContentLoaded', () => {
    KNNAnimation.init();
});

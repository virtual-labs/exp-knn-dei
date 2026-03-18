/**
 * Mode Selection Flow Controller
 * Handles: Mode Selection Loading -> Mode Selection Screen -> Choice Loading -> Target Mode
 * 
 * Design: All new screens (mode selection, choice loading, parametric) are OVERLAYS
 * that sit above the playground elements. The playground header/content/loading are
 * always present in the container but covered by overlays when not active.
 */

// ==================== OVERLAY MANAGEMENT ====================
// Only manages our new overlay screens. Playground elements are NOT touched.

function hideAllOverlays() {
    const overlayIds = [
        'modeSelectionLoading',
        'modeSelectionScreen',
        'choiceLoadingScreen',
        'parametricAnalysisScreen'
    ];
    overlayIds.forEach(id => {
        const el = document.getElementById(id);
        if (el) el.style.display = 'none';
    });
}

function showOverlay(id) {
    const el = document.getElementById(id);
    if (el) el.style.display = 'flex';
}

// ==================== ANIMATED LOADING SCREEN ====================

function runLoadingAnimation(progressId, textId, percentId, messages, onComplete) {
    const progressEl = document.getElementById(progressId);
    const textEl = document.getElementById(textId);
    const percentEl = document.getElementById(percentId);

    // Reset
    if (progressEl) progressEl.style.width = '0%';
    if (percentEl) percentEl.textContent = '0%';
    if (textEl) textEl.textContent = messages[0] || 'Loading...';

    let progress = 0;
    const interval = setInterval(() => {
        progress += Math.random() * 4 + 2;
        if (progress > 100) progress = 100;

        if (progressEl) progressEl.style.width = `${progress}%`;
        if (percentEl) percentEl.textContent = `${Math.floor(progress)}%`;
        if (textEl) {
            if (progress < 30) textEl.textContent = messages[0] || 'Loading...';
            else if (progress < 70) textEl.textContent = messages[1] || 'Processing...';
            else textEl.textContent = messages[2] || 'Almost ready...';
        }

        if (progress >= 100) {
            clearInterval(interval);
            setTimeout(onComplete, 300);
        }
    }, 40);
}

// ==================== MODE SELECTION FLOW ====================

function showModeSelectionLoading() {
    hideAllOverlays();
    showOverlay('modeSelectionLoading');

    runLoadingAnimation(
        'modeLoadingProgress',
        'modeLoadingText',
        'modeLoadingPercent',
        ['Establishing connection...', 'Loading experiment modes...', 'Ready!'],
        () => {
            hideAllOverlays();
            showOverlay('modeSelectionScreen');
        }
    );
}

function showModeSelection() {
    hideAllOverlays();
    showOverlay('modeSelectionScreen');
}

// ==================== MODE CHOICE HANDLER ====================

function handleModeChoice(choice) {
    hideAllOverlays();
    showOverlay('choiceLoadingScreen');

    const titleEl = document.getElementById('choiceLoadingTitle');

    if (choice === 'parametric') {
        if (titleEl) titleEl.textContent = 'Launching Parametric Analysis';
        runLoadingAnimation(
            'choiceLoadingProgress',
            'choiceLoadingText',
            'choiceLoadingPercent',
            ['Configuring analysis environment...', 'Loading dataset parameters...', 'Initializing visualizer...'],
            () => {
                hideAllOverlays();
                showOverlay('parametricAnalysisScreen');
                if (window.ParametricAnalysis) {
                    window.ParametricAnalysis.show();
                }
            }
        );
    } else if (choice === 'playground') {
        if (titleEl) titleEl.textContent = 'Launching Deep Dive Playground';
        runLoadingAnimation(
            'choiceLoadingProgress',
            'choiceLoadingText',
            'choiceLoadingPercent',
            ['Setting up playground...', 'Preparing visualizations...', 'Loading data...'],
            () => {
                // Hide all our overlays -- the playground elements are already in the container
                hideAllOverlays();

                // Reset the playground's loading screen so it can show again
                const knnLoadingScreen = document.getElementById('knnLoadingScreen');
                if (knnLoadingScreen) {
                    knnLoadingScreen.classList.remove('hidden');
                    knnLoadingScreen.style.display = '';  // Clear any inline override
                }

                // Reset loading progress bar
                const loadingProgress = document.getElementById('loadingProgress');
                if (loadingProgress) loadingProgress.style.width = '0%';

                // Init the KNN visualizer (it has its own loading animation)
                if (window.initKNNVisualizer) {
                    window.initKNNVisualizer();
                }
            }
        );
    }
}

// ==================== NAVIGATION HANDLERS ====================

function handleBackToModeSelection() {
    // From Parametric Analysis or Playground -> back to Mode Selection
    if (window.ParametricAnalysis) {
        window.ParametricAnalysis.hide();
    }
    // Stop playground auto-run if active
    if (typeof STATE !== 'undefined' && STATE.isAutoRunning) {
        clearInterval(STATE.autoIntervalId);
        STATE.isAutoRunning = false;
    }
    // Show mode selection overlay (covers playground elements)
    hideAllOverlays();
    showOverlay('modeSelectionScreen');
}

function handleBackToExperimentFromModeSelection() {
    // From Mode Selection -> back to Experiment (hide entire container)
    hideAllOverlays();
    const container = document.getElementById('knnAnimationContainer');
    if (container) {
        container.style.display = 'none';
    }
}

// ==================== GLOBAL EXPOSURE ====================

window.handleModeChoice = handleModeChoice;
window.handleBackToModeSelection = handleBackToModeSelection;
window.handleBackToExperimentFromModeSelection = handleBackToExperimentFromModeSelection;
window.showModeSelectionLoading = showModeSelectionLoading;
window.showModeSelection = showModeSelection;

console.log('[MODE_SELECTION] Mode selection controller loaded');

/**
 * Shared KNN Computation Engine
 * Contains IRIS dataset and core distance/scaling/KNN algorithms
 */

const KNNEngine = {
    // ==================== IRIS DATASET (RAW) ====================
    // 150 samples (50 per class), 4 features
    IRIS_RAW: {
        features: [
            [5.1,3.5,1.4,0.2],[4.9,3.0,1.4,0.2],[4.7,3.2,1.3,0.2],[4.6,3.1,1.5,0.2],[5.0,3.6,1.4,0.2],
            [5.4,3.9,1.7,0.4],[4.6,3.4,1.4,0.3],[5.0,3.4,1.5,0.2],[4.4,2.9,1.4,0.2],[4.9,3.1,1.5,0.1],
            [5.4,3.7,1.5,0.2],[4.8,3.4,1.6,0.2],[4.8,3.0,1.4,0.1],[4.3,3.0,1.1,0.1],[5.8,4.0,1.2,0.2],
            [5.7,4.4,1.5,0.4],[5.4,3.9,1.3,0.4],[5.1,3.5,1.4,0.3],[5.7,3.8,1.7,0.3],[5.1,3.8,1.5,0.3],
            [5.4,3.4,1.7,0.2],[5.1,3.7,1.5,0.4],[4.6,3.6,1.0,0.2],[5.1,3.3,1.7,0.5],[4.8,3.4,1.9,0.2],
            [5.0,3.0,1.6,0.2],[5.0,3.4,1.6,0.4],[5.2,3.5,1.5,0.2],[5.2,3.4,1.4,0.2],[4.7,3.2,1.6,0.2],
            [4.8,3.1,1.6,0.2],[5.4,3.4,1.5,0.4],[5.2,4.1,1.5,0.1],[5.5,4.2,1.4,0.2],[4.9,3.1,1.5,0.2],
            [5.0,3.2,1.2,0.2],[5.5,3.5,1.3,0.2],[4.9,3.6,1.4,0.1],[4.4,3.0,1.3,0.2],[5.1,3.4,1.5,0.2],
            [5.0,3.5,1.3,0.3],[4.5,2.3,1.3,0.3],[4.4,3.2,1.3,0.2],[5.0,3.5,1.6,0.6],[5.1,3.8,1.9,0.4],
            [4.8,3.0,1.4,0.3],[5.1,3.8,1.6,0.2],[4.6,3.2,1.4,0.2],[5.3,3.7,1.5,0.2],[5.0,3.3,1.4,0.2],
            [7.0,3.2,4.7,1.4],[6.4,3.2,4.5,1.5],[6.9,3.1,4.9,1.5],[5.5,2.3,4.0,1.3],[6.5,2.8,4.6,1.5],
            [5.7,2.8,4.5,1.3],[6.3,3.3,4.7,1.6],[4.9,2.4,3.3,1.0],[6.6,2.9,4.6,1.3],[5.2,2.7,3.9,1.4],
            [5.0,2.0,3.5,1.0],[5.9,3.0,4.2,1.5],[6.0,2.2,4.0,1.0],[6.1,2.9,4.7,1.4],[5.6,2.9,3.6,1.3],
            [6.7,3.1,4.4,1.4],[5.6,3.0,4.5,1.5],[5.8,2.7,4.1,1.0],[6.2,2.2,4.5,1.5],[5.6,2.5,3.9,1.1],
            [5.9,3.2,4.8,1.8],[6.1,2.8,4.0,1.3],[6.3,2.5,4.9,1.5],[6.1,2.8,4.7,1.2],[6.4,2.9,4.3,1.3],
            [6.6,3.0,4.4,1.4],[6.8,2.8,4.8,1.4],[6.7,3.0,5.0,1.7],[6.0,2.9,4.5,1.5],[5.7,2.6,3.5,1.0],
            [5.5,2.4,3.8,1.1],[5.5,2.4,3.7,1.0],[5.8,2.7,3.9,1.2],[6.0,2.7,5.1,1.6],[5.4,3.0,4.5,1.5],
            [6.0,3.4,4.5,1.6],[6.7,3.1,4.7,1.5],[6.3,2.3,4.4,1.3],[5.6,3.0,4.1,1.3],[5.5,2.5,4.0,1.3],
            [5.5,2.6,4.4,1.2],[6.1,3.0,4.6,1.4],[5.8,2.6,4.0,1.2],[5.0,2.3,3.3,1.0],[5.6,2.7,4.2,1.3],
            [5.7,3.0,4.2,1.2],[5.7,2.9,4.2,1.3],[6.2,2.9,4.3,1.3],[5.1,2.5,3.0,1.1],[5.7,2.8,4.1,1.3],
            [6.3,3.3,6.0,2.5],[5.8,2.7,5.1,1.9],[7.1,3.0,5.9,2.1],[6.3,2.9,5.6,1.8],[6.5,3.0,5.8,2.2],
            [7.6,3.0,6.6,2.1],[4.9,2.5,4.5,1.7],[7.3,2.9,6.3,1.8],[6.7,2.5,5.8,1.8],[7.2,3.6,6.1,2.5],
            [6.5,3.2,5.1,2.0],[6.4,2.7,5.3,1.9],[6.8,3.0,5.5,2.1],[5.7,2.5,5.0,2.0],[5.8,2.8,5.1,2.4],
            [6.4,3.2,5.3,2.3],[6.5,3.0,5.5,1.8],[7.7,3.8,6.7,2.2],[7.7,2.6,6.9,2.3],[6.0,2.2,5.0,1.5],
            [6.9,3.2,5.7,2.3],[5.6,2.8,4.9,2.0],[7.7,2.8,6.7,2.0],[6.3,2.7,4.9,1.8],[6.7,3.3,5.7,2.1],
            [7.2,3.2,6.0,1.8],[6.2,2.8,4.8,1.8],[6.1,3.0,4.9,1.8],[6.4,2.8,5.6,2.1],[7.2,3.0,5.8,1.6],
            [7.4,2.8,6.1,1.9],[7.9,3.8,6.4,2.0],[6.4,2.8,5.6,2.2],[6.3,2.8,5.1,1.5],[6.1,2.6,5.6,1.4],
            [7.7,3.0,6.1,2.3],[6.3,3.4,5.6,2.4],[6.4,3.1,5.5,1.8],[6.0,3.0,4.8,1.8],[6.9,3.1,5.4,2.1],
            [6.7,3.1,5.6,2.4],[6.9,3.1,5.1,2.3],[5.8,2.7,5.1,1.9],[6.8,3.2,5.9,2.3],[6.7,3.3,5.7,2.5],
            [6.7,3.0,5.2,2.3],[6.3,2.5,5.0,1.9],[6.5,3.0,5.2,2.0],[6.2,3.4,5.4,2.3],[5.9,3.0,5.1,1.8]
        ],
        targets: [
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
            1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
            2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2
        ],
        trainIndices: [
            60,80,128,114,48,32,22,91,31,72,118,4,29,18,134,77,117,90,
            56,110,23,113,44,17,39,67,121,147,120,127,146,97,42,36,14,
            116,64,53,78,86,63,88,69,7,82,1,35,43,10,3,
            129,108,135,106,55,68,21,109,84,76,130,41,105,143,
            149,73,8,46,95,122,37,66,145,131,40,111,92,0,133,
            57,132,9,34,51,74,104,71,81,99,138,93,83,26,
            52,89,140,11,142,96,33,103,50,61,141,30
        ],
        testIndices: [
            144,101,119,102,148,126,75,85,12,115,25,28,123,136,19,
            137,70,20,24,5,65,15,98,107,58,54,87,79,62,6,
            112,139,94,38,125,16,45,2,27,59,100,13,47,124,49
        ],
        classNames: ['Setosa', 'Versicolor', 'Virginica'],
        featureNames: ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    },

    // ==================== CORE ALGORITHMS ====================

    euclideanDistance: function(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
        return Math.sqrt(sum);
    },

    manhattanDistance: function(a, b) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
        return sum;
    },

    standardScale: function(trainData, testData) {
        const nFeatures = trainData[0].length;
        const means = new Array(nFeatures).fill(0);
        const stds = new Array(nFeatures).fill(0);

        for (const row of trainData) {
            for (let j = 0; j < nFeatures; j++) means[j] += row[j];
        }
        for (let j = 0; j < nFeatures; j++) means[j] /= trainData.length;

        for (const row of trainData) {
            for (let j = 0; j < nFeatures; j++) stds[j] += (row[j] - means[j]) ** 2;
        }
        for (let j = 0; j < nFeatures; j++) stds[j] = Math.sqrt(stds[j] / trainData.length);

        const scaledTrain = trainData.map(row =>
            row.map((val, j) => stds[j] === 0 ? 0 : (val - means[j]) / stds[j])
        );
        const scaledTest = testData.map(row =>
            row.map((val, j) => stds[j] === 0 ? 0 : (val - means[j]) / stds[j])
        );
        return { scaledTrain, scaledTest, means, stds };
    },

    runKNN: function(trainX, trainY, testX, testY, k, distanceFn) {
        const predictions = [];
        for (const testPoint of testX) {
            const distances = trainX.map((trainPoint, idx) => ({
                distance: distanceFn(testPoint, trainPoint),
                label: trainY[idx]
            }));
            distances.sort((a, b) => a.distance - b.distance);
            const kNearest = distances.slice(0, k);
            const votes = {};
            for (const neighbor of kNearest) {
                votes[neighbor.label] = (votes[neighbor.label] || 0) + 1;
            }
            let maxVotes = 0, predicted = 0;
            for (const [label, count] of Object.entries(votes)) {
                if (count > maxVotes) { maxVotes = count; predicted = parseInt(label); }
            }
            predictions.push(predicted);
        }
        return { predictions, metrics: this.computeMetrics(testY, predictions, 3) };
    },

    computeMetrics: function(trueLabels, predictions, numClasses) {
        const n = trueLabels.length;
        let correct = 0;
        for (let i = 0; i < n; i++) if (trueLabels[i] === predictions[i]) correct++;
        const accuracy = correct / n;

        const tp = new Array(numClasses).fill(0);
        const fp = new Array(numClasses).fill(0);
        const fn = new Array(numClasses).fill(0);
        for (let i = 0; i < n; i++) {
            const actual = trueLabels[i], pred = predictions[i];
            if (pred === actual) tp[actual]++;
            else { fp[pred]++; fn[actual]++; }
        }

        let precisionSum = 0, recallSum = 0, f1Sum = 0;
        for (let c = 0; c < numClasses; c++) {
            const prec = tp[c] + fp[c] > 0 ? tp[c] / (tp[c] + fp[c]) : 0;
            const rec = tp[c] + fn[c] > 0 ? tp[c] / (tp[c] + fn[c]) : 0;
            const f1 = prec + rec > 0 ? 2 * prec * rec / (prec + rec) : 0;
            precisionSum += prec; recallSum += rec; f1Sum += f1;
        }
        return {
            accuracy,
            precision: precisionSum / numClasses,
            recall: recallSum / numClasses,
            f1: f1Sum / numClasses
        };
    },

    /**
     * Compute ROC Curve data (TPR, FPR, AUC) per class
     * Uses neighbor proportions as probability estimates
     */
    computeROCCurve: function(trainX, trainY, testX, testY, k, distanceFn) {
        const numClasses = 3;
        const rocData = {};

        // 1. Get probability estimates (proportion of votes) for each test point
        const probabilities = testX.map(testPoint => {
            const distances = trainX.map((trainPoint, idx) => ({
                distance: distanceFn(testPoint, trainPoint),
                label: trainY[idx]
            }));
            distances.sort((a, b) => a.distance - b.distance);
            const kNearest = distances.slice(0, k);
            const probs = new Array(numClasses).fill(0);
            kNearest.forEach(n => probs[n.label]++);
            return probs.map(count => count / k);
        });

        // 2. Compute curve for each class
        for (let c = 0; c < numClasses; c++) {
            const scores = probabilities.map(p => p[c]);
            const actuals = testY.map(y => y === c ? 1 : 0);

            // Sort by scores descending
            const combined = scores.map((s, i) => ({ s, a: actuals[i] })).sort((a, b) => b.s - a.s);

            const tpr = [0], fpr = [0];
            const totalPos = actuals.filter(a => a === 1).length;
            const totalNeg = actuals.length - totalPos;

            let tp = 0, fp = 0;
            for (let i = 0; i < combined.length; i++) {
                if (combined[i].a === 1) tp++; else fp++;
                tpr.push(tp / totalPos);
                fpr.push(fp / totalNeg);
            }

            // Simple trapezoidal AUC
            let auc = 0;
            for (let i = 1; i < tpr.length; i++) {
                auc += (fpr[i] - fpr[i - 1]) * (tpr[i] + tpr[i - 1]) / 2;
            }

            rocData[c] = { fpr, tpr, auc };
        }

        return rocData;
    },

    findKNearestForPoint: function(testFeatures, trainX, trainY, k, distanceFn) {
        const distances = trainX.map((trainPoint, idx) => ({
            distance: distanceFn(testFeatures, trainPoint),
            label: trainY[idx],
            idx
        }));
        distances.sort((a, b) => a.distance - b.distance);
        return distances.slice(0, k);
    }
};

window.KNNEngine = KNNEngine;

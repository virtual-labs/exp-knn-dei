const fs = require('fs');
const path = require('path');

const sourcePath = 'f:\\ML Virtual Labs\\KNN\\artifacts\\plot_data.json';
const destPath = 'f:\\ML Virtual Labs\\GitHub Repos\\exp-knn-dei\\experiment\\simulation\\js\\knn_data.js';

try {
    const data = fs.readFileSync(sourcePath, 'utf8');
    const jsContent = `const KNN_PLOT_DATA = ${data};\n`;
    
    // Ensure directory exists
    const dir = path.dirname(destPath);
    if (!fs.existsSync(dir)){
        fs.mkdirSync(dir, { recursive: true });
    }

    fs.writeFileSync(destPath, jsContent, 'utf8');
    console.log(`Successfully converted ${sourcePath} to ${destPath}`);
} catch (err) {
    console.error(`Error: ${err.message}`);
    process.exit(1);
}

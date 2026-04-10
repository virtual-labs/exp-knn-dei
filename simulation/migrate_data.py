import json
import os

source_path = r"f:\ML Virtual Labs\KNN\artifacts\plot_data.json"
dest_path = r"f:\ML Virtual Labs\GitHub Repos\exp-knn-dei\experiment\simulation\js\knn_data.js"

try:
    with open(source_path, 'r', encoding='utf-8') as f:
        data = f.read()
    
    js_content = f"const KNN_PLOT_DATA = {data};\n"
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(js_content)
        
    print(f"Successfully converted {source_path} to {dest_path}")
except Exception as e:
    print(f"Error: {e}")

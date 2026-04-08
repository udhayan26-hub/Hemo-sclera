import pandas as pd
import numpy as np
import cv2
import os
import joblib
import matplotlib.pyplot as plt

try:
    from engine.color_logic import analyze_anemia_eyelid
except ImportError:
    print("Engine modules missing or not found.")

def generate_hero_graph():
    print("Generating Hero Graph (Real vs predicted Hb)...")
    df = pd.read_csv("master_metadata.csv")
    anemia_df = df[df['type'].str.contains("Anemia")].dropna(subset=['hb_value'])
    
    if not os.path.exists("anemia_rf_model.pkl"):
        print("Model not trained yet!")
        return
        
    model = joblib.load("anemia_rf_model.pkl")
    
    real_hb = []
    pred_hb = []
    
    for _, row in anemia_df.iterrows():
        img_path = row['image_path']
        base_dir = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        # Determine mask path
        mask_path = os.path.join(base_dir, f"{base_name}_palpebral.png")
        if not os.path.exists(mask_path):
            candidates = [f for f in os.listdir(base_dir) if f.endswith('_palpebral.png') and 'forniceal' not in f]
            if candidates:
                mask_path = os.path.join(base_dir, candidates[0])
            else:
                continue
                
        # Load and predict
        try:
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            a_star = analyze_anemia_eyelid(img, mask)
            if a_star is not None:
                predicted = model.predict([[a_star]])[0]
                real_hb.append(row['hb_value'])
                pred_hb.append(predicted)
        except Exception:
            continue
            
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.scatter(real_hb, pred_hb, color='blue', alpha=0.5, edgecolor='k')
    
    # Perfect line
    min_val = min(min(real_hb), min(pred_hb)) - 1
    max_val = max(max(real_hb), max(pred_hb)) + 1
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Accuracy Line')
    
    plt.title("Diagnostics vs Machine Output (Hb g/dL)")
    plt.xlabel("Real Hemoglobin (Validation Data)")
    plt.ylabel("Predicted Hemoglobin (Random Forest)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.savefig("hero_graph.png", dpi=300, bbox_inches='tight')
    print("Saved 'hero_graph.png' successfully.")

if __name__ == "__main__":
    generate_hero_graph()

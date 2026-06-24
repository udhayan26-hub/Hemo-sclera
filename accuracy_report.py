import pandas as pd
import joblib
import cv2
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, accuracy_score

# Mocking the engine calls if they fail to import for some reason
try:
    from engine.color_logic import analyze_anemia_eyelid, analyze_jaundice_sclera
    from engine.sclera_seg import isolate_sclera
except ImportError:
    print("Engine modules not found.")
    exit(1)

def run_evaluation():
    print("--- Hemo-Sclera Accuracy Report ---")
    if not os.path.exists('master_metadata.csv'):
        print("Data missing.")
        return

    df = pd.read_csv('master_metadata.csv')
    
    # Anemia
    anemia_df = df[df['type'].str.contains('Anemia')].dropna(subset=['hb_value'])
    if os.path.exists('anemia_rf_model.pkl'):
        model = joblib.load('anemia_rf_model.pkl')
        real, pred = [], []
        for _, row in anemia_df.iterrows():
            img_path = row['image_path']
            if not os.path.exists(img_path): continue
            img = cv2.imread(img_path)
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(os.path.dirname(img_path), f"{base_name}_palpebral.png")
            if not os.path.exists(mask_path): continue
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            a_star = analyze_anemia_eyelid(img, mask)
            if a_star is not None:
                real.append(row['hb_value'])
                pred.append(model.predict([[a_star]])[0])
        
        if real:
            mae = mean_absolute_error(real, pred)
            print(f"Anemia (Hb) Regression MAE: {mae:.2f} g/dL (Target < 1.2)")

    # Jaundice
    jaundice_df = df[df['type'].str.contains('Jaundice')]
    if os.path.exists('jaundice_lr_model.pkl'):
        model = joblib.load('jaundice_lr_model.pkl')
        real, pred = [], []
        for _, row in jaundice_df.iterrows():
            img_path = row['image_path']
            if not os.path.exists(img_path): continue
            try:
                _, mask = isolate_sclera(img_path)
                img = cv2.imread(img_path)
                b_star = analyze_jaundice_sclera(img, mask)
                if b_star is not None:
                    prob = model.predict_proba([[b_star]])[0][1]
                    pred.append(1 if prob > 0.5 else 0)
                    real.append(row['label'])
            except: continue
        
        if real:
            acc = accuracy_score(real, pred)
            print(f"Jaundice Risk Classification Accuracy: {acc*100:.2f}% (Target > 90%)")

if __name__ == "__main__":
    run_evaluation()

import pandas as pd
import joblib
import cv2
import numpy as np
import os
from sklearn.metrics import mean_absolute_error, accuracy_score, classification_report
try:
    from engine.color_logic import analyze_anemia_eyelid, analyze_jaundice_sclera
    from engine.sclera_seg import isolate_sclera
except ImportError:
    print("Engine modules not found. Ensure you are running from the project root.")
    exit(1)

def evaluate():
    print("--- Evaluating Hemo-Sclera Models ---")
    if not os.path.exists('master_metadata.csv'):
        print("Error: master_metadata.csv not found.")
        return

    df = pd.read_csv('master_metadata.csv')
    
    # 1. Anemia Model Evaluation
    print("\n[Anemia Model]")
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
            
            if not os.path.exists(mask_path):
                # Try fallback
                base_dir = os.path.dirname(img_path)
                candidates = [f for f in os.listdir(base_dir) if f.endswith('_palpebral.png') and 'forniceal' not in f]
                if candidates:
                    mask_path = os.path.join(base_dir, candidates[0])
                else: continue
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            a_star = analyze_anemia_eyelid(img, mask)
            if a_star is not None:
                real.append(row['hb_value'])
                pred.append(model.predict([[a_star]])[0])
        
        if real:
            mae = mean_absolute_error(real, pred)
            print(f"Samples Tested: {len(real)}")
            print(f"Mean Absolute Error (MAE): {mae:.2f} g/dL")
            print(f"Target MAE: < 1.20 g/dL")
            if mae < 1.20:
                print("Status: ✅ WITHIN TARGET")
            else:
                print("Status: ⚠️ ABOVE TARGET")
    else:
        print("Anemia model not found.")

    # 2. Jaundice Model Evaluation
    print("\n[Jaundice Model]")
    jaundice_df = df[df['type'].str.contains('Jaundice')]
    if os.path.exists('jaundice_lr_model.pkl'):
        model = joblib.load('jaundice_lr_model.pkl')
        real, pred = [], []
        for _, row in jaundice_df.iterrows():
            img_path = row['image_path']
            if not os.path.exists(img_path): continue
            
            try:
                # Dynamic Sclera Segmentation
                _, mask = isolate_sclera(img_path)
                img = cv2.imread(img_path)
                b_star = analyze_jaundice_sclera(img, mask)
                if b_star is not None:
                    # Risk prediction
                    prob = model.predict_proba([[b_star]])[0][1]
                    pred.append(1 if prob > 0.5 else 0)
                    real.append(row['label'])
            except:
                continue
        
        if real:
            acc = accuracy_score(real, pred)
            print(f"Samples Tested: {len(real)}")
            print(f"Accuracy: {acc*100:.2f}%")
            print(f"Target Accuracy: > 90.00%")
            if acc > 0.90:
                print("Status: ✅ WITHIN TARGET")
            else:
                print("Status: ⚠️ BELOW TARGET")
            print("\nClassification Report:")
            print(classification_report(real, pred, target_names=["Normal", "Jaundice"]))
    else:
        print("Jaundice model not found.")

if __name__ == "__main__":
    evaluate()

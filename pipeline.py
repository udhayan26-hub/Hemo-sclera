import argparse
import sys
import os

# Import Engine Modules
from engine.data_prep import main as run_data_prep
from engine.train_yolo import process_anemia_masks_for_yolo, run_smoke_test, create_yolo_dataset_structure
from engine.color_logic import analyze_anemia_eyelid, analyze_jaundice_sclera
from engine.sclera_seg import isolate_sclera
from engine.diagnostic_models import AnemiaModel, JaundiceModel
import pandas as pd
import cv2
import numpy as np

def run_vision_smoke_test():
    """Runs the YOLO smoke test for eyelid segmentation."""
    metadata = "master_metadata.csv"
    if not os.path.exists(metadata):
        print("Data Prep must be run first! Use --task run_data_prep")
        sys.exit(1)
        
    dataset_dir = create_yolo_dataset_structure(os.getcwd())
    yaml_config = process_anemia_masks_for_yolo(metadata, dataset_dir)
    run_smoke_test(yaml_config)

def train_color_models():
    """Trains the Anemia Regressor and Jaundice Classifier models."""
    metadata = "master_metadata.csv"
    if not os.path.exists(metadata):
        print("Data Prep must be run first! Use --task run_data_prep")
        sys.exit(1)
        
    df = pd.read_csv(metadata)
    
    # 1. Anemia Processing
    print("--- Training Proprietary Anemia Model ---")
    anemia_records = df[df['type'].str.contains("Anemia")]
    a_stars = []
    hbs = []
    
    for _, row in anemia_records.iterrows():
        img_path = row['image_path']
        base_dir = os.path.dirname(img_path)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        
        mask_path = os.path.join(base_dir, f"{base_name}_palpebral.png")
        if not os.path.exists(mask_path):
            candidates = [f for f in os.listdir(base_dir) if f.endswith('_palpebral.png') and 'forniceal' not in f]
            if candidates:
                mask_path = os.path.join(base_dir, candidates[0])
            else:
                continue
                
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        a_star = analyze_anemia_eyelid(img, mask)
        if a_star is not None:
            try:
                hb_val = float(row['hb_value'])
                if not np.isnan(hb_val):
                    a_stars.append(a_star)
                    hbs.append(hb_val)
            except (ValueError, TypeError):
                continue
            
    if a_stars:
        anemia_model = AnemiaModel()
        anemia_model.train(a_stars, hbs)
        anemia_model.save()
    else:
        print("Could not extract enough valid anemia features to train.")

    # 2. Jaundice Processing
    print("\n--- Training Proprietary Jaundice Risk Model ---")
    jaundice_records = df[df['type'].str.contains("Jaundice")]
    b_stars = []
    labels = []
    
    for _, row in jaundice_records.iterrows():
        img_path = row['image_path']
        try:
            # We use Hough Circles to dynamically mask out the iris for Jaundice
            _, mask = isolate_sclera(img_path)
            img = cv2.imread(img_path)
            b_star = analyze_jaundice_sclera(img, mask)
            if b_star is not None:
                b_stars.append(b_star)
                labels.append(row['label'])
        except Exception as e:
            continue
            
    if b_stars:
        jaundice_model = JaundiceModel()
        jaundice_model.train(b_stars, labels)
        jaundice_model.save()
    else:
        print("Could not extract enough valid jaundice features to train.")

def predict_sample(image_path, disease_type):
    """
    Given an open patient image and a target disease strategy, predict clinical values.
    """
    if disease_type == "jaundice":
        print(f"Executing High-Risk Jaundice Triage on: {image_path}")
        _, mask = isolate_sclera(image_path)
        img = cv2.imread(image_path)
        b_star = analyze_jaundice_sclera(img, mask)
        
        j_model = JaundiceModel()
        try:
            j_model.model = __import__('joblib').load("jaundice_lr_model.pkl")
            risk, prob = j_model.predict_risk(b_star)
            print(f">>> Outcome: {risk} (Probability: {prob*100:.1f}%) | Detected Sclera b*: {b_star:.2f}")
        except FileNotFoundError:
            print("Model not trained yet! Run: pipeline.py --task train_models")
            
    elif disease_type == "anemia":
        print("Anemia prediction directly from image requires the YOLOv11 segmentation to extract the eyelid mask dynamically.")
        # In full production, this would call YOLO to get the mask first.
        print("For this smoke-test environment, the predictive pipeline focuses on Jaundice, or pre-masked Anemia vectors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hemo-Sclera Vision AI Pipeline")
    parser.add_argument("--task", type=str, required=True, 
                        choices=["run_data_prep", "vision_smoke_test", "train_models", "predict"],
                        help="The execution phase of the pipeline.")
    
    parser.add_argument("--image", type=str, help="Path to patient image for prediction.")
    parser.add_argument("--disease", type=str, choices=["anemia", "jaundice"], help="Disease target for prediction.")

    args = parser.parse_args()

    print("========================================")
    print(f" Hemo-Sclera System | Task: {args.task}")
    print("========================================\n")

    if args.task == "run_data_prep":
        run_data_prep()
    elif args.task == "vision_smoke_test":
        run_vision_smoke_test()
    elif args.task == "train_models":
        train_color_models()
    elif args.task == "predict":
        if not args.image or not args.disease:
            print("Error: --predict requires --image and --disease arguments.")
        else:
            predict_sample(args.image, args.disease)

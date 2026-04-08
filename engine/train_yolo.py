import os
import cv2
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def create_yolo_dataset_structure(base_dir):
    """Creates the necessary folder structure for a YOLO dataset."""
    dataset_dir = os.path.join(base_dir, "yolo_anemia")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)
    return dataset_dir

def mask_to_polygon(mask):
    """Converts a binary mask to normalized YOLO polygon coordinates with noise filtering."""
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    
    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Filter out tiny contours that are likely noise (e.g., < 10 points or tiny area)
    if cv2.contourArea(largest_contour) < 500 or len(largest_contour) < 10:
        return []
        
    # Normalize coordinates
    h, w = mask.shape
    polygon = []
    for point in largest_contour:
        x, y = point[0]
        polygon.append(f"{x/w:.6f} {y/h:.6f}")
        
    return polygon

def clean_yolo_workspace(base_dir):
    """Wipes old dataset and runs to ensure a high-fidelity fresh start."""
    yolo_dir = os.path.join(base_dir, "yolo_anemia")
    runs_dir = os.path.join(base_dir, "runs")
    
    for d in [yolo_dir, runs_dir]:
        if os.path.exists(d):
            print(f"Cleaning existing directory: {d}")
            shutil.rmtree(d)

def process_anemia_masks_for_yolo(metadata_path, output_dir):
    """Prepares the dataset by parsing masks into YOLO txt format."""
    print("Preparing YOLOv11 Segmentation Dataset...")
    df = pd.read_csv(metadata_path)
    
    # Filter only Anemia images
    anemia_df = df[df['type'].str.contains('Anemia', na=False)].copy()
    
    # Simple split: first 80% train, rest val
    anemia_df = anemia_df.sample(frac=1, random_state=42).reset_index(drop=True)
    train_split_idx = int(len(anemia_df) * 0.8)
    
    processed_count = 0
    for idx, row in anemia_df.iterrows():
        img_path = row['image_path']
        if not os.path.exists(img_path):
            continue
            
        base_name = os.path.basename(img_path)
        name_no_ext = os.path.splitext(base_name)[0]
        folder_path = os.path.dirname(img_path)
        
        # Locate the palpebral mask
        # Usually it's named 'basename_palpebral.png'
        mask_path = os.path.join(folder_path, f"{name_no_ext}_palpebral.png")
        if not os.path.exists(mask_path):
            # Try just looking for any file ending in _palpebral.png in that folder
            candidates = [f for f in os.listdir(folder_path) if f.endswith('_palpebral.png') and 'forniceal' not in f]
            if candidates:
                mask_path = os.path.join(folder_path, candidates[0])
            else:
                continue

        # Load mask and convert to polygon
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
            
        # Ensure mask is strictly binary
        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        polygon = mask_to_polygon(binary_mask)
        
        if not polygon:
            continue

        split = "train" if idx < train_split_idx else "val"
        
        # Paths for output
        dest_img_path = os.path.join(output_dir, "images", split, base_name)
        dest_label_path = os.path.join(output_dir, "labels", split, f"{name_no_ext}.txt")
        
        # Copy image
        shutil.copy(img_path, dest_img_path)
        
        # Write YOLO label (Class 0)
        with open(dest_label_path, "w") as f:
            f.write(f"0 {' '.join(polygon)}\n")
            
        processed_count += 1
        
    print(f"Dataset compiled: {processed_count} eyelid masks processed.")
    
    # Generate yaml file
    yaml_content = f"""path: {os.path.abspath(output_dir)}
train: images/train
val: images/val
names:
  0: palpebral
"""
    yaml_path = os.path.join(output_dir, "anemia_yolo.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    return yaml_path

def run_smoke_test(yaml_path):
    """Runs a tiny/nano YOLO model for a smoke test."""
    print("Initiating YOLOv11 Smoke Test...")
    model = YOLO("yolo11n-seg.pt")
    
    # Train for just 3 epochs to verify pipeline integrity
    results = model.train(
        data=yaml_path,
        epochs=30,
        imgsz=640,
        batch=4,
        device="cpu", # Force CPU for local smoke test if GPU isn't optimized, or let it auto-select
        project="runs/segmentation",
        name="anemia_smoke_test"
    )
    print("Smoke Test Completed Successfully.")

if __name__ == "__main__":
    metadata_file = "master_metadata.csv"
    if not os.path.exists(metadata_file):
        print(f"Error: {metadata_file} not found. Run data_prep.py first.")
    else:
        clean_yolo_workspace(os.getcwd())
        dataset_dir = create_yolo_dataset_structure(os.getcwd())
        yaml_config = process_anemia_masks_for_yolo(metadata_file, dataset_dir)
        run_smoke_test(yaml_config)

import os
import pandas as pd

def get_column(df, possible_names):
    for col in df.columns:
        clean_col = str(col).strip().lower()
        if any(p.lower() == clean_col for p in possible_names):
            return col
    return None

def process_anemia_dataset(base_path, dataset_type, excel_filename):
    data = []
    excel_path = os.path.join(base_path, excel_filename)
    if not os.path.exists(excel_path):
        print(f"Warning: {excel_path} not found.")
        return data

    print(f"Processing {excel_path}...")
    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        print(f"Error reading {excel_path}: {e}")
        return data

    id_col = get_column(df, ["Number", "ID", "Patient ID", "Patient", "Patient_ID"])
    hb_col = get_column(df, ["Hb", "Hb Value", "Hb_Value", "Hemoglobin", "Hb_g_dL", "Hb (g/dL)", "Hb (g/dl)", "Hb_g_dl"])
    
    if id_col is None:
        id_col = df.columns[0]
        print(f"  Fallback: ID col -> {id_col}")
    if hb_col is None:
        # Try to find common Hb markers
        for c in df.columns:
            if 'hb' in str(c).lower():
                hb_col = c
                break
        if hb_col is None:
            hb_col = df.columns[1] # fallback to second col
        print(f"  Fallback: Hb col -> {hb_col}")

    matched_count = 0
    for _, row in df.iterrows():
        if pd.isna(row[id_col]):
            continue
            
        patient_id = str(row[id_col]).strip()
        # Handle cases where pandas reads '1' as '1.0'
        if patient_id.endswith('.0'):
            patient_id = patient_id[:-2]
            
        hb_value = row[hb_col]
        # skip if no valid Hb
        if pd.isna(hb_value):
            continue

        folder_path = os.path.join(base_path, patient_id)
        if os.path.isdir(folder_path):
            images = [f for f in os.listdir(folder_path) if f.casefold().endswith(('.jpg', '.png', '.jpeg'))]
            # Need the original image, ignore the masks
            original_images = [img for img in images if not any(mask_tag in img.lower() for mask_tag in ['_palpebral', '_forniceal', '_mask'])]
            
            if original_images:
                # Take the first matched image
                img_path = os.path.join(folder_path, original_images[0])
                data.append({
                    "image_path": img_path,
                    "hb_value": hb_value,
                    "label": 0, # Placeholder for anemia set (we use hb_value regression, not classification)
                    "type": f"Anemia_{dataset_type}"
                })
                matched_count += 1
            else:
                print(f"  Missing original image for Patient ID {patient_id}")
        else:
            pass # Some patients in excel might not have a corresponding folder

    print(f"Matched {matched_count} records from {dataset_type} dataset.")
    return data

def process_jaundice_dataset(base_path):
    data = []
    base_path = os.path.abspath(base_path)
    print(f"DEBUG: Processing Jaundice from {base_path}...")
    
    if not os.path.isdir(base_path):
        print(f"DEBUG: Jaundice base path not found: {base_path}")
        return data

    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(base_path, split)
        if not os.path.isdir(split_path):
            print(f"DEBUG: Split {split} not found: {split_path}")
            continue
            
        print(f"DEBUG: Scanning split: {split}")
        subdirs = os.listdir(split_path)
        for class_name, label in [('Normal', 0), ('Jaundice', 1)]:
            matched_dir = next((d for d in subdirs if d.lower() == class_name.lower()), None)
            if not matched_dir:
                print(f"DEBUG: Class {class_name} not found in {subdirs}")
                continue
                
            class_path = os.path.join(split_path, matched_dir)
            if not os.path.isdir(class_path):
                continue
                
            img_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            print(f"DEBUG: Found {len(img_files)} images in {class_path}")
            for img_name in img_files:
                data.append({
                    "image_path": os.path.join(class_path, img_name),
                    "hb_value": None,
                    "label": label,
                    "type": "Jaundice_Set"
                })
    print(f"SUCCESS: Collected {len(data)} Jaundice records.")
    return data

def main():
    root_dir = os.path.abspath(os.getcwd())
    base_dir = os.path.join(root_dir, "Datasets")
    print(f"DEBUG: Root: {root_dir}")
    print(f"DEBUG: Base Datasets: {base_dir}")
    
    all_data = []
    
    an_india = os.path.join(base_dir, "anemia", "India")
    anemia_india = process_anemia_dataset(an_india, "India", "India.xlsx")
    all_data.extend(anemia_india)
    print(f"DEBUG: Added India {len(anemia_india)}. Total: {len(all_data)}")
    
    an_italy = os.path.join(base_dir, "anemia", "Italy")
    anemia_italy = process_anemia_dataset(an_italy, "Italy", "Italy.xlsx")
    all_data.extend(anemia_italy)
    print(f"DEBUG: Added Italy {len(anemia_italy)}. Total: {len(all_data)}")
    
    j_path = os.path.join(base_dir, "jaundice")
    jaundice_data = process_jaundice_dataset(j_path)
    all_data.extend(jaundice_data)
    print(f"DEBUG: Added Jaundice {len(jaundice_data)}. Total: {len(all_data)}")
    
    if not all_data:
        print("CRITICAL: No data collected at all!")
        return

    df_master = pd.DataFrame(all_data)
    out_path = os.path.join(root_dir, "master_metadata.csv")
    df_master.to_csv(out_path, index=False)
    print(f"\nSUCCESS: Generated '{out_path}' with {len(df_master)} total records.")
    
if __name__ == "__main__":
    main()

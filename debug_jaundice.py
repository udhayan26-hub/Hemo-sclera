import pandas as pd
import cv2
import os

from engine.sclera_seg import isolate_sclera
from engine.color_logic import analyze_jaundice_sclera

df = pd.read_csv('master_metadata.csv')
jdf = df[df['type'].str.contains('Jaundice')]
print("Jaundice rows in CSV:", len(jdf))

success = 0
for _, r in jdf.iterrows():
    img_path = r['image_path']
    if not os.path.exists(img_path):
        print("Missing file:", img_path)
        continue
    try:
        _, m = isolate_sclera(img_path)
        img = cv2.imread(img_path)
        b = analyze_jaundice_sclera(img, m)
        if b is not None and b > 0:
            success += 1
    except Exception as e:
        print('Error on', img_path, e)
        
print('Loaded successfully:', success)

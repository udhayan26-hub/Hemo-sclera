import os
import cv2
import glob
import numpy as np

try:
    from engine.sclera_seg import isolate_sclera
    from engine.color_logic import analyze_jaundice_sclera
except ImportError:
    pass

def test_group(path, count=5):
    images = glob.glob(os.path.join(path, "*.jpg")) + glob.glob(os.path.join(path, "*.png"))
    results = []
    
    for img_path in images[:count]:
        try:
            _, mask = isolate_sclera(img_path)
            img = cv2.imread(img_path)
            b_star = analyze_jaundice_sclera(img, mask)
            if b_star is not None:
                results.append(b_star)
        except Exception as e:
            print(f"Error on {img_path}: {e}")
            
    avg = np.mean(results) if results else 0
    print(f"Group: {os.path.basename(path)}")
    print(f"Values tested: {results}")
    print(f"Average b* : {avg:.2f}")

print("Running Scientific Sanity Check...")
test_group("Datasets/jaundice/valid/Normal", 5)
print("---")
test_group("Datasets/jaundice/valid/Jaundice", 5)

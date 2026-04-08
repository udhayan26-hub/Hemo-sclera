import os
import pandas as pd

base = "Datasets"
print(f"CWD: {os.getcwd()}")
print(f"Base exists: {os.path.exists(base)}")

j_path = os.path.join(base, "jaundice")
print(f"Jaundice path: {j_path}")
print(f"Jaundice exists: {os.path.exists(j_path)}")

if os.path.exists(j_path):
    print(f"Jaundice contents: {os.listdir(j_path)}")
    for split in ['train', 'valid', 'test']:
        sp = os.path.join(j_path, split)
        print(f"  Split {split} exists: {os.path.exists(sp)}")
        if os.path.exists(sp):
            print(f"    {split} contents: {os.listdir(sp)}")

a_path = os.path.join(base, "anemia")
print(f"Anemia path: {a_path}")
print(f"Anemia exists: {os.path.exists(a_path)}")
if os.path.exists(a_path):
    print(f"Anemia contents: {os.listdir(a_path)}")
    it_path = os.path.join(a_path, "Italy")
    print(f"    Italy path exists: {os.path.exists(it_path)}")
    if os.path.exists(it_path):
        print(f"    Italy contents: {os.listdir(it_path)}")

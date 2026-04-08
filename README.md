# Hemo-Sclera: Multimodal Diagnostic AI
**An Enterprise-Grade Triage System for Non-Invasive Anemia & Jaundice Screening**

## 1. Overview & Impact
In rural regions (such as deep deployments in India), access to standard CBC (Complete Blood Count) or Bilirubin lab clinics is heavily restricted. This system aims to provide a rapid "Triage Gateway" using basic phone imaging.

By algorithmically stripping away lighting interference and mapping purely to physiological signs on the human body (eyelid paleness for hemoglobin, scleral yellowness for bilirubin), this system acts as a front-line warning before a patient is routed to a distant urban clinic.

## 2. Core Architecture
**Modular Pipeline Ecosystem**
Rather than a single monolithic script, this system uses a heavily organized set of core modules orchestrated via `pipeline.py`.
- `engine/data_prep.py`: Integrates Excel clinical metrics with physical OS image directories.
- `engine/train_yolo.py`: Translates legacy 1-channel binary masks into polygon arrays expected by cutting-edge CV models, performing smoke-tests in local execution.
- `engine/sclera_seg.py`: Hough Circle Transforms isolate the iris.
- `engine/color_logic.py`: Handles all structural manipulation and CV color logic.
- `engine/diagnostic_models.py`: Packages Scikit-Learn logic.

You can launch discrete tasks: `python pipeline.py --task <task_name>`

## 3. The Color Science (CIE L*a*b* vs RGB)
**Why not RGB?** Standard RGB spaces merge light intensity and color natively. A dark picture looks redder. A bright flash looks whiter.
**The Native L*a*b Advantage:** 
By converting images into `L*a*b*`, we isolate the `L*` (Light) component out completely. 
- The `a*` channel directly maps onto a Green $\rightarrow$ Red axis, acting as a proxy for Oxygenation/Hemoglobin counts.
- The `b*` channel directly maps onto a Blue $\rightarrow$ Yellow axis, acting as a proxy for Bilirubin clusters indicating possible hepatic issues.

## 4. UI Dashboard Summary

A clinical interface powered by Streamlit (`app.py`). It enforces safety by disclaiming standard liability, maps predicted Hemoglobin metrics against true validation states using Random Forest, and determines a Triage (0 vs 1) risk scale using a specialized Logistic Regression.

---

### Security Disclaimer
This represents an experimental approach. The UI displays **"This is a Triage Assistant for screening purposes..."** to ensure any evaluator understands this supplements, rather than replaces, standard clinical procedures.

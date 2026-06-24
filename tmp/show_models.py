import joblib
import numpy as np

def show_model_info(path, model_type):
    print(f"\n--- Model: {path} ({model_type}) ---")
    try:
        model = joblib.load(path)
        print(f"Type: {type(model)}")
        if hasattr(model, "get_params"):
            print("Parameters:")
            for k, v in model.get_params().items():
                print(f"  {k}: {v}")
        
        if model_type == "Anemia (RF)":
            if hasattr(model, "feature_importances_"):
                print(f"Feature Importances: {model.feature_importances_}")
            if hasattr(model, "n_estimators"):
                print(f"N Estimators: {model.n_estimators}")
        
        elif model_type == "Jaundice (LR)":
            if hasattr(model, "coef_"):
                print(f"Coefficients: {model.coef_}")
            if hasattr(model, "intercept_"):
                print(f"Intercept: {model.intercept_}")
            if hasattr(model, "classes_"):
                print(f"Classes: {model.classes_}")
                
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    show_model_info("anemia_rf_model.pkl", "Anemia (RF)")
    show_model_info("jaundice_lr_model.pkl", "Jaundice (LR)")

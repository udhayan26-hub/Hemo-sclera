import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, classification_report
import joblib
import os

class AnemiaModel:
    def __init__(self):
        # Using Random Forest for regression (Hb Level)
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
    def train(self, X, y):
        """
        Train the Anemia Predictor. 
        X expected shape (N, n_features) - e.g. [[a_star_value], ...]
        y expected shape (N,) - [Hb_value, ...]
        """
        X = np.array(X).reshape(-1, 1) if len(np.shape(X)) == 1 else np.array(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        print(f"Anemia Model Trained. MSE: {mse:.2f}, MAE: {mae:.2f} g/dL")
        
    def predict(self, a_star_value):
        X = np.array([[a_star_value]])
        return self.model.predict(X)[0]
        
    def save(self, path="anemia_rf_model.pkl"):
        joblib.dump(self.model, path)
        print(f"Proprietary Anemia model saved to {path}")


class JaundiceModel:
    def __init__(self):
        # Using Logistic Regression for Risk Triage Strategy
        self.model = LogisticRegression(random_state=42)
        
    def train(self, X, y):
        """
        Train the Jaundice Risk Triage Predictor.
        X expected shape (N, n_features) - e.g. [[b_star_value], ...]
        y expected shape (N,) - 1 for Jaundice, 0 for Normal
        """
        X = np.array(X).reshape(-1, 1) if len(np.shape(X)) == 1 else np.array(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        
        preds = self.model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        print(f"Jaundice Triage Model Trained. Accuracy: {acc*100:.2f}%")
        print(classification_report(y_test, preds))
        
    def predict_risk(self, b_star_value):
        X = np.array([[b_star_value]])
        prob = self.model.predict_proba(X)[0][1] # Probability of Class 1
        return "High Risk" if prob > 0.5 else "Low Risk", prob
        
    def save(self, path="jaundice_lr_model.pkl"):
        joblib.dump(self.model, path)
        print(f"Proprietary Jaundice model saved to {path}")

if __name__ == "__main__":
    # Synthetic smoke test
    print("Testing Diagnostic Models Compilation...")
    anemia = AnemiaModel()
    anemia.train([[110], [120], [130], [140], [150], [160]], [9.0, 9.5, 12.0, 13.5, 14.5, 15.0])
    
    jaundice = JaundiceModel()
    jaundice.train([[110], [120], [165], [175], [180]], [0, 0, 1, 1, 1])

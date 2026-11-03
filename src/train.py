import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from . import config
from .preprocess import load_and_merge_data

def main():
    print("Loading and processing raw data...")
    df = load_and_merge_data()
    
    X = df[config.FEATURES]
    y = df["label"]
    
    print(f"Total windows: {len(X)}")
    print(f"Normal: {(y == 0).sum()}, Fault: {(y == 1).sum()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=y
    )
    
    print("\nTraining Random Forest...")
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=config.RANDOM_SEED)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=config.CLASSES))
    
    # Save Model
    model_path = os.path.join(config.ML_DIR, "vibration_rf.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved to {model_path}")
    
    # Plot Feature Importance
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    sns.barplot(x=importances, y=config.FEATURES)
    plt.title("Feature Importance for Vibration Fault Detection")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
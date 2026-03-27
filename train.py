import os
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

# --- CONFIGURATION ---
NUM_CLASSES = 234

def train_xgboost():
    os.makedirs("models", exist_ok=True)
    
    print("Loading pre-extracted embeddings from disk...")
    # LOAD THE .NPY FILES
    try:
        X_embeddings = np.load("X_embeddings.npy")
        y_raw_labels = np.load("y_labels.npy")
    except FileNotFoundError:
        print("Error: Could not find .npy files. Run extract_features.py first!")
        return

    print(f"Loaded shape: X={X_embeddings.shape}, y={y_raw_labels.shape}")

    # Encode Labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw_labels)
    
    # Dynamically count the exact number of classes we ended up with
    actual_num_classes = len(np.unique(y_encoded))
    print(f"Detected {actual_num_classes} unique bird classes.")

    # Save the encoder so you can translate predictions back later
    with open("label_encode.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Train/Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        X_embeddings, y_encoded, test_size=0.2, random_state=42, stratify=None
    )

    # Train XGBoost
    print("Training XGBoost classifier...")
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob', 
        num_class=NUM_CLASSES,      
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',          
        random_state=42
    )

    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=10
    )

    # Evaluate
    val_preds = xgb_clf.predict(X_val)
    val_probs = xgb_clf.predict_proba(X_val)
    
    print("\n--- Validation Results ---")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print(f"Log Loss: {log_loss(y_val, val_probs):.4f}")

    # Save Model
    model_save_path = "bird_xgb_model.json"
    xgb_clf.save_model(model_save_path)
    print(f"\nModel successfully saved to {model_save_path}")

if __name__ == "__main__":
    train_xgboost()
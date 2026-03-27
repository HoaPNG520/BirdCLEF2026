import os
import pickle
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss

# ==========================================
# 1. CONFIGURATION & SETUP
# ==========================================
PERCH_TF_HUB_URL = "https://tfhub.dev/google/bird-vocalization-classifier/4"
SAMPLE_RATE = 32000
DURATION = 5.0 # seconds
AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)
NUM_CLASSES = 234

# Ensure the models directory exists
os.makedirs("models", exist_ok=True)

# ==========================================
# 2. FEATURE EXTRACTION (PERCH)
# ==========================================
print("Loading Perch model from TF Hub...")
perch_model = hub.load(PERCH_TF_HUB_URL)
# # Check if the model has a signature or is callable
# if hasattr(perch_model, '__call__'):
#     print("Model loaded successfully and is callable.")
# else:
#     print("Model loaded, but might not be an executable signature.")

def load_and_preprocess_audio(file_path):
    """Loads an audio file, resamples to 32kHz, and pads/crops to exactly 5 seconds."""
    # librosa.load automatically resamples if sr is provided
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    
    # Pad with zeros if shorter than 5 seconds
    if len(waveform) < AUDIO_LENGTH:
        padding = AUDIO_LENGTH - len(waveform)
        waveform = np.pad(waveform, (0, padding), 'constant')
    # Crop if longer than 5 seconds (taking the first 5s for training)
    elif len(waveform) > AUDIO_LENGTH:
        waveform = waveform[:AUDIO_LENGTH]
        
    return waveform

def extract_embeddings(df, audio_dir):
    """Passes audio through Perch to extract 1280-d embeddings."""
    embeddings_list = []
    labels_list = []
    
    print(f"Extracting features for {len(df)} files...")
    for index, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        
        try:
            # 1. Preprocess audio
            waveform = load_and_preprocess_audio(file_path)
            
            # 2. Add batch dimension: shape becomes (1, 160000)
            waveform_batched = tf.expand_dims(waveform, axis=0)
            
            # 3. Pass through Perch
            # Perch returns a dictionary. We want the 'embeddings' key.
            model_output = perch_model.infer_tf(waveform_batched)
            embedding = model_output['embeddings'].numpy()[0] # Shape: (1280,)
            
            embeddings_list.append(embedding)
            labels_list.append(row['primary_label'])
            
            if (index + 1) % 50 == 0:
                print(f"Processed {index + 1}/{len(df)} files...")
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return np.array(embeddings_list), np.array(labels_list)

# ==========================================
# 3. MAIN PIPELINE
# ==========================================
if __name__ == "__main__":
    # --- A. Load Metadata ---
    # Replace with your actual train.csv and audio folder path
    csv_path = "data/raw/train.csv" 
    audio_folder = "data/raw/audio_files/"
    
    print("Loading metadata...")
    # For testing the script, you might want to use df.head(100)
    train_df = pd.read_csv(csv_path) 
    
    # --- B. Extract Features ---
    X_embeddings, y_raw_labels = extract_embeddings(train_df, audio_folder)
    print(f"Extracted shape: X={X_embeddings.shape}, y={y_raw_labels.shape}")
    
    # --- C. Encode Labels ---
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_raw_labels)
    
    # Save the label encoder to translate XGBoost numbers back to bird IDs later
    with open("models/label_encode.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
        
    # --- D. Train/Validation Split ---
    X_train, X_val, y_train, y_val = train_test_split(
        X_embeddings, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # --- E. Train XGBoost ---
    print("Training XGBoost classifier...")
    xgb_clf = xgb.XGBClassifier(
        objective='multi:softprob',  # Crucial: outputs probabilities for each class
        num_class=NUM_CLASSES,       # 234 in your case
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method='hist',          # Extremely fast for 1280-column datasets
        random_state=42
    )
    
    # Fit with early stopping
    xgb_clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=15,
        verbose=10
    )
    
    # --- F. Evaluate & Save ---
    val_preds = xgb_clf.predict(X_val)
    val_probs = xgb_clf.predict_proba(X_val)
    
    print("\n--- Validation Results ---")
    print(f"Accuracy: {accuracy_score(y_val, val_preds):.4f}")
    print(f"Log Loss: {log_loss(y_val, val_probs):.4f}")
    
    # Save the trained XGBoost model
    model_save_path = "models/bird_xgb_model.json"
    xgb_clf.save_model(model_save_path)
    print(f"\nModel successfully saved to {model_save_path}")
    print("Label Encoder successfully saved to models/label_encoder.pkl")
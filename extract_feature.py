import os
import numpy as np
import pandas as pd
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from configs.config import *
from extract_feature import *
from data.dataset import load_df_clean, load_label2idx

# --- CONFIGURATION ---
PERCH_TF_HUB_URL = "https://tfhub.dev/google/bird-vocalization-classifier/4"
SAMPLE_RATE = 32000
DURATION = 5.0
AUDIO_LENGTH = int(SAMPLE_RATE * DURATION)
CSV_PATH = BASE_DIR_ARTIFACT / "df_clean.csv"
AUDIO_FOLDER = AUDIO_DIR 

print("Loading Perch model from TF Hub...")
perch_model = hub.load(PERCH_TF_HUB_URL)

def load_and_preprocess_audio(file_path):
    waveform, _ = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    if len(waveform) < AUDIO_LENGTH:
        waveform = np.pad(waveform, (0, AUDIO_LENGTH - len(waveform)), 'constant')
    elif len(waveform) > AUDIO_LENGTH:
        waveform = waveform[:AUDIO_LENGTH]
    return waveform

def extract_embeddings(df, audio_dir):
    embeddings_list, labels_list = [], []
    
    print(f"Extracting features for {len(df)} files. This will take a while...")
    for index, row in df.iterrows():
        file_path = os.path.join(audio_dir, row['filename'])
        try:
            waveform = load_and_preprocess_audio(file_path)
            waveform_batched = tf.expand_dims(waveform, axis=0)
            
            # Pass through Perch
            model_output = perch_model.infer_tf(waveform_batched)
            embedding = model_output['embeddings'].numpy()[0]
            
            embeddings_list.append(embedding)
            labels_list.append(row['encoded_label'])
            
            if (index + 1) % 50 == 0:
                print(f"Processed {index + 1}/{len(df)} files...")
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    return np.array(embeddings_list), np.array(labels_list)
def run_extraction():
    df =  load_df_clean()
    label2idx = load_label2idx()
    df['encoded_label'] = df['primary_label'].map(label2idx)
    X_embeddings, y_labels = extract_embeddings(df, AUDIO_FOLDER)


# Choose your save directory (use 'data/' if local, '/kaggle/working/' if on Kaggle)
    save_dir = "/kaggle/working/" # Change to "/kaggle/working/" if running in a Kaggle Notebook
    
    # Ensure the folder actually exists before saving
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the arrays
    np.save(os.path.join(save_dir, "X_embeddings.npy"), X_embeddings)
    np.save(os.path.join(save_dir, "y_labels.npy"), y_labels)
    
    print(f"Success! Files saved to {save_dir}")

import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Extract Features from Audio
def extract_features(file_path):
    try:
        # Load audio file (ensure soundfile is installed to avoid PySoundFile warning)
        y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds of audio
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)
        chroma = np.mean(librosa.feature.chroma_stft(y=y, sr=sr).T, axis=0)
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr).T, axis=0)
        return np.hstack((mfcc, chroma, spectral_contrast))
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Step 2: Load Dataset
def load_dataset(dataset_path):
    data = []
    genres = os.listdir(dataset_path)  # Get all genre subfolders
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith('.wav'):  # Ensure only WAV files are processed
                    file_path = os.path.join(genre_path, file)
                    features = extract_features(file_path)
                    if features is not None:
                        data.append([features, genre])
    return pd.DataFrame(data, columns=["features", "label"])

# Define the dataset path
dataset_path = "C:/AI projects/music_genre_detection/Data/genres_original"  # Replace with your dataset path
df = load_dataset(dataset_path)

# Step 3: Prepare Data for Training
X = np.array(df["features"].tolist())
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df["label"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 6: Save the Model
joblib.dump(model, "music_genre_model.pkl")
print("Model saved as music_genre_model.pkl")

# Step 7: Predict a Genre for New Audio
def predict_genre(file_path, model_path="music_genre_model.pkl"):
    model = joblib.load(model_path)
    features = extract_features(file_path)
    if features is not None:
        prediction = model.predict([features])
        return label_encoder.inverse_transform(prediction)[0]
    else:
        return "Error in processing audio file"

# Test with a new audio file
test_audio = "C:/AI projects/music_genre_detection/Data/genres_original/disco/disco.wav"  # Replace with your test audio file

# Check if the test file exists
if os.path.exists(test_audio):
    predicted_genre = predict_genre(test_audio)
    print(f"The predicted genre is: {predicted_genre}")
else:
    print(f"The test audio file does not exist at: {test_audio}")


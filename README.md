# Music Genre Classification using Machine Learning
you can download dataset from  https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification
This project aims to classify music genres from audio files using machine learning. The dataset used in this project is the GTZAN Genre Dataset, which contains 1,000 audio tracks categorized into 10 genres, such as blues, classical, jazz, rock, and more. The model uses various features extracted from the audio files to predict the genre of the song.

Project Structure:
Audio Feature Extraction: Features such as MFCC (Mel-frequency cepstral coefficients) are extracted from each audio file using librosa, a Python library for audio analysis.
Model Training: The features are used to train a machine learning model using the Random Forest classifier.
Model Evaluation: The trained model is evaluated on a test set, and its accuracy is reported, along with a detailed classification report.
Predictions: The model can predict the genre of a new audio file.
Technologies Used:
Python: The primary programming language used for building and training the model.
librosa: For audio processing and feature extraction.
scikit-learn: For building and evaluating the machine learning model.
joblib: For saving the trained model.
Features:
Audio file preprocessing and feature extraction (MFCC).
Machine learning model for classification (Random Forest).
Model evaluation using accuracy and classification reports.
Model saving for future predictions.
Getting Started:
Clone the repository.
Install the required dependencies:
bash
Copy code
pip install -r requirements.txt
Run music.py to train the model and make predictions on audio files.
Results:
The model achieves an accuracy of approximately 65.5% on the test dataset.
The classification report provides precision, recall, and F1-score for each genre.
Future Improvements:
Experiment with different models (e.g., SVM, XGBoost).
Implement feature engineering techniques for better accuracy.
Expand the dataset for more diverse training data.

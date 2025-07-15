🎤 Speech Emotion Recognition using CNN on TESS Dataset
This project aims to recognize human emotions from speech audio signals using the TESS (Toronto Emotional Speech Set) dataset. The pipeline involves data preprocessing, audio augmentation, feature extraction, CNN-based classification, and performance evaluation.

📁 Dataset
Name: TESS - Toronto Emotional Speech Set

Source: Kaggle Dataset Link

Classes: 14 emotion labels derived from speaker ID and emotion type:

OAF_angry, YAF_happy, YAF_sad, OAF_neutral, YAF_fear, YAF_neutral, YAF_angry, OAF_Pleasant_surprise, YAF_pleasant_surprised, OAF_happy, OAF_Fear, YAF_disgust, OAF_disgust, OAF_Sad

📊 Data Preprocessing
🧪 Exploratory Analysis
Visualized waveforms using librosa.display.waveshow

Extracted duration, zero-crossing rate, and sample rate

🔉 Data Augmentation
To improve generalization and robustness, the following augmentations were applied:

Noise injection

Time stretching

Pitch shifting

Shifting in time

🧠 Feature Extraction
Using librosa, we extracted:

Zero Crossing Rate

Chroma STFT

MFCCs (Mel-frequency cepstral coefficients)

Root Mean Square Energy

Mel Spectrogram

Each audio file results in a 162-dimensional feature vector.

📦 Model Architecture
A 1D Convolutional Neural Network (CNN) was built using Keras:

text
Copy
Edit
Input → Conv1D (256) → MaxPool → Conv1D (256) → MaxPool 
→ Conv1D (128) → MaxPool → Dropout 
→ Conv1D (64) → MaxPool 
→ Flatten → Dense (32) → Dropout 
→ Dense (14) with softmax
Loss: categorical_crossentropy

Optimizer: Adam

Activation: ReLU (hidden), Softmax (output)

🏋️ Model Training
Epochs: 5

Batch size: 64

Train/Test split: 90% / 10%

Early Results:

Final accuracy: ~95.2%

Val loss: ~0.10

📈 Performance Evaluation
✅ Confusion Matrix
A heatmap was plotted for true vs predicted labels, showing strong diagonal dominance (high accuracy).

📋 Classification Report
text
Copy
Edit
              precision    recall  f1-score   support

       OAF_Fear       1.00      0.99      0.99        67
OAF_Pleasant_surprise 0.94      0.68      0.79        50
     ...
        YAF_sad       1.00      0.98      0.99        58

    accuracy                           0.95       840
   macro avg       0.96      0.95      0.95
weighted avg       0.96      0.95      0.95
🔧 Requirements
bash
Copy
Edit
tensorflow
keras
librosa
matplotlib
pandas
numpy
scikit-learn
seaborn
🧪 How to Run
Clone the repository or use in Google Colab

Ensure Kaggle API token is uploaded (kaggle.json)

Run the full pipeline:

Data download

Preprocessing & augmentation

Feature extraction

Model training

Evaluation

📂 Project Structure
kotlin
Copy
Edit
├── kaggle.json
├── TESS Toronto emotional speech set data/
├── New_Wav_Set.csv
├── speech-emotion-recognition.keras
├── model.png
├── confusion_matrix.png
└── notebook.ipynb
📌 Future Improvements
Use a larger and more diverse dataset

Try transformer-based audio classification

Deploy via web/mobile app for real-time predictions

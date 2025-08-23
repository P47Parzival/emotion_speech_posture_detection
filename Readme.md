# 🤖 Real-Time AI Interview Coach

This project is a web-based, real-time interview practice tool designed to provide instant feedback on a user's performance. It uses a combination of **traditional machine learning** and **deep learning** to analyze three key aspects of an interview: facial emotion, body posture, and speech.

## ✨ Features

* **Multi-Modal Feedback**: Provides comprehensive, parallel analysis of video and audio streams.
* **Emotion Detection**:
   * Analyzes the user's facial expression in real-time.
   * Classifies emotions into categories like Happy, Sad, Neutral, etc.
   * Provides live feedback to encourage a more confident and positive demeanor (e.g., "Smile to appear confident!").
* **Posture Analysis**:
   * Uses background subtraction to create a real-time silhouette of the user.
   * Classifies posture (e.g., Sitting, Standing).
   * Offers rule-based suggestions for better posture during an interview (e.g., "Sit up straight!").
* **Speech & Grammar Analysis**:
   * Transcribes the user's speech live using a deep learning model.
   * Provides basic grammar suggestions to help improve sentence structure and word choice.

## 🛠️ Technology Stack

This project was built with a specific set of tools to showcase both classic and modern AI techniques.

* **Backend**: Flask, Flask-SocketIO
* **Frontend**: HTML5, CSS3, JavaScript, Socket.IO Client
* **Computer Vision**: OpenCV
* **Traditional Machine Learning**: Scikit-learn (SVM), ensemble model (Random forest and XGBoost),XGBoost
* **Deep Learning (Speech)**: PyTorch, Hugging Face Transformers (Wav2Vec2)
* **Audio Processing**: PyAudio, Librosa
* **Natural Language Processing**: NLTK
* **Data Handling**: NumPy, Pandas

## 🚀 Getting Started

Follow these instructions to get the project running on your local machine.

### 1. Prerequisites

* Python 3.8+
* A webcam and a microphone
* System-level dependency for PyAudio:
   * **macOS**: `brew install portaudio`
   * **Debian/Ubuntu**: `sudo apt-get install portaudio19-dev`

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/P47Parzival/emotion_speech_posture_detection.git
   cd emotion_speech_posture_detection
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install the required Python packages:** A `requirements.txt` file is recommended for easier installation. You can create one with the following command:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download NLTK data:** Open a Python interpreter and run the following commands:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('averaged_perceptron_tagger')
   ```

### 3. Required Model Files

Make sure the following pre-trained model and configuration files are placed in the root directory of the project:

* `emotion_model.pkl` (Trained SVM model for better accuracy and also for faster model training RF and XGBoost used)
* `scaler.pkl` (Scikit-learn scaler for emotion features)
* `posture_model.pkl` (Trained XGBoost model for posture)
* `haarcascade_frontalface_default.xml` (OpenCV face detection cascade)

## 🏃‍♀️ How to Use

1. **Run the Flask application:**
   ```bash
   python app.py
   ```

   The first time you run the application, it will download the Wav2Vec2 speech recognition model from Hugging Face (approx. 380MB). This may take a few minutes.

2. **Open your browser:** Navigate to `http://127.0.0.1:5000`.

3. **Grant permissions:** Your browser will ask for permission to use your camera and microphone. Please allow access.

4. **Start practicing!** The three feedback panels will activate, providing you with live insights into your interview performance.

## 📁 Project Structure

```
/ai-interview-coach
├── app.py                      # Main Flask application
├── emotion_model.pkl           # Trained emotion model
├── scaler.pkl                  # Scaler for emotion features
├── posture_model.pkl           # Trained posture model
├── haarcascade_frontalface_default.xml # Face detection file
├── requirements.txt            # Project dependencies
└── /templates
    └── index.html              # Frontend HTML and JavaScript
```
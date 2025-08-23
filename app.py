import cv2
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
from skimage import feature
from skimage.feature import hog
import joblib
import time
import threading
import queue

# Libraries for speech model 
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pyaudio
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng')

# --- 1. INITIALIZATION ---
app = Flask(__name__)
socketio = SocketIO(app)

# Emotion model config
try:
    emotion_model = joblib.load('./Emotion detection/ensemble_model_fitted.pkl')
    scaler = joblib.load('./Emotion detection/scaler_new.pkl')
    pca = joblib.load('./Emotion detection/pca_new.pkl')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("❌ Error: haarcascade_frontalface_default.xml not loaded correctly")
        exit()
    print("✅ Model, scaler, PCA, and cascade classifier loaded successfully!")
except Exception as e:
    print(f"❌ Error loading emotion files: {e}")
    exit()

# Posture model config
try:
    posture_model = joblib.load('./Posture detection/posture_model.pkl')
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)
    print("✅ Posture detection files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading posture files: {e}")
    exit()

# Single shared camera
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not camera.isOpened():
    print("❌ Cannot open camera")
    exit()
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
camera.set(cv2.CAP_PROP_FPS, 30)
print("✅ Shared camera initialized")

# --- Load Speech Recognition Model (Wav2Vec2) ---
try:
    # Use a smaller, faster model for real-time inference
    model_name = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    print("✅ Speech recognition model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading speech model: {e}")

# Global frame buffer and lock for thread safety
latest_frame = None
frame_lock = threading.Lock()

# Background thread to read frames
def read_frames():
    global latest_frame
    while True:
        success, frame = camera.read()
        if success and frame is not None:
            with frame_lock:
                latest_frame = frame.copy()
        else:
            print("❌ Failed to read frame in background thread")
            time.sleep(0.1)

# Start background thread
threading.Thread(target=read_frames, daemon=True).start()

# Emotion, Posture labels mapping
emotion_labels = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
posture_labels = {0: 'Bending', 1: 'Lying', 2: 'Sitting', 3: 'Standing'}

# --- 2. FEATURE EXTRACTION FUNCTIONS ---
def get_lbp_features(image):
    lbp = feature.local_binary_pattern(image, P=24, R=8, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def get_hog_features(image):
    features = hog(image, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), block_norm="L2-Hys", visualize=False)
    return features

def extract_hu_moments_from_mask(mask):
    moments = cv2.moments(mask)
    hu_moments = cv2.HuMoments(moments)
    hu_moments = -1 * np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-7)
    return hu_moments.flatten()

# --- Simple Grammar Checker ---
def check_grammar(text):
    """A very basic grammar checker for interview practice."""
    suggestions = []
    tokens = nltk.word_tokenize(text.lower())
    tagged = nltk.pos_tag(tokens)
    
    # Rule 1: Check for past tense after "I"
    for i, (word, tag) in enumerate(tagged):
        if i > 0 and tagged[i-1][0] == 'i' and tag.startswith('VBP'): # e.g., "I go"
            suggestions.append(f"Suggestion: For past events, use past tense (e.g., 'I went' instead of 'I go').")

    # Rule 2: Check for missing 'a' or 'the' before a noun
    for i, (word, tag) in enumerate(tagged):
        if tag.startswith('NN') and (i == 0 or not tagged[i-1][0] in ['a', 'an', 'the', 'my', 'your']):
             suggestions.append(f"Suggestion: Consider using an article like 'a' or 'the' before '{word}'.")
             
    return list(set(suggestions)) # Return unique suggestions

# --- 3. REAL-TIME SPEECH-TO-TEXT THREAD ---
class SpeechToText:
    def __init__(self):
        self.audio_format = pyaudio.paInt16
        self.channels = 1
        self.rate = 16000
        self.chunk = 1024
        self.audio_interface = pyaudio.PyAudio()
        self.stream = None
        self.is_running = False
        self.audio_queue = queue.Queue()

    def start(self):
        self.is_running = True
        self.stream = self.audio_interface.open(format=self.audio_format,
                                                channels=self.channels,
                                                rate=self.rate,
                                                input=True,
                                                frames_per_buffer=self.chunk,
                                                stream_callback=self._fill_queue)
        self.stream.start_stream()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        print("🎙️  Microphone stream started.")

    def stop(self):
        self.is_running = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio_interface.terminate()
        print("🎤 Microphone stream stopped.")

    def _fill_queue(self, in_data, frame_count, time_info, status):
        self.audio_queue.put(in_data)
        return (None, pyaudio.paContinue)

    def _process_audio(self):
        buffer = []
        while self.is_running:
            try:
                data = self.audio_queue.get(timeout=1)
                buffer.append(data)

                # Process in chunks of ~1 second
                if len(buffer) >= (self.rate // self.chunk):
                    audio_data = b''.join(buffer)
                    buffer = []
                    
                    # Convert byte data to writable torch tensor
                    waveform_np = np.frombuffer(audio_data, dtype=np.int16).copy()  # .copy() makes it writable
                    waveform = torch.from_numpy(waveform_np).float()
                    
                    # Resample if necessary (though we open stream at 16k)
                    if self.rate != 16000:
                        resampler = torchaudio.transforms.Resample(self.rate, 16000)
                        waveform = resampler(waveform)

                    # Process and transcribe
                    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt", padding=True)
                    with torch.no_grad():
                        logits = model(**inputs).logits
                    
                    predicted_ids = torch.argmax(logits, dim=-1)
                    transcription = processor.batch_decode(predicted_ids)[0]
                    
                    if transcription:
                        suggestions = check_grammar(transcription)
                        # Emit to the client via WebSocket
                        socketio.emit('speech_update', {
                            'transcription': transcription,
                            'suggestions': suggestions
                        })
            except queue.Empty:
                continue

stt = SpeechToText()

# --- 3. VIDEO STREAMING GENERATOR ---
def generate_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                print("❌ No frame available for emotion feed")
                time.sleep(0.1)
                continue
            frame = latest_frame.copy()
        
        # Convert frame to grayscale for face detection and LBP
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # Process each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            roi_resized = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            lbp_features = get_lbp_features(roi_resized)
            hog_features = get_hog_features(roi_resized)
            combined_features = np.hstack([lbp_features, hog_features])
            scaled_features = scaler.transform(combined_features.reshape(1, -1))
            pca_features = pca.transform(scaled_features)
            prediction = emotion_model.predict(pca_features)
            predicted_emotion = emotion_labels[prediction[0]]
            feedback_text = f"Emotion: {predicted_emotion}"
            if predicted_emotion in ['Sad', 'Angry', 'Fear']:
                feedback_text += " - Try to smile for a confident look!"
            elif predicted_emotion == 'Happy':
                feedback_text += " - Great smile!"
            cv2.putText(frame, feedback_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            print("❌ Failed to encode frame for emotion feed")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- Posture Feed Generator ---
def generate_posture_frames():
    global latest_frame
    while True:
        with frame_lock:
            if latest_frame is None:
                print("❌ No frame available for posture feed")
                time.sleep(0.1)
                continue
            frame = latest_frame.copy()

        # Apply background subtraction
        fg_mask = bg_subtractor.apply(frame)
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.erode(fg_mask, kernel, iterations=1)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 5000:
                cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 3)
                contour_mask = np.zeros(frame.shape[:2], dtype="uint8")
                cv2.drawContours(contour_mask, [largest_contour], -1, 255, -1)
                hu_features = extract_hu_moments_from_mask(contour_mask)
                prediction = posture_model.predict(hu_features.reshape(1, -1))
                predicted_posture = posture_labels[prediction[0]]
                feedback_text = f"Posture: {predicted_posture}"
                if predicted_posture == 'Sitting':
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    aspect_ratio = h / w
                    if aspect_ratio < 1.5:
                        feedback_text += " - Sit up straight!"
                    else:
                        feedback_text += " - Good posture!"
                cv2.putText(frame, feedback_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ret:
            print("❌ Failed to encode frame for posture feed")
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- 4. FLASK ROUTES ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/posture_feed')
def posture_feed():
    return Response(generate_posture_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- SocketIO Events for Speech ---
@socketio.on('connect')
def handle_connect(auth=None):
    print('Client connected')
    stt.start()

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')
    stt.stop()

# --- 5. MAIN EXECUTION ---
if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"❌ Flask app error: {e}")
    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("✅ Shared camera released")
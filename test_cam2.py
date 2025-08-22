import cv2
from flask import Flask, Response
import numpy as np

app = Flask(__name__)

# Initialize camera
camera = cv2.VideoCapture(0)

def generate_frames(process_type="raw"):
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            if process_type == "processed":
                # Example: Convert to grayscale
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Convert grayscale to RGB for streaming
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed_raw')
def video_feed_raw():
    return Response(generate_frames(process_type="raw"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed_processed')
def video_feed_processed():
    return Response(generate_frames(process_type="processed"),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return """
    <html>
        <body>
            <h1>Two Camera Feeds from Single Camera</h1>
            <div>
                <h2>Raw Feed</h2>
                <img src="/video_feed_raw" width="45%">
                <h2>Processed Feed (Grayscale)</h2>
                <img src="/video_feed_processed" width="45%">
            </div>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(debug=True)
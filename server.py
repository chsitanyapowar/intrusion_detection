from flask import Flask, render_template, Response, request, jsonify
import cv2
import dlib
import numpy as np
import pickle
import threading
import os
import time
from datetime import datetime
from config import PREDICTOR_PATH, FACE_RECOG_MODEL_PATH, ENCODINGS_FILE

app = Flask(__name__, template_folder="templates")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)

with open(ENCODINGS_FILE, "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

cap = None  # Initialize camera as None
frame_lock = threading.Lock()
unknown_face_detected = False
unknown_face_start_time = None
photo_capture_times = []

def process_frame(frame):
    global unknown_face_detected, unknown_face_start_time, photo_capture_times
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    current_time = time.time()

    for face in faces:
        shape = sp(gray, face)
        face_descriptor = facerec.compute_face_descriptor(frame, shape)
        face_encoding = np.array(face_descriptor)
        min_distance = float("inf")
        name = "Unknown"

        for i, known_encoding in enumerate(known_face_encodings):
            distance = np.linalg.norm(known_encoding - face_encoding)
            if distance < min_distance and distance < 0.5:
                min_distance = distance
                name = known_face_names[i]

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if name == "Unknown":
            if not unknown_face_detected:
                unknown_face_detected = True
                unknown_face_start_time = current_time
            elif current_time - unknown_face_start_time > 5:
                threading.Thread(target=save_photo, args=(frame,)).start()
                if current_time - unknown_face_start_time > 10:
                    threading.Thread(target=save_photo, args=(frame,)).start()
                    unknown_face_detected = False  # Reset after capturing the second photo
        else:
            unknown_face_detected = False

    return frame

def save_photo(frame):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = r"E:\project\Intrusion_Detection\unknowns"
    os.makedirs(save_path, exist_ok=True)
    photo_path = os.path.join(save_path, f"unknown_{timestamp}.jpg")
    cv2.imwrite(photo_path, frame)
    photo_capture_times.append(timestamp)
    print(f"Photo saved: {photo_path}")

def generate_frames():
    global cap
    if cap is None:
        cap = cv2.VideoCapture(0)  # Start camera if not already running
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set width
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set height
        cap.set(cv2.CAP_PROP_FPS, 15)  # Set frame rate

    frame_count = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)

        # Process every 2nd frame to reduce load
        if frame_count % 2 == 0:
            with frame_lock:
                frame = process_frame(frame)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        frame_count += 1

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Open the camera
        return jsonify({"status": "Camera Started"})
    return jsonify({"status": "Camera Already Running"})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    global cap
    if cap is not None:
        cap.release()
        cv2.destroyAllWindows()
        cap = None
        return jsonify({"status": "Camera Stopped"})
    return jsonify({"status": "Camera Not Running"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)

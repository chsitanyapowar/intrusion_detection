import cv2
import dlib
import numpy as np
import pickle
from config import PREDICTOR_PATH, FACE_RECOG_MODEL_PATH, ENCODINGS_FILE

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)

with open(ENCODINGS_FILE, "rb") as f:
    known_face_encodings, known_face_names = pickle.load(f)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
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
            print("⚠️ Unknown person detected!")
    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

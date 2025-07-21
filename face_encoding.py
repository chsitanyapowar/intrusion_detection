import cv2
import dlib
import os
import numpy as np
import pickle
from config import PREDICTOR_PATH, FACE_RECOG_MODEL_PATH, DATASET_PATH, ENCODINGS_FILE

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(PREDICTOR_PATH)
facerec = dlib.face_recognition_model_v1(FACE_RECOG_MODEL_PATH)

def encode_faces(dataset_path):
    known_face_encodings = []
    known_face_names = []

    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_path):
            continue
        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            image = cv2.imread(img_path)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                shape = sp(gray, face)
                face_descriptor = facerec.compute_face_descriptor(image, shape)
                known_face_encodings.append(np.array(face_descriptor))
                known_face_names.append(person_name)
    return known_face_encodings, known_face_names

print("Encoding authorized faces... Please wait.")
known_face_encodings, known_face_names = encode_faces(DATASET_PATH)

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump((known_face_encodings, known_face_names), f)

print("Face encoding completed and saved.")
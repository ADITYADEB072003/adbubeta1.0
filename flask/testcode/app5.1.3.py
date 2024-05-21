import cv2
from flask import Flask, render_template, Response
import face_recognition
import pickle
import numpy as np
import os

app = Flask(__name__)

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def load_known_encodings():
    known_encodings_file = 'known_encodings.pkl'
    if os.path.exists(known_encodings_file):
        with open(known_encodings_file, 'rb') as f:
            known_encodings = pickle.load(f)
    else:
        known_encodings = {}
    return known_encodings

def recognize_faces(frame, known_encodings):
    rgb_frame = frame[:, :, ::-1]  # Convert BGR to RGB (OpenCV uses BGR)

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_encodings['encodings'], face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_encodings['names'][first_match_index]

        # Draw a rectangle around the face
        top, right, bottom, left = face_locations[0]
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 255, 0), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    return frame

@app.route('/')
def index():
    return render_template('index1.html')

@app.route('/video_feed')
def video_feed():
    known_encodings = load_known_encodings()
    return Response(recognize_faces(generate_frames(), known_encodings), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5002)

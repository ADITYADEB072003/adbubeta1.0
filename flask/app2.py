import logging
from flask import Flask, render_template, Response
import cv2
import face_recognition
import numpy as np
import os

app = Flask(__name__)
camera = cv2.VideoCapture(0)

known_face_encodings = []
known_face_names = []

# Function to load face encodings and names from images in subfolders
def load_known_faces():
    base_dir = "/Users/adityadebchowdhury/Desktop/Desktop - Aditya’s MacBook Air/opencv2/faceapi/labels"
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            for filename in os.listdir(folder_path):
                image_path = os.path.join(folder_path, filename)
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    # Load image and encode face
                    try:
                        face_image = face_recognition.load_image_file(image_path)
                        face_encoding = face_recognition.face_encodings(face_image)
                        if len(face_encoding) > 0:
                            known_face_encodings.append(face_encoding[0])
                            known_face_names.append(folder)
                        else:
                            logging.warning(f"No face detected in {image_path}")
                    except Exception as e:
                        logging.error(f"Error processing {image_path}: {str(e)}")

# Generator function to process video frames
def gen_frames():  
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]  # Convert BGR to RGB

            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                # Compare faces with known faces
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Find the best match
                if True in matches:
                    match_index = matches.index(True)
                    name = known_face_names[match_index]

                face_names.append(name)

            # Display results on the frame
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw rectangle around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw label with name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            # Encode frame as JPEG and yield for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    load_known_faces()  # Load known faces at startup
    app.run(debug=True)

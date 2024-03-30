from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

app = Flask(__name__)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) > 0:
            encode = face_recognition.face_encodings(img, face_locations)[0]
            encodeList.append(encode)
    return encodeList

def markAttendance(studentID, studentName, folder_name, marked_students):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    if studentID not in marked_students:
        csv_path = os.path.join(folder_name, 'Attendance.csv')
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([studentID, studentName, dtString])
        print(f'Attendance recorded: {studentID}, {studentName}, {dtString}')
        marked_students.add(studentID)
    else:
        print(f'{studentID} - {studentName} already marked attendance.')

def gen_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/mark_attendance', methods=['POST'])
def mark_attendance():
    studentID = request.json.get('studentID')
    studentName = request.json.get('studentName')
    folder_name = request.json.get('folderName')

    if studentID is None or studentName is None or folder_name is None:
        return jsonify({'error': 'Student ID, name, and folder name are required parameters.'}), 400

    markAttendance(studentID, studentName, folder_name, marked_students)

    return jsonify({'message': f'Attendance recorded for {studentID}, {studentName} in {folder_name}.'}), 200

if __name__ == '__main__':
    app.run(debug=True)

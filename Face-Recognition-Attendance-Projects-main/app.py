from flask import Flask, render_template, request
import cv2
import numpy as np
import face_recognition
from datetime import datetime
import csv
import os
app = Flask(__name__)

# Path to the directory containing student images
path = 'Student_Images'
images = []
studentIDs = []
studentNames = []
myList = os.listdir(path)

# Load the student images and their corresponding IDs and names
for file in myList:
    img_path = os.path.join(path, file)
    if os.path.isfile(img_path):
        student_img = cv2.imread(img_path)
        images.append(student_img)
        studentID, studentName = os.path.splitext(file)[0].split('_')
        studentIDs.append(studentID)
        studentNames.append(studentName)

# Function to find encodings of the student images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark attendance and store in CSV
def markAttendance(studentID, studentName, marked_students):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    if studentID not in marked_students:
        with open('Attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([studentID, studentName, dtString])
        print(f'Attendance recorded: {studentID}, {studentName}, {dtString}')
        marked_students.add(studentID)
    else:
        print(f'{studentID} - {studentName} already marked attendance.')

# Encoding the student images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set to store marked students
marked_students = set()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/take_attendance', methods=['POST'])
def take_attendance():
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Detect faces in the captured frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Loop through the detected faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            studentID = studentIDs[matchIndex]
            studentName = studentNames[matchIndex]
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f"{studentID} - {studentName}", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            # Check if 'r' key is pressed to record attendance
            if cv2.waitKey(1) & 0xFF == ord('r'):
                markAttendance(studentID, studentName, marked_students)

    cv2.imwrite('static/attendance_capture.jpg', img)  # Save the captured frame
    return render_template('attendance.html')

if __name__ == '__main__':
    app.run(debug=True)

import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

# Path to the directory containing student images
path = '/Users/adityadebchowdhury/Desktop/Desktop - Aditya’s MacBook Air/opencv2/Face-Recognition-Attendance-Projects-main/Training_images/output'
images = []
studentIDs = []
studentNames = []

# Load images from subfolders
for root, dirs, files in os.walk(path):
    for directory in dirs:
        subdir_path = os.path.join(root, directory)
        for file in os.listdir(subdir_path):
            img_path = os.path.join(subdir_path, file)
        if os.path.isfile(img_path):
             student_img = cv2.imread(img_path)
        if student_img is None:
            print(f"Error: Unable to load image from {img_path}")
            continue  # Skip to the next image if loading fails

        # Extract studentID and studentName from filename
        studentID, studentName, *_ = os.path.splitext(file)[0].split('_')

        images.append(student_img)
        studentIDs.append(studentID)
        studentNames.append(studentName)

# Function to find encodings of the student images
# Function to find encodings of the student images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find face locations in the image
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) > 0:
            # If faces are detected, extract encodings
            encode = face_recognition.face_encodings(img, face_locations)[0]
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

while True:
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

    # Display the frame
    cv2.imshow('College Attendance System', img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam
cap.release()
cv2.destroyAllWindows()

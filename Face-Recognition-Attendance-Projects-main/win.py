import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import csv

path = r'C:\Users\adity\OneDrive\Documents\opencv2\Face-Recognition-Attendance-Projects-main\Student_Images'
images = []
studentIDs = []
studentNames = []

# Load images from subfolders and extract student IDs and names from folder names
for root, dirs, files in os.walk(path):
    for directory in dirs:
        subdir_path = os.path.join(root, directory)
        if os.path.isdir(subdir_path):
            studentID, studentName = directory.split('_')
            for file in os.listdir(subdir_path):
                img_path = os.path.join(subdir_path, file)
                if os.path.isfile(img_path):
                    # Attempt to load the image
                    student_img = cv2.imread(img_path)
                    if student_img is not None:  # Check if the image is successfully loaded
                        images.append(student_img)
                        studentIDs.append(studentID)
                        studentNames.append(studentName)
                    else:
                        print(f"Error loading image: {img_path}")

# Function to find encodings of the student images
def findEncodings(images):
    encodeList = []
    for img in images:
        # Convert image to RGB format
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Find face locations in the image
        face_locations = face_recognition.face_locations(img_rgb)
        if face_locations:
            # Extract face encodings
            encode = face_recognition.face_encodings(img_rgb, face_locations)[0]
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

# Main loop for real-time face recognition
while True:
    success, img = cap.read()
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Detect faces in the captured frame
    facesCurFrame = face_recognition.face_locations(img_rgb)
    encodesCurFrame = face_recognition.face_encodings(img_rgb, facesCurFrame)
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
    cv2.imshow('College Attendance System', img)
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

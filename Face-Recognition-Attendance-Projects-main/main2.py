import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path to the directory containing training images
path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)

# Load the training images and their corresponding class names
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# Function to find encodings of the training images
def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

# Function to mark attendance
def markAttendance(name, marked_names):
    if name not in marked_names:
        with open('Attendance.csv', 'a') as f:
            now = datetime.now()
            dtString = now.strftime('%Y-%m-%d %H:%M:%S')
            line = f'{name},{dtString}\n'
            f.write(line)
            print(f'Attendance recorded: {line}')
        marked_names.append(name)
    else:
        print(f'{name} already marked attendance.')

# Encoding the training images
encodeListKnown = findEncodings(images)
print('Encoding Complete')

# Initialize webcam
cap = cv2.VideoCapture(0)

marked_names = []

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
            name = classNames[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Check if 'r' key is pressed to record attendance
            if cv2.waitKey(1) & 0xFF == ord('r'):
                markAttendance(name, marked_names)

    # Display the frame
    cv2.imshow('Webcam', img)
    
    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

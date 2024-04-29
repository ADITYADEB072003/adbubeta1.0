import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path = 'Training_images'
images = []
classNames = []
myList = os.listdir(path)
print(myList)

for cl in myList:
    img_path = os.path.join(path, cl)
    try:
        curImg = cv2.imread(img_path)
        if curImg is None:
            print(f"Error: Unable to load image from {img_path}")
            continue  # Skip to the next image if loading fails
        images.append(curImg)
        classNames.append(os.path.splitext(cl)[0])
    except Exception as e:
        print(f"Error loading image '{img_path}': {str(e)}")

print(classNames)


def findEncodings(images):
    encodeList = []
    for img in images:
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            face_encodings = face_recognition.face_encodings(img)
            if len(face_encodings) > 0:
                encode = face_encodings[0]
                encodeList.append(encode)
            else:
                print("No face detected in the image.")
        except Exception as e:
            print(f"Error processing image: {str(e)}")
    return encodeList


def markAttendance(name):
    with open('Attendance.csv', 'a') as f:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        line = f'{name},{dtString}\n'
        f.write(line)
        print(f'Attendance recorded: {line}')


encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
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
                markAttendance(name)

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

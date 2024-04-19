import cv2
import os
import face_recognition
from datetime import datetime
import csv

# Path to the directory containing student images
path = 'Student_Images'

# Load known student images and IDs from subfolders
known_students = {}
for root, dirs, files in os.walk(path):
    for directory in dirs:
        studentID, studentName = directory.split('_')
        student_images = []
        for file in os.listdir(os.path.join(root, directory)):
            img_path = os.path.join(root, directory, file)
            img = cv2.imread(img_path)
            if img is not None:
                # Resize image for faster processing (optional)
                img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
                
                # Extract face encodings if a face is detected
                face_encodings = face_recognition.face_encodings(img)
                if len(face_encodings) > 0:
                    known_students[studentID] = {
                        'name': studentName,
                        'encodings': face_encodings
                    }
                else:
                    print(f"No face detected in {img_path}")

# Function to recognize faces using face_recognition library
def recognize_student_faces(frame, known_students):
    # Find face locations in the frame
    face_locations = face_recognition.face_locations(frame)
    
    # Check if any faces are detected
    if len(face_locations) == 0:
        return None, None

    # Encode the faces in the current frame
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Initialize variables to store the matched student ID and face coordinates
    matched_studentID = None
    min_distance = float('inf')
    best_face_coords = None

    # Iterate over each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Iterate over known students to find the best match
        for studentID, student_data in known_students.items():
            for known_encoding in student_data['encodings']:
                # Compare face encodings using face_recognition library
                distance = face_recognition.face_distance([known_encoding], face_encoding)

                # Update matched student if current distance is lower
                if distance < min_distance:
                    min_distance = distance
                    matched_studentID = studentID
                    best_face_coords = (left, top, right, bottom)

    # Return the matched student ID and best face coordinates
    return matched_studentID, best_face_coords

# Function to mark attendance and store in CSV
def mark_attendance(studentID, studentName):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')
    with open('Attendance.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([studentID, studentName, dtString])
    print(f'Attendance recorded: {studentID}, {studentName}, {dtString}')

# Initialize webcam
cap = cv2.VideoCapture(0)

# Main loop for real-time face recognition
while True:
    # Capture frame from webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to RGB (for face_recognition library)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Recognize student faces in the frame
    studentID, face_coords = recognize_student_faces(rgb_frame, known_students)

    # Display rectangle and student name if a match is found
    if studentID is not None and face_coords is not None:
        left, top, right, bottom = face_coords
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        studentName = known_students[studentID]['name']
        cv2.putText(frame, studentName, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Mark attendance if 'r' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('r'):
            mark_attendance(studentID, studentName)

    # Display the frame
    cv2.imshow('Attendance System', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close all windows
cap.release()
cv2.destroyAllWindows()

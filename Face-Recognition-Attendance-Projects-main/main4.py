import cv2                      # Importing the OpenCV library for image processing
import numpy as np              # Importing NumPy for numerical operations
import face_recognition         # Importing the face_recognition library for face detection and recognition
import os                       # Importing the os module for interacting with the file system
from datetime import datetime  # Importing the datetime module for timestamping attendance records
import csv                      # Importing the csv module for writing data to CSV files

# Path to the directory containing student images
path = 'Student_Images'

# Lists to store student images, IDs, and names
images = []
studentIDs = []
studentNames = []

# Load images from subfolders and extract student IDs and names from folder names
for root, dirs, files in os.walk(path):  # Traverse through the directory tree
    for directory in dirs:               # Iterate over subdirectories
        subdir_path = os.path.join(root, directory)  # Construct subdirectory path
        if os.path.isdir(subdir_path):   # Check if it's a directory
            # Extract student ID and name from folder name
            studentID, studentName = directory.split('_')
            for file in os.listdir(subdir_path):
             img_path = os.path.join(subdir_path, file)
            if os.path.isfile(img_path):
             student_img = cv2.imread(img_path)
             if student_img is None:
                    print(f"Error: Unable to load image from {img_path}")
                    continue  # Skip to the next image if loading fails

        images.append(student_img)
        studentIDs.append(studentID)
        studentNames.append(studentName)


# Function to find encodings of the student images
def findEncodings(images):
    encodeList = []  # Initialize list to store face encodings
    for img in images:  # Iterate over the list of images
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB format
        # Find face locations in the image
        face_locations = face_recognition.face_locations(img)
        if len(face_locations) > 0:  # If faces are detected
            # Extract face encodings
            encode = face_recognition.face_encodings(img, face_locations)[0]
            encodeList.append(encode)  # Append face encoding to the list
    return encodeList

# Function to mark attendance and store in CSV
def markAttendance(studentID, studentName, marked_students):
    now = datetime.now()  # Get current date and time
    dtString = now.strftime('%Y-%m-%d %H:%M:%S')  # Format date and time as string
    if studentID not in marked_students:  # Check if student ID is not already marked
        # Open the CSV file in append mode
        with open('Attendance.csv', 'a', newline='') as f:
            writer = csv.writer(f)  # Create CSV writer object
            # Write student ID, name, and timestamp to the CSV file
            writer.writerow([studentID, studentName, dtString])
        # Print message indicating attendance recorded
        print(f'Attendance recorded: {studentID}, {studentName}, {dtString}')
        marked_students.add(studentID)  # Add student ID to the set of marked students
    else:
        # Print message indicating student already marked attendance
        print(f'{studentID} - {studentName} already marked attendance.')

# Encoding the student images
encodeListKnown = findEncodings(images)  # Call findEncodings function to encode student images
print('Encoding Complete')  # Print message indicating encoding is complete

# Initialize webcam
cap = cv2.VideoCapture(0)  # Open the default camera (index 0)

# Set to store marked students
marked_students = set()  # Initialize an empty set to store marked students

# Main loop for real-time face recognition
while True:  # Infinite loop to continuously capture frames
    success, img = cap.read()  # Read a frame from the webcam
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)  # Resize the frame
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB format

    # Detect faces in the captured frame
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    # Loop through the detected faces
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        # Compare the face with known faces
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:  # If match found with known faces
            studentID = studentIDs[matchIndex]  # Get the student ID
            studentName = studentNames[matchIndex]  # Get the student name
            # Unpack the face location coordinates
            y1, x2, y2, x1 = faceLoc
            # Scale the coordinates back to original size
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            # Draw rectangle around the face
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw filled rectangle for displaying text
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            # Put text (student ID and name) on the image
            cv2.putText(img, f"{studentID} - {studentName}", (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

            # Check if 'r' key is pressed to record attendance
            if cv2.waitKey(1) & 0xFF == ord('r'):
                markAttendance(studentID, studentName, marked_students)

    # Display the frame with detected faces and student information
    cv2.imshow('College Attendance System', img)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

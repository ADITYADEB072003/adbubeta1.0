press r to record and to quite

## add images in training folder
# College Attendance System

This Python script utilizes OpenCV and the face_recognition library to create a college attendance system using a webcam. It detects faces in real-time, matches them with known student images, and records their attendance in a CSV file.

## Setup

### Libraries Required
- `cv2` (OpenCV): for capturing images and processing frames from the webcam.
- `numpy`: for numerical operations.
- `face_recognition`: for face detection and recognition.
- `os`: for interacting with the file system.
- `datetime`: for timestamping attendance records.
- `csv`: for writing attendance data to a CSV file.

## Functionality

### Loading Student Images
- `path = 'Student_Images'`: Specifies the path to the directory containing student images.
- The script loads student images from subfolders, extracting student IDs and names from folder names.

### Encoding Student Images
- The `findEncodings(images)` function extracts face encodings from student images using the face_recognition library.

### Marking Attendance
- The `markAttendance(studentID, studentName, marked_students)` function records attendance and stores it in a CSV file.
- It includes the student ID, name, and timestamp.

## Main Loop
- `cap = cv2.VideoCapture(0)`: Initializes the webcam for capturing frames.
- `encodeListKnown = findEncodings(images)`: Encodes the student images.
- The main loop continuously captures frames from the webcam:
  - Detects faces in the captured frame.
  - Compares detected faces with known faces.
  - Draws rectangles around recognized faces and displays student ID and name.
  - Records attendance if the 'r' key is pressed.
  - Displays the frame with detected faces and student information.
  - Ends the loop if the 'q' key is pressed.
- The webcam is released and OpenCV windows are closed after the loop ends.

## Running the Script
1. Ensure that the required libraries are installed (`cv2`, `numpy`, `face_recognition`, `os`, `datetime`, `csv`).
2. Place student images in the `Student_Images` directory with subfolders named as `studentID_studentName`.
3. Run the script.
4. Press the 'r' key to mark attendance for recognized students.
5. Press the 'q' key to quit the application.


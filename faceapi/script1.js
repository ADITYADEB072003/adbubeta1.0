
const fs = require('fs');

const video = document.getElementById("video");
let detectedFacesData = []; // Array to store detected face data

// Load Face-API models and start webcam
Promise.all([
  faceapi.nets.ssdMobilenetv1.loadFromUri("/models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("/models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("/models"),
]).then(startWebcam);

// Function to start webcam and set video stream
function startWebcam() {
  navigator.mediaDevices
    .getUserMedia({ video: true, audio: false })
    .then((stream) => {
      video.srcObject = stream;
    })
    .catch((error) => {
      console.error(error);
    });
}

// Function to load labeled face descriptions from subfolder names
async function getLabeledFaceDescriptions() {
  const labelsPath = './labels'; // Path to the directory containing subfolders with labels
  const labels = fs.readdirSync(labelsPath);
  const fs = require('fs');

  const labeledDescriptorsPromises = labels.map(async (label) => {
    try {
      const labelName = label.split('_')[1]; // Extract label from subfolder name
      const descriptions = await loadFaceDescriptor(label);
      return new faceapi.LabeledFaceDescriptors(labelName, [descriptions]);
    } catch (error) {
      console.error(`Error loading face descriptor for ${label}:`, error);
      return null;
    }
  });

  // Resolve all promises concurrently
  return Promise.all(labeledDescriptorsPromises);
}

// Function to load face descriptor for a labeled folder
async function loadFaceDescriptor(label) {
  try {
    const imgPath = `./labels/${label}/1.jpg`; // Load descriptor for the first image
    const img = await faceapi.fetchImage(imgPath);
    const detections = await faceapi
      .detectSingleFace(img)
      .withFaceLandmarks()
      .withFaceDescriptor();
    return detections.descriptor;
  } catch (error) {
    console.error(`Error loading face descriptor for ${label}: ${error.message}`);
    return null;
  }
}

// Function to format and download CSV file
function writeToCSV(data) {
  const csvContent = "Label,Timestamp\n" + data.map(row => row.join(",")).join("\n");
  const encodedUri = encodeURI("data:text/csv;charset=utf-8," + csvContent);
  const link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "detected_faces.csv");
  document.body.appendChild(link);
  link.click();
}

// Event listener when video playback starts
video.addEventListener("play", async () => {
  const labeledFaceDescriptors = await getLabeledFaceDescriptions();
  const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors);

  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);

  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  // Interval for face detection and recognition
  setInterval(async () => {
    const detections = await faceapi
      .detectAllFaces(video)
      .withFaceLandmarks()
      .withFaceDescriptors();

    const resizedDetections = faceapi.resizeResults(detections, displaySize);

    canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);

    resizedDetections.forEach((detection) => {
      const result = faceMatcher.findBestMatch(detection.descriptor);
      const box = detection.detection.box;
      const label = result.label;

      // Record detected face label and timestamp
      const timestamp = new Date().toISOString();
      detectedFacesData.push([label, timestamp]);

      const drawBox = new faceapi.draw.DrawBox(box, {
        label: label,
      });
      drawBox.draw(canvas);
    });
  }, 1000); // Update every 1 second

  // Button to trigger CSV download
  const downloadButton = document.createElement("button");
  downloadButton.textContent = "Download CSV";
  downloadButton.addEventListener("click", () => {
    writeToCSV(detectedFacesData);
    detectedFacesData = []; // Clear data after writing to CSV
  });
  document.body.appendChild(downloadButton);
});

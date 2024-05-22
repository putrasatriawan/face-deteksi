const video = document.getElementById("video");
const lastSentTime = {};
let recognizedFaces = {};
// Pre-load models
Promise.all([
  faceapi.nets.mtcnn.loadFromUri("./models"),
  faceapi.nets.faceLandmark68Net.loadFromUri("./models"),
  faceapi.nets.faceRecognitionNet.loadFromUri("./models"),
  faceapi.nets.ssdMobilenetv1.loadFromUri("./models")
]).then(startVideo);

function startVideo() {
  navigator.mediaDevices
    .getUserMedia({ video: {} })
    .then((stream) => (video.srcObject = stream))
    .catch((err) => console.error(err));
}

video.addEventListener("play", () => {
  const canvas = faceapi.createCanvasFromMedia(video);
  document.body.append(canvas);
  const displaySize = { width: video.width, height: video.height };
  faceapi.matchDimensions(canvas, displaySize);

  // Define known faces
  const knownFaces = [
    { name: "Niki Marco", imageURL: "Image/Niki.JPG" },
    { name: "Arizal", imageURL: "Image/Arizal.jpeg" },
    { name: "Putra", imageURL: "Image/Putra.jpg" },
    // Add more persons with their respective images
  ];

  // Preload images and compute descriptors
  Promise.all(
    knownFaces.map(async (face) => {
      const image = await faceapi.fetchImage(face.imageURL);
      const detections = await faceapi
        .detectSingleFace(image)
        .withFaceLandmarks()
        .withFaceDescriptor();

      if (!detections) {
        throw new Error(`No face detected in ${face.imageURL}`);
      }

      return new faceapi.LabeledFaceDescriptors(
        face.name,
        [detections.descriptor]
      );
    })
  )
    .then((labeledDescriptors) => {
      // Initialize face matcher
      const faceMatcher = new faceapi.FaceMatcher(labeledDescriptors);

      setInterval(async () => {
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.MtcnnOptions())
          .withFaceLandmarks()
          .withFaceDescriptors();
        const resizedDetections = faceapi.resizeResults(
          detections,
          displaySize
        );

        canvas
          .getContext("2d")
          .clearRect(0, 0, canvas.width, canvas.height);
        faceapi.draw.drawDetections(canvas, resizedDetections);
        faceapi.draw.drawFaceLandmarks(canvas, resizedDetections);

        // Face recognition
        resizedDetections.forEach((detection) => {
          const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
          const text = bestMatch.distance < 0.4 ? "unknown" : bestMatch.toString();
          const { x, y, width, height } = detection.detection.box;
          const currentTime = Date.now();

          // Draw text with name
          canvas.getContext("2d").font = "24px Arial";
          canvas.getContext("2d").fillStyle = "#ffffff";
          canvas.getContext("2d").fillText(text, x + 5, y + height + 24);

          if (bestMatch.distance >= 0.4 && bestMatch.label !== "unknown") {
          const name = bestMatch.label;

          if (!recognizedFaces[name]) {
            recognizedFaces[name] = currentTime; // Start recognition timer
          } else if (currentTime - recognizedFaces[name] >= 2000) {
          // If recognized for at least 2 seconds, send to server
          if (
            !lastSentTime[name] ||
            currentTime - lastSentTime[name] >= 3600000 // 1 hour in milliseconds
          ) {
            sendToServer(name, new Date().toISOString());
            lastSentTime[name] = currentTime;
            }
            delete recognizedFaces[name]; // Reset recognition timer
          }
          } else {
          // Reset recognition timer if not recognized
            recognizedFaces = {};
          }
        });
      }, 100);
    })
    .catch((err) => console.error(err));
});

function sendToServer(name, time) {
  // Data to be sent to the server
  const data = {
    name: name,
    time: time
  };

  console.log('Data:', data);
}
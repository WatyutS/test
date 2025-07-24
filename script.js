let video = document.getElementById("webcam");
let alertBox = document.getElementById("alert-box");
let model = null;
let camera = null;

const soundMap = {
  drowsy: document.getElementById("sound-drowsy"),
  distracted: document.getElementById("sound-distracted"),
  cigarette: document.getElementById("sound-cigarette"),
  phone: document.getElementById("sound-phone"),
  child: document.getElementById("sound-child") // ใช้สำหรับ multi-face
};

let peakdow = 0;
let distractCount = 0;
const threshold = 8;
const distractThreshold = 8;

let active = {
  drowsy: false,
  distracted: false,
  cigarette: false,
  phone: false,
  multiFace: false
};

function calcEAR(p1, p2, p3, p4, p5, p6) {
  const v1 = Math.hypot(p2.x - p6.x, p2.y - p6.y);
  const v2 = Math.hypot(p3.x - p5.x, p3.y - p5.y);
  const h = Math.hypot(p1.x - p4.x, p1.y - p4.y);
  return (v1 + v2) / (2.0 * h);
}

function getEAR(landmarks) {
  const left = calcEAR(landmarks[33], landmarks[160], landmarks[158], landmarks[133], landmarks[153], landmarks[144]);
  const right = calcEAR(landmarks[362], landmarks[385], landmarks[387], landmarks[263], landmarks[373], landmarks[380]);
  return (left + right) / 2.0;
}

const faceMesh = new FaceMesh({
  locateFile: file => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
  maxNumFaces: 5,
  refineLandmarks: true,
  minDetectionConfidence: 0.6,
  minTrackingConfidence: 0.6
});

faceMesh.onResults(results => {
  if (!results.multiFaceLandmarks) return;

  const faceCount = results.multiFaceLandmarks.length;

  if (faceCount > 1 && !active.multiFace) {
    soundMap.child.play();
    alertBox.innerText = "👶 พบหลายใบหน้า!";
    alertBox.style.display = "block";
    active.multiFace = true;
    console.log("🧍‍🧍‍🧍 Multi-face Detected: " + faceCount);
  } else if (faceCount <= 1) {
    active.multiFace = false;
  }

  for (const landmarks of results.multiFaceLandmarks) {
    const ear = getEAR(landmarks);
    peakdow = ear < 0.2 ? peakdow + 1 : Math.max(0, peakdow - 1);

    const noseX = landmarks[1].x;
    const centerX = (landmarks[33].x + landmarks[263].x) / 2;
    const deviation = Math.abs(noseX - centerX);
    distractCount = deviation > 0.08 ? distractCount + 1 : Math.max(0, distractCount - 1);
  }

  if (peakdow >= threshold && !active.drowsy) {
    soundMap.drowsy.play();
    alertBox.innerText = "😴 ตรวจพบหลับใน!";
    alertBox.style.display = "block";
    active.drowsy = true;
    console.log("😴 Drowsiness: 100%");
  } else if (peakdow < threshold) {
    active.drowsy = false;
  }

  if (distractCount >= distractThreshold && !active.distracted) {
    soundMap.distracted.play();
    alertBox.innerText = "😵 ตรวจพบเหม่อลอย!";
    alertBox.style.display = "block";
    active.distracted = true;
    console.log("😵 Distraction: 100%");
  } else if (distractCount < distractThreshold) {
    active.distracted = false;
  }
});

async function loadTFModel() {
  const url = "./model/";
  model = await tmImage.load(url + "model.json", url + "metadata.json");
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
  await video.play();

  camera = new Camera(video, {
    onFrame: async () => {
      if (faceMesh && video.readyState >= 2) await faceMesh.send({ image: video });
      if (!model) return;

      const prediction = await model.predict(video);
      const preds = {};
      prediction.forEach(p => preds[p.className.toLowerCase()] = p.probability);

      // เฉพาะ phone และ cigarette
      ["phone", "cigarette"].forEach(label => {
        const prob = preds[label] || 0;
        console.log(`${label}: ${(prob * 100).toFixed(1)}%`);
        if (prob > 0.98 && !active[label]) {
          soundMap[label].play();
          alertBox.innerText = label === "phone" ? "📱 ใช้โทรศัพท์!" : "🚬 สูบบุหรี่!";
          alertBox.style.display = "block";
          active[label] = true;
        } else if (prob < 0.5) {
          active[label] = false;
        }
      });
    },
    width: 640,
    height: 480
  });

  camera.start();
}

function stopCamera() {
  if (camera) {
    camera.stop();
    camera = null;
  }
  if (video.srcObject) {
    video.srcObject.getTracks().forEach(track => track.stop());
    video.srcObject = null;
  }
  peakdow = 0;
  distractCount = 0;
  alertBox.style.display = "none";
}

window.addEventListener("DOMContentLoaded", async () => {
  await loadTFModel();
});

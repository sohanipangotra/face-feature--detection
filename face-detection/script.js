// =========================================================
//  FaceAI — script.js
//  Face Detection · Landmark Contours · CVIP Analysis
// =========================================================

const MODEL_URL = 'https://cdn.jsdelivr.net/gh/justadudewhohacks/face-api.js@master/weights';

// ── State ─────────────────────────────────────────────────
let modelsLoaded  = false;
let cameraStream  = null;
let detectingLoop = null;
let fpsTimer      = null;
let frameCount    = 0;
let lastFpsTime   = performance.now();
let autoFreezeTimer    = null;   // auto-stop after N seconds
let countdownInterval  = null;   // countdown tick

// DOM refs
const video         = document.getElementById('video');
const cameraCanvas  = document.getElementById('cameraCanvas');
const imageCanvas   = document.getElementById('imageCanvas');
const uploadedImg   = document.getElementById('uploadedImg');
const loadingOverlay= document.getElementById('loadingOverlay');
const progressBar   = document.getElementById('progressBar');
const startCamBtn   = document.getElementById('startCamBtn');
const stopCamBtn    = document.getElementById('stopCamBtn');
const snapshotBtn   = document.getElementById('snapshotBtn');
const faceCount     = document.getElementById('faceCount');
const fpsDisplay    = document.getElementById('fpsDisplay');
const confDisplay   = document.getElementById('confidenceDisplay');
const modeDisplay   = document.getElementById('modeDisplay');
const scanLine      = document.getElementById('scanLine');
const noFaceMsg     = document.getElementById('noFaceMsg');
const detectionGrid = document.getElementById('detectionGrid');
const uploadZone    = document.getElementById('uploadZone');
const imageResult   = document.getElementById('imageResult');

// ── Load Models ────────────────────────────────────────────
async function loadModels() {
  try {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL),
      faceapi.nets.faceExpressionNet.loadFromUri(MODEL_URL),
      faceapi.nets.ageGenderNet.loadFromUri(MODEL_URL),
    ]);
    modelsLoaded = true;
    progressBar.style.width = '100%';
    setTimeout(() => { loadingOverlay.classList.add('hidden'); }, 500);
  } catch (err) {
    console.error('Model loading failed:', err);
    document.querySelector('.loading-sub').textContent = 'Error loading models. Check your internet connection.';
    document.querySelector('.spinner').style.borderTopColor = 'var(--red)';
  }
}

// ── Tab Switcher ───────────────────────────────────────────
function switchTab(tab) {
  document.getElementById('tabCamera').classList.toggle('active', tab === 'camera');
  document.getElementById('tabImage').classList.toggle('active', tab === 'image');
  document.getElementById('panelCamera').classList.toggle('active', tab === 'camera');
  document.getElementById('panelImage').classList.toggle('active', tab === 'image');
  if (tab === 'image') { stopCamera(); setMode('Image'); }
  else { setMode('Idle'); clearDetectionCards(); updateStats(0, '--', '--%'); }
}

// ── Camera ─────────────────────────────────────────────────
async function startCamera() {
  if (!modelsLoaded) return alert('Models are still loading, please wait…');
  // Reset button label in case it was changed to 'Scan Again'
  startCamBtn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <circle cx="12" cy="12" r="10"/><polygon points="10,8 16,12 10,16"/>
    </svg>
    Start Camera`;
  // Clear the frozen canvas so it doesn't show stale frame
  const ctx = cameraCanvas.getContext('2d');
  ctx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
  clearDetectionCards();
  updateStats(0, '--', '--%');
  try {
    cameraStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false
    });
    video.srcObject = cameraStream;
    await video.play();
    startCamBtn.classList.add('hidden');
    stopCamBtn.classList.remove('hidden');
    snapshotBtn.disabled = false;
    scanLine.classList.add('active');
    setMode('Live');
    detectingLoop = setInterval(detectFromCamera, 120);
    fpsTimer = setInterval(() => {
      const now = performance.now();
      fpsDisplay.textContent = Math.round(frameCount / ((now - lastFpsTime) / 1000));
      frameCount = 0; lastFpsTime = now;
    }, 1000);

    // ── Start 5-second countdown then auto-freeze ──
    startCountdown(5);
    autoFreezeTimer = setTimeout(() => { if (cameraStream) freezeOnFace(); }, 5500);
  } catch (err) { alert('Could not access camera: ' + err.message); }
}

function stopCamera() {
  if (autoFreezeTimer)   { clearTimeout(autoFreezeTimer);    autoFreezeTimer = null; }
  if (countdownInterval) { clearInterval(countdownInterval); countdownInterval = null; }
  removeCountdownEl();
  if (detectingLoop) clearInterval(detectingLoop);
  if (fpsTimer)      clearInterval(fpsTimer);
  if (cameraStream)  cameraStream.getTracks().forEach(t => t.stop());
  cameraStream = detectingLoop = fpsTimer = null;
  cameraCanvas.getContext('2d').clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);
  startCamBtn.classList.remove('hidden');
  stopCamBtn.classList.add('hidden');
  snapshotBtn.disabled = true;
  scanLine.classList.remove('active');
  noFaceMsg.classList.remove('visible');
  setMode('Idle');
  updateStats(0, '--', '--%');
  clearDetectionCards();
}

// ── Detection Loop ─────────────────────────────────────────
async function detectFromCamera() {
  if (!video.videoWidth) return;
  cameraCanvas.width  = video.videoWidth;
  cameraCanvas.height = video.videoHeight;

  const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 320, scoreThreshold: 0.5 });
  const detections = await faceapi
    .detectAllFaces(video, options)
    .withFaceLandmarks(true)
    .withFaceExpressions()
    .withAgeAndGender();

  frameCount++;
  const ctx = cameraCanvas.getContext('2d');
  ctx.clearRect(0, 0, cameraCanvas.width, cameraCanvas.height);

  // Draw mirrored video frame onto the analysis canvas (right panel)
  ctx.save();
  ctx.translate(cameraCanvas.width, 0);
  ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, cameraCanvas.width, cameraCanvas.height);
  ctx.restore();

  if (!detections.length) {
    noFaceMsg.classList.add('visible');
    updateStats(0, null, '--%');
    clearDetectionCards();
    return;
  }
  noFaceMsg.classList.remove('visible');

  const resized = faceapi.resizeResults(detections, { width: cameraCanvas.width, height: cameraCanvas.height });
  drawDetections(ctx, resized, true);

  const avgConf = detections.reduce((s, d) => s + d.detection.score, 0) / detections.length;
  updateStats(detections.length, null, (avgConf * 100).toFixed(1) + '%');
  renderDetectionCards(detections);
  // (auto-freeze handled by countdown timer started in startCamera)
}

// ── Freeze on Detection ────────────────────────────────────
// Stops the stream but keeps the analyzed frame visible on canvas
function freezeOnFace() {
  if (autoFreezeTimer)   { clearTimeout(autoFreezeTimer);    autoFreezeTimer = null; }
  if (countdownInterval) { clearInterval(countdownInterval); countdownInterval = null; }
  removeCountdownEl();
  if (detectingLoop) { clearInterval(detectingLoop); detectingLoop = null; }
  if (fpsTimer)      { clearInterval(fpsTimer);      fpsTimer = null; }
  if (cameraStream)  { cameraStream.getTracks().forEach(t => t.stop()); cameraStream = null; }

  // Show frozen badge on analysis pane
  showFrozenBadge();

  scanLine.classList.remove('active');
  setMode('Captured ✓');

  // Switch buttons: hide Stop, show Scan Again
  stopCamBtn.classList.add('hidden');
  startCamBtn.classList.remove('hidden');
  startCamBtn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.29"/>
    </svg>
    Scan Again`;
  snapshotBtn.disabled = false;
}

function showFrozenBadge() {
  // Remove any existing badge
  const old = document.getElementById('frozenBadge');
  if (old) old.remove();
  const badge = document.createElement('div');
  badge.id = 'frozenBadge';
  badge.className = 'frozen-badge';
  badge.innerHTML = `<span>📸 Captured</span>`;
  const paneBox = cameraCanvas.parentElement;
  if (paneBox) paneBox.appendChild(badge);
  // Fade out after 2.5s
  setTimeout(() => badge.classList.add('fade-out'), 2000);
  setTimeout(() => badge.remove(), 2800);
}

// ── Countdown Timer ────────────────────────────────────────
function startCountdown(seconds) {
  let remaining = seconds;
  showCountdownEl(remaining);
  countdownInterval = setInterval(() => {
    remaining--;
    if (remaining > 0) {
      showCountdownEl(remaining);
    } else {
      clearInterval(countdownInterval);
      countdownInterval = null;
      showCountdownEl(0); // show camera emoji briefly
      setTimeout(removeCountdownEl, 600);
    }
  }, 1000);
}

function showCountdownEl(n) {
  // Re-create element each tick to restart CSS animation
  removeCountdownEl();
  const el = document.createElement('div');
  el.id = 'countdownOverlay';
  el.className = 'countdown-overlay' + (n <= 2 && n > 0 ? ' urgent' : '');
  el.textContent = n === 0 ? '\ud83d\udcf8' : n;
  const paneBox = cameraCanvas.parentElement;
  if (paneBox) paneBox.appendChild(el);
}

function removeCountdownEl() {
  const el = document.getElementById('countdownOverlay');
  if (el) el.remove();
}

// ═══════════════════════════════════════════════════════════
//  CVIP UTILITY FUNCTIONS
// ═══════════════════════════════════════════════════════════

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function midpoint(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
function avgPts(pts) {
  return { x: pts.reduce((s,p)=>s+p.x,0)/pts.length, y: pts.reduce((s,p)=>s+p.y,0)/pts.length };
}

// 1. Eye Aspect Ratio (EAR) ─ blink / eye-open detection
//    EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
function eyeAspectRatio(eyePts) {
  const A = dist(eyePts[1], eyePts[5]);
  const B = dist(eyePts[2], eyePts[4]);
  const C = dist(eyePts[0], eyePts[3]);
  if (C === 0) return 0;
  return (A + B) / (2.0 * C);
}

// 2. Mouth Aspect Ratio (MAR) ─ open/yawn detection
function mouthAspectRatio(pts) {
  const lips = pts.slice(48, 60); // outer lip
  const A = dist(lips[2],  lips[10]);
  const B = dist(lips[4],  lips[8]);
  const C = dist(lips[0],  lips[6]);
  if (C === 0) return 0;
  return (A + B) / (2.0 * C);
}

// 3. Head Pose (Yaw / Pitch / Roll) from landmarks
function estimateHeadPose(pts) {
  const leftEye   = avgPts(pts.slice(36, 42));
  const rightEye  = avgPts(pts.slice(42, 48));
  const eyeMid    = midpoint(leftEye, rightEye);
  const noseTip   = pts[30];
  const chin      = pts[8];
  const jawLeft   = pts[0];
  const jawRight  = pts[16];

  // Roll: tilt of eye line
  const roll = Math.atan2(rightEye.y - leftEye.y, rightEye.x - leftEye.x) * 180 / Math.PI;

  // Yaw: nose tip offset from eye midpoint (normalized by face width)
  const faceW = dist(jawLeft, jawRight) || 1;
  const yaw   = ((noseTip.x - eyeMid.x) / faceW) * 65;

  // Pitch: eye-mid to nose vs nose to chin ratio
  const eyeToNose  = dist(eyeMid, noseTip);
  const noseToChin = dist(noseTip, chin);
  const total      = eyeToNose + noseToChin || 1;
  const pitch      = ((eyeToNose / total) - 0.45) * 80;

  return {
    yaw:   yaw.toFixed(1),
    pitch: pitch.toFixed(1),
    roll:  roll.toFixed(1),
  };
}

// 4. Face Symmetry Score (0-100%)
function faceSymmetry(pts) {
  const noseTip = pts[30];
  const pairs = [
    [pts[0],  pts[16]],
    [pts[1],  pts[15]],
    [pts[3],  pts[13]],
    [pts[17], pts[26]],
    [pts[19], pts[24]],
    [pts[36], pts[45]],
    [pts[39], pts[42]],
    [pts[31], pts[35]],
  ];
  const diffs = pairs.map(([l, r]) => {
    const dL = dist(l, noseTip), dR = dist(r, noseTip);
    return Math.abs(dL - dR) / (((dL + dR) / 2) || 1);
  });
  const avg = diffs.reduce((s, d) => s + d, 0) / diffs.length;
  return Math.max(0, Math.min(100, (1 - avg * 4) * 100)).toFixed(1);
}

// 5. Inter-Pupillary Distance (IPD) in pixels
function interPupillaryDistance(pts) {
  const leftPupil  = avgPts(pts.slice(36, 42));
  const rightPupil = avgPts(pts.slice(42, 48));
  return dist(leftPupil, rightPupil).toFixed(1);
}

// 6. Face Shape from landmark geometry
function detectFaceShape(positions) {
  const jaw            = positions.slice(0, 17);
  const jawWidth       = dist(jaw[4], jaw[12]);
  const cheekWidth     = dist(jaw[1], jaw[15]);
  const foreheadEst    = dist(positions[17], positions[26]);
  const browMid        = midpoint(positions[19], positions[24]);
  const faceHeight     = dist(browMid, jaw[8]);
  const heightToWidth  = faceHeight / (cheekWidth || 1);
  const jawRatio       = jawWidth   / (cheekWidth || 1);
  const foreheadRatio  = foreheadEst/ (cheekWidth || 1);
  const chinWidth      = dist(jaw[6], jaw[10]);
  const chinRatio      = chinWidth  / (cheekWidth || 1);

  if (heightToWidth < 1.05)                                        return 'Round';
  if (heightToWidth > 1.55)                                        return 'Oblong';
  if (jawRatio > 0.82 && foreheadRatio > 0.75)                     return 'Square';
  if (foreheadRatio > jawRatio + 0.12 && chinRatio < 0.45)        return 'Heart';
  if (cheekWidth > foreheadEst * 1.1 && cheekWidth > jawWidth * 1.12) return 'Diamond';
  return 'Oval';
}

// 7. Skin Tone from bounding-box pixel average — skin-pixel filtered
function estimateSkinTone(srcElement, box, mirrored, canvasW, canvasH) {
  try {
    const off = document.createElement('canvas');
    const sw  = srcElement.videoWidth  || srcElement.naturalWidth  || canvasW;
    const sh  = srcElement.videoHeight || srcElement.naturalHeight || canvasH;
    off.width = sw; off.height = sh;
    const offCtx = off.getContext('2d', { willReadFrequently: true });
    if (mirrored) { offCtx.translate(sw, 0); offCtx.scale(-1, 1); }
    offCtx.drawImage(srcElement, 0, 0, sw, sh);
    if (mirrored) offCtx.setTransform(1, 0, 0, 1, 0, 0);

    const dx = mirrored ? canvasW - box.x - box.width : box.x;
    const scaleX = sw / canvasW, scaleY = sh / canvasH;

    // Sample cheeks + forehead region (skip hair at top, chin at bottom)
    const bx = Math.max(0, Math.floor((dx + box.width  * 0.18) * scaleX));
    const by = Math.max(0, Math.floor((box.y + box.height * 0.18) * scaleY));
    const bw = Math.max(1, Math.floor(box.width  * 0.64 * scaleX));
    const bh = Math.max(1, Math.floor(box.height * 0.50 * scaleY));

    const { data } = offCtx.getImageData(bx, by, bw, bh);

    let rSum = 0, gSum = 0, bSum = 0, skinCount = 0;

    for (let i = 0; i < data.length; i += 4) {
      const r = data[i], g = data[i+1], b = data[i+2];
      const lum = 0.299*r + 0.587*g + 0.114*b;

      // Skin filter rules (works across all skin tones):
      //  - not too dark (shadow/hair) or too bright (overexposed)
      //  - red channel dominates over blue (skin property)
      //  - not too gray/desaturated (background)
      if (lum < 40 || lum > 245) continue;       // skip shadows & blown-out
      if (r < 50) continue;                        // no dark-dominant pixels
      if (r < g || r < b) continue;               // skin: R >= G and R >= B
      if ((r - b) < 10) continue;                  // skin: noticeable R-B gap
      if (Math.abs(r - g) < 5 && Math.abs(g - b) < 5) continue; // bland grey

      rSum += r; gSum += g; bSum += b; skinCount++;
    }

    // Fallback: no skin pixels found → use all sampled pixels
    if (skinCount < 30) {
      let n = 0;
      for (let i = 0; i < data.length; i += 4) {
        rSum += data[i]; gSum += data[i+1]; bSum += data[i+2]; n++;
      }
      skinCount = n;
    }

    const r = Math.round(rSum / skinCount);
    const g = Math.round(gSum / skinCount);
    const b = Math.round(bSum / skinCount);
    const luminance = 0.299*r + 0.587*g + 0.114*b;

    // Simplified skin tone classification: Only Fair or Brown
    let tone;
    if (luminance > 135) tone = 'Fair';
    else                 tone = 'Brown';

    return { tone, r, g, b };
  } catch (e) {
    return { tone: 'Unknown', r: 128, g: 100, b: 80 };
  }
}


// ═══════════════════════════════════════════════════════════
//  CANVAS DRAWING
// ═══════════════════════════════════════════════════════════

function mirrorPts(pts, cw) { return pts.map(p => ({ x: cw - p.x, y: p.y })); }

function drawDetections(ctx, detections, mirrored = false, srcElement = null) {
  const colors = ['#a855f7','#06b6d4','#10b981','#f59e0b','#ec4899'];

  detections.forEach((det, i) => {
    const box   = det.detection.box;
    const color = colors[i % colors.length];
    const x = mirrored ? ctx.canvas.width - box.x - box.width : box.x;
    const y = box.y, w = box.width, h = box.height;

    // Bounding box
    ctx.save();
    ctx.shadowColor = color; ctx.shadowBlur = 20;
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.lineJoin = 'round';
    ctx.strokeRect(x, y, w, h);

    // Corner accents
    const cs = 18; ctx.lineWidth = 4;
    [['tl', x, y, x+cs, y, x, y+cs],
     ['tr', x+w, y, x+w-cs, y, x+w, y+cs],
     ['bl', x, y+h, x, y+h-cs, x+cs, y+h],
     ['br', x+w, y+h, x+w, y+h-cs, x+w-cs, y+h]
    ].forEach(([, ax, ay, bx, by, cx2, cy]) => {
      ctx.beginPath();
      ctx.moveTo(bx, by); ctx.lineTo(ax, ay); ctx.lineTo(cx2, cy);
      ctx.stroke();
    });
    ctx.restore();

    // Label
    const label = `#${i+1} · ${Math.round(det.detection.score*100)}%`;
    ctx.font = 'bold 13px Inter, sans-serif';
    const tw = ctx.measureText(label).width;
    const lx = x, ly = box.y - 34 < 0 ? box.y + 8 : box.y - 34;
    ctx.fillStyle = color + 'cc';
    ctx.beginPath(); ctx.roundRect(lx, ly, tw+20, 26, 6); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.fillText(label, lx+10, ly+17);

    if (det.landmarks) {
      const raw = det.landmarks.positions;
      const pts = mirrored ? mirrorPts(raw, ctx.canvas.width) : raw;

      // Landmark contours
      drawContour(ctx, pts.slice(0,17),  false, color,    2.5, 0.9);
      drawContour(ctx, pts.slice(17,22), false, '#f9a8d4',1.8, 0.8);
      drawContour(ctx, pts.slice(22,27), false, '#f9a8d4',1.8, 0.8);
      drawContour(ctx, pts.slice(27,31), false, '#93c5fd',1.8, 0.7);
      drawContour(ctx, pts.slice(31,36), false, '#93c5fd',1.8, 0.7);
      drawContour(ctx, pts.slice(36,42), true,  '#6ee7b7',1.8, 0.85);
      drawContour(ctx, pts.slice(42,48), true,  '#6ee7b7',1.8, 0.85);
      drawContour(ctx, pts.slice(48,60), true,  '#fca5a5',2,   0.85);
      drawContour(ctx, pts.slice(60,68), true,  '#fca5a5',1.4, 0.7);

      // Landmark dots
      ctx.fillStyle = color + '55';
      pts.forEach(pt => { ctx.beginPath(); ctx.arc(pt.x, pt.y, 1.5, 0, Math.PI*2); ctx.fill(); });

      // ── Head Pose Axes ─────────────────────────────────────
      drawHeadPoseAxes(ctx, pts, mirrored);

      // ── Eye open/close markers ─────────────────────────────
      const earL = eyeAspectRatio(pts.slice(36, 42));
      const earR = eyeAspectRatio(pts.slice(42, 48));
      const EAR_THRESH = 0.22;
      drawEyeMarker(ctx, avgPts(pts.slice(36,42)), earL < EAR_THRESH);
      drawEyeMarker(ctx, avgPts(pts.slice(42,48)), earR < EAR_THRESH);

      // ── Symmetry axis ─────────────────────────────────────
      drawSymmetryAxis(ctx, pts, color);
    }
  });
}

// Draw roll/yaw/pitch indicator at nose tip
function drawHeadPoseAxes(ctx, pts, mirrored) {
  const nose   = pts[30];
  const pose   = estimateHeadPose(pts);
  const len    = 40;
  const rollRad = parseFloat(pose.roll) * Math.PI / 180;
  const yawRad  = parseFloat(pose.yaw)  * Math.PI / 180;

  ctx.save();
  ctx.lineWidth = 2.5; ctx.lineCap = 'round';

  // X axis (roll — red)
  ctx.strokeStyle = '#f87171'; ctx.shadowColor = '#f87171'; ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(nose.x, nose.y);
  ctx.lineTo(nose.x + len * Math.cos(rollRad), nose.y + len * Math.sin(rollRad));
  ctx.stroke();

  // Y axis (pitch — green)
  ctx.strokeStyle = '#4ade80'; ctx.shadowColor = '#4ade80';
  ctx.beginPath();
  ctx.moveTo(nose.x, nose.y);
  ctx.lineTo(nose.x + len * Math.sin(yawRad), nose.y - len * Math.cos(rollRad));
  ctx.stroke();

  // Z axis (yaw — blue)
  ctx.strokeStyle = '#60a5fa'; ctx.shadowColor = '#60a5fa';
  ctx.beginPath();
  ctx.moveTo(nose.x, nose.y);
  ctx.lineTo(nose.x - len * Math.sin(yawRad * 0.8), nose.y + len * 0.2);
  ctx.stroke();

  // Label
  ctx.shadowBlur = 0;
  ctx.font = '10px Inter, sans-serif';
  ctx.fillStyle = '#f87171'; ctx.fillText('X', nose.x + len * Math.cos(rollRad) + 3, nose.y + len * Math.sin(rollRad));
  ctx.fillStyle = '#4ade80'; ctx.fillText('Y', nose.x + len * Math.sin(yawRad) + 3, nose.y - len * Math.cos(rollRad));
  ctx.fillStyle = '#60a5fa'; ctx.fillText('Z', nose.x - len * Math.sin(yawRad * 0.8) - 10, nose.y + len * 0.2);
  ctx.restore();
}

// Draw open/blink circle at eye center
function drawEyeMarker(ctx, center, isClosed) {
  ctx.save();
  ctx.beginPath();
  ctx.arc(center.x, center.y, 5, 0, Math.PI * 2);
  ctx.strokeStyle = isClosed ? '#f87171' : '#4ade80';
  ctx.shadowColor = isClosed ? '#f87171' : '#4ade80';
  ctx.shadowBlur  = 10;
  ctx.lineWidth   = 2;
  ctx.stroke();
  ctx.restore();
}

// Draw vertical symmetry axis from forehead to chin
function drawSymmetryAxis(ctx, pts, color) {
  const top  = pts[27]; // top of nose bridge ≈ forehead center
  const chin = pts[8];
  ctx.save();
  ctx.setLineDash([4, 4]);
  ctx.strokeStyle = color + '55';
  ctx.lineWidth   = 1;
  ctx.beginPath();
  ctx.moveTo(top.x, top.y - 20);
  ctx.lineTo(chin.x, chin.y + 10);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
}

// Smooth spline contour
function drawContour(ctx, pts, closed, color, lineWidth, alpha) {
  if (pts.length < 2) return;
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color; ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round'; ctx.lineJoin = 'round';
  ctx.shadowColor = color; ctx.shadowBlur = 8;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length - 1; i++) {
    const mx = (pts[i].x + pts[i+1].x) / 2;
    const my = (pts[i].y + pts[i+1].y) / 2;
    ctx.quadraticCurveTo(pts[i].x, pts[i].y, mx, my);
  }
  const last = pts[pts.length - 1];
  if (closed) {
    ctx.quadraticCurveTo(last.x, last.y, (last.x+pts[0].x)/2, (last.y+pts[0].y)/2);
    ctx.closePath();
  } else {
    ctx.lineTo(last.x, last.y);
  }
  ctx.stroke();
  ctx.restore();
}

// ═══════════════════════════════════════════════════════════
//  IMAGE DETECTION
// ═══════════════════════════════════════════════════════════

function onDragOver(e)  { e.preventDefault(); uploadZone.classList.add('drag-over'); }
function onDragLeave(e) { uploadZone.classList.remove('drag-over'); }
function onDrop(e)      { e.preventDefault(); uploadZone.classList.remove('drag-over'); processImageFile(e.dataTransfer.files[0]); }
function onFileSelected(e) { processImageFile(e.target.files[0]); }

function processImageFile(file) {
  if (!file || !file.type.startsWith('image/')) return alert('Please select a valid image file.');
  if (!modelsLoaded) return alert('Models are still loading, please wait…');
  const reader = new FileReader();
  reader.onload = async (e) => {
    uploadedImg.src = e.target.result;
    uploadedImg.onload = async () => {
      uploadZone.classList.add('hidden');
      imageResult.classList.remove('hidden');
      setMode('Analyzing…');
      await detectFromImage();
      setMode('Image');
    };
  };
  reader.readAsDataURL(file);
}

async function detectFromImage() {
  const options = new faceapi.TinyFaceDetectorOptions({ inputSize: 512, scoreThreshold: 0.4 });
  const detections = await faceapi
    .detectAllFaces(uploadedImg, options)
    .withFaceLandmarks(true)
    .withFaceExpressions()
    .withAgeAndGender();

  const displaySize = { width: uploadedImg.width, height: uploadedImg.height };
  imageCanvas.width  = displaySize.width;
  imageCanvas.height = displaySize.height;
  const ctx = imageCanvas.getContext('2d');
  ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  // Draw the original image onto the analysis canvas, then overlay detections
  ctx.drawImage(uploadedImg, 0, 0, displaySize.width, displaySize.height);

  if (!detections.length) {
    updateStats(0, '--', '--%'); clearDetectionCards();
    noFaceMsg.classList.add('visible'); return;
  }
  noFaceMsg.classList.remove('visible');
  const resized = faceapi.resizeResults(detections, displaySize);
  drawDetections(ctx, resized, false, uploadedImg);

  const avgConf = detections.reduce((s, d) => s + d.detection.score, 0) / detections.length;
  updateStats(detections.length, '--', (avgConf * 100).toFixed(1) + '%');
  renderDetectionCards(detections, uploadedImg, false, imageCanvas.width, imageCanvas.height);
}

function resetImage() {
  uploadZone.classList.remove('hidden');
  imageResult.classList.add('hidden');
  uploadedImg.src = '';
  document.getElementById('fileInput').value = '';
  const ctx = imageCanvas.getContext('2d');
  if (ctx) ctx.clearRect(0, 0, imageCanvas.width, imageCanvas.height);
  clearDetectionCards(); updateStats(0, '--', '--%');
  setMode('Idle'); noFaceMsg.classList.remove('visible');
}

// ── Snapshot ───────────────────────────────────────────────
function takeSnapshot() {
  const canvas = document.createElement('canvas');
  const vw = video.videoWidth, vh = video.videoHeight;
  canvas.width = vw; canvas.height = vh;
  const ctx = canvas.getContext('2d');
  ctx.translate(vw, 0); ctx.scale(-1, 1);
  ctx.drawImage(video, 0, 0, vw, vh);
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(cameraCanvas, 0, 0, vw, vh);
  const link = document.createElement('a');
  link.download = `faceai-snapshot-${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}

// ═══════════════════════════════════════════════════════════
//  UI
// ═══════════════════════════════════════════════════════════

function updateStats(faces, fps, conf) {
  faceCount.textContent = faces;
  if (fps !== null) fpsDisplay.textContent = fps;
  confDisplay.textContent = conf;
}
function setMode(str) { modeDisplay.textContent = str; }

function clearDetectionCards() {
  detectionGrid.innerHTML = `
    <div class="empty-state">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2">
        <circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/>
      </svg>
      <p>Start the camera or upload an image to begin detection</p>
    </div>`;
}

// Face shape profiles
const FACE_SHAPE_PROFILES = {
  Oval:    { emoji:'✨',color:'#a855f7',trait:'Balanced & Adaptable',feeling:'Content · Calm · Open-minded',desc:'Naturally diplomatic and versatile. Radiates quiet confidence.' },
  Round:   { emoji:'🌟',color:'#f59e0b',trait:'Warm & Sociable',feeling:'Joyful · Empathetic · Nurturing',desc:'Kind-hearted and emotionally expressive. Brings warmth to every room.' },
  Square:  { emoji:'💪',color:'#06b6d4',trait:'Determined & Ambitious',feeling:'Focused · Driven · Confident',desc:'Strong-willed and decisive. Channels emotions into purposeful action.' },
  Heart:   { emoji:'❤️',color:'#ec4899',trait:'Creative & Passionate',feeling:'Romantic · Imaginative · Sensitive',desc:'Deeply emotional and intuitive. Pours passion into everything.' },
  Oblong:  { emoji:'🎯',color:'#10b981',trait:'Thoughtful & Disciplined',feeling:'Reflective · Serene · Methodical',desc:'Organized and introspective. Processes emotions through careful thought.' },
  Diamond: { emoji:'💎',color:'#38bdf8',trait:'Precise & Perceptive',feeling:'Analytical · Curious · Discerning',desc:'Detail-oriented and highly perceptive. Loves uncovering hidden patterns.' },
};

const EMOTION_COLORS = {
  happy:'emo-happy', surprised:'emo-surprised', neutral:'emo-neutral',
  sad:'emo-sad', angry:'emo-angry', fearful:'emo-fearful', disgusted:'emo-disgusted',
};

// SVG outline for each detected face shape
function faceShapeSVG(shape) {
  const col = (FACE_SHAPE_PROFILES[shape] || {}).color || '#a855f7';
  const glowId = 'g' + shape;
  const defs = `<defs><filter id="${glowId}"><feGaussianBlur stdDeviation="3" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>`;
  const attr = `fill="none" stroke="${col}" stroke-width="2.5" filter="url(#${glowId})"`;
  let body = '';
  switch (shape) {
    case 'Oval':    body = `<ellipse cx="44" cy="54" rx="32" ry="44" ${attr}/>`; break;
    case 'Round':   body = `<ellipse cx="44" cy="50" rx="38" ry="38" ${attr}/>`; break;
    case 'Square':  body = `<rect x="10" y="12" width="68" height="76" rx="8" ${attr}/>`; break;
    case 'Heart':   body = `<path d="M44 82 C14 58 8 28 28 20 C36 17 44 26 44 26 C44 26 52 17 60 20 C80 28 74 58 44 82Z" ${attr}/>`; break;
    case 'Oblong':  body = `<ellipse cx="44" cy="54" rx="24" ry="46" ${attr}/>`; break;
    case 'Diamond': body = `<polygon points="44,8 78,50 44,92 10,50" ${attr}/>`; break;
    default:        body = `<ellipse cx="44" cy="54" rx="32" ry="44" ${attr}/>`;
  }
  return `<svg width="88" height="100" xmlns="http://www.w3.org/2000/svg">${defs}${body}</svg>`;
}

function renderDetectionCards(detections, srcElement = video, mirrored = true, cw = null, ch = null) {
  detectionGrid.innerHTML = '';
  const canvasW = cw || cameraCanvas.width;
  const canvasH = ch || cameraCanvas.height;

  detections.forEach((det, i) => {
    const conf  = (det.detection.score * 100).toFixed(1);
    const age   = det.age  ? Math.round(det.age)  : '?';
    const gend  = det.gender ? capitalize(det.gender) : '?';
    const gendC = det.genderProbability ? (det.genderProbability * 100).toFixed(0) : '?';

    const shape   = det.landmarks ? detectFaceShape(det.landmarks.positions) : 'Unknown';
    const profile = FACE_SHAPE_PROFILES[shape] || {};

    // CVIP metrics
    let cvip = null;
    if (det.landmarks) {
      const pts = det.landmarks.positions;
      const earL = eyeAspectRatio(pts.slice(36, 42));
      const earR = eyeAspectRatio(pts.slice(42, 48));
      const EAR_THRESH = 0.22;
      const mar   = mouthAspectRatio(pts);
      const pose  = estimateHeadPose(pts);
      const sym   = faceSymmetry(pts);
      const ipd   = interPupillaryDistance(pts);
      const skin  = srcElement ? estimateSkinTone(srcElement, det.detection.box, mirrored, canvasW, canvasH) : null;
      cvip = { earL, earR, ear_thresh: EAR_THRESH, mar, pose, sym, ipd, skin };
    }

    const emos = Object.entries(det.expressions).sort((a,b)=>b[1]-a[1]).slice(0,5);
    const domEmo = emos[0] ? capitalize(emos[0][0]) : '?';
    const domPct = emos[0] ? (emos[0][1]*100).toFixed(0) : '?';

    const emotionBars = emos.map(([name, val]) => `
      <div class="emotion-bar-row">
        <span class="emotion-name">${name}</span>
        <div class="emotion-track">
          <div class="emotion-fill ${EMOTION_COLORS[name]||'emo-neutral'}" style="width:${(val*100).toFixed(1)}%"></div>
        </div>
        <span class="emotion-pct">${(val*100).toFixed(0)}%</span>
      </div>`).join('');

    const cvipHTML = cvip ? `
      <div class="cvip-section">
        <div class="cvip-title">📐 CVIP Analysis</div>
        <div class="cvip-grid">

          <div class="cvip-item">
            <div class="cvip-label">Head Yaw <span class="cvip-hint">(left/right turn)</span></div>
            <div class="cvip-value ${Math.abs(cvip.pose.yaw) > 20 ? 'warn' : ''}">
              ${Math.abs(cvip.pose.yaw)}° ${parseFloat(cvip.pose.yaw) >= 0 ? 'Left' : 'Right'}
            </div>
          </div>
          <div class="cvip-item">
            <div class="cvip-label">Head Pitch <span class="cvip-hint">(up/down tilt)</span></div>
            <div class="cvip-value ${Math.abs(cvip.pose.pitch) > 15 ? 'warn' : ''}">
              ${Math.abs(cvip.pose.pitch)}° ${parseFloat(cvip.pose.pitch) >= 0 ? 'Up' : 'Down'}
            </div>
          </div>
          <div class="cvip-item">
            <div class="cvip-label">Head Roll <span class="cvip-hint">(side tilt)</span></div>
            <div class="cvip-value ${Math.abs(cvip.pose.roll) > 10 ? 'warn' : ''}">
              ${Math.abs(cvip.pose.roll)}° ${parseFloat(cvip.pose.roll) >= 0 ? 'CW' : 'CCW'}
            </div>
          </div>
          <div class="cvip-item">
            <div class="cvip-label">Symmetry <span class="cvip-hint">(L vs R match)</span></div>
            <div class="cvip-value">${cvip.sym}%</div>
          </div>

          <div class="cvip-item">
            <div class="cvip-label">Left Eye <span class="cvip-hint">(EAR ratio)</span></div>
            <div class="cvip-value eye-status ${cvip.earL < cvip.ear_thresh ? 'closed' : 'open'}">
              ${cvip.earL < cvip.ear_thresh ? '😑 Closed' : '👁 Open'} <span>${cvip.earL.toFixed(2)}</span>
            </div>
          </div>
          <div class="cvip-item">
            <div class="cvip-label">Right Eye <span class="cvip-hint">(EAR ratio)</span></div>
            <div class="cvip-value eye-status ${cvip.earR < cvip.ear_thresh ? 'closed' : 'open'}">
              ${cvip.earR < cvip.ear_thresh ? '😑 Closed' : '👁 Open'} <span>${cvip.earR.toFixed(2)}</span>
            </div>
          </div>
          <div class="cvip-item">
            <div class="cvip-label">Mouth <span class="cvip-hint">(MAR ratio)</span></div>
            <div class="cvip-value ${cvip.mar > 0.25 ? 'warn' : 'open'}">
              ${cvip.mar > 0.25 ? '😮 Open' : '😐 Closed'} <span>${cvip.mar.toFixed(2)}</span>
            </div>
          </div>
          <div class="cvip-item">
            <div class="cvip-label">Pupil Dist <span class="cvip-hint">(IPD in px)</span></div>
            <div class="cvip-value">${cvip.ipd} px</div>
          </div>

          ${cvip.skin ? `
          <div class="cvip-item cvip-full">
            <div class="cvip-label">Skin Tone <span class="cvip-hint">(avg face region)</span></div>
            <div class="cvip-value skin-row">
              <span class="skin-swatch" style="background:rgb(${cvip.skin.r},${cvip.skin.g},${cvip.skin.b})"></span>
              ${cvip.skin.tone}
              <span class="skin-rgb">RGB(${cvip.skin.r},${cvip.skin.g},${cvip.skin.b})</span>
            </div>
          </div>` : ''}
        </div>
      </div>` : '';

    detectionGrid.insertAdjacentHTML('beforeend', `
      <div class="detection-card">
        <div class="card-header">
          <span class="face-id">FACE #${i+1}</span>
          <span class="confidence-badge">✓ ${conf}%</span>
        </div>
        <div class="attr-grid">
          <div class="attr-item"><span class="attr-label">Age</span><span class="attr-value">~${age} yrs</span></div>
          <div class="attr-item"><span class="attr-label">Gender</span><span class="attr-value">${gend} <small style="color:var(--text-muted);font-weight:400">${gendC}%</small></span></div>
          <div class="attr-item"><span class="attr-label">Emotion</span><span class="attr-value">${domEmo}</span></div>
          <div class="attr-item"><span class="attr-label">Intensity</span><span class="attr-value">${domPct}%</span></div>
        </div>
        <div class="shape-profile" style="border-color:${profile.color||'#666'}22;background:${profile.color||'#666'}11;">
          <div class="shape-header">
            <span class="shape-svg">${faceShapeSVG(shape)}</span>
            <div>
              <div class="shape-name" style="color:${profile.color||'#fff'}">${shape} Face</div>
              <div class="shape-trait">${profile.trait||''}</div>
            </div>
          </div>
          <div class="shape-feeling">${profile.feeling||''}</div>
          <p class="shape-desc">${profile.desc||''}</p>
        </div>
        ${cvipHTML}
        <div class="emotions-title">Expression Analysis</div>
        ${emotionBars}
      </div>`);
  });
}

function capitalize(str) { return str ? str.charAt(0).toUpperCase() + str.slice(1) : str; }

// ── Init ───────────────────────────────────────────────────
loadModels();

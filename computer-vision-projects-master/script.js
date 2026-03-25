// DOM Elements
const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const previewZone = document.getElementById('previewZone');
const resetBtn = document.getElementById('resetBtn');
const downloadBtn = document.getElementById('downloadBtn');

const originalCanvas = document.getElementById('originalCanvas');
const grayscaleCanvas = document.getElementById('grayscaleCanvas');
const ctxOriginal = originalCanvas.getContext('2d');
const ctxGrayscale = grayscaleCanvas.getContext('2d');

let originalImage = new Image();

// Event Listeners for Upload Interaction
browseBtn.addEventListener('click', (e) => {
    e.preventDefault();
    fileInput.click();
});

uploadZone.addEventListener('click', () => {
    fileInput.click();
});

uploadZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => {
    uploadZone.classList.remove('dragover');
});

uploadZone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');

    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        handleFile(e.target.files[0]);
    }
});

// Reset Button
resetBtn.addEventListener('click', () => {
    uploadZone.style.display = 'flex';
    previewZone.style.display = 'none';
    fileInput.value = '';
    // Clear canvas
    ctxOriginal.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
    ctxGrayscale.clearRect(0, 0, grayscaleCanvas.width, grayscaleCanvas.height);
});

// Download Functionality
downloadBtn.addEventListener('click', () => {
    const dataURL = grayscaleCanvas.toDataURL('image/png');
    const a = document.createElement('a');
    a.href = dataURL;
    a.download = 'visionary-grayscale.png';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
});

// Core Processing Logic
function handleFile(file) {
    const reader = new FileReader();

    reader.onload = function(e) {
        originalImage.onload = function() {
            // Setup layouts and constraints
            const maxWidth = 800;
            const maxHeight = 800;
            
            let width = originalImage.width;
            let height = originalImage.height;

            // Maintain aspect ratio while scaling to fit max bounds
            if (width > maxWidth || height > maxHeight) {
                const ratio = Math.min(maxWidth / width, maxHeight / height);
                width = width * ratio;
                height = height * ratio;
            }

            // Set canvas dimensions
            originalCanvas.width = width;
            originalCanvas.height = height;
            grayscaleCanvas.width = width;
            grayscaleCanvas.height = height;

            // Draw original image onto original canvas
            ctxOriginal.drawImage(originalImage, 0, 0, width, height);

            // Execute the grayscale conversion
            applyGrayscaleFilter(width, height);

            // UI Transitions
            uploadZone.style.display = 'none';
            previewZone.style.display = 'flex';
        }
        originalImage.src = e.target.result;
    }

    reader.readAsDataURL(file);
}

// Vision logic: Convert pixels to Grayscale
function applyGrayscaleFilter(width, height) {
    // Retrieve image matrix (pixel data)
    const imageData = ctxOriginal.getImageData(0, 0, width, height);
    const data = imageData.data;

    // Loop through every pixel
    // the array comes sequentially like: R, G, B, A, R, G, B, A...
    for (let i = 0; i < data.length; i += 4) {
        const r = data[i];
        const g = data[i + 1];
        const b = data[i + 2];

        // Advanced grayscale formula (luminance method matches human color perception)
        const luminance = (0.299 * r) + (0.587 * g) + (0.114 * b);

        // Set Red, Green, and Blue identical to achieve grayscale
        data[i] = luminance;       // R
        data[i + 1] = luminance;   // G
        data[i + 2] = luminance;   // B
        // Alpha (data[i + 3]) remains unchanged
    }

    // Paint mutated data onto right-side canvas
    ctxGrayscale.putImageData(imageData, 0, 0);
}

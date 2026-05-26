# 🛡️ DeepGuard

**Multi-Modal Deepfake Detection & Media Verification System**

DeepGuard is a state-of-the-art, fully local deepfake detection pipeline designed to authenticate media across three distinct modalities: **Visual Artifacts**, **Audio Synthesis**, and **Temporal Inconsistencies**. By processing videos and images entirely on your local GPU, it ensures absolute privacy without relying on cloud APIs.

---

## ✨ Key Features

- **Multi-Modal Inference:** Fuses predictions from visual, audio, and temporal models to generate a highly accurate overall authenticity score.
- **Hardware Accelerated:** Leverages PyTorch with CUDA support to offload heavy neural network processing to your dedicated NVIDIA GPU (e.g., RTX 3050).
- **Fast Face Extraction:** Uses MTCNN for rapid batch face detection and extraction.
- **Sleek React UI:** A beautifully designed frontend with drag-and-drop media uploading, dark mode aesthetics, and dynamic result visualizations.
- **Local SSD Pipeline:** Training and inference scripts are built to run directly from your local SSD, avoiding the bottlenecks and caching issues of streaming large Hugging Face datasets on Windows.

---

## 🏗️ Architecture

The project is divided into three main components:

1. **/frontend**
   - Built with **React**, **Vite**, and **TailwindCSS**.
   - Handles the user interface, media uploading, and rendering the analytical breakdown of the AI's prediction.
2. **/backend**
   - A **FastAPI** server that acts as the bridge between the React frontend and the PyTorch models.
   - Handles file routing and returns structured JSON predictions.
3. **/ml**
   - The core intelligence of the system.
   - Contains the training pipelines (`train_visual.py`), dataset loaders (`dataset.py`, `download_dataset.py`), and the multi-modal fusion architecture.

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- Node.js & npm
- An NVIDIA GPU with CUDA Toolkit installed (Highly Recommended for inference speed).

### 2. Machine Learning Setup
```bash
cd ml
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Backend Setup
Open a new terminal:
```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

### 4. Frontend Setup
Open a third terminal:
```bash
cd frontend
npm install
npm run dev
```

The application will now be accessible at `http://localhost:5173`.

---

## 🧠 Training Your Own Models

DeepGuard provides an end-to-end local training pipeline. To train the visual classification model:

1. Download the OpenRL DeepFakeFace dataset using the included script:
   ```bash
   python ml/data/download_dataset.py
   ```
2. Initiate the training loop:
   ```bash
   python ml/training/train_visual.py
   ```
Weights will be automatically saved to `ml/training/weights/visual_best.pt`.

---

## 🔒 Privacy & Data

Because DeepGuard runs its deep neural networks entirely locally, your uploaded videos and images never leave your machine. Temporary media used for processing is instantly discarded from the `temp_uploads` directory after analysis.

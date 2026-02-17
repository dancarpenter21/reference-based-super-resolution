# Reference-Based Super-Resolution

The use case for this web app is when the user has a copmlete low-resolution video of a scene, and a high-resolution video of the same scene that is missing frames, and wants to use the high-resolution video to train a model to upscale the complete low-resolution video.

## The Problem

Two videos of the same scene are provided, one low-resolution and one high-resolution. The high resolution video is missing frames, typically in segments a couple seconds long at a time. The low-resoution video is complete, but obviously low-resolution. The goal is to train a model to upscale the low-resolution video to high-resolution, using the high-resolution video as a reference so that the upscaled video is the same resolution as the high-resolution reference video, but has all the frames that the low-resolution video has.

## The Solution

This system uses a reference-based super-resolution approach. It consists of:
- **Backend**: A FastAPI application that handles video processing and model inference.
- **Frontend**: A React application for user interaction (uploading videos, viewing results).
- **ML Engine**: The core machine learning logic for training and inference.

## Project Structure

```
.
├── backend/            # FastAPI backend
│   ├── app/            # Application logic
│   ├── Dockerfile      # Backend container definition
│   └── requirements.txt
├── frontend/           # React frontend
│   ├── src/
│   └── package.json
├── ml_engine/          # Machine Learning models and training scripts
├── data/               # Data directory
├── scripts/            # Utility scripts
└── docker-compose.yml  # Docker composition for backend service
```

## Prerequisites

- **Docker & Docker Compose** (Recommended for backend)
- **Python 3.10+** (If running backend locally)
- **Node.js 18+** (For frontend)

## Installation & Running

### Option A: Using Docker (Recommended)

Run the entire application (Frontend + Backend) with a single command:

```bash
docker compose up --build
```

- **Frontend**: `http://localhost:5173`
- **Backend API**: `http://localhost:8000`
- **API Docs**: `http://localhost:8000/docs`

### Option B: Running Locally

If you prefer to run services individually without Docker:

#### 1. Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the server:
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

#### 2. Frontend

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

## Data Preparation

For training or testing, you may need to prepare your video data.

- **Extract Frames**: Use the `scripts/prepare_data.py` script to extract frames from a video file.
  ```bash
  python scripts/prepare_data.py /path/to/your/video.mp4
  ```
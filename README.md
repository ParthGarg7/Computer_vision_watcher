# The Watcher - Computer Vision Pipeline

> **Intelligent multi-layer surveillance and face analysis pipeline built entirely on local, offline-first open-source models. No data ever leaves the machine.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7%2B%20cu128-orange.svg)](https://pytorch.org)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.13-green.svg)](https://opencv.org)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n--face-Ultralytics-purple.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-AGPL--3.0-lightgrey.svg)](LICENSE)

---

## What Is This?

The Watcher is a modular, 9-layer computer vision pipeline that ingests live camera streams or recorded video, detects faces in real time, identifies individuals, analyses expressions, and persists the results to a structured database. All processing runs locally on a single machine.

**MVP Status (v0.5.0-alpha): Layers 1–7 are complete and working.**

| Layer | Name | Status |
|-------|------|--------|
| **1** | Ingestion | ✅ Done |
| **2** | Preprocessing | ✅ Done |
| **3** | Face Detection | ✅ Done |
| **4** | Identity (InsightFace + DeepSORT + FAISS) | ✅ Done |
| **5** | Expression Analysis (hsemotion-onnx) | ✅ Done |
| **6** | Analytics & Business Logic (session metrics + alerts) | ✅ Done |
| **7** | Storage (SQLite MVP; PostgreSQL/TimescaleDB at scale) | ✅ Done |
| 8 | API (FastAPI) | 🔜 Planned |
| 9 | Frontend Dashboard | 🔜 Planned |

---

## Architecture (Layers 1–3)

```
Camera / RTSP / Video File
          │
          ▼
  ┌───────────────┐
  │   Layer 1     │   cv2.VideoCapture
  │   Ingestion   │   Raw BGR frames (H, W, 3) uint8
  └───────┬───────┘
          │ FrameContext (original_frame, camera_id, timestamp, frame_seq)
          ▼
  ┌───────────────┐
  │   Layer 2     │   BGR → RGB, Resize 640×640
  │ Preprocessing │   Builds Frame Context Object with original_shape metadata
  └───────┬───────┘
          │ FrameContext + preprocessed_frame + original_shape + resized_shape
          ▼
  ┌───────────────┐
  │   Layer 3     │   YOLOv8n-face (GPU via CUDA)
  │  Detection    │   Bounding boxes + confidence + face crops
  └───────┬───────┘
          │ FrameContext + detections list
          ▼
  Live preview window  +  optional annotated video output
```

**Key design principle:** A single `FrameContext` object travels through all layers, accumulating output from each. No layer modifies or replaces the object - it only adds fields.

---

## Hardware Requirements

| Component | Minimum | This Build |
|-----------|---------|------------|
| GPU | CPU-only (slower) | NVIDIA RTX 5060 (sm_120 Blackwell) |
| CUDA | Not required | 13.2 (driver) / 12.8 (PyTorch runtime) |
| VRAM | 2 GB | 8.5 GB |
| RAM | 4 GB free | 16 GB system |
| OS | Windows 10+, Linux | Windows 11 |

> **GPU Note:** The RTX 50-series (Blackwell) requires PyTorch ≥ 2.7.0 built with CUDA 12.8 (`cu128`). See setup instructions below.

---

## Prerequisites

- Python **3.10, 3.11, 3.12, or 3.14**
- Git
- An NVIDIA GPU with CUDA driver 525+ (or run on CPU at reduced FPS)

---

## Setup

### 1. Clone the repository
```bash
git clone https://github.com/ParthGarg7/Computer_vision_watcher.git
cd Computer_vision_watcher
```

### 2. Create a virtual environment
```powershell
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # Linux/macOS
```

### 3. Install PyTorch (GPU - RTX 50-series / CUDA 12.8)
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

> For **CPU only**: `pip install torch torchvision torchaudio`  
> For **older CUDA (12.1)**: use `--index-url https://download.pytorch.org/whl/cu121`

### 4. Install all other dependencies
```powershell
pip install -r requirements.txt
```

### 5. Download the face detection model
```powershell
python -c "
from huggingface_hub import hf_hub_download
import shutil
path = hf_hub_download(repo_id='arnabdhar/YOLOv8-Face-Detection', filename='model.pt')
shutil.copy(path, 'models/yolov8n-face.pt')
print('Model saved to models/yolov8n-face.pt')
"
```

### 6. Verify the environment
```powershell
python -c "
import cv2, torch, ultralytics, os
print('OpenCV:', cv2.__version__)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')
print('Ultralytics:', ultralytics.__version__)
print('Model exists:', os.path.exists('models/yolov8n-face.pt'))
"
```

---

## Running

### Interactive terminal menu
```powershell
python main.py
```
```
  Select Input Source:
  [1]  Webcam         (your built-in camera)
  [2]  Recorded Video (opens file browser)
  [3]  RTSP Stream    (enter URL manually)
  [Q]  Quit
```

### With validation output (saves annotated video)
```powershell
python main.py --validate
```

### Direct source (skip menu)
```powershell
python main.py --source 0              # webcam at index 0
python main.py --source path/to/video.mp4
python main.py --source rtsp://user:pass@ip:554/stream
```

### Preview window controls
| Key | Action |
|-----|--------|
| `Q` | Quit / stop the pipeline |
| `F` | Toggle fullscreen |
| `X` (window button) | Close and stop cleanly |

---

## RAM Checker Utility

A standalone utility to check system RAM usage before running the pipeline:

```powershell
python scripts/ram_checker.py           # Single snapshot
python scripts/ram_checker.py --watch   # Live refresh every 3s
python scripts/ram_checker.py --top 15  # Show top 15 processes
```

> This script is fully standalone - copy it to any project. Requires only `psutil` (`pip install psutil`).

---

## Project Structure

```
Computer_vision_watcher/
├── main.py                              # Entry point - terminal menu
├── requirements.txt                     # Python dependencies
│
├── models/                              # Model weights (gitignored)
│   └── yolov8n-face.pt                 # 6.2 MB - download separately
│
├── output/                             # Annotated output videos (gitignored)
│
├── scripts/
│   └── ram_checker.py                  # Standalone RAM usage inspector
│
├── src/
│   ├── core/
│   │   └── frame_context.py            # FrameContext + Detection dataclasses
│   │
│   ├── layer1_ingestion/
│   │   └── capture.py                  # cv2.VideoCapture wrapper (webcam/file/RTSP)
│   │
│   ├── layer2_preprocessing/
│   │   └── preprocessor.py             # BGR→RGB, resize, FrameContext builder
│   │
│   └── layer3_detection/
│       ├── detector.py                 # YOLOv8n-face inference + coord scaling
│       └── validator.py               # Draw detections + save annotated video
│
├── tests/                              # (Placeholder - unit tests coming in v0.2)
└── Documents/                          # Architecture documentation (.docx)
```

---

## How It Works - Key Technical Details

### The Frame Context Object
Every frame travels as a single `FrameContext` dataclass from Layer 1 through all downstream layers. It carries:
- `original_frame` - raw BGR array, never modified, used for drawing and cropping
- `preprocessed_frame` - RGB 640×640 ready for YOLOv8
- `original_shape` / `resized_shape` - required for coordinate scaling
- `camera_id`, `timestamp`, `frame_seq` - metadata for tracking and storage
- `detections` - list of `Detection` objects populated by Layer 3

### Coordinate Scaling (Layer 3 critical step)
YOLOv8 returns bounding boxes in the **resized** input space (640×640). These are scaled back to the original frame's coordinate space before drawing or passing downstream:
```
x_original = x_resized × (original_W / 640)
y_original = y_resized × (original_H / 640)
```

### GPU Usage
- **VRAM:** YOLOv8 model weights and inference tensors run entirely on the GPU
- **RAM:** Frame buffers (numpy arrays), FrameContext objects, and cv2 drawing operations run on CPU
- **Observed:** ~75 FPS inference on RTX 5060 (post warmup), ~30 FPS end-to-end with webcam at 640×480

### Privacy & Security
- All models run **100% locally**. No frame data, embeddings, or metadata are sent externally.
- RTSP streams are the primary network attack surface. Always use authenticated URLs.
- Face crops produced by Layer 3 constitute biometric data. Access controls and encryption apply from Layer 3 onward (implemented in Layer 7).

---

## Testing

The unit test suite covers the identity store (FAISS), Layer 5 expression
logic (label mapping, throttling, smoothing — with a mocked model, no
downloads), Layer 6 analytics, Layer 7 storage, and the Layer 1-3 contracts.
No GPU or model weights required; runs in under a second.

```powershell
python -m unittest discover tests
```

The heavier visual validators (real models, annotated output video) remain
available: `python validate_layer4.py`, `python validate_layer5.py`.

---

## Roadmap

- **v0.2.0** ✅ Layer 4: Identity (InsightFace ArcFace + DeepSORT tracking + FAISS search)
- **v0.3.0** ✅ Layer 5: Expression Analysis (hsemotion-onnx)
- **v0.4.0** ✅ Layer 6: Analytics & Business Logic (session metrics, presence/threshold alerts)
- **v0.5.0** ✅ Layer 7: Storage (SQLite MVP schema mirroring PostgreSQL + TimescaleDB; FAISS persistence in Layer 4; Redis at scale-up)
- **v0.8.0** - Layer 8: REST API (FastAPI)
- **v1.0.0** - Layer 9: Frontend Dashboard, full production MVP

---

## License

[GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) - see the LICENSE file for details.

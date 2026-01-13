# Image and Video Generation API

A FastAPI-based REST API for generating images and videos using Stable Diffusion AI models.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)

## Features

- **Image Generation**: Create AI-generated images from text prompts
- **Video Generation**: Create videos from multiple scene descriptions with smooth transitions
- **Multiple Output Formats**: Support for file download and base64 encoding
- **GPU Acceleration**: Automatic detection and use of CUDA, MPS, or CPU
- **Customizable Parameters**: Control inference steps, FPS, resolution, and more

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Required Dependencies

```bash
pip install fastapi uvicorn torch diffusers transformers accelerate pydantic opencv-python pillow
```

### Optional (for GPU acceleration)

**NVIDIA GPU:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Apple Silicon (M1/M2/M3):**
```bash
# PyTorch with MPS support is included in the standard installation
```

## Quick Start

1. **Clone or download the API code**

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the server**
   ```bash
   python app.py
   ```

4. **Access the API**
   - API Base URL: `http://localhost:8000`
   - Interactive Docs: `http://localhost:8000/docs`
   - Alternative Docs: `http://localhost:8000/redoc`

## API Endpoints

### 1. Root Endpoint

**GET /** - Health check and API information

```bash
curl http://localhost:8000/
```

**Response:**
```json
{
  "status": "running",
  "device": "cuda",
  "model": "segmind/tiny-sd",
  "endpoints": {
    "generate_image": "/generate",
    "generate_video": "/generate-video",
    "video_info": "/generate-video/info"
  }
}
```

**Use Case:** Verify the API is running and check which device is being used (CPU/GPU).

---

### 2. Generate Image

**POST /generate** - Create a single AI-generated image from a text prompt

**Request Body:**
```json
{
  "prompt": "A beautiful sunset over the ocean with dolphins jumping",
  "num_inference_steps": 25,
  "return_base64": false
}
```

**Parameters:**
- `prompt` (string, required): Text description of the image you want to generate
- `num_inference_steps` (integer, optional, default: 25): Number of denoising steps. Higher = better quality but slower (range: 10-50)
- `return_base64` (boolean, optional, default: false): If true, returns base64-encoded image instead of file

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cute robot playing guitar",
    "num_inference_steps": 30,
    "return_base64": false
  }' \
  --output image.png
```

**Example with Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A magical forest with glowing mushrooms",
        "num_inference_steps": 25,
        "return_base64": False
    }
)

with open("generated_image.png", "wb") as f:
    f.write(response.content)
```

**Response (file mode):**
Returns PNG image file for download.

**Response (base64 mode):**
```json
{
  "success": true,
  "prompt": "A cute robot playing guitar",
  "steps": 30,
  "image_base64": "iVBORw0KGgoAAAANS...",
  "format": "png"
}
```

**Use Cases:**
- Generate single images for social media
- Create artwork from descriptions
- Prototype visual concepts
- Integration with design workflows

---

### 3. Generate Video

**POST /generate-video** - Create a video from multiple scene descriptions

**Request Body:**
```json
{
  "scenes": [
    "A peaceful sunrise over mountains with mist",
    "A busy city street at noon with people walking",
    "A calm beach at sunset with waves"
  ],
  "num_inference_steps": 25,
  "duration_per_scene": 3.0,
  "fps": 24,
  "width": 512,
  "height": 512,
  "transition_frames": 12
}
```

**Parameters:**
- `scenes` (array of strings, required): List of text descriptions for each scene. Minimum 2 scenes required
- `num_inference_steps`
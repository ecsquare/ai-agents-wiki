# 🎨 AI Image Generation API

A FastAPI service for generating images from text prompts using Stable Diffusion (Tiny-SD model). Optimized for Apple Silicon (M1/M2/M3), NVIDIA GPUs (CUDA), and CPU.

## 🚀 Features

- **Fast API endpoints** for image generation
- **Multiple device support**: MPS (Apple Silicon), CUDA (NVIDIA), CPU
- **Flexible output**: Base64 or file response
- **Ready for n8n integration**
- **Lightweight model**: Uses Segmind Tiny-SD (~500MB)
- **Auto device detection**: Automatically uses the best available hardware

## 📋 Prerequisites

- Python 3.8+
- For Apple Silicon: macOS 12.3+
- For NVIDIA GPUs: CUDA-compatible GPU with drivers installed

## 🛠️ Installation

### 1. Clone or Create Project Directory

```bash
mkdir image-generation-api
cd image-generation-api
```

### 2. Create Virtual Environment (Recommended)

```bash
python -m venv venv

# Activate on macOS/Linux:
source venv/bin/activate

# Activate on Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install fastapi uvicorn diffusers torch transformers accelerate pillow
```

### 4. Save the API Code

Save the FastAPI code as `main.py` in your project directory.

## 🎯 Usage

### Start the Server

```bash
# Method 1: Direct Python
python main.py

# Method 2: Using Uvicorn
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# For production (with multiple workers):
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 2
```

The API will be available at: `http://localhost:8000`

### API Documentation

Once running, visit:
- **Interactive API docs**: http://localhost:8000/docs
- **Alternative docs**: http://localhost:8000/redoc

## 📡 API Endpoints

### 1. Health Check
```bash
GET /
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "mps"
}
```

### 2. Generate Image
```bash
POST /generate
```

**Request Body:**
```json
{
  "prompt": "Portrait of a cat wearing a suit and tie, digital art",
  "num_inference_steps": 25,
  "return_base64": false
}
```

**Parameters:**
- `prompt` (string, required): Text description of the image to generate
- `num_inference_steps` (integer, optional): Number of denoising steps (default: 25)
  - Lower (10-20): Faster, lower quality
  - Medium (25-35): Balanced (recommended)
  - Higher (40-50): Slower, better quality
- `return_base64` (boolean, optional): Return base64 encoded image (default: false)

**Response (file mode):**
Returns PNG image file directly

**Response (base64 mode):**
```json
{
  "success": true,
  "prompt": "Portrait of a cat wearing a suit and tie",
  "steps": 25,
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "format": "png"
}
```

## 🧪 Testing Examples

### Using cURL

```bash
# Health check
curl http://localhost:8000/health

# Generate image (returns file)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains, photorealistic",
    "num_inference_steps": 30
  }' \
  --output image.png

# Generate image (returns base64)
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A futuristic city at night, cyberpunk style",
    "num_inference_steps": 25,
    "return_base64": true
  }'
```

### Using Python

```python
import requests
import base64
from PIL import Image
from io import BytesIO

# Generate image
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A cute robot playing guitar, cartoon style",
        "num_inference_steps": 25,
        "return_base64": True
    }
)

# Decode and save base64 image
if response.status_code == 200:
    data = response.json()
    image_data = base64.b64decode(data["image_base64"])
    image = Image.open(BytesIO(image_data))
    image.save("output.png")
    print("Image saved!")
```

### Using JavaScript/Node.js

```javascript
const axios = require('axios');
const fs = require('fs');

async function generateImage() {
  const response = await axios.post('http://localhost:8000/generate', {
    prompt: 'A magical forest with glowing mushrooms',
    num_inference_steps: 25,
    return_base64: true
  });
  
  const imageBuffer = Buffer.from(response.data.image_base64, 'base64');
  fs.writeFileSync('output.png', imageBuffer);
  console.log('Image saved!');
}

generateImage();
```

## 🔗 n8n Integration

### Setup in n8n:

1. **Add HTTP Request Node**
2. **Configure the node:**
   - **Method**: POST
   - **URL**: `http://your-server-ip:8000/generate`
   - **Authentication**: None (add if needed)
   - **Body Content Type**: JSON
   
3. **Request Body:**
```json
{
  "prompt": "{{ $json.prompt }}",
  "num_inference_steps": 25,
  "return_base64": true
}
```

4. **Process the Response:**
   - The `image_base64` field contains the PNG image
   - Use "Move Binary Data" node to convert to file
   - Or send directly to other services (Slack, email, etc.)

### Example n8n Workflow:

```
[Webhook/Trigger] → [HTTP Request to API] → [Move Binary Data] → [Save to Google Drive/Email/etc.]
```

## ⚙️ Configuration

### Change Model

Edit `main.py` to use a different model:

```python
pipeline = DiffusionPipeline.from_pretrained(
    "segmind/SSD-1B",  # or "runwayml/stable-diffusion-v1-5"
    torch_dtype=torch.float16
)
```

### Change Port

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

### Enable CORS (for web apps)

Add to `main.py`:

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## 🐳 Docker Deployment (Optional)

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:

```bash
docker build -t image-gen-api .
docker run -p 8000:8000 image-gen-api
```

## 🔧 Troubleshooting

### Model Download Issues
- First run downloads ~500MB model
- Ensure stable internet connection
- Model is cached in `~/.cache/huggingface/`

### Memory Issues
- Reduce `num_inference_steps` to 15-20
- Use smaller batch sizes
- On Mac, close other memory-intensive apps

### Slow Generation
- **Mac M2**: 15-25 seconds is normal for 25 steps
- Use fewer steps (15-20) for faster generation
- Ensure MPS is being used (check logs)

### MPS Not Available on Mac
```bash
# Check PyTorch MPS support
python -c "import torch; print(torch.backends.mps.is_available())"

# Update PyTorch if needed
pip install --upgrade torch
```

## 📊 Performance Benchmarks

Approximate generation times (25 steps):

| Device | Time |
|--------|------|
| NVIDIA RTX 4090 | ~2-3s |
| NVIDIA RTX 3060 | ~5-8s |
| Apple M2 Ultra | ~10-15s |
| Apple M2 | ~15-25s |
| CPU (i7) | ~60-120s |

## 📝 License

This project uses:
- FastAPI (MIT License)
- Diffusers (Apache 2.0)
- Segmind Tiny-SD model (check model license on HuggingFace)

## 🤝 Contributing

Feel free to submit issues and enhancement requests!

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Segmind Tiny-SD Model](https://huggingface.co/segmind/tiny-sd)
- [n8n Documentation](https://docs.n8n.io/)

## 💡 Tips

- Start with 20-25 steps for testing, increase for final images
- Use descriptive prompts for better results
- Add style keywords like "digital art", "photorealistic", "cartoon style"
- Keep prompts under 77 tokens for best results

---

**Happy Image Generating! 🎨✨**
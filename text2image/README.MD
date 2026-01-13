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
  "hero_description": "a young woman with long red hair, green eyes, wearing a blue jacket",
  "scenes": [
    "standing on a mountain peak at sunrise",
    "walking through a busy city street at noon",
    "sitting on a beach at sunset"
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
- `hero_description` (string, optional): **NEW!** Detailed description of your main character/subject to maintain consistency across all scenes. This description is automatically added to each scene prompt.
- `scenes` (array of strings, required): List of scene descriptions (what the character is doing or where they are). Minimum 2 scenes required
- `num_inference_steps` (integer, optional, default: 25): Quality of each generated image (10-50)
- `duration_per_scene` (float, optional, default: 3.0): How long each scene appears in seconds
- `fps` (integer, optional, default: 24): Frames per second (standard: 24, smooth: 30, cinematic: 24)
- `width` (integer, optional, default: 512): Video width in pixels (recommend: 512, 768, or 1024)
- `height` (integer, optional, default: 512): Video height in pixels (recommend: 512, 768, or 1024)
- `transition_frames` (integer, optional, default: 12): Number of frames for crossfade transition between scenes

**How Character Consistency Works:**
When you provide a `hero_description`, the API automatically prepends it to each scene. For example:
- Hero: "a young woman with long red hair, green eyes, wearing a blue jacket"
- Scene: "standing on a mountain peak"
- **Final prompt:** "a young woman with long red hair, green eyes, wearing a blue jacket, standing on a mountain peak"

This helps the AI generate the same character across all scenes!

**Tips for Best Character Consistency:**
1. **Be specific**: Include details like hair color, eye color, clothing, age, body type
2. **Use consistent style**: Add style keywords like "photorealistic", "anime style", "cartoon"
3. **Mention distinctive features**: Tattoos, glasses, accessories, facial hair
4. **Keep it detailed but concise**: 15-30 words works best

**Example Character Descriptions:**
- "a muscular man with short black hair, wearing a red superhero cape and blue suit"
- "a young girl with blonde pigtails, freckles, wearing overalls and a yellow t-shirt, cartoon style"
- "an elderly wizard with long white beard, purple robes, holding a wooden staff, fantasy art style"

**Example with cURL:**
```bash
curl -X POST "http://localhost:8000/generate-video" \
  -H "Content-Type: application/json" \
  -d '{
    "hero_description": "a brave knight in silver armor with a red cloak",
    "scenes": [
      "standing in front of a castle gate",
      "fighting a dragon in a cave",
      "celebrating victory in a village square"
    ],
    "duration_per_scene": 4.0,
    "fps": 30
  }' \
  --output video.mp4
```

**Example with Python:**
```python
import requests

response = requests.post(
    "http://localhost:8000/generate-video",
    json={
        "hero_description": "a cyberpunk hacker with neon blue mohawk, wearing VR goggles and leather jacket",
        "scenes": [
            "typing on holographic keyboard in dark room",
            "running through rainy neon-lit streets",
            "overlooking the city from a rooftop at night"
        ],
        "num_inference_steps": 30,
        "duration_per_scene": 5.0,
        "fps": 24,
        "width": 768,
        "height": 768,
        "transition_frames": 15
    }
)

with open("hacker_story.mp4", "wb") as f:
    f.write(response.content)
```

**Response:**
Returns MP4 video file with metadata in headers:
- `X-Video-Scenes`: Number of scenes in the video
- `X-Video-Duration`: Total duration in seconds
- `X-Video-FPS`: Frames per second

**Use Cases:**
- Create visual stories or narratives
- Generate marketing videos
- Produce AI-powered slideshows
- Create concept videos for presentations
- Automated video content generation

---

### 4. Get Video Info

**POST /generate-video/info** - Preview video specifications without generating

**Request Body:**
Same as `/generate-video` endpoint (including optional `hero_description`).

**Example:**
```bash
curl -X POST "http://localhost:8000/generate-video/info" \
  -H "Content-Type: application/json" \
  -d '{
    "hero_description": "a brave knight in silver armor",
    "scenes": ["Scene 1", "Scene 2", "Scene 3"],
    "duration_per_scene": 3.0,
    "fps": 24,
    "transition_frames": 12
  }'
```

**Response:**
```json
{
  "scenes_count": 3,
  "frames_per_scene": 72,
  "transition_frames": 12,
  "total_frames": 240,
  "total_duration_seconds": 10.0,
  "fps": 24,
  "resolution": "512x512",
  "estimated_generation_time_seconds": 15
}
```

**Use Cases:**
- Preview video duration before generating
- Calculate resource requirements
- Plan video structure
- Estimate processing time

---

### 5. Health Check

**GET /health** - Check if the service is running properly

```bash
curl http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

**Use Cases:**
- Monitor API availability
- Check if model is loaded
- Verify GPU/CPU status
- Integration with monitoring systems

---

## Usage Examples

### Example 1: Generate a Single Image

```python
import requests

# Generate an image
response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A cyberpunk city at night with neon lights",
        "num_inference_steps": 35
    }
)

# Save the image
with open("cyberpunk_city.png", "wb") as f:
    f.write(response.content)

print("Image saved as cyberpunk_city.png")
```

### Example 2: Generate a Story Video with Consistent Character

**IMPORTANT:** Always include the `hero_description` field in your request to maintain character consistency!

```python
import requests

# Step 1: Define your hero/character with detailed description
hero = "a young adventurer with brown messy hair, wearing a green tunic and brown boots, fantasy art style"

# Step 2: Define story scenes (just describe the action/setting)
story_scenes = [
    "standing at the entrance of a dark cave, holding a torch",
    "exploring inside the cave with glowing crystals",
    "discovering an ancient treasure chest",
    "running away from a giant spider",
    "celebrating victory outside the cave at sunset"
]

# Step 3: Generate the video - MUST include hero_description in the request!
response = requests.post(
    "http://localhost:8000/generate-video",
    json={
        "hero_description": hero,  # ← THIS IS REQUIRED for character consistency
        "scenes": story_scenes,
        "duration_per_scene": 4.0,
        "fps": 30,
        "width": 768,
        "height": 768,
        "num_inference_steps": 30
    }
)

# Save the video
with open("adventure_story.mp4", "wb") as f:
    f.write(response.content)

print("Video saved! Your hero appears consistently in all scenes.")
```

### Example 3: Video WITHOUT Character (Landscape/Scene-based)

If you don't need a consistent character, simply omit the `hero_description`:

```python
import requests

# For videos without characters, just use scene descriptions
landscape_scenes = [
    "A peaceful sunrise over mountains with mist",
    "A busy city street at noon with cars",
    "A calm ocean beach at sunset with waves"
]

response = requests.post(
    "http://localhost:8000/generate-video",
    json={
        # No hero_description needed for landscapes
        "scenes": landscape_scenes,
        "duration_per_scene": 3.0,
        "fps": 24
    }
)

with open("landscape_video.mp4", "wb") as f:
    f.write(response.content)
```

### Example 4: Batch Image Generation

```python
import requests
import time

prompts = [
    "A red sports car",
    "A blue ocean wave",
    "A green forest landscape",
    "A purple sunset sky"
]

for i, prompt in enumerate(prompts):
    response = requests.post(
        "http://localhost:8000/generate",
        json={"prompt": prompt}
    )
    
    with open(f"image_{i+1}.png", "wb") as f:
        f.write(response.content)
    
    print(f"Generated image {i+1}/{len(prompts)}")
    time.sleep(1)  # Brief pause between requests
```

### Example 5: Using Base64 for Web Integration

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/generate",
    json={
        "prompt": "A beautiful landscape",
        "return_base64": True
    }
)

data = response.json()
base64_image = data["image_base64"]

# Now you can use this in HTML
html = f'<img src="data:image/png;base64,{base64_image}" />'
print("Base64 image ready for web use")
```

## Configuration

### Adjusting Model Quality

For higher quality images, increase `num_inference_steps`:
- **Fast (10-15 steps)**: Quick generation, lower quality
- **Balanced (20-30 steps)**: Good quality, reasonable speed
- **High Quality (35-50 steps)**: Best quality, slower generation

### Video Resolution Recommendations

- **512x512**: Fast, good for testing
- **768x768**: Better quality, moderate speed
- **1024x1024**: High quality, slower (requires more VRAM)

### FPS Guidelines

- **24 fps**: Cinematic, film-like
- **30 fps**: Smooth, standard video
- **60 fps**: Very smooth (larger file sizes)

## Troubleshooting

### Model Not Loading

**Issue:** API starts but model doesn't load

**Solution:**
- Check GPU memory availability
- Try using CPU mode by modifying device selection
- Ensure all dependencies are installed

### Out of Memory Errors

**Issue:** CUDA out of memory or similar errors

**Solutions:**
- Reduce image resolution (use 512x512 instead of 1024x1024)
- Lower `num_inference_steps`
- Close other GPU-intensive applications
- Use CPU mode if GPU memory is insufficient

### Slow Generation

**Issue:** Images or videos take too long to generate

**Solutions:**
- Reduce `num_inference_steps` (try 15-20)
- Use smaller resolution (512x512)
- Ensure GPU is being used (check `/health` endpoint)
- Reduce number of scenes in videos

### Character Consistency Issues

**Issue:** Character looks different in each scene despite using hero_description

**ROOT CAUSE:** The `hero_description` field was not included in the request body!

**Solutions:**
1. **ALWAYS include `hero_description` in your request** - This is the most important step!
   ```python
   # ✅ CORRECT - includes hero_description
   json={
       "hero_description": "a tall woman with red hair wearing a blue jacket",
       "scenes": ["walking in park", "sitting in cafe"]
   }
   
   # ❌ WRONG - missing hero_description  
   json={
       "scenes": ["a tall woman walking in park", "a tall woman sitting in cafe"]
   }
   ```

2. Make your hero description more detailed and specific
3. Include style keywords (e.g., "photorealistic", "3D render", "anime style")
4. Increase `num_inference_steps` to 35-40 for better adherence to prompts
5. Keep hero description under 30 words but pack it with distinctive details
6. Add "consistent character" to your hero description

**Example of a COMPLETE request with hero:**
```python
response = requests.post(
    "http://localhost:8000/generate-video",
    json={
        "hero_description": "consistent character, a tall athletic woman with shoulder-length blonde hair in a ponytail, wearing black tactical gear and sunglasses, photorealistic style",  # ← MUST INCLUDE THIS!
        "scenes": [
            "running through a hallway",
            "jumping over obstacles", 
            "standing on a rooftop"
        ],
        "num_inference_steps": 35
    }
)
```

### Video Quality Issues

**Issue:** Video looks choppy or low quality

**Solutions:**
- Increase `num_inference_steps` (30-40)
- Increase `duration_per_scene` for smoother playback
- Increase `transition_frames` for smoother transitions
- Use higher resolution (768x768 or 1024x1024)

### Connection Errors

**Issue:** Cannot connect to API

**Solutions:**
- Verify server is running: `curl http://localhost:8000/health`
- Check firewall settings
- Ensure correct port (8000) is not in use
- Try accessing from `http://127.0.0.1:8000` instead

## Performance Tips

1. **Batch Processing**: For multiple images, send requests sequentially with small delays
2. **Caching**: The model stays loaded, so subsequent requests are faster
3. **GPU Optimization**: Ensure CUDA/MPS is properly configured for best performance
4. **Resource Management**: Close the API when not in use to free GPU memory

## Support & Contributing

For issues, questions, or contributions:
- Check the troubleshooting section above
- Review the interactive API docs at `/docs`
- Ensure all dependencies are up to date

## License

This API uses open-source models and libraries. Please review their respective licenses.
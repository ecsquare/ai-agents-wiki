from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
import uuid
from pathlib import Path
from typing import List
import cv2
import numpy as np
from PIL import Image

app = FastAPI(title="Image and Video Generation API")

# Global variable for pipeline
pipeline = None
device = None

class ImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 25
    return_base64: bool = False

class VideoRequest(BaseModel):
    scenes: List[str] = Field(..., description="List of scene descriptions/prompts")
    hero_description: str = Field(default="", description="Description of main character to maintain consistency")
    num_inference_steps: int = 25
    duration_per_scene: float = Field(3.0, description="Duration in seconds for each scene")
    fps: int = Field(24, description="Frames per second")
    width: int = Field(512, description="Video width")
    height: int = Field(512, description="Video height")
    transition_frames: int = Field(12, description="Number of frames for crossfade")

@app.on_event("startup")
async def load_model():
    """Load the model when the API starts"""
    global pipeline, device
    
    print("Loading model...")
    
    # Set device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    print(f"Using device: {device}")
    
    # Load pipeline
    pipeline = DiffusionPipeline.from_pretrained(
        "segmind/tiny-sd", 
        torch_dtype=torch.float16
    )
    pipeline = pipeline.to(device)
    
    print(f"Model loaded on: {pipeline.device}")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "device": str(device),
        "model": "segmind/tiny-sd",
        "endpoints": {
            "generate_image": "/generate",
            "generate_video": "/generate-video",
            "video_info": "/generate-video/info"
        }
    }

@app.post("/generate")
async def generate_image(request: ImageRequest):
    """
    Generate an image from a text prompt
    
    Example:
    {
        "prompt": "Portrait of a cat wearing a suit",
        "num_inference_steps": 25,
        "return_base64": false
    }
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        print(f"Generating image for prompt: {request.prompt}")
        image = pipeline(
            request.prompt, 
            num_inference_steps=request.num_inference_steps
        ).images[0]
        
        if request.return_base64:
            buffered = BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return {
                "success": True,
                "prompt": request.prompt,
                "steps": request.num_inference_steps,
                "image_base64": img_str,
                "format": "png"
            }
        else:
            output_dir = Path("generated_images")
            output_dir.mkdir(exist_ok=True)
            
            filename = f"{uuid.uuid4()}.png"
            filepath = output_dir / filename
            image.save(filepath)
            
            return FileResponse(
                filepath,
                media_type="image/png",
                filename=filename
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

def create_crossfade_transition(img1: np.ndarray, img2: np.ndarray, num_frames: int) -> List[np.ndarray]:
    """Create a crossfade transition between two images"""
    transition_frames = []
    for i in range(num_frames):
        alpha = i / num_frames
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        transition_frames.append(blended)
    return transition_frames

@app.post("/generate-video")
async def generate_video(request: VideoRequest):
    """
    Generate a video from a list of scene descriptions
    
    Example:
    {
        "scenes": [
            "A peaceful sunrise over mountains",
            "A busy city street at noon",
            "A calm beach at sunset"
        ],
        "num_inference_steps": 25,
        "duration_per_scene": 3.0,
        "fps": 24,
        "width": 512,
        "height": 512,
        "transition_frames": 12
    }
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if len(request.scenes) < 2:
        raise HTTPException(status_code=400, detail="At least 2 scenes required")
    
    try:
        print(f"Generating video with {len(request.scenes)} scenes...")
        
        # Create output directories
        output_dir = Path("generated_videos")
        images_dir = output_dir / "temp_images"
        output_dir.mkdir(exist_ok=True)
        images_dir.mkdir(exist_ok=True)
        
        # Generate images for each scene
        generated_images = []
        for idx, scene_prompt in enumerate(request.scenes):
            scene_prompt = f"{scene_prompt}" if request.hero_description else scene_prompt
            print(f"Generating scene {idx + 1}/{len(request.scenes)}: {scene_prompt}")
            
            image = pipeline(
                scene_prompt,
                num_inference_steps=request.num_inference_steps
            ).images[0]
            
            # Resize to requested dimensions
            image = image.resize((request.width, request.height), Image.LANCZOS)
            
            # Convert to numpy array (RGB to BGR for OpenCV)
            img_array = np.array(image)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            generated_images.append(img_array)
        
        # Create video
        video_filename = f"video_{uuid.uuid4()}.mp4"
        video_path = output_dir / video_filename
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            str(video_path),
            fourcc,
            request.fps,
            (request.width, request.height)
        )
        
        # Calculate frames per scene
        frames_per_scene = int(request.duration_per_scene * request.fps)
        
        # Write video with transitions
        for idx in range(len(generated_images)):
            current_image = generated_images[idx]
            
            # Write frames for current scene
            for _ in range(frames_per_scene):
                video_writer.write(current_image)
            
            # Add transition to next scene (if not last scene)
            if idx < len(generated_images) - 1:
                next_image = generated_images[idx + 1]
                transition_frames = create_crossfade_transition(
                    current_image,
                    next_image,
                    request.transition_frames
                )
                for frame in transition_frames:
                    video_writer.write(frame)
        
        video_writer.release()
        
        # Calculate video metadata
        total_frames = (frames_per_scene * len(generated_images)) + \
                      (request.transition_frames * (len(generated_images) - 1))
        total_duration = total_frames / request.fps
        
        print(f"\nVideo generated successfully: {video_path}")
        print("="*80 + "\n")
        
        # Create metadata about prompts used
        response_headers = {
            "X-Video-Scenes": str(len(request.scenes)),
            "X-Video-Duration": str(total_duration),
            "X-Video-FPS": str(request.fps),
            "X-Hero-Used": "true" if request.hero_description else "false"
        }
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=video_filename,
            headers=response_headers
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")

@app.post("/generate-video/info")
async def generate_video_info(request: VideoRequest):
    """Get video specs without generating"""
    frames_per_scene = int(request.duration_per_scene * request.fps)
    total_frames = (frames_per_scene * len(request.scenes)) + \
                  (request.transition_frames * (len(request.scenes) - 1))
    total_duration = total_frames / request.fps
    
    return {
        "scenes_count": len(request.scenes),
        "frames_per_scene": frames_per_scene,
        "transition_frames": request.transition_frames,
        "total_frames": total_frames,
        "total_duration_seconds": total_duration,
        "fps": request.fps,
        "resolution": f"{request.width}x{request.height}",
        "estimated_generation_time_seconds": len(request.scenes) * 5
    }

@app.get("/health")
async def health_check():
    """Check service health"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": str(device) if device else "not set"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
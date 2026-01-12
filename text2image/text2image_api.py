from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from diffusers import DiffusionPipeline
import torch
import base64
from io import BytesIO
import uuid
from pathlib import Path

app = FastAPI(title="Image Generation API")

# Global variable for pipeline
pipeline = None
device = None

class ImageRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 25
    return_base64: bool = False  # If True, returns base64 encoded image

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
        "model": "segmind/tiny-sd"
    }

@app.post("/generate")
async def generate_image(request: ImageRequest):
    """
    Generate an image from a text prompt
    
    Example request:
    {
        "prompt": "Portrait of a cat wearing a suit",
        "num_inference_steps": 25,
        "return_base64": false
    }
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        # Generate image
        print(f"Generating image for prompt: {request.prompt}")
        image = pipeline(
            request.prompt, 
            num_inference_steps=request.num_inference_steps
        ).images[0]
        
        if request.return_base64:
            # Return base64 encoded image (good for n8n)
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
            # Save to file and return file path
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

@app.get("/health")
async def health_check():
    """Check if the service is healthy"""
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "device": str(device) if device else "not set"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
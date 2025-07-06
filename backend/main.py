# backend/app/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import torch
import numpy as np
from model import create_model
from midi_utils import midi_to_piano_roll, piano_roll_to_midi  # Remove the long path

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (create dummy for now)
model = create_model()
model.eval()

@app.post("/api/harmonize")
async def harmonize(midi: UploadFile = File(...)):
    # Save uploaded MIDI
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_in:
        shutil.copyfileobj(midi.file, tmp_in)
        input_path = tmp_in.name

    # Process MIDI
    try:
        # Convert to piano roll (now returns 13 features)
        melody_features = midi_to_piano_roll(input_path)
        
        # Run model
        with torch.no_grad():
            input_tensor = torch.FloatTensor(melody_features).unsqueeze(0)
            harmony = model(input_tensor)
            harmony_np = harmony.squeeze(0).numpy()  # Shape: (32, 12)
        
        # Extract melody pitch classes (first 12 columns) from melody_features
        melody_pitch_classes = melody_features[:, :12]  # Shape: (32, 12)
        
        # Create separate tracks: melody and harmony
        output_path = input_path.replace(".mid", "_harmonized.mid")
        piano_roll_to_midi(melody_pitch_classes, harmony_np, output_path)
        
        return FileResponse(output_path, filename="harmonized.mid")
        
    except Exception as e:
        print(f"Error: {e}")
        # Fallback: return original file
        return FileResponse(input_path, filename="harmonized.mid")
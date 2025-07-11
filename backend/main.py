# backend/main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import shutil
import torch
import numpy as np
from model import create_model, load_model
from data_processor import MeasureBasedChordProcessor
from midi_utils import piano_roll_to_midi_chords
import music21
import os

app = FastAPI()

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create processor for inference
processor = MeasureBasedChordProcessor()

@app.post("/api/harmonize")
async def harmonize(midi: UploadFile = File(...), model: str = Form("creative")):
    # Save uploaded MIDI
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_in:
        shutil.copyfileobj(midi.file, tmp_in)
        input_path = tmp_in.name

    try:
        print(f"=== HARMONIZATION REQUEST ===")
        print(f"Requested model: '{model}'")
        print(f"Model type: {type(model)}")
        print(f"Bach model file exists: {os.path.exists('models/chord_harmonizer.pt')}")
        
        # Parse MIDI to music21 score
        score = music21.converter.parse(input_path)
        
        # Detect key
        detected_key = processor.detect_key(score)
        print(f"Detected key: {detected_key}")
        
        # Extract melody part
        parts = list(score.parts)
        melody_part = parts[0] if parts else None
        
        if melody_part:
            # Extract features with actual length
            melody_features, chord_labels, actual_length = processor.extract_measure_based_features(melody_part, detected_key)
            
            print(f"Processing {actual_length} beats of melody")
            
            # Choose harmonization method based on selected model
            print(f"Checking condition: model == 'bach' and os.path.exists('models/chord_harmonizer.pt')")
            print(f"model == 'bach': {model == 'bach'}")
            print(f"os.path.exists('models/chord_harmonizer.pt'): {os.path.exists('models/chord_harmonizer.pt')}")
            
            if model == "bach" and os.path.exists("models/chord_harmonizer.pt"):
                print("✅ Using Bach neural network model")
                bach_model = load_model("models/chord_harmonizer.pt")
                
                # Pad melody_features to model's expected size
                model_input_size = 32
                
                if melody_features.shape[0] < model_input_size:
                    padding = np.zeros((model_input_size - melody_features.shape[0], melody_features.shape[1]))
                    padded_features = np.vstack([melody_features, padding])
                    print(f"Padded features from {melody_features.shape} to {padded_features.shape}")
                else:
                    padded_features = melody_features[:model_input_size]
                
                with torch.no_grad():
                    input_tensor = torch.FloatTensor(padded_features).unsqueeze(0)
                    
                    # Get model predictions
                    chord_logits = bach_model(input_tensor)
                    print(f"Bach model raw logits shape: {chord_logits.shape}")
                    print(f"Bach model raw logits[0][0]: {chord_logits[0][0]}")
                    
                    # Apply temperature sampling
                    temperature = 2.0
                    chord_probs = torch.softmax(chord_logits / temperature, dim=-1)
                    
                    # Convert to numpy
                    chord_probs = chord_probs.squeeze(0).numpy()
                    
                    # Only use predictions for actual melody length
                    chord_probs = chord_probs[:actual_length]
                    
                    # Debug: Show raw probabilities
                    print(f"Bach model probabilities[0]: {chord_probs[0]}")
                    print(f"Bach model probabilities[5]: {chord_probs[5] if len(chord_probs) > 5 else 'N/A'}")
                    
                    # Debug model predictions
                    predicted_chords = [np.argmax(chord_probs[i]) for i in range(actual_length)]
                    print(f"Bach model predicted: {predicted_chords}")
                    
                    # CRITICAL: Check if this is the same as creative engine
                    creative_chords = [np.argmax(chord_labels[i]) for i in range(actual_length)]
                    print(f"Creative engine would predict: {creative_chords}")
                    print(f"Are they identical? {predicted_chords == creative_chords}")
                    
                    # Report diversity but don't fallback
                    unique_chords = len(set(predicted_chords))
                    print(f"✅ Bach model output: {unique_chords} unique chords")
                    if unique_chords == 1:
                        print(f"   Note: Bach model is producing uniform output (all chord {predicted_chords[0]})")
                    
                    # USE BACH MODEL OUTPUT REGARDLESS
                    # chord_probs is already set correctly above
            
            elif model == "bach":
                print("❌ Bach model requested but not found, falling back to creative engine")
                chord_probs = chord_labels[:actual_length].copy()
                predicted_chords = [np.argmax(chord_probs[i]) for i in range(actual_length)]
                print(f"Creative fallback prediction: {predicted_chords}")
                
            else:  # model == "creative" or default
                print(f"✅ Using creative chord progression engine (model='{model}')")
                chord_probs = chord_labels[:actual_length].copy()
                predicted_chords = [np.argmax(chord_probs[i]) for i in range(actual_length)]
                print(f"Creative engine prediction: {predicted_chords}")
            print(f"Final chord sequence: {predicted_chords}")
            print(f"Chord diversity: {len(set(predicted_chords))} unique chords out of 7")
            
            # Create harmonized MIDI
            output_path = input_path.replace(".mid", "_harmonized.mid")
            piano_roll_to_midi_chords(input_path, chord_probs, detected_key, output_path)
            
            return FileResponse(output_path, filename="harmonized.mid")
        else:
            return FileResponse(input_path, filename="harmonized.mid")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return FileResponse(input_path, filename="harmonized.mid")

@app.get("/api/models")
async def get_available_models():
    """Get list of available models"""
    models = [
        {
            "id": "creative",
            "name": "Creative Engine (Rule-Based)",
            "description": "Analyzes melody and applies music theory rules",
            "available": True
        },
        {
            "id": "bach",
            "name": "Bach Neural Network", 
            "description": "Neural network trained on Bach chorales",
            "available": os.path.exists("models/chord_harmonizer.pt")
        }
    ]
    return {"models": models}
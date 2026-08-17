# backend/main.py
"""FastAPI service: upload a melody MIDI file, get back a harmonized MIDI file."""
import logging
import os
import shutil
import tempfile

import music21
import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from data_processor import MeasureBasedChordProcessor
from midi_utils import piano_roll_to_midi_chords
from model import load_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("harmonizer")

BACH_MODEL_PATH = "models/chord_harmonizer.pt"

MODEL_INFO = {
    "creative": {
        "name": "Creative Engine (Rule-Based)",
        "description": "Analyzes the melody and applies music theory rules",
    },
    "bach": {
        "name": "Bach Neural Network",
        "description": "BiLSTM trained on Bach chorales; falls back to the Creative Engine if unavailable",
    },
}

app = FastAPI(title="Harmonizer API")

# CORS_ORIGINS: comma-separated allow-list, defaults to "*" for local/demo use.
cors_origins = os.environ.get("CORS_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if cors_origins == "*" else [o.strip() for o in cors_origins.split(",")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

processor = MeasureBasedChordProcessor()


def _predict_chord_probs(model_name, melody_features, chord_labels, actual_length):
    """Return per-beat chord probabilities, shape (actual_length, 7), for the requested model."""
    if model_name == "bach" and os.path.exists(BACH_MODEL_PATH):
        bach_model = load_model(BACH_MODEL_PATH)
        with torch.no_grad():
            input_tensor = torch.FloatTensor(melody_features).unsqueeze(0)
            logits = bach_model(input_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        return probs[:actual_length]

    if model_name == "bach":
        logger.warning("Bach model requested but %s is missing; using the creative engine instead", BACH_MODEL_PATH)

    return chord_labels[:actual_length].copy()


@app.post("/api/harmonize")
async def harmonize(midi: UploadFile = File(...), model: str = Form("creative")):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_in:
        shutil.copyfileobj(midi.file, tmp_in)
        input_path = tmp_in.name

    try:
        score = music21.converter.parse(input_path)
        detected_key = processor.detect_key(score)

        parts = list(score.parts)
        melody_part = parts[0] if parts else None
        if melody_part is None:
            return FileResponse(input_path, filename="harmonized.mid")

        melody_features, chord_labels, actual_length = processor.extract_measure_based_features(melody_part, detected_key)
        chord_probs = _predict_chord_probs(model, melody_features, chord_labels, actual_length)

        output_path = input_path.replace(".mid", "_harmonized.mid")
        piano_roll_to_midi_chords(input_path, chord_probs, detected_key, output_path)
        return FileResponse(output_path, filename="harmonized.mid")

    except Exception:
        logger.exception("Harmonization failed; returning the original melody unchanged")
        return FileResponse(input_path, filename="harmonized.mid")


@app.get("/api/models")
async def get_available_models():
    """List harmonization models and whether each is currently available."""
    return {
        "models": [
            {
                "id": model_id,
                "name": info["name"],
                "description": info["description"],
                "available": model_id != "bach" or os.path.exists(BACH_MODEL_PATH),
            }
            for model_id, info in MODEL_INFO.items()
        ]
    }

from fastapi import FastAPI
from pydantic import BaseModel

class PredictResponse(BaseModel):
    chords: list[str] = []  # empty list for now

app = FastAPI(title="Harmonizer Model Stub", version="0.0.1")

@app.post("/predict", response_model=PredictResponse)
async def predict_stub():
    """Return an empty chord sequence; replaced later by ONNX inference."""
    return PredictResponse()
# HarmonAIzer

A full-stack MIDI harmonizer that adds chord accompaniment to recorded or uploaded melodies. The React interface
sends MIDI to a FastAPI service, which detects the key and generates a downloadable harmonized MIDI file.

Choose between:

- **Creative Engine**: derives chord choices from key and melody features using music-theory rules.
- **Bach Model**: runs a PyTorch chord model trained from Bach chorales when the model artifact is available.

## Architecture

- `frontend/`: React 19, TypeScript, Vite, Web MIDI tooling, and an interactive piano.
- `backend/`: FastAPI endpoint for MIDI upload, key detection, harmonization, and MIDI export.
- `backend/data_processor.py`: `music21` feature extraction and rule-based chord labels.
- `backend/model.py`: PyTorch sequence model used by the Bach engine.
- `backend/midi_utils.py`: writes generated chord tracks into the returned MIDI.

## Technical highlights

- Preserves the uploaded melody while adding a generated chord track.
- Supports deterministic rule-based output and learned chord prediction behind the same API.
- Detects the input key before selecting and rendering diatonic chord accompaniment.
- Provides in-browser MIDI recording, playback controls, model selection, and download.

## Run locally

Python 3.12 and Node.js are recommended.

```powershell
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install fastapi "uvicorn[standard]" python-multipart torch numpy music21 mido
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

The frontend currently calls the backend at `http://localhost:8000`.

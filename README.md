# HarmonAIzer

A full-stack MIDI harmonizer: record or upload a melody in the browser, and get back a
downloadable MIDI file with a generated chord accompaniment track.

Two harmonization engines are available behind the same API:

- **Creative Engine** - a deterministic music-theory engine. It detects the key, then
  picks the best-fitting diatonic chord for each measure using chord-tone/passing-tone
  scoring plus a bonus for common functional progressions (e.g. V -> I).
- **Bach Neural Network** - a small bidirectional LSTM trained on ~400 Bach chorales.
  Unlike a rule engine, it's trained on the *actual* 4-part harmony Bach wrote (extracted
  via `music21`'s `chordify()`), not just melody heuristics, so it learns genuine
  harmonic style rather than reproducing the Creative Engine's own rules.

## Architecture

- `frontend/` - React 19 + TypeScript + Vite. Records melodies from a virtual/MIDI
  keyboard onto a piano-roll grid, exports MIDI, and calls the backend to harmonize.
- `backend/` - FastAPI service: MIDI upload -> key detection -> chord prediction -> MIDI
  export with a generated accompaniment track.
  - `data_processor.py` - melody feature extraction and the rule-based Creative Engine.
  - `model.py` - the BiLSTM chord model architecture.
  - `train.py` - trains the model on the Bach corpus and saves `models/chord_harmonizer.pt`.
  - `midi_utils.py` - renders a predicted chord sequence into a MIDI accompaniment track.

## Run with Docker Compose

The simplest way to run the whole stack:

```powershell
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000

The frontend container serves the built app via nginx, which also reverse-proxies
`/api/*` to the backend container - no CORS configuration needed.

## Run locally for development

Python 3.12+ and Node.js 20+ are recommended.

```powershell
# Backend
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend (new terminal)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 - Vite's dev server proxies `/api/*` to `http://localhost:8000`
(see `vite.config.ts`), so the frontend and backend can run independently.

## Retraining the Bach model

The trained checkpoint is committed at `backend/models/chord_harmonizer.pt`, so this step
is optional. To retrain from scratch:

```powershell
cd backend
.\.venv\Scripts\Activate.ps1
python train.py
```

This reprocesses the Bach corpus bundled with `music21`, trains for up to a few hundred
epochs with early stopping, and overwrites `models/chord_harmonizer.pt`.

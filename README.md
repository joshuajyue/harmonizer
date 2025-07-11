# HarmonAIzer

A MIDI harmonizer that adds chord progressions to melody lines.

## What it does

Upload a MIDI file with a melody and get back the same melody with chord accompaniment. Choose between:

- **Creative Engine**: Uses music theory rules to pick chords
- **Bach Model**: Neural network trained on Bach chorales

## Quick Demo

### Setup
```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python main.py

# Frontend (new terminal)
cd frontend
npm install
npm start

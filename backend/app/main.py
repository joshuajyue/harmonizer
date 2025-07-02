# filepath: backend/main.py
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import tempfile
import shutil

app = FastAPI()

@app.post("/api/harmonize")
async def harmonize(midi: UploadFile = File(...)):
    # Save uploaded MIDI to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as tmp_in:
        shutil.copyfileobj(midi.file, tmp_in)
        input_path = tmp_in.name

    # TODO: Run your ML model here and save output to output_path
    output_path = input_path.replace(".mid", "_harm.mid")
    shutil.copy(input_path, output_path)  # Dummy: just copies input for now

    return FileResponse(output_path, filename="harmonized.mid")
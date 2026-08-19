# HarmonAIzer backend

Run from the repository root:

```bash
python -m venv backend/.venv
backend/.venv/bin/pip install -r backend/requirements-dev.txt
HARMONIZER_ENABLE_DEV_ENGINE=1 backend/.venv/bin/uvicorn backend.main:app --reload
```

The repository root must be the working directory because `backend`, `contracts`,
and `ml` are sibling packages. If already inside `backend/`, run `cd ..` first.

The development engine returns `contracts/examples/harmonize.response.json`. It
exists for realistic service smoke tests; all musical decisions belong in `ml/`.

## API

- `POST /api/v1/harmonize` and `GET /api/v1/engines`
- `POST /api/v1/render` and `GET /api/v1/synths`
- `POST /api/v1/transcribe` (`audio` multipart field; `file` is also accepted).
  Supply tempo explicitly or it is estimated from note-onset intervals.
- `POST /api/v1/midi/import`
- `POST /api/v1/midi/export?tempo=<bpm>` (tempo is required)
- `GET /api/v1/health`

Engine modules are imported lazily from `ml.engines`; the API never owns an
engine list or falls back to a different harmonizer. CPU-heavy harmonization,
rendering, and pitch tracking run in FastAPI's worker thread pool.

## Audio design

`sf2` prefers the FluidSynth CLI and a configured/system SoundFont. A small
in-process wavetable renderer is the guaranteed fallback, so preview WAV
rendering also works on machines without FluidSynth.

`ddsp` is deliberately optional. DDSP-SVC currently advertises much lower
hardware use than so-vits-svc and near-RVC training speed, while RVC remains a
good low-latency alternative. Both require model files and hardware-specific
PyTorch installs, and model/data licenses vary. so-vits-svc is archived and its
stable repository is AGPL-3.0. Consequently none of these stacks is imported by
the base service. Instead, configure a lazy adapter:

```text
HARMONIZER_DDSP_ADAPTER=my_voice_backend:adapter
```

The object must be callable, or expose `render`, with keyword arguments
`voices`, `tempo`, `timbre`, `guide_audio`, and `sample_rate`, and return WAV
bytes. It may expose `is_available()`. This seam supports DDSP-SVC or RVC
without coupling the API image to either project's dependency lock.

If neural rendering is unavailable, a `ddsp` request tries optional WORLD
resynthesis from `backend/timbres/<id>.wav`, then `sf2`. The actual backend and
fallback reason are returned in `X-HarmonAIzer-*` headers. Install WORLD with
`requirements-voice.txt`. Only use voice samples and model weights you are
authorized to process.

The design follows the current primary projects:
[DDSP-SVC](https://github.com/yxlllc/DDSP-SVC),
[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI),
[so-vits-svc](https://github.com/svc-develop-team/so-vits-svc),
[WORLD/pyworld](https://github.com/JeremyCCHsu/Python-Wrapper-for-World-Vocoder),
and [FluidSynth](https://github.com/FluidSynth/fluidsynth). Transcription uses
librosa pYIN, with libsndfile first and a bundled imageio-ffmpeg binary for
browser WebM/Opus. A separate system ffmpeg installation is not required.

The contract has no reference-audio/model-upload type, so user voice enrollment
is intentionally not invented here: `timbre` resolves only server-authorized
models/samples. MIDI export also takes tempo as a query parameter because
`HarmonizeResponse` does not carry tempo.

Dependency resolution was checked against PyPI: `fastapi==0.141.1` and
`music21==10.5.0` exist, while `torch==2.13.0` does not resolve.
This service no longer imports music21 or torch, so neither heavy package is in
the base image; ML/neural adapters own their hardware-specific dependencies.
To build an image with the ML training extras anyway, pass
`--build-arg HARMONIZER_INSTALL_ML_EXTRAS=1`:

```bash
docker build --build-arg HARMONIZER_INSTALL_ML_EXTRAS=1 \
  --file backend/Dockerfile .
```

## Configuration

| Variable | Purpose |
| --- | --- |
| `HARMONIZER_SOUNDFONT` | Explicit `.sf2` path |
| `HARMONIZER_DDSP_ADAPTER` | Lazy `module:object` neural adapter |
| `HARMONIZER_TIMBRE_DIR` | Authorized WORLD reference WAV directory |
| `HARMONIZER_MAX_UPLOAD_BYTES` | Upload limit (default 25 MiB) |
| `HARMONIZER_MAX_RENDER_SECONDS` | Render/transcription limit (default 180s) |
| `CORS_ORIGINS` | Optional comma-separated local-dev origins |

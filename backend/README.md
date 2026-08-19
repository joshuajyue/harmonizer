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
  Supply tempo explicitly or it is estimated from note-onset intervals. By
  default, the complete melody is shifted by whole octaves to best fit the
  soprano/melody working range, MIDI 60-79. Use `normalizeOctave=false` to keep
  the sung register or `octaveShift=<signed octaves>` to force a global shift;
  the manual shift takes precedence. Responses report
  `X-HarmonAIzer-Octave-Shift` and the pre-shift MIDI median in
  `X-HarmonAIzer-Detected-Median-Pitch`.
- `POST /api/v1/midi/import`
- `POST /api/v1/midi/export?tempo=<bpm>&numerator=<n>&denominator=<d>`
  (tempo and complete meter are required)
- `GET /api/v1/health`

Engine modules are imported lazily from `ml.engines` and optional `ml.reharm`;
the API never owns an engine list or falls back to a different harmonizer.
CPU-heavy harmonization, rendering, and pitch tracking run in FastAPI's worker
thread pool.

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
bytes. It may expose `is_available()`. Adapter renders are serialized because
most model runtimes are stateful and not safely reentrant. This seam supports
DDSP-SVC or RVC without coupling the API image to either project's dependency
lock.

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
models/samples. MIDI export takes tempo and meter as query parameters because
`HarmonizeResponse` carries neither.

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
| `HARMONIZER_MAX_JSON_BODY_BYTES` | JSON request limit (default 2 MiB) |
| `HARMONIZER_MAX_MIDI_NOTES` | Imported MIDI note limit (default 10,000) |
| `HARMONIZER_MAX_RENDER_SECONDS` | Render/transcription limit (default 180s) |
| `HARMONIZER_MAX_RENDER_NOTES` | Render note limit (default 10,000) |
| `HARMONIZER_MAX_RENDER_WORK_SECONDS` | Summed note-duration limit (default 1,440 note-seconds) |
| `CORS_ORIGINS` | Optional comma-separated local-dev origins |

## Project status

### Built and verified

- The complete contract API is implemented with registry-driven engine
  discovery, clean 503 engine failures, response validation, and no disguised
  harmony-engine fallback.
- WAV rendering supports FluidSynth plus a guaranteed wavetable fallback.
  Optional neural/WORLD backends load lazily, serialize model access, and
  report every fallback through `X-HarmonAIzer-*` provenance headers.
- WAV, MP3, and browser WebM/Opus transcription uses bundled FFmpeg and pYIN.
  Register normalization shifts the whole contour by octaves toward MIDI 60-79
  and reports both the shift and original median.
- MIDI import/export preserves explicit tempo, meter, and key metadata. Import
  preflight bounds event/note complexity before mido allocates message objects.
- Request-body, render-span, note-count, and summed-note-duration limits bound
  memory and CPU work. Optional response fields are omitted rather than emitted
  as contract-incompatible `null`.
- The Docker Compose stack has been built and exercised end to end: both
  services became healthy, nginx preserved `/api/v1/*`, registry harmonization
  completed through the web proxy, and FluidSynth produced valid WAV.
- Current validation: 54 backend tests pass, the contract drift guard passes,
  and the live API plus Vite proxy both return healthy responses.

### Known limitations

- High-quality voice synthesis is not bundled. `ddsp` needs an authorized,
  separately installed adapter/model; WORLD needs `pyworld` and a configured
  reference WAV. The base service remains fully functional without either.
- The frozen contract has no voice-enrollment or reference-audio upload model,
  so timbres are server-authorized IDs rather than user uploads.
- Rendering is synchronous. Hard work limits keep preview requests bounded,
  but a production neural deployment may still warrant an asynchronous job
  API with polling.
- Transcription is intentionally monophonic. Tempo estimation needs multiple
  usable onsets; clients should send their known project tempo when possible.
- Tests currently emit a Starlette deprecation warning for the legacy
  `fastapi.testclient` compatibility import; behavior is passing and unaffected.

### Recommended next work

1. Build a licensed, deployment-specific DDSP-SVC or RVC adapter against the
   documented seam and benchmark quality/latency before choosing a default.
2. Add a real recorded-voice transcription corpus spanning registers, noise,
   browsers, and microphones to tune pYIN segmentation and confidence handling.
3. Add the already verified Compose smoke path to CI when Docker runners are
   available.
4. If the API contract expands, add explicit voice enrollment and asynchronous
   render jobs rather than inventing undocumented request fields.

### Decisions to preserve

- Missing or failed harmony engines return an honest 503; they never silently
  substitute another engine.
- Musical register normalization is one global multiple-of-12 shift—never
  note-wise clamping or non-octave transposition.
- Neural dependencies and model weights stay optional and out of the base
  image.
- Fallbacks, substitutions, and normalization decisions are reported rather
  than hidden.

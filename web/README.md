# HarmonAIzer Studio

The v2 frontend is a fresh React + TypeScript + Vite application centered on
the voiced SATB result rather than file download.

## Run

```bash
npm install
npm run dev
```

Development mock mode is enabled by default, so both comparison engines,
editable voices, chords, violations, transcription, and rendered audio work
without a backend. It reads the schema-validated request and response directly
from `contracts/examples/` through Vite middleware. Set
`VITE_USE_MOCK_API=false` to use the Vite `/api/*` proxy to
`http://127.0.0.1:8000`.

```bash
npm test
npm run lint
npm run build
```

## Architecture

- **Zustand slices** separate project data, engine comparisons, transport, and
  editor UI state. Components subscribe only to the state they use; no root
  component owns or threads the studio state.
- **Canvas piano roll** draws the active result, beat grids, chord analysis,
  and violation markers without creating a DOM element per note.
- **Web Audio scheduler** uses `AudioContext.currentTime` and a short look-ahead
  scheduler for note and metronome timing.
- **Typed API client** is the only network boundary and imports all shared music
  and engine types directly from `contracts/types.ts`.

## Shortcuts

| Key | Action |
| --- | --- |
| Space | Play / pause; cancel an active count-in |
| Home | Return to start |
| Left / Right | Move the selection or playhead by the current snap value |
| Up / Down | Transpose selected notes; hold Shift for an octave |
| / | Toggle loop |
| M | Toggle metronome |
| R | Start, cancel, or stop recording |
| 1 / 2 | Show result A / B |
| Cmd/Ctrl + Enter | Harmonize the active result |
| Delete / Backspace | Delete selected notes |
| Escape | Clear selection or exit lane focus |

The virtual piano reserves the chromatic row
`A W S E D F T G Y H U J K O L P ;`; global character shortcuts are checked
against that set at startup.

## Note input

- **Record** (default) previews keyboard and MIDI input until recording starts.
  Recording supports an Off, one-bar, or two-bar meter-aware count-in.
- **Place** inserts a one-beat note on each keyboard or MIDI note-on and
  advances the playhead by one beat.
- **Snap starts** quantizes selected onsets to the current grid while preserving
  every note's duration.

## Project status

### Built and working

- Editable canvas SATB piano roll with drag, resize, delete, marquee selection,
  bulk movement, lane focus, chord ribbon, and timeline violation markers.
- One-result-at-a-time A/B engine comparison with independent engines,
  violation counts, editable results, and `1` / `2` switching.
- Virtual piano, Web MIDI, MIDI import, microphone transcription, octave
  restoration, Record/Place modes, and raw held-note capture.
- Off, one-bar, and two-bar meter-aware count-ins. Input previews but is not
  captured during the count-in, and recording starts at the original beat.
- AudioContext-scheduled local playback, live loop/tempo/meter/metronome
  updates, distinct voice timbres, backend audio rendering, and MIDI export.
- Typed API access, realistic default mock mode, and the
  `http://127.0.0.1:8000` development proxy.

### Known limitations and unfinished work

- No known blocking frontend defect is open at this checkpoint.
- `npm test` covers shortcut collisions and onset-only quantization, but there
  is no automated end-to-end suite for marquee edits, recording, count-in, or
  A/B switching.
- Web MIDI and microphone behavior still require final checks with physical
  devices and browser permission prompts. The local bass patch also merits a
  subjective laptop-speaker check.

### Recommended next work

1. Add browser regression tests for marquee delete/move, Place versus Record,
   count-in cancellation, onset-only quantization, and A/B switching.
2. Run a physical MIDI keyboard, microphone, and laptop-speaker acceptance pass.
3. Continue musician-led polish from observed use rather than adding
   unrequested workflow features.

### Decisions to preserve

- Only the active A/B result is drawn, edited, and played; results are never
  overlaid.
- Record is the default input mode. Place is an explicit alternate mode.
- Quantization moves selected onsets only and never changes duration.
- Count-in time is pre-roll and is never added to recorded beat positions.
- Harmonization stays behind the typed API; the frontend contains no music
  theory or voicing logic.

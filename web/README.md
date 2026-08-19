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

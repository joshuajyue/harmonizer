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
- **Canvas piano roll** draws all notes, beat grids, chord analysis, comparison
  layers, and violation markers without creating a DOM element per note.
- **Web Audio scheduler** uses `AudioContext.currentTime` and a short look-ahead
  scheduler for note and metronome timing.
- **Typed API client** is the only network boundary and imports all shared music
  and engine types directly from `contracts/types.ts`.

## Shortcuts

| Key | Action |
| --- | --- |
| Space | Play / pause |
| Home | Return to start |
| Left / Right | Move by the current snap value |
| L | Toggle loop |
| M | Toggle metronome |
| R | Compare both engines |
| 1 / 2 / 0 | Show A / B / overlay |
| Delete | Delete selected note |
| Escape | Clear note selection |

The virtual piano also maps chromatic notes from `A` through `;`.

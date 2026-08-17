// frontend/src/soundfont-player.d.ts
// Minimal type declarations for the untyped "soundfont-player" package - just enough
// of its API surface to type-check the code in this app.
declare module "soundfont-player" {
  export interface SoundfontNode {
    stop: (time?: number) => void;
  }

  export interface SoundfontInstrument {
    play: (note: string | number, when?: number, options?: { duration?: number; gain?: number }) => SoundfontNode;
    stop: (time?: number) => void;
  }

  const Soundfont: {
    instrument: (ctx: AudioContext, name: string) => Promise<SoundfontInstrument>;
  };

  export default Soundfont;
}

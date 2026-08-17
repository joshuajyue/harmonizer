// frontend/src/audio/audioContext.ts
// Cross-browser AudioContext creation (older Safari only exposes the prefixed
// `webkitAudioContext` constructor instead of the standard `AudioContext`).
interface WindowWithWebkitAudio extends Window {
  webkitAudioContext?: typeof AudioContext;
}

export function createAudioContext(): AudioContext {
  const win = window as WindowWithWebkitAudio;
  const AudioContextCtor = window.AudioContext || win.webkitAudioContext;
  return new AudioContextCtor();
}

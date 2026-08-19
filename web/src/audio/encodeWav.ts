function writeAscii(view: DataView, offset: number, value: string) {
  for (let index = 0; index < value.length; index += 1) {
    view.setUint8(offset + index, value.charCodeAt(index));
  }
}

function encodeMono(buffer: AudioBuffer) {
  const bytes = new ArrayBuffer(44 + buffer.length * 2);
  const view = new DataView(bytes);
  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + buffer.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, buffer.sampleRate, true);
  view.setUint32(28, buffer.sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, buffer.length * 2, true);

  const channels = Array.from(
    { length: buffer.numberOfChannels },
    (_value, index) => buffer.getChannelData(index),
  );
  for (let frame = 0; frame < buffer.length; frame += 1) {
    let sample = 0;
    for (const channel of channels) sample += channel[frame];
    sample = Math.max(-1, Math.min(1, sample / channels.length));
    view.setInt16(44 + frame * 2, sample * 0x7fff, true);
  }
  return new Blob([bytes], { type: "audio/wav" });
}

export async function transcodeRecordingToWav(recording: Blob) {
  if (recording.type === "audio/wav" || recording.type === "audio/x-wav") {
    return recording;
  }
  const context = new AudioContext();
  try {
    const decoded = await context.decodeAudioData(await recording.arrayBuffer());
    return encodeMono(decoded);
  } finally {
    await context.close();
  }
}

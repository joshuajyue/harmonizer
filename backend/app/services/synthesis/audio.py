from __future__ import annotations

import wave
from io import BytesIO

import numpy as np


def encode_pcm16_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    if samples.ndim == 1:
        samples = samples[:, np.newaxis]
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = np.asarray(np.rint(clipped * 32767), dtype="<i2")
    output = BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(pcm.shape[1])
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return output.getvalue()


def is_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"

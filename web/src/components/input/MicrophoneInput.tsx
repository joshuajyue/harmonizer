import { Mic } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { TimeSignature } from "../../../../contracts/types";
import {
  apiClient,
  type TranscriptionOptions,
  type TranscriptionResult,
} from "../../api/client";
import { transcodeRecordingToWav } from "../../audio/encodeWav";
import { useStudioStore } from "../../store";
import { midiToName } from "../../utils/music";
import {
  MicrophoneActions,
  type MicrophoneStatus,
} from "./MicrophoneActions";

interface RecordedTake {
  audio: Blob;
  tempo: number;
  timeSignature: TimeSignature;
}

export function MicrophoneInput() {
  const recorderRef = useRef<MediaRecorder | undefined>(undefined);
  const streamRef = useRef<MediaStream | undefined>(undefined);
  const chunksRef = useRef<Blob[]>([]);
  const startedRef = useRef(0);
  const lastTakeRef = useRef<RecordedTake | undefined>(undefined);
  const [status, setStatus] = useState<MicrophoneStatus>("idle");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string>();
  const [notice, setNotice] = useState<string>();
  const [registerMode, setRegisterMode] = useState<"auto" | "sung">("auto");
  const replaceTranscribedMelody = useStudioStore(
    (state) => state.replaceTranscribedMelody,
  );
  const transcriptionRegister = useStudioStore(
    (state) => state.transcriptionRegister,
  );
  const tempo = useStudioStore((state) => state.melody.tempo);
  const storedTimeSignature = useStudioStore(
    (state) => state.melody.timeSignature,
  );
  const timeSignature = storedTimeSignature ?? {
    numerator: 4,
    denominator: 4,
  };

  useEffect(() => {
    if (status !== "recording") return;
    const id = window.setInterval(
      () => setElapsed((performance.now() - startedRef.current) / 1000),
      100,
    );
    return () => window.clearInterval(id);
  }, [status]);

  useEffect(
    () => () => {
      for (const track of streamRef.current?.getTracks() ?? []) track.stop();
    },
    [],
  );

  async function start() {
    setError(undefined);
    setNotice(undefined);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const recorder = new MediaRecorder(stream);
      streamRef.current = stream;
      recorderRef.current = recorder;
      chunksRef.current = [];
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunksRef.current.push(event.data);
      };
      recorder.onstop = () => void transcribe(recorder.mimeType);
      startedRef.current = performance.now();
      setElapsed(0);
      setStatus("recording");
      recorder.start(200);
    } catch (cause) {
      setError(
        cause instanceof Error
          ? cause.message
          : "Microphone permission was denied.",
      );
      setStatus("error");
    }
  }

  function stop() {
    if (recorderRef.current?.state === "recording") {
      setStatus("transcribing");
      recorderRef.current.stop();
    }
  }

  async function transcribe(type: string) {
    for (const track of streamRef.current?.getTracks() ?? []) track.stop();
    try {
      const currentMelody = useStudioStore.getState().melody;
      const recording = await transcodeRecordingToWav(
        new Blob(chunksRef.current, { type }),
      );
      const take = {
        audio: recording,
        tempo: currentMelody.tempo,
        timeSignature: currentMelody.timeSignature ?? {
          numerator: 4,
          denominator: 4,
        },
      };
      lastTakeRef.current = take;
      await submitTake(take, {
        normalizeOctave: registerMode === "auto",
      });
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Transcription failed.",
      );
      setStatus("error");
    }
  }

  async function submitTake(
    take: RecordedTake,
    options: TranscriptionOptions,
  ) {
    const result = await apiClient.transcribe(
      take.audio,
      {
        tempo: take.tempo,
        timeSignature: take.timeSignature,
      },
      options,
    );
    replaceTranscribedMelody(
      result.melody,
      {
        detectedOctaveShift: result.octaveShift,
        detectedMedianPitch: result.detectedMedianPitch,
      },
      "Microphone take",
    );
    setNotice(describeOctaveDecision(result));
    setStatus("idle");
  }

  async function restoreSungRegister() {
    const take = lastTakeRef.current;
    if (!take) return;
    setStatus("transcribing");
    setError(undefined);
    try {
      await submitTake(take, { normalizeOctave: false });
      setRegisterMode("sung");
    } catch (cause) {
      setError(
        cause instanceof Error ? cause.message : "Transcription failed.",
      );
      setStatus("error");
    }
  }

  return (
    <div className="microphone-input">
      <div className={`mic-orb ${status === "recording" ? "recording" : ""}`}>
        <Mic size={20} />
      </div>
      <div className="microphone-copy">
        <strong>
          {status === "recording"
            ? `Recording · ${elapsed.toFixed(1)}s`
            : status === "transcribing"
              ? "Extracting melody…"
              : "Sing or play a monophonic line"}
        </strong>
        <span>
          Sent at {tempo} BPM · {timeSignature.numerator}/
          {timeSignature.denominator} to{" "}
          <code>POST /api/v1/transcribe</code>.
        </span>
        {error && <small>{error}</small>}
        {notice && <small className="octave-decision">{notice}</small>}
      </div>
      <MicrophoneActions
        status={status}
        registerMode={registerMode}
        canRestoreSungRegister={Boolean(
          transcriptionRegister &&
            transcriptionRegister.detectedOctaveShift !== 0 &&
            lastTakeRef.current,
        )}
        onRegisterMode={setRegisterMode}
        onStart={() => void start()}
        onStop={stop}
        onRestoreSungRegister={() => void restoreSungRegister()}
      />
    </div>
  );
}

function describeOctaveDecision(result: TranscriptionResult) {
  const detected =
    result.detectedMedianPitch === undefined
      ? "your sung register"
      : `around ${midiToName(Math.round(result.detectedMedianPitch))}`;
  if (result.octaveShift === 0) {
    return `Kept ${detected}; no octave shift was applied.`;
  }
  const direction = result.octaveShift > 0 ? "up" : "down";
  const amount = Math.abs(result.octaveShift);
  return `Detected ${detected}; transposed ${direction} ${amount} octave${amount === 1 ? "" : "s"} into the melody range.`;
}

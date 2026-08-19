import { LoaderCircle, Mic, Square } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { apiClient } from "../../api/client";
import { useStudioStore } from "../../store";

export function MicrophoneInput() {
  const recorderRef = useRef<MediaRecorder | undefined>(undefined);
  const streamRef = useRef<MediaStream | undefined>(undefined);
  const chunksRef = useRef<Blob[]>([]);
  const startedRef = useRef(0);
  const [status, setStatus] = useState<
    "idle" | "recording" | "transcribing" | "error"
  >("idle");
  const [elapsed, setElapsed] = useState(0);
  const [error, setError] = useState<string>();
  const replaceMelody = useStudioStore((state) => state.replaceMelody);

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
      const melody = await apiClient.transcribe(
        new Blob(chunksRef.current, { type }),
      );
      replaceMelody(melody, "Microphone take");
      setStatus("idle");
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
          Audio is sent to <code>POST /api/v1/transcribe</code>.
        </span>
        {error && <small>{error}</small>}
      </div>
      {status === "recording" ? (
        <button type="button" className="record-stop" onClick={stop}>
          <Square size={12} fill="currentColor" />
          Stop & transcribe
        </button>
      ) : (
        <button
          type="button"
          onClick={() => void start()}
          disabled={status === "transcribing"}
        >
          {status === "transcribing" ? (
            <LoaderCircle size={13} className="spin" />
          ) : (
            <Mic size={13} />
          )}
          {status === "transcribing" ? "Transcribing" : "Start recording"}
        </button>
      )}
    </div>
  );
}

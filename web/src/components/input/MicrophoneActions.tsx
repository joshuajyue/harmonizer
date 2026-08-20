import { LoaderCircle, Mic, Square } from "lucide-react";

export type MicrophoneStatus =
  | "idle"
  | "recording"
  | "transcribing"
  | "error";

interface MicrophoneActionsProps {
  status: MicrophoneStatus;
  registerMode: "auto" | "sung";
  canRestoreSungRegister: boolean;
  onRegisterMode: (mode: "auto" | "sung") => void;
  onStart: () => void;
  onStop: () => void;
  onRestoreSungRegister: () => void;
}

export function MicrophoneActions({
  status,
  registerMode,
  canRestoreSungRegister,
  onRegisterMode,
  onStart,
  onStop,
  onRestoreSungRegister,
}: MicrophoneActionsProps) {
  const busy = status === "recording" || status === "transcribing";

  return (
    <div className="microphone-actions">
      <label className="register-mode-select">
        <span>Register</span>
        <select
          value={registerMode}
          onChange={(event) =>
            onRegisterMode(event.currentTarget.value as "auto" | "sung")
          }
          disabled={busy}
        >
          <option value="auto">Fit melody lane</option>
          <option value="sung">Keep sung register</option>
        </select>
      </label>
      {status === "recording" ? (
        <button type="button" className="record-stop" onClick={onStop}>
          <Square size={12} fill="currentColor" />
          Stop & transcribe
        </button>
      ) : (
        <button
          type="button"
          onClick={onStart}
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
      {canRestoreSungRegister && (
        <button
          type="button"
          className="sung-register-button"
          onClick={onRestoreSungRegister}
          disabled={busy}
        >
          Use sung register
        </button>
      )}
    </div>
  );
}

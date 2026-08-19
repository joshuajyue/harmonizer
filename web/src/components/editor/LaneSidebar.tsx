import { Eye, EyeOff } from "lucide-react";
import type { VoiceName } from "../../../../contracts/types";
import { useStudioStore } from "../../store";
import { voiceLabel, VOICE_COLORS, VOICE_ORDER } from "../../utils/music";
import {
  CHORD_HEIGHT,
  LANE_HEIGHT,
  RULER_HEIGHT,
  SIDEBAR_WIDTH,
} from "./rollGeometry";

export function LaneSidebar() {
  const visibility = useStudioStore((state) => state.voiceVisibility);
  const mute = useStudioStore((state) => state.voiceMute);
  const solo = useStudioStore((state) => state.voiceSolo);
  const toggleVisibility = useStudioStore(
    (state) => state.toggleVoiceVisibility,
  );
  const toggleMute = useStudioStore((state) => state.toggleVoiceMute);
  const toggleSolo = useStudioStore((state) => state.toggleVoiceSolo);

  return (
    <div className="lane-sidebar" style={{ width: SIDEBAR_WIDTH }}>
      <div className="lane-sidebar-ruler" style={{ height: RULER_HEIGHT }}>
        TRACKS
      </div>
      <div className="lane-control melody-control" style={{ height: LANE_HEIGHT }}>
        <span className="lane-color melody-dot" />
        <div>
          <strong>Melody</strong>
          <small>source · editable</small>
        </div>
        <span className="source-badge">IN</span>
      </div>
      {VOICE_ORDER.map((voice) => (
        <VoiceLaneControl
          key={voice}
          voice={voice}
          visible={visibility[voice]}
          muted={mute[voice]}
          soloed={solo[voice]}
          onVisibility={() => toggleVisibility(voice)}
          onMute={() => toggleMute(voice)}
          onSolo={() => toggleSolo(voice)}
        />
      ))}
      <div className="chord-sidebar" style={{ height: CHORD_HEIGHT }}>
        <span>RN</span>
        <strong>Chords</strong>
      </div>
    </div>
  );
}

interface VoiceLaneControlProps {
  voice: VoiceName;
  visible: boolean;
  muted: boolean;
  soloed: boolean;
  onVisibility: () => void;
  onMute: () => void;
  onSolo: () => void;
}

function VoiceLaneControl({
  voice,
  visible,
  muted,
  soloed,
  onVisibility,
  onMute,
  onSolo,
}: VoiceLaneControlProps) {
  return (
    <div
      className={`lane-control ${visible ? "" : "lane-hidden"}`}
      style={{ height: LANE_HEIGHT }}
    >
      <span className="lane-color" style={{ background: VOICE_COLORS[voice] }} />
      <div className="lane-name">
        <strong>{voiceLabel(voice)}</strong>
        <small>generated voice</small>
      </div>
      <div className="lane-actions">
        <button
          type="button"
          className="lane-icon-button"
          onClick={onVisibility}
          aria-label={`${visible ? "Hide" : "Show"} ${voice}`}
          aria-pressed={visible}
        >
          {visible ? <Eye size={13} /> : <EyeOff size={13} />}
        </button>
        <button
          type="button"
          className={`lane-letter-button ${muted ? "active mute" : ""}`}
          onClick={onMute}
          aria-label={`Mute ${voice}`}
          aria-pressed={muted}
        >
          M
        </button>
        <button
          type="button"
          className={`lane-letter-button ${soloed ? "active solo" : ""}`}
          onClick={onSolo}
          aria-label={`Solo ${voice}`}
          aria-pressed={soloed}
        >
          S
        </button>
      </div>
    </div>
  );
}

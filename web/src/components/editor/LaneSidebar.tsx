import { Eye, EyeOff, Maximize2, Minimize2 } from "lucide-react";
import type { VoiceName } from "../../../../contracts/types";
import { useStudioStore, type FocusedLane } from "../../store";
import {
  voiceLabel,
  VOICE_COLORS,
  VOICE_ORDER,
} from "../../utils/music";
import {
  CHORD_HEIGHT,
  LANE_HEIGHT,
  RULER_HEIGHT,
  SIDEBAR_WIDTH,
  type RollLayout,
} from "./rollGeometry";
import { FocusPianoKeyboard } from "./FocusPianoKeyboard";

interface LaneSidebarProps {
  layout: RollLayout;
}

export function LaneSidebar({ layout }: LaneSidebarProps) {
  const visibility = useStudioStore((state) => state.voiceVisibility);
  const mute = useStudioStore((state) => state.voiceMute);
  const solo = useStudioStore((state) => state.voiceSolo);
  const setFocusedLane = useStudioStore((state) => state.setFocusedLane);
  const toggleVisibility = useStudioStore(
    (state) => state.toggleVoiceVisibility,
  );
  const toggleMute = useStudioStore((state) => state.toggleVoiceMute);
  const toggleSolo = useStudioStore((state) => state.toggleVoiceSolo);

  if (layout.focusedLane) {
    const voice =
      layout.focusedLane === "melody" ? undefined : layout.focusedLane;
    return (
      <div
        className="lane-sidebar focus-lane-sidebar"
        style={{ width: SIDEBAR_WIDTH }}
      >
        <div
          className="focus-sidebar-ruler"
          style={{ height: RULER_HEIGHT }}
        >
          <button
            type="button"
            className="focus-exit-button"
            onClick={() => setFocusedLane(undefined)}
            aria-label="Exit lane focus"
            title="Exit focus (Escape)"
          >
            <Minimize2 size={12} />
            All
          </button>
          <span
            className={`lane-color ${voice ? "" : "melody-dot"}`}
            style={voice ? { background: VOICE_COLORS[voice] } : undefined}
          />
          <strong>
            {voice ? voiceLabel(voice) : "Melody"}
          </strong>
          {voice && (
            <div className="focus-audio-actions">
              <button
                type="button"
                className={`lane-letter-button ${mute[voice] ? "active mute" : ""}`}
                onClick={() => toggleMute(voice)}
                aria-label={`Mute ${voice}`}
                aria-pressed={mute[voice]}
              >
                M
              </button>
              <button
                type="button"
                className={`lane-letter-button ${solo[voice] ? "active solo" : ""}`}
                onClick={() => toggleSolo(voice)}
                aria-label={`Solo ${voice}`}
                aria-pressed={solo[voice]}
              >
                S
              </button>
            </div>
          )}
        </div>
        <FocusPianoKeyboard lane={layout.focusedLane} layout={layout} />
        <ChordSidebar />
      </div>
    );
  }

  return (
    <div className="lane-sidebar" style={{ width: SIDEBAR_WIDTH }}>
      <div className="lane-sidebar-ruler" style={{ height: RULER_HEIGHT }}>
        TRACKS
      </div>
      <div
        className="lane-control melody-control"
        style={{ height: LANE_HEIGHT }}
        onDoubleClick={() => setFocusedLane("melody")}
      >
        <span className="lane-color melody-dot" />
        <div>
          <strong>Melody</strong>
          <small>source · editable</small>
        </div>
        <div className="melody-lane-actions">
          <span className="source-badge">IN</span>
          <FocusButton lane="melody" onFocus={setFocusedLane} />
        </div>
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
          onFocus={() => setFocusedLane(voice)}
        />
      ))}
      <ChordSidebar />
    </div>
  );
}

function ChordSidebar() {
  return (
    <div className="chord-sidebar" style={{ height: CHORD_HEIGHT }}>
      <span>RN</span>
      <strong>Chords</strong>
    </div>
  );
}

function FocusButton({
  lane,
  onFocus,
}: {
  lane: FocusedLane;
  onFocus: (lane: FocusedLane) => void;
}) {
  return (
    <button
      type="button"
      className="lane-icon-button"
      onClick={() => onFocus(lane)}
      aria-label={`Focus ${lane} lane`}
      title="Focus lane"
    >
      <Maximize2 size={12} />
    </button>
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
  onFocus: () => void;
}

function VoiceLaneControl({
  voice,
  visible,
  muted,
  soloed,
  onVisibility,
  onMute,
  onSolo,
  onFocus,
}: VoiceLaneControlProps) {
  return (
    <div
      className={`lane-control ${visible ? "" : "lane-hidden"}`}
      style={{ height: LANE_HEIGHT }}
      onDoubleClick={onFocus}
    >
      <span className="lane-color" style={{ background: VOICE_COLORS[voice] }} />
      <div className="lane-name">
        <strong>{voiceLabel(voice)}</strong>
        <small>generated voice</small>
      </div>
      <div className="lane-actions four-actions">
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
        <FocusButton lane={voice} onFocus={() => onFocus()} />
      </div>
    </div>
  );
}

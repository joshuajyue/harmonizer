import type {
  HarmonizeResponse,
  Melody,
  Note,
  Violation,
  VoiceName,
} from "../../../../contracts/types";
import type {
  ComparisonSlotId,
  ComparisonView,
  SelectedNote,
} from "../../store";
import type { Rect, RollLayout } from "./rollGeometry";

export interface NoteHit {
  rect: Rect;
  note: Note;
  source: "melody" | "voice";
  index: number;
  slot?: ComparisonSlotId;
  voice?: VoiceName;
  editable: boolean;
}

export interface ViolationHit {
  rect: Rect;
  violation: Violation;
  slot: ComparisonSlotId;
}

export interface DrawModel {
  melody: Melody;
  resultA?: HarmonizeResponse;
  resultB?: HarmonizeResponse;
  viewMode: ComparisonView;
  activeSlot: ComparisonSlotId;
  selectedNotes: SelectedNote[];
  pxPerBeat: number;
  duration: number;
  loopEnabled: boolean;
  loopStart: number;
  loopEnd: number;
  voiceVisibility: Record<VoiceName, boolean>;
  layout: RollLayout;
  empty: boolean;
}

export interface DrawResult {
  noteHits: NoteHit[];
  violationHits: ViolationHit[];
  model: DrawModel;
}

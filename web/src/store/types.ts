import type {
  EngineInfo,
  HarmonizeResponse,
  KeySignature,
  Melody,
  Note,
  TimeSignature,
  VoiceName,
} from "../../../contracts/types";

export type ComparisonSlotId = "A" | "B";
export type ComparisonView = ComparisonSlotId | "overlay";
export type AsyncStatus = "idle" | "loading" | "ready" | "error";
export type InputTab = "piano" | "midi" | "file" | "microphone";
export type FocusedLane = "melody" | VoiceName;

export interface ComparisonSlot {
  engineId: string;
  result?: HarmonizeResponse;
  requestRevision?: number;
  status: AsyncStatus;
  error?: string;
}

export interface SelectedNote {
  source: "melody" | "voice";
  index: number;
  slot?: ComparisonSlotId;
  voice?: VoiceName;
}

export interface SelectionOrigin {
  selection: SelectedNote;
  note: Note;
}

export interface TranscriptionRegister {
  detectedOctaveShift: number;
  currentOctaveShift: number;
  detectedMedianPitch?: number;
}

export interface ProjectSlice {
  projectName: string;
  melody: Melody;
  melodyRevision: number;
  transcriptionRegister?: TranscriptionRegister;
  setProjectName: (name: string) => void;
  setTempo: (tempo: number) => void;
  setTimeSignature: (signature: TimeSignature) => void;
  setKey: (key?: KeySignature) => void;
  replaceMelody: (melody: Melody, projectName?: string) => void;
  replaceTranscribedMelody: (
    melody: Melody,
    register: Omit<TranscriptionRegister, "currentOctaveShift">,
    projectName?: string,
  ) => void;
  shiftMelodyOctave: (octaves: number) => void;
  restoreSungRegister: () => void;
  addMelodyNote: (note: Note) => number;
  updateMelodyNote: (index: number, patch: Partial<Note>) => void;
  deleteMelodyNote: (index: number) => void;
  clearMelody: () => void;
}

export interface ComparisonSlice {
  engines: EngineInfo[];
  enginesStatus: AsyncStatus;
  enginesError?: string;
  slots: Record<ComparisonSlotId, ComparisonSlot>;
  viewMode: ComparisonView;
  activeSlot: ComparisonSlotId;
  setEnginesLoading: () => void;
  setEngines: (engines: EngineInfo[]) => void;
  setEnginesError: (message: string) => void;
  setSlotEngine: (slot: ComparisonSlotId, engineId: string) => void;
  setSlotLoading: (slot: ComparisonSlotId) => void;
  setSlotResult: (
    slot: ComparisonSlotId,
    result: HarmonizeResponse,
    revision: number,
  ) => void;
  setSlotError: (slot: ComparisonSlotId, message: string) => void;
  setViewMode: (view: ComparisonView) => void;
  setActiveSlot: (slot: ComparisonSlotId) => void;
  updateVoiceNote: (
    slot: ComparisonSlotId,
    voice: VoiceName,
    index: number,
    patch: Partial<Note>,
  ) => void;
  deleteVoiceNote: (
    slot: ComparisonSlotId,
    voice: VoiceName,
    index: number,
  ) => void;
}

export interface TransportSlice {
  isPlaying: boolean;
  currentBeat: number;
  loopEnabled: boolean;
  loopStart: number;
  loopEnd: number;
  loopRangeCustomized: boolean;
  metronomeEnabled: boolean;
  setPlaying: (playing: boolean) => void;
  setCurrentBeat: (beat: number) => void;
  setLoopEnabled: (enabled: boolean) => void;
  setLoopRange: (start: number, end: number, customized?: boolean) => void;
  setMetronomeEnabled: (enabled: boolean) => void;
}

export interface EditorSlice {
  pxPerBeat: number;
  snap: number;
  voiceVisibility: Record<VoiceName, boolean>;
  voiceMute: Record<VoiceName, boolean>;
  voiceSolo: Record<VoiceName, boolean>;
  focusedLane?: FocusedLane;
  selectedNotes: SelectedNote[];
  inputTab: InputTab;
  inputDockOpen: boolean;
  setZoom: (pxPerBeat: number) => void;
  setSnap: (snap: number) => void;
  toggleVoiceVisibility: (voice: VoiceName) => void;
  toggleVoiceMute: (voice: VoiceName) => void;
  toggleVoiceSolo: (voice: VoiceName) => void;
  setFocusedLane: (lane?: FocusedLane) => void;
  setSelectedNotes: (notes: SelectedNote[]) => void;
  clearSelection: () => void;
  deleteSelectedNotes: () => void;
  transformSelectedNotes: (
    origins: SelectionOrigin[],
    deltaBeats: number,
    deltaPitch: number,
  ) => void;
  nudgeSelectedNotes: (deltaBeats: number) => void;
  transposeSelectedNotes: (semitones: number) => void;
  setSelectedNotesDuration: (duration: number) => void;
  setInputTab: (tab: InputTab) => void;
  setInputDockOpen: (open: boolean) => void;
}

export type StudioStore = ProjectSlice &
  ComparisonSlice &
  TransportSlice &
  EditorSlice;

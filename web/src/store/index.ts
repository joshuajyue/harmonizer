import { create } from "zustand";
import { createComparisonSlice } from "./comparisonSlice";
import { createEditorSlice } from "./editorSlice";
import { createProjectSlice } from "./projectSlice";
import { createTransportSlice } from "./transportSlice";
import type { StudioStore } from "./types";

export const useStudioStore = create<StudioStore>()((...args) => ({
  ...createProjectSlice(...args),
  ...createComparisonSlice(...args),
  ...createTransportSlice(...args),
  ...createEditorSlice(...args),
}));

export type {
  ComparisonSlotId,
  ComparisonView,
  FocusedLane,
  InputTab,
  SelectedNote,
  SelectionOrigin,
} from "./types";

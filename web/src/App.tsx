import { StudioHeader } from "./components/app/StudioHeader";
import { ComparisonPanel } from "./components/comparison/ComparisonPanel";
import { PianoRoll } from "./components/editor/PianoRoll";
import { InputDock } from "./components/input/InputDock";
import { TransportBar } from "./components/transport/TransportBar";
import { useKeyboardShortcuts } from "./hooks/useKeyboardShortcuts";
import { useStudioBootstrap } from "./hooks/useStudioBootstrap";

export function App() {
  useStudioBootstrap();
  useKeyboardShortcuts();

  return (
    <main className="studio-shell">
      <StudioHeader />
      <TransportBar />
      <div className="workspace">
        <div className="editor-column">
          <PianoRoll />
          <InputDock />
        </div>
        <ComparisonPanel />
      </div>
    </main>
  );
}

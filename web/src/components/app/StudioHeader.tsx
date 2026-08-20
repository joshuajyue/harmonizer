import { AudioLines, Command, Radio } from "lucide-react";
import { apiClient } from "../../api/client";
import { useStudioStore } from "../../store";
import { RenderControl } from "../transport/RenderControl";

export function StudioHeader() {
  const projectName = useStudioStore((state) => state.projectName);
  const setProjectName = useStudioStore((state) => state.setProjectName);

  return (
    <header className="studio-header">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">
          <AudioLines size={17} />
        </div>
        <div>
          <strong>HarmonAIzer</strong>
          <span>VOICING STUDIO · V2</span>
        </div>
      </div>
      <div className="project-divider" />
      <label className="project-name">
        <span>PROJECT</span>
        <input
          value={projectName}
          onChange={(event) => setProjectName(event.currentTarget.value)}
          aria-label="Project name"
        />
      </label>
      <div className="header-spacer" />
      <div
        className={`connection-badge ${apiClient.isMock ? "mock" : "live"}`}
        title={
          apiClient.isMock
            ? "Using local API fixtures"
            : "Connected through the Vite API proxy"
        }
      >
        <Radio size={11} />
        {apiClient.isMock ? "Fixture session" : "Backend live"}
      </div>
      <div
        className="shortcut-hint"
        title="Space play · R record · / loop · M metronome · 1/2 results · Cmd/Ctrl+Enter harmonize"
      >
        <Command size={12} />
        <span>Space</span>
      </div>
      <RenderControl />
    </header>
  );
}

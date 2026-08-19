import type { Violation } from "../../../../contracts/types";
import type { ComparisonSlotId } from "../../store";

interface ViolationScoreProps {
  slot: ComparisonSlotId;
  engineName: string;
  violations?: Violation[];
  latency?: number;
  stale: boolean;
}

export function ViolationScore({
  slot,
  engineName,
  violations,
  latency,
  stale,
}: ViolationScoreProps) {
  const counts = {
    error: violations?.filter((item) => item.severity === "error").length ?? 0,
    warning:
      violations?.filter((item) => item.severity === "warning").length ?? 0,
    info: violations?.filter((item) => item.severity === "info").length ?? 0,
  };
  const total = counts.error + counts.warning + counts.info;

  return (
    <article className={`violation-score slot-${slot.toLowerCase()}`}>
      <div className="score-heading">
        <span className="slot-letter">{slot}</span>
        <div>
          <strong>{engineName || "Choose engine"}</strong>
          <small>
            {latency !== undefined ? `${latency} ms` : "Not generated"}
            {stale ? " · stale" : ""}
          </small>
        </div>
      </div>
      <div className="score-total">
        <strong>{violations ? total : "—"}</strong>
        <span>violations</span>
      </div>
      <div className="severity-row" aria-label={`${total} total violations`}>
        <span className="severity-count error">{counts.error} E</span>
        <span className="severity-count warning">{counts.warning} W</span>
        <span className="severity-count info">{counts.info} I</span>
      </div>
    </article>
  );
}

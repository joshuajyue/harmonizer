import { Magnet } from "lucide-react";

export function QuantizeSelectionButton({
  onClick,
}: {
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      className="quantize-selection"
      onClick={onClick}
      title="Snap note starts to the grid without changing their lengths"
    >
      <Magnet size={11} />
      Snap starts
    </button>
  );
}

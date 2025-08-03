import { useRef, useState } from "react";

interface HarmonizerPanelProps {
  isOpen: boolean;
  onClose: () => void;
  midiBlob: Blob | null;
}

const HarmonizerPanel: React.FC<HarmonizerPanelProps> = ({ isOpen, onClose, midiBlob }) => {
  const [harmonizedUrl, setHarmonizedUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState<string>("creative");
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Model descriptions
  const modelDescriptions: { [key: string]: string } = {
    'creative': 'Analyzes melody and applies music theory rules for harmonization',
    'bach': 'Neural network trained on Bach chorales (if available, falls back to Creative)'
  };

  const harmonizeMidi = async () => {
    if (!midiBlob) {
      alert("No MIDI data available. Please record some notes first.");
      return;
    }

    setLoading(true);
    setHarmonizedUrl(null);

    try {
      const formData = new FormData();
      formData.append("midi", midiBlob, "melody.mid");
      formData.append("model", selectedModel);

      const response = await fetch("http://localhost:8000/api/harmonize", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const harmonizedBlob = await response.blob();
      const url = URL.createObjectURL(harmonizedBlob);
      setHarmonizedUrl(url);
    } catch (error) {
      console.error("Error harmonizing MIDI:", error);
      alert("Failed to harmonize MIDI. Make sure the backend is running.");
    } finally {
      setLoading(false);
    }
  };

  const downloadHarmonized = () => {
    if (harmonizedUrl) {
      const a = document.createElement("a");
      a.href = harmonizedUrl;
      a.download = `harmonized_${selectedModel}.mid`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    }
  };

  const playHarmonized = () => {
    if (harmonizedUrl && fileInputRef.current) {
      // Create a temporary audio element to play the MIDI
      // Note: This requires a MIDI player or conversion to audio
      const audio = new Audio(harmonizedUrl);
      audio.play().catch(err => {
        console.error("Could not play harmonized MIDI:", err);
        alert("Cannot play MIDI directly. Please download and use a MIDI player.");
      });
    }
  };

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        right: 0,
        width: "400px",
        height: "100vh",
        background: "rgba(255,255,255,0.95)",
        backdropFilter: "blur(10px)",
        borderLeft: "1px solid rgba(0,0,0,0.1)",
        boxShadow: "-4px 0 24px rgba(0,0,0,0.1)",
        zIndex: 1000,
        display: "flex",
        flexDirection: "column",
        padding: "24px",
        transform: isOpen ? "translateX(0)" : "translateX(100%)",
        transition: "transform 0.3s ease-in-out",
      }}
    >
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "24px" }}>
        <h2 style={{
          margin: 0,
          fontSize: "24px",
          fontWeight: 700,
          color: "#333",
          fontFamily: "'SF Pro Display', sans-serif"
        }}>
          AI Harmonizer
        </h2>
        <button
          onClick={onClose}
          style={{
            background: "none",
            border: "none",
            fontSize: "24px",
            cursor: "pointer",
            color: "#666",
            padding: "4px 8px",
            borderRadius: "4px",
            transition: "background 0.2s"
          }}
          onMouseEnter={(e) => e.currentTarget.style.background = "#f0f0f0"}
          onMouseLeave={(e) => e.currentTarget.style.background = "none"}
        >
          ×
        </button>
      </div>

      {/* Model Selection */}
      <div style={{ marginBottom: "24px" }}>
        <label style={{
          display: "block",
          fontSize: "16px",
          fontWeight: 600,
          color: "#444",
          marginBottom: "8px"
        }}>
          Choose Harmonization Model:
        </label>
        <select
          value={selectedModel}
          onChange={(e) => setSelectedModel(e.target.value)}
          style={{
            width: "100%",
            padding: "12px",
            border: "1.5px solid #b0bec5",
            borderRadius: "8px",
            fontSize: "16px",
            background: "#f8fafc",
            color: "#222",
            outline: "none",
            cursor: "pointer"
          }}
        >
          <option value="creative">Creative Engine</option>
          <option value="bach">Bach AI Model</option>
        </select>
        <div style={{
          fontSize: "14px",
          color: "#666",
          marginTop: "8px",
          fontStyle: "italic",
          lineHeight: "1.4"
        }}>
          {modelDescriptions[selectedModel]}
        </div>
      </div>

      {/* Status */}
      <div style={{ marginBottom: "24px" }}>
        <div style={{
          padding: "12px",
          background: midiBlob ? "#e8f5e8" : "#fff3e0",
          border: `1px solid ${midiBlob ? "#4caf50" : "#ff9800"}`,
          borderRadius: "8px",
          fontSize: "14px",
          color: midiBlob ? "#2e7d32" : "#f57c00"
        }}>
          {midiBlob ? "✓ MIDI data ready for harmonization" : "⚠ No MIDI data available. Record some notes first."}
        </div>
      </div>

      {/* Harmonize Button */}
      <button
        onClick={harmonizeMidi}
        disabled={loading || !midiBlob}
        style={{
          background: (!midiBlob || loading) 
            ? "#ccc" 
            : "linear-gradient(90deg, #4fd1c5 0%, #1976d2 100%)",
          color: "#fff",
          border: "none",
          borderRadius: "10px",
          padding: "16px 32px",
          fontSize: "18px",
          fontWeight: 600,
          cursor: (!midiBlob || loading) ? "not-allowed" : "pointer",
          marginBottom: "16px",
          transition: "all 0.2s",
          opacity: (!midiBlob || loading) ? 0.6 : 1
        }}
      >
        {loading ? "Harmonizing..." : "Harmonize MIDI"}
      </button>

      {/* Results */}
      {harmonizedUrl && (
        <div style={{
          padding: "16px",
          background: "#f0f8ff",
          borderRadius: "8px",
          border: "1px solid #1976d2"
        }}>
          <h3 style={{ margin: "0 0 12px 0", color: "#1976d2", fontSize: "16px" }}>
            ✓ Harmonization Complete!
          </h3>
          <div style={{ display: "flex", gap: "8px", flexDirection: "column" }}>
            <button
              onClick={downloadHarmonized}
              style={{
                background: "#1976d2",
                color: "#fff",
                border: "none",
                borderRadius: "6px",
                padding: "8px 16px",
                fontSize: "14px",
                cursor: "pointer",
                transition: "background 0.2s"
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = "#1565c0"}
              onMouseLeave={(e) => e.currentTarget.style.background = "#1976d2"}
            >
              Download Harmonized MIDI
            </button>
            <button
              onClick={playHarmonized}
              style={{
                background: "#4caf50",
                color: "#fff",
                border: "none",
                borderRadius: "6px",
                padding: "8px 16px",
                fontSize: "14px",
                cursor: "pointer",
                transition: "background 0.2s"
              }}
              onMouseEnter={(e) => e.currentTarget.style.background = "#43a047"}
              onMouseLeave={(e) => e.currentTarget.style.background = "#4caf50"}
            >
              Try to Play (Browser Dependent)
            </button>
          </div>
        </div>
      )}

      {/* Info */}
      <div style={{
        marginTop: "auto",
        fontSize: "12px",
        color: "#888",
        textAlign: "center",
        lineHeight: "1.4"
      }}>
        Upload your recorded melody and get AI-generated chord progressions in Bach's style or using music theory rules.
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept=".mid,.midi"
        style={{ display: "none" }}
      />
    </div>
  );
};

export default HarmonizerPanel;

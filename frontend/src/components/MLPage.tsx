import React, { useRef, useState } from "react";

// Shared style for Apple-like look
const pageStyle: React.CSSProperties = {
  minHeight: "100vh",
  width: "100vw",
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  background: "linear-gradient(135deg, #e8ecf3 0%, #cfd8dc 100%)",
  fontFamily: "'SF Pro Display', 'Segoe UI', 'Arial Rounded MT Bold', Arial, sans-serif",
  color: "#222",
  overflow: "auto",
};

const cardStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.85)",
  borderRadius: 18,
  boxShadow: "0 4px 24px 0 rgba(60,60,60,0.08)",
  padding: "40px 48px",
  marginTop: 100,
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  minWidth: 360,
};

const buttonStyle: React.CSSProperties = {
  background: "linear-gradient(90deg, #4fd1c5 0%, #1976d2 100%)",
  color: "#fff",
  border: "none",
  borderRadius: 10,
  padding: "10px 32px",
  fontSize: 18,
  fontWeight: 600,
  boxShadow: "0 2px 8px rgba(80,120,180,0.08)",
  cursor: "pointer",
  marginTop: 24,
  marginBottom: 8,
  transition: "background 0.2s, box-shadow 0.2s",
};

const inputStyle: React.CSSProperties = {
  border: "1.5px solid #b0bec5",
  borderRadius: 8,
  padding: "10px 14px",
  fontSize: 16,
  marginBottom: 18,
  background: "#f8fafc",
  color: "#222",
  outline: "none",
  width: "100%",
  boxSizing: "border-box",
};

const MLPage: React.FC = () => {
  const [inputMidi, setInputMidi] = useState<File | null>(null);
  const [harmonizedUrl, setHarmonizedUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setInputMidi(e.target.files[0]);
      setHarmonizedUrl(null);
    }
  };

  const handleHarmonize = async () => {
    if (!inputMidi) return;
    setLoading(true);
    setHarmonizedUrl(null);

    const formData = new FormData();
    formData.append("midi", inputMidi);

    try {
      const res = await fetch("/api/harmonize", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) throw new Error("Failed to harmonize");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      setHarmonizedUrl(url);
    } catch (err) {
      alert("Harmonization failed.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={pageStyle}>
      <header style={{
        position: "absolute",
        top: 32,
        right: 48,
        zIndex: 10,
      }}>
        <h1 style={{
          fontFamily: "'SF Pro Display', 'Montserrat', Arial, sans-serif",
          fontWeight: 800,
          fontSize: "2.2rem",
          letterSpacing: "0.1em",
          color: "#1976d2",
          textShadow: "0 2px 8px #b0c4de",
          margin: 0,
        }}>
          HarmonAIzer
        </h1>
      </header>
      <div style={cardStyle}>
        <h2 style={{
          color: "#1976d2",
          marginBottom: 24,
          fontWeight: 700,
          fontSize: 28,
          letterSpacing: "0.03em"
        }}>
          ML Harmonizer
        </h2>
        <input
          type="file"
          accept=".mid,audio/midi"
          ref={fileInputRef}
          onChange={handleFileChange}
          style={inputStyle}
        />
        <button
          onClick={handleHarmonize}
          disabled={!inputMidi || loading}
          style={{
            ...buttonStyle,
            opacity: !inputMidi || loading ? 0.6 : 1,
            pointerEvents: !inputMidi || loading ? "none" : "auto"
          }}
        >
          {loading ? "Harmonizing..." : "Harmonize"}
        </button>
        {harmonizedUrl && (
          <a
            href={harmonizedUrl}
            download="harmonized.mid"
            style={{
              ...buttonStyle,
              display: "inline-block",
              background: "linear-gradient(90deg, #1976d2 0%, #4fd1c5 100%)",
              marginTop: 16,
              textAlign: "center",
              textDecoration: "none",
              fontWeight: 700,
              fontSize: 18,
            }}
          >
            Download Harmonized MIDI
          </a>
        )}
      </div>
    </div>
  );
};

export default MLPage;
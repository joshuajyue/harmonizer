import React from "react";
import { BrowserRouter as Router, Routes, Route, Link } from "react-router-dom";
import MidiInput from "./components/MidiInput";
import MLPage from "./components/MLPage";

function App() {
  return (
    <Router>
      <div>
        <h1>Harmonizer</h1>
        <nav>
          <Link to="/">Go to MIDI Input</Link>
          <Link to="/ml">Go to ML Harmonizer</Link>
        </nav>
        <Routes>
          <Route path="/" element={<MidiInput />} />
          <Route path="/ml" element={<MLPage />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

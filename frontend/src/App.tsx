import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MidiInput from "./components/MidiInput";

function App() {
  return (
    <Router>
      <div>
        <Routes>
          <Route path="/" element={<MidiInput />} />
        </Routes>
      </div>
    </Router>
  );
}

export default App;

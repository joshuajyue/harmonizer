import React, { useEffect, useRef } from "react";
import { Renderer, Stave, StaveNote, Formatter } from "vexflow";

type MelodyRendererProps = {
  notes: { key: string; duration: string }[]; // e.g., [{key: "c/4", duration: "q"}]
};

const MelodyRenderer: React.FC<MelodyRendererProps> = ({ notes }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    containerRef.current.innerHTML = ""; // Clear previous rendering

    const renderer = new Renderer(containerRef.current, Renderer.Backends.SVG);
    renderer.resize(500, 120);
    const context = renderer.getContext();
    const stave = new Stave(10, 40, 480);
    stave.addClef("treble").setContext(context).draw();

    const vfNotes = notes.map(
      (n) => new StaveNote({ keys: [n.key], duration: n.duration })
    );

    Formatter.FormatAndDraw(context, stave, vfNotes);
  }, [notes]);

  return <div ref={containerRef} />;
};

export default MelodyRenderer;
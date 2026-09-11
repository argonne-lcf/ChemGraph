import { useEffect, useRef, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";
import type { GLViewer } from "3dmol";
import type { Artifact } from "./api";

export default function MoleculeViewer({ artifact }: { artifact: Artifact }) {
  const element = useRef<HTMLDivElement>(null);
  const viewer = useRef<GLViewer | null>(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);
  const [playing, setPlaying] = useState(false);
  const [frames, setFrames] = useState(1);
  const [frame, setFrame] = useState(0);
  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    let instance: GLViewer | null = null;
    const resize = new ResizeObserver(() => {
      instance?.resize();
      instance?.render();
    });
    setReady(false);
    setError("");
    setPlaying(false);
    setFrame(0);
    Promise.all([
      import("3dmol"),
      fetch(artifact.preview_url!, { signal: controller.signal }).then(
        async (response) => {
          if (!response.ok)
            throw new Error(
              "This structure could not be previewed. Download the original file below.",
            );
          return response.text();
        },
      ),
    ])
      .then(([mol, xyz]) => {
        if (disposed || !element.current) return;
        instance = mol.createViewer(element.current, {
          backgroundColor: "#f4f7fa",
          antialias: true,
        });
        viewer.current = instance;
        instance.addModelsAsFrames(xyz, "xyz");
        instance.setStyle(
          {},
          { stick: { radius: 0.13 }, sphere: { scale: 0.28 } },
        );
        instance.zoomTo();
        instance.rotate(45, "y");
        instance.rotate(15, "x");
        instance.render();
        setFrames(instance.getNumFrames());
        setReady(true);
        resize.observe(element.current);
      })
      .catch((error) => {
        if (!disposed)
          setError(
            error instanceof Error ? error.message : "Viewer unavailable.",
          );
      });
    return () => {
      disposed = true;
      controller.abort();
      resize.disconnect();
      instance?.stopAnimate();
      instance?.clear();
      viewer.current = null;
      if (element.current) element.current.replaceChildren();
    };
  }, [artifact.id, artifact.preview_url]);
  const play = () => {
    if (playing) viewer.current?.stopAnimate();
    else viewer.current?.animate({ loop: "forward", interval: 120 });
    setPlaying(!playing);
  };
  return (
    <div className="viewer-card">
      <div
        ref={element}
        className="molecule-canvas"
        role="img"
        aria-label={`3D molecular structure: ${artifact.name}`}
      />
      {!ready && (
        <div className="viewer-message" role="status">
          {error || "Loading molecular structure…"}
        </div>
      )}
      <div className="viewer-controls">
        <span>
          {frames > 1
            ? `${frames} trajectory frames`
            : "Drag to rotate · Scroll to zoom"}
        </span>
        {frames > 1 && (
          <button
            className="icon-button"
            disabled={!ready}
            onClick={play}
            aria-label={playing ? "Pause trajectory" : "Play trajectory"}
          >
            {playing ? <Pause size={16} /> : <Play size={16} />}
          </button>
        )}
        <button
          className="icon-button"
          disabled={!ready}
          aria-label="Reset molecular view"
          onClick={() => {
            viewer.current?.zoomTo();
            viewer.current?.render();
          }}
        >
          <RotateCcw size={16} />
        </button>
      </div>
      {frames > 1 && (
        <input
          className="frame-slider"
          type="range"
          min={0}
          max={frames - 1}
          value={frame}
          aria-label="Trajectory frame"
          onChange={(event) => {
            const value = Number(event.target.value);
            setFrame(value);
            setPlaying(false);
            viewer.current?.stopAnimate();
            void viewer.current
              ?.setFrame(value)
              .then(() => viewer.current?.render());
          }}
        />
      )}
    </div>
  );
}

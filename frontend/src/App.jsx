import React, { useState, useEffect, useRef, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { GlowingEffect } from "./components/ui/glowing-effect";
import { ShinyButton } from "./components/ui/shiny-button";
import { CoreSpinLoader } from "./components/ui/core-spin-loader";

// Typewriter hook — streams text char by char
function useTypewriter(text, speed = 18, active = true) {
  const [displayed, setDisplayed] = useState("");
  const [done, setDone] = useState(false);
  useEffect(() => {
    if (!active || !text) { setDisplayed(""); setDone(false); return; }
    setDisplayed("");
    setDone(false);
    let i = 0;
    const id = setInterval(() => {
      i++;
      setDisplayed(text.slice(0, i));
      if (i >= text.length) { clearInterval(id); setDone(true); }
    }, speed);
    return () => clearInterval(id);
  }, [text, speed, active]);
  return { displayed, done };
}

const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

function ShutterBlock({ children, delay = 0 }) {
  // We use Framer Motion to create the 3-slice glitch effect on the entire block of children
  return (
    <div className="relative inline-flex flex-col overflow-hidden group">
      {/* Main Character / Block */}
      <motion.div
        initial={{ opacity: 0, filter: "blur(10px)" }}
        animate={{ opacity: 1, filter: "blur(0px)" }}
        transition={{ delay: delay + 0.3, duration: 0.8 }}
      >
        {children}
      </motion.div>

      {/* Top Slice Layer */}
      <motion.div
        initial={{ x: "-100%", opacity: 0 }}
        animate={{ x: "100%", opacity: [0, 1, 0] }}
        transition={{
          duration: 0.7,
          delay: delay,
          ease: "easeInOut",
        }}
        className="absolute inset-0 z-10 pointer-events-none"
        style={{
          clipPath: "polygon(0 0, 100% 0, 100% 35%, 0 35%)",
          filter: "hue-rotate(90deg) saturate(1.5) brightness(1.2)"
        }}
      >
        {children}
      </motion.div>

      {/* Middle Slice Layer */}
      <motion.div
        initial={{ x: "100%", opacity: 0 }}
        animate={{ x: "-100%", opacity: [0, 1, 0] }}
        transition={{
          duration: 0.7,
          delay: delay + 0.1,
          ease: "easeInOut",
        }}
        className="absolute inset-0 z-10 pointer-events-none"
        style={{
          clipPath: "polygon(0 35%, 100% 35%, 100% 65%, 0 65%)",
          filter: "brightness(2) contrast(1.2)"
        }}
      >
        {children}
      </motion.div>

      {/* Bottom Slice Layer */}
      <motion.div
        initial={{ x: "-100%", opacity: 0 }}
        animate={{ x: "100%", opacity: [0, 1, 0] }}
        transition={{
          duration: 0.7,
          delay: delay + 0.2,
          ease: "easeInOut",
        }}
        className="absolute inset-0 z-10 pointer-events-none"
        style={{
          clipPath: "polygon(0 65%, 100% 65%, 100% 100%, 0 100%)",
          filter: "hue-rotate(-90deg) saturate(1.5) brightness(1.2)"
        }}
      >
        {children}
      </motion.div>
    </div>
  );
}

const exposeFeatures = [
  {
    title: "FIRST CONTACT",
    desc: "The container is frozen. Raw frames and audio tracks are separated into independent analysis vectors.",
  },
  {
    title: "VISUAL FORENSICS",
    desc: "A vision transformer scans sampled frames for artifacts, converting anomalies into a raw visual probability tensor.",
  },
  {
    title: "VOICE SPECTRUM",
    desc: "Wav2Vec2 isolates cloned voices and spectral synthesis fingerprints. Broken into 5s scenes.",
  },
  {
    title: "PIPELINE TRACE",
    desc: "Metadata extraction reveals missing camera tags and mismatches, exposing re-encoding toolchains.",
  },
  {
    title: "FUSION ENGINE",
    desc: "A rule-guided matrix weighs modalities. Strong modality-specific signals trigger safeguard overrides logic.",
  },
  {
    title: "LLM CROSS-EXAM",
    desc: "A local reasoning engine audits the numeric matrix, outputting a narrative explanation for analysts.",
  },
];

// FlipHeading takes inspiration from next-reveal to apply the 3D flip-up elastic bounce
function FlipHeading({ text, Component = "h1", className = "" }) {
  const words = text.split(" ");
  return (
    <Component className={`flex flex-wrap gap-[0.25em] ${className}`} aria-label={text} style={{ perspective: "800px" }}>
      {words.map((word, wordIndex) => (
        <span key={wordIndex} className="inline-flex">
          {word.split("").map((char, charIndex) => {
            const globalIndex = words.slice(0, wordIndex).join("").length + charIndex;
            return (
              <span
                key={`${wordIndex}-${charIndex}`}
                className="flip-char font-serif font-bold"
                style={{ "--index": globalIndex }}
              >
                {char}
              </span>
            );
          })}
        </span>
      ))}
    </Component>
  );
}

function formatProb(value) {
  if (value == null) return "N/A";
  return value.toFixed(2);
}

function RevealBlock({ children }) {
  return (
    <motion.div
      variants={{
        hidden: { opacity: 0, rotateX: -90, y: 40, transformOrigin: "bottom center" },
        visible: { opacity: 1, rotateX: 0, y: 0, transition: { type: "spring", stiffness: 100, damping: 15 } }
      }}
      style={{ perspective: 1000 }}
    >
      {children}
    </motion.div>
  );
}

function ProgressBar({ label, value, subtitle }) {
  const pct = Math.round((value ?? 0) * 100);
  // Color shifting based on probability
  const getGlow = (p) => {
    if (p < 30) return "rgba(0, 240, 255, 0.8)"; // Clean / cyan
    if (p < 70) return "rgba(255, 165, 0, 0.8)"; // Sus / orange
    return "rgba(255, 0, 60, 0.8)"; // Fake / magenta
  };

  return (
    <div className="relative p-6 border border-white/10 bg-black/40 backdrop-blur-sm group">
      {/* Label above */}
      <div className="text-[#00F0FF] font-mono tracking-[0.2em] text-sm font-bold mb-4 drop-shadow-[0_0_8px_rgba(0,240,255,0.4)] uppercase">
        {label}
      </div>

      {/* Subtitle / Details */}
      <div className="text-[#888] font-mono tracking-widest text-xs uppercase mb-3">
        {subtitle}
      </div>

      {/* Track */}
      <div className="relative h-[2px] w-full bg-white/10 mt-4 rounded-full">
        {/* Fill */}
        <div
          className="absolute top-0 left-0 h-full rounded-full transition-all duration-1000 ease-in-out"
          style={{
            width: `${pct}%`,
            backgroundColor: getGlow(pct),
            boxShadow: `0 0 10px ${getGlow(pct)}, 0 0 20px ${getGlow(pct)}`
          }}
        >
          {/* Glowing dot at the end ─● */}
          <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1.5 h-1.5 bg-white rounded-full shadow-[0_0_8px_4px_rgba(255,255,255,0.5)]"></div>
        </div>
      </div>

      {/* Percentage (Bottom right) */}
      <div className="text-right mt-3 text-[#00F0FF] font-mono text-xs tracking-widest drop-shadow-[0_0_5px_rgba(0,240,255,0.3)]">
        {pct}% PROBABILITY
      </div>
    </div>
  );
}

// ─── LLM Audit Terminal Panel ────────────────────────────────────────────────
function LlmAuditPanel({ audit }) {
  const { displayed, done } = useTypewriter(audit.explanation, 16, true);
  const isConsistent = audit.consistency?.toUpperCase() === "CONSISTENT";
  const confidenceColor = {
    HIGH: "#00F0FF",
    MEDIUM: "#FFA500",
    LOW: "#FF003C",
  }[audit.confidence_level?.toUpperCase()] ?? "#828282";

  return (
    <div className="border border-white/10 bg-black/60 backdrop-blur-md overflow-hidden">
      {/* Header bar */}
      <div className="flex items-center gap-3 px-5 py-3 border-b border-white/10 bg-white/[0.02]">
        <span className="w-2 h-2 rounded-full bg-[#00F0FF] animate-pulse" />
        <span className="font-mono text-[0.7rem] uppercase tracking-[0.25em] text-[#00F0FF]">LLM Intelligence Audit</span>
        <span className="ml-auto font-mono text-[0.6rem] text-[#545454] tracking-widest">EXPOSE-AI // FORENSIC MATRIX</span>
      </div>

      {/* Field rows */}
      <div className="grid grid-cols-2 border-b border-white/10">
        <div className="px-5 py-4 border-r border-white/10">
          <div className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-[#545454] mb-1">MODALITY CONSISTENCY</div>
          <div className={`font-mono text-[0.95rem] font-bold ${isConsistent ? 'text-[#00F0FF]' : 'text-[#FFA500]'}`}>
            {audit.consistency?.toUpperCase()}
          </div>
        </div>
        <div className="px-5 py-4">
          <div className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-[#545454] mb-1">AUDIT CONFIDENCE</div>
          <div className="font-mono text-[0.95rem] font-bold" style={{ color: confidenceColor }}>
            {audit.confidence_level?.toUpperCase()}
          </div>
        </div>
      </div>

      {/* Typewriter explanation */}
      <div className="px-5 py-5">
        <div className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-[#545454] mb-3">NARRATIVE SUMMARY</div>
        <p className="font-mono text-[0.82rem] leading-[1.75] text-[#a0a0a0]">
          {displayed}
          {!done && (
            <motion.span
              animate={{ opacity: [1, 0] }}
              transition={{ repeat: Infinity, duration: 0.6 }}
              className="inline-block w-[2px] h-[0.9em] bg-[#00F0FF] ml-0.5 align-middle"
            />
          )}
        </p>
      </div>

      {/* Warnings */}
      {Array.isArray(audit.warnings) && audit.warnings.length > 0 && (
        <div className="border-t border-white/10 px-5 py-4 space-y-2">
          {audit.warnings.map((w, i) => (
            <div key={i} className="flex items-start gap-2 font-mono text-[0.72rem] text-[#FFA500]">
              <span className="shrink-0 mt-0.5 opacity-60">//</span>
              <span>{w.toUpperCase()}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [spectrogramUrl, setSpectrogramUrl] = useState(null);
  const [spectrogramLoading, setSpectrogramLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("dashboard");
  const [currentFeatureIndex, setCurrentFeatureIndex] = useState(0);

  // Auto-advance the feature cards every 3 seconds
  useEffect(() => {
    if (activeTab === "dashboard") {
      const timer = setInterval(() => {
        setCurrentFeatureIndex((prev) => (prev + 1) % exposeFeatures.length);
      }, 3000);
      return () => clearInterval(timer);
    }
  }, [activeTab]);

  // GSAP References
  const appRef = useRef(null);

  useEffect(() => {
    // Cinematic entrance animation when app loads or tab changes
    if (window.gsap) {
      window.gsap.fromTo(
        ".gsap-fade-up",
        { y: 30, opacity: 0 },
        { y: 0, opacity: 1, duration: 0.8, stagger: 0.1, ease: "power4.out" }
      );
    }
  }, [activeTab]);

  const handleFileChange = (e) => {
    const selected = e.target.files?.[0];
    setResult(null);
    setError("");
    setSpectrogramUrl(null);
    if (!selected) {
      setFile(null);
      setPreviewUrl(null);
      return;
    }
    setFile(selected);
    setPreviewUrl(URL.createObjectURL(selected));
    
    // Automatically trigger detection
    handleAnalyze(selected);
  };

  const handleAnalyze = async (selectedFile = null) => {
    const targetFile = selectedFile || file;
    if (!targetFile) {
      setError("Please select a file first.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", targetFile);

      const res = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        throw new Error(data.detail || "Analysis failed. Please try again.");
      }

      const data = await res.json();
      setResult(data);

      // Fetch spectrogram in parallel (don't block main result)
      setSpectrogramLoading(true);
      setSpectrogramUrl(null);
      try {
        const specForm = new FormData();
        specForm.append("file", targetFile);
        const specRes = await fetch(`${API_BASE_URL}/api/spectrogram`, {
          method: "POST",
          body: specForm,
        });
        if (specRes.ok) {
          const blob = await specRes.blob();
          setSpectrogramUrl(URL.createObjectURL(blob));
        }
      } catch (_) {
        // silently ignore — spectrogram is optional
      } finally {
        setSpectrogramLoading(false);
      }

      // Animate results on load
      setTimeout(() => {
        if (window.gsap) {
          window.gsap.fromTo(".results-grid > *",
            { opacity: 0, x: 20 },
            { opacity: 1, x: 0, duration: 0.6, stagger: 0.1, ease: "power3.out" }
          );
        }
      }, 100);

    } catch (err) {
      setError(err.message || "Unexpected error. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const metadataLabel = (score) => {
    if (score < 0.3) return "Clean";
    if (score < 0.6) return "Re-encoded (Common)";
    if (score < 0.8) return "Suspicious";
    return "Strongly Synthetic";
  };

  const isDeepfake = result?.label === "DEEPFAKE";

  return (
    <div className="min-h-screen flex flex-col relative z-10 bg-[#050505] text-white font-['Inter',sans-serif]" ref={appRef}>

      {/* Background Architectures */}
      <div className="noise-overlay"></div>
      <div className="drafting-grid"></div>

      {/* Glassmorphism Header */}
      <header className="px-6 md:px-12 py-6 backdrop-blur-xl bg-[#050505]/60 border-b border-white/10 sticky top-0 z-50 flex flex-col md:flex-row justify-between items-center gap-4 md:gap-0">
        <div className="flex flex-col gap-1">
          <ShutterBlock>
            <div className="flex items-center gap-4">
              <h1 className="text-2xl md:text-3xl uppercase bg-gradient-to-br from-white to-[#A0A0A0] bg-clip-text text-transparent tracking-wider font-bold font-['Syne',sans-serif] m-0">EXPOSE AI</h1>
              <span className="font-mono text-[0.65rem] px-2 py-1 border border-[#00F0FF] text-[#00F0FF] rounded-sm uppercase tracking-[0.1em] bg-[#00F0FF]/5">v2.0 Beta</span>
            </div>
            <p className="m-0 font-mono text-[#828282] text-xs md:text-sm uppercase tracking-[0.15em] mt-1">
              Multimodal Deepfake Detector // Forensic Matrix
            </p>
          </ShutterBlock>
        </div>

        <div className="flex gap-2 md:gap-4 border border-white/10 p-1 rounded bg-black/50">
          <button
            type="button"
            className={`relative overflow-hidden rounded-sm px-4 md:px-6 py-2 md:py-2.5 font-mono text-xs md:text-sm uppercase tracking-widest transition-all duration-300 before:content-['//_'] before:transition-colors hover:bg-white/5 hover:text-white ${activeTab === "dashboard" ? "bg-white/10 text-white before:text-[#00F0FF]" : "bg-transparent text-[#828282] before:text-[#545454]"}`}
            onClick={() => setActiveTab("dashboard")}
          >
            Dashboard
          </button>
          <button
            type="button"
            className={`relative overflow-hidden rounded-sm px-4 md:px-6 py-2 md:py-2.5 font-mono text-xs md:text-sm uppercase tracking-widest transition-all duration-300 before:content-['//_'] before:transition-colors hover:bg-white/5 hover:text-white ${activeTab === "detector" ? "bg-white/10 text-white before:text-[#00F0FF]" : "bg-transparent text-[#828282] before:text-[#545454]"}`}
            onClick={() => setActiveTab("detector")}
          >
            Detector Core
          </button>
        </div>
      </header>

      <main className="flex-1 px-6 md:px-12 py-8 md:py-12 max-w-[1600px] mx-auto w-full">
        {activeTab === "dashboard" && (
          <section className="dashboard">
            <FlipHeading
              text="System Overview"
              Component="h2"
              className="text-3xl md:text-4xl lg:text-[5rem] font-light font-['Syne',sans-serif] mb-8 flex items-center gap-4 before:content-[''] before:block before:w-10 before:h-1 before:bg-[#00F0FF] tracking-tight gsap-fade-up"
            />
            <p className="text-base md:text-[1.15rem] leading-[1.6] text-white/70 font-light max-w-[800px] mb-12 gsap-fade-up">
              Expose AI is a multimodal deepfake detection system inspired by
              modern forensic platforms. It is built to answer a single
              question with courtroom-ready clarity:
              <strong> "What really happened in this media?"</strong>
            </p>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-16 gsap-fade-up">
              <div className="metrics-item group flex-1 bg-[#0f0f11]/65 backdrop-blur-md border border-white/10 p-6 relative transition-all duration-300 hover:-translate-y-1 hover:border-white/20">
                <GlowingEffect
                  spread={40}
                  glow={true}
                  disabled={false}
                  proximity={64}
                  inactiveZone={0.01}
                  borderWidth={3}
                />
                <div className="relative z-10">
                  <span className="block font-mono text-[0.7rem] uppercase tracking-widest text-[#828282] mb-2">Modalities</span>
                  <span className="block font-['Syne',sans-serif] text-2xl md:text-[1.8rem] text-white font-bold">3-Way Fusion</span>
                </div>
              </div>
              <div className="metrics-item group flex-1 bg-[#0f0f11]/65 backdrop-blur-md border border-white/10 p-6 relative transition-all duration-300 hover:-translate-y-1 hover:border-white/20">
                <GlowingEffect
                  spread={40}
                  glow={true}
                  disabled={false}
                  proximity={64}
                  inactiveZone={0.01}
                  borderWidth={3}
                />
                <div className="relative z-10">
                  <span className="block font-mono text-[0.7rem] uppercase tracking-widest text-[#828282] mb-2">Analysis Speed</span>
                  <span className="block font-['Syne',sans-serif] text-2xl md:text-[1.8rem] text-white font-bold">800ms / Clip</span>
                </div>
              </div>
              <div className="metrics-item group flex-1 bg-[#0f0f11]/65 backdrop-blur-md border border-white/10 p-6 relative transition-all duration-300 hover:-translate-y-1 hover:border-white/20">
                <GlowingEffect
                  spread={40}
                  glow={true}
                  disabled={false}
                  proximity={64}
                  inactiveZone={0.01}
                  borderWidth={3}
                />
                <div className="relative z-10">
                  <span className="block font-mono text-[0.7rem] uppercase tracking-widest text-[#828282] mb-2">Deployment</span>
                  <span className="block font-['Syne',sans-serif] text-2xl md:text-[1.8rem] text-white font-bold">Node API</span>
                </div>
              </div>
              <div className="metrics-item group flex-1 bg-[#0f0f11]/65 backdrop-blur-md border border-white/10 p-6 relative transition-all duration-300 hover:-translate-y-1 hover:border-white/20">
                <GlowingEffect
                  spread={40}
                  glow={true}
                  disabled={false}
                  proximity={64}
                  inactiveZone={0.01}
                  borderWidth={3}
                />
                <div className="relative z-10">
                  <span className="block font-mono text-[0.7rem] uppercase tracking-widest text-[#828282] mb-2">Explainability</span>
                  <span className="block font-['Syne',sans-serif] text-2xl md:text-[1.8rem] text-white font-bold">LLM Audit</span>
                </div>
              </div>
            </div>

            <FlipHeading
              text="How Expose AI Thinks"
              Component="h1"
              className="text-4xl font-light font-['Syne',sans-serif] mb-8 flex items-center gap-4 before:content-[''] before:block before:w-10 before:h-1 before:bg-[#00F0FF] tracking-tight gsap-fade-up"
            />
            <div className="relative h-[400px] md:h-[450px] lg:h-[500px] w-full max-w-2xl mx-auto gsap-fade-up">
              <AnimatePresence mode="wait">
                <motion.div
                  key={currentFeatureIndex}
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
                  className="absolute h-full w-full inset-0 section-card overflow-hidden group "
                >
                  {/* Oversized fading animated index background */}
                  <div className="absolute -left-2 md:-left-4 top-1/2 -translate-y-1/2 text-[8rem] md:text-[10rem] font-bold text-[#10878C]/5 select-none pointer-events-none leading-none tracking-tighter transition-all duration-500 z-0">
                    <motion.span
                      initial={{ opacity: 0, scale: 0.8, filter: "blur(5px)" }}
                      animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
                      className="block"
                    >
                      {String(currentFeatureIndex + 1).padStart(2, "0")}
                    </motion.span>
                  </div>

                  {/* Main Card Content */}
                  <div className="relative z-10 flex flex-col items-center justify-center h-full px-4 md:px-8 text-center pt-8 md:pt-0">
                    <div className="section-header mb-6 w-full flex justify-center">
                      <span className="section-title text-[#10878C] flex items-center gap-3 text-lg font-bold tracking-widest">
                        <span className="w-2 h-2 rounded-full bg-[#10878C] animate-pulse" />
                        {String(currentFeatureIndex + 1).padStart(2, "0")} // {exposeFeatures[currentFeatureIndex].title}
                      </span>
                    </div>

                    {/* Word-by-word reveal for description */}
                    <div className="dashboard-text text-base md:text-xl leading-relaxed text-white/90 max-w-lg px-2">
                      {exposeFeatures[currentFeatureIndex].desc.split(" ").map((word, i) => (
                        <span key={i} className="inline-block">
                          <motion.span
                            className="inline-block"
                            initial={{ opacity: 0, y: 15, filter: "blur(4px)" }}
                            animate={{
                              opacity: 1,
                              y: 0,
                              filter: "blur(0px)",
                              transition: {
                                duration: 0.5,
                                delay: i * 0.04, // smooth scatter delay
                                ease: [0.22, 1, 0.36, 1]
                              }
                            }}
                          >
                            {word}
                          </motion.span>
                          <span className="inline-block w-[0.25em]" />
                        </span>
                      ))}
                    </div>

                    {/* Navigation Line Indicator */}
                    <div className="absolute bottom-6 left-0 right-0 flex justify-center gap-1 md:gap-2 px-4 md:px-12">
                      {exposeFeatures.map((_, dotIdx) => (
                        <div
                          key={dotIdx}
                          className={`h-1 flex-1 max-w-[4rem] rounded-full transition-all duration-500 ${dotIdx === currentFeatureIndex
                            ? "bg-[#10878C]"
                            : "bg-white/10"
                            }`}
                        />
                      ))}
                    </div>
                  </div>
                </motion.div>
              </AnimatePresence>
            </div>

            <div className="flex justify-center w-full mt-20 md:mt-40 mb-12 md:mb-24 px-4">
              <ShinyButton
                onClick={() => setActiveTab("detector")}
                className="w-full max-w-[400px] py-4 md:py-5 px-4 md:px-8 rounded-2xl font-mono tracking-widest text-xs md:text-sm uppercase flex items-center justify-center text-center leading-tight md:leading-normal"
              >
                INITIALIZE DETECTOR CORE
              </ShinyButton>
            </div>
          </section>
        )}

        {activeTab === "detector" && (
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.2fr] gap-8 items-start">
            <section className="flex flex-col gsap-fade-up">
              <h2 className="font-['Syne',sans-serif] text-2xl md:text-[1.8rem] font-bold mb-2">Data Ingestion</h2>
              <p className="text-sm text-[#828282] mb-8">
                Upload target media container for multimodal manipulation analysis.
              </p>

              <label className="block border border-dashed border-white/10 bg-black/30 p-6 md:p-12 text-center cursor-pointer transition-all duration-300 relative hover:border-[#00F0FF] hover:bg-[#00F0FF]/5 group">
                <input
                  type="file"
                  accept="video/*,audio/*,image/*"
                  className="hidden"
                  onChange={handleFileChange}
                />
                <i className="ph-light ph-file text-5xl block mb-4 text-[#828282] transition-colors duration-300 group-hover:text-white"></i>
                <div className="block font-mono text-sm text-white mb-2">
                  {file ? file.name : "DRAG & DROP CONTAINER OR CLICK TO BROWSE"}
                </div>
                <div className="block text-xs text-[#545454]">
                  [ MEDIA ] // Max size: 50MB
                </div>
              </label>

              <div className="mt-4">
                <ShinyButton
                  onClick={handleAnalyze}
                  disabled={loading || !file}
                  className="w-full py-4 md:py-5 px-4 md:px-8 rounded-2xl font-mono tracking-widest text-xs md:text-sm uppercase flex items-center justify-center text-center"
                >
                  {loading ? "EXECUTING MATRIX..." : "INITIATE SCAN"}
                </ShinyButton>
              </div>

              {error && <div className="p-4 border border-[#FF003C] text-[#FF003C] bg-[#FF003C]/5 font-mono text-sm mt-4">{error}</div>}

              {previewUrl && (
                <div className="bg-[#0f0f11]/65 border border-white/10 p-8 relative backdrop-blur-md transition-all duration-400 overflow-hidden mt-8 hover:-translate-y-1 hover:border-white/20 section-card-fx">
                  <div className="mb-4 flex justify-between items-center">
                    <span className="font-mono text-[0.85rem] uppercase tracking-widest text-[#00F0FF]">TARGET PREVIEW</span>
                  </div>
                  {file?.type?.startsWith('image/') ? (
                    <img
                      className="w-full aspect-video bg-black border border-white/10 object-contain"
                      src={previewUrl}
                      alt="Target Preview"
                    />
                  ) : file?.type?.startsWith('audio/') ? (
                    <div className="w-full h-[200px] bg-black border border-white/10 flex items-center justify-center p-8">
                       <audio className="w-full" src={previewUrl} controls />
                    </div>
                  ) : (
                    <video
                      className="w-full aspect-video bg-black border border-white/10 object-contain"
                      src={previewUrl}
                      controls
                    />
                  )}


                  {/* Spectrogram below the video */}
                  {(spectrogramLoading || spectrogramUrl) && (
                    <div className="mt-6 border-t border-white/10 pt-5">
                      <div className="font-mono text-[0.6rem] uppercase tracking-[0.2em] text-[#545454] mb-3">MEL SPECTROGRAM — AUDIO FORENSIC TRACE</div>
                      {spectrogramLoading ? (
                        <div className="w-full h-[140px] bg-white/5 animate-pulse rounded flex items-center justify-center">
                          <span className="font-mono text-[0.65rem] text-[#545454] tracking-widest">GENERATING SPECTROGRAM...</span>
                        </div>
                      ) : (
                        <motion.div
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ duration: 0.6 }}
                        >
                          <img
                            src={spectrogramUrl}
                            alt="Audio Mel Spectrogram"
                            className="w-full rounded border border-white/10"
                          />
                        </motion.div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </section>

            <section className="flex flex-col gsap-fade-up">
              <h2 className="font-['Syne',sans-serif] text-2xl md:text-[1.8rem] font-bold mb-2">Forensic Matrix</h2>
              <p className="text-sm text-[#828282] mb-8">
                Live readout of modality tensors and final LLM Audit.
              </p>

              {loading ? (
                <div className="bg-[#0f0f11]/65 border border-white/10 p-8 md:p-16 relative backdrop-blur-md transition-all duration-400 overflow-hidden text-center section-card-fx min-h-[300px] md:min-h-[400px] flex items-center justify-center">
                  <CoreSpinLoader />
                </div>
              ) : !result ? (
                <div className="bg-[#0f0f11]/65 border border-white/10 p-8 md:p-16 relative backdrop-blur-md transition-all duration-400 overflow-hidden opacity-50 text-center section-card-fx">
                  <i className="ph-light ph-fingerprint text-4xl md:text-5xl block mb-4 text-[#828282] opacity-30"></i>
                  <p className="text-sm md:text-base lg:text-[1.05rem] leading-[1.6] text-white/70 font-light">AWAITING TARGET DATA</p>
                </div>
              ) : null}

              {!loading && result && (
                <motion.div
                  className="flex flex-col gap-6 mb-8"
                  initial="hidden"
                  animate="visible"
                  variants={{
                    visible: { transition: { staggerChildren: 0.15 } }
                  }}
                >
                  <RevealBlock>
                    {result.video_prob != null ? (
                      <ProgressBar
                        label="VISUAL TENSOR"
                        value={result.video_prob}
                        subtitle={`Visual anomaly detection probability: ${formatProb(result.video_prob)} `}
                      />
                    ) : (
                      <div className="relative p-6 border border-white/10 bg-black/40 backdrop-blur-sm group">
                        <div className="text-[#00F0FF] font-mono tracking-[0.2em] text-sm font-bold mb-4 drop-shadow-[0_0_8px_rgba(0,240,255,0.4)] uppercase">
                          VISUAL TENSOR
                        </div>
                        <div className="font-mono text-xs uppercase text-[#828282] mb-1">
                          NO USABLE VISUAL TRACK DETECTED
                        </div>
                      </div>
                    )}
                  </RevealBlock>

                  <RevealBlock>
                    <div className="bg-[#0f0f11]/65 border border-white/10 p-8 relative backdrop-blur-md transition-all duration-400 overflow-hidden hover:-translate-y-1 hover:border-white/20 section-card-fx">
                      <div className="mb-4 flex justify-between items-center">
                        <span className="font-mono text-[0.85rem] uppercase tracking-widest text-[#00F0FF]">AUDIO / SPECTRAL TENSOR</span>
                      </div>
                      <div className="font-mono text-xs uppercase text-[#828282] mb-1">
                        {result.audio_prob == null
                          ? "NO USABLE AUDIO TRACK DETECTED"
                          : `Vocal synthesis spoofing probability: ${formatProb(result.audio_prob)} `}
                      </div>
                      {result.audio_prob != null && (
                        <div className="w-full h-[2px] bg-white/10 mt-4 relative">
                          <div
                            className="h-full bg-white shadow-[0_0_10px_rgba(255,255,255,0.5)] transition-all duration-1000 relative custom-progress-after"
                            style={{
                              width: `${Math.round((result.audio_prob ?? 0) * 100)}% `,
                              background: result.audio_prob > 0.6 ? '#FF003C' : '#00F0FF',
                              boxShadow: `0 0 15px ${result.audio_prob > 0.6 ? '#FF003C' : '#00F0FF'} `
                            }}
                          ></div>
                          <div className="mt-2 font-mono text-[0.75rem] text-right text-[#00F0FF]">
                            {Math.round((result.audio_prob ?? 0) * 100)}% PROBABILITY
                          </div>
                        </div>
                      )}


                    </div>
                  </RevealBlock>

                  <RevealBlock>
                    <ProgressBar
                      label="PIPELINE FORENSICS"
                      value={result.meta_prob}
                      subtitle={`[${metadataLabel(result.meta_prob).toUpperCase()}]Score: ${formatProb(result.meta_prob)} `}
                    />
                  </RevealBlock>

                  <RevealBlock>
                    <div className="p-6 md:p-8 border border-white/10 bg-gradient-to-b from-white/5 to-transparent">
                      <h3 className="text-[1.1rem] md:text-[1.2rem] mb-6 font-mono uppercase text-white">SYSTEM CONCLUSION</h3>
                      <div className="flex flex-col md:flex-row gap-3 md:gap-4 mb-6">
                        <span className={`px-4 py-1.5 font-mono text-[0.7rem] md:text-[0.75rem] uppercase tracking-[0.1em] border text-center md:text-left ${isDeepfake ? "text-[#FF003C] border-[#FF003C] bg-[#FF003C]/10" : "text-[#00F0FF] border-[#00F0FF] bg-[#00F0FF]/10"} `}>
                          STATE: {isDeepfake ? "SYNTHETIC / DEEPFAKE" : "AUTHENTIC"}
                        </span>
                        <span className="px-4 py-1.5 font-mono text-[0.7rem] md:text-[0.75rem] uppercase tracking-[0.1em] border border-white/10 text-[#828282] text-center md:text-left">
                          SUBTYPE: {result.detected_type}
                        </span>
                      </div>

                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
                        <div>
                          <span className="font-mono text-[0.7rem] uppercase text-[#545454] block mb-1">ABSOLUTE SCORE</span>
                          <span className="text-[1.8rem] font-['Syne',sans-serif] block text-white font-bold">
                            {(result.final_score ?? 0).toFixed(2)}
                          </span>
                        </div>
                        <div>
                          <span className="font-mono text-[0.7rem] uppercase text-[#545454] block mb-1">SAFEGUARD OVERRIDE</span>
                          <span className="text-[1.2rem] font-['Syne',sans-serif] block text-white">
                            {result.override_triggered ? "ACTIVE" : "INACTIVE"}
                          </span>
                        </div>
                      </div>
                      <p className="text-[0.95rem] leading-[1.6] text-[#828282]">{result.reason}</p>
                    </div>
                  </RevealBlock>

                  {result.llm_audit && (
                    <RevealBlock>
                      <LlmAuditPanel audit={result.llm_audit} />
                    </RevealBlock>
                  )}
                </motion.div>
              )}
            </section>
          </div>
        )}
      </main>

      <footer className="px-6 md:px-12 py-6 border-t border-white/10 flex flex-col md:flex-row justify-between items-center gap-4 font-mono text-[0.75rem] text-[#545454] uppercase tracking-[0.1em]">
        <span>EXPOSE AI · MULTIMODAL DEEPFAKE FORENSICS</span>
        <span>ENGINE: FASTAPI / VITE // MATRICES ACTIVE</span>
      </footer>
    </div>
  );
}

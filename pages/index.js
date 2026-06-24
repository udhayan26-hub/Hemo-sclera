import Head from "next/head";
import { useState, useRef, useCallback, useEffect } from "react";

// ── Constants ──────────────────────────────────────────────────────────────
const RISK_CONFIG = {
  LOW: {
    icon: "✅",
    label: "Low Risk",
    description: "Scleral chromaticity is within normal physiological limits.",
    recommendations: [
      {
        icon: "💙",
        title: "Continue Routine Care",
        text: "No immediate action required. Maintain regular health check-ups.",
      },
      {
        icon: "👁️",
        title: "Monitor If Symptoms Appear",
        text: "If you experience fatigue, dizziness, or pale skin — consult a doctor regardless of this AI result.",
      },
    ],
  },
  MEDIUM: {
    icon: "⚠️",
    label: "Moderate Risk",
    description: "Some indicators detected. Medical monitoring is advised.",
    recommendations: [
      {
        icon: "🩺",
        title: "Schedule a Check-up",
        text: "Consult a healthcare professional for a confirmatory blood test within the week.",
      },
      {
        icon: "📋",
        title: "Track Symptoms",
        text: "Monitor for worsening symptoms: fatigue, yellowing, dark urine, abdominal pain.",
      },
    ],
  },
  HIGH: {
    icon: "🚨",
    label: "High Risk",
    description: "Critical indicators detected. Immediate medical review required.",
    recommendations: [
      {
        icon: "🏥",
        title: "Seek Immediate Medical Attention",
        text: "Schedule an urgent appointment with a healthcare professional for confirmatory lab tests.",
      },
      {
        icon: "🧪",
        title: "Confirmatory Tests Required",
        text: "Request a Hemoglobin/CBC blood test (Anemia) or Bilirubin blood test (Jaundice).",
      },
    ],
  },
  UNKNOWN: {
    icon: "❓",
    label: "Unknown",
    description: "Unable to complete analysis. Please retry with a clearer image.",
    recommendations: [],
  },
};

const LOADING_STEPS = [
  "Encoding image for secure transmission...",
  "Connecting to OpenRouter AI endpoint...",
  "Running multimodal vision analysis...",
  "Extracting scleral chromaticity data...",
  "Generating clinical assessment...",
];

// ── Helpers ────────────────────────────────────────────────────────────────
function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      // Strip the data URL prefix (data:image/jpeg;base64,)
      const base64 = reader.result.split(",")[1];
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

// ── Components ─────────────────────────────────────────────────────────────
function LoadingCard({ step }) {
  return (
    <div className="loading-card card">
      <div className="spinner-ring" />
      <p style={{ fontSize: "0.9rem", fontWeight: 600, color: "var(--text-primary)" }}>
        Consulting Neural Engine...
      </p>
      <div className="loading-steps">
        {LOADING_STEPS.map((s, i) => (
          <div
            key={i}
            className={`loading-step ${i < step ? "done" : i === step ? "active" : ""}`}
          >
            <div className="dot" />
            <span>{i < step ? "✓ " : ""}{s}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ResultsPanel({ result, task }) {
  const riskLevel = result.risk_level || "UNKNOWN";
  const config = RISK_CONFIG[riskLevel] || RISK_CONFIG.UNKNOWN;
  const confidence = Number(result.confidence_score) || 0;

  return (
    <>
      {/* Risk Banner */}
      <div className={`risk-banner ${riskLevel}`}>
        <span className="risk-icon">{config.icon}</span>
        <div className="risk-text">
          <h3>{config.label} — {task === "jaundice" ? "Jaundice" : "Anemia"} Screening</h3>
          <p>{config.description}</p>
        </div>
        <span className={`risk-badge ${riskLevel}`}>{riskLevel}</span>
      </div>

      {/* Assessment + Confidence */}
      <div className="result-card">
        <div className="result-card-title">👁️ Visual Assessment</div>
        <p className="assessment-text">"{result.sclera_color_assessment}"</p>
      </div>

      <div className="result-card">
        <div className="result-card-title">📊 AI Confidence</div>
        <div className="confidence-value">{confidence}%</div>
        <div className="confidence-label">Neural certainty score</div>
        <div className="meter-bar">
          <div className="meter-fill" style={{ width: `${confidence}%` }} />
        </div>
      </div>

      {/* Clinical Reasoning */}
      <div className="result-card reasoning-text">
        <div className="result-card-title">🧠 Clinical Reasoning</div>
        <p>{result.clinical_reasoning}</p>
        {result._model_used && (
          <div className="model-tag">Model: {result._model_used}</div>
        )}
      </div>

      {/* Recommendations */}
      {config.recommendations.length > 0 && (
        <div className="reco-card" style={{ gridColumn: "1 / -1" }}>
          <div className="reco-title">🩺 Next Steps &amp; Recommendations</div>
          <div className="reco-items">
            {config.recommendations.map((r, i) => (
              <div className="reco-item" key={i}>
                <span className="reco-icon">{r.icon}</span>
                <p>
                  <strong>{r.title}:</strong> {r.text}
                </p>
              </div>
            ))}
          </div>
        </div>
      )}
    </>
  );
}

// ── Main Page ──────────────────────────────────────────────────────────────
export default function Home() {
  const [task, setTask] = useState("jaundice");
  const [onsetHistory, setOnsetHistory] = useState("N/A or Not Yellow");
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [loadingStep, setLoadingStep] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragover, setDragover] = useState(false);
  const fileInputRef = useRef(null);

  // Cycle loading steps
  useEffect(() => {
    if (!loading) return;
    setLoadingStep(0);
    const interval = setInterval(() => {
      setLoadingStep((s) => Math.min(s + 1, LOADING_STEPS.length - 1));
    }, 1200);
    return () => clearInterval(interval);
  }, [loading]);

  const handleFile = useCallback((f) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Please upload a JPG, JPEG, or PNG image.");
      return;
    }
    setFile(f);
    setPreviewUrl(URL.createObjectURL(f));
    setResult(null);
    setError(null);
  }, []);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragover(false);
      const f = e.dataTransfer.files[0];
      handleFile(f);
    },
    [handleFile]
  );

  const handleAnalyze = async () => {
    if (!file) return;
    setLoading(true);
    setResult(null);
    setError(null);

    try {
      const imageBase64 = await fileToBase64(file);
      const response = await fetch("/api/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ imageBase64, task, onsetHistory }),
      });

      const data = await response.json();

      if (!response.ok || data.error) {
        throw new Error(data.error_detail || `Server error ${response.status}`);
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <Head>
        <title>Hemo-Sclera AI Triage — Non-Invasive Diagnostic Dashboard</title>
        <meta
          name="description"
          content="Enterprise-grade AI triage for non-invasive anemia and jaundice screening using multimodal scleral analysis."
        />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>👁️</text></svg>" />
      </Head>

      <div className="app-wrapper">
        {/* Header */}
        <header className="header">
          <div className="header-inner">
            <span className="header-logo">👁️</span>
            <div className="header-title">
              <h1>Hemo-Sclera Multimodal Diagnostic Dashboard</h1>
              <p>Enterprise-Grade Triage for Non-Invasive Anemia &amp; Jaundice Screening</p>
            </div>
            <div className="header-badge">AI Triage v2.0</div>
          </div>
        </header>

        {/* Main Layout */}
        <main className="main-layout">
          {/* ── Sidebar ── */}
          <aside className="sidebar">
            {/* Disclaimer */}
            <div className="disclaimer">
              ⚠️ <strong>Clinical Disclaimer:</strong> This is a triage assistant for
              screening purposes only. It does <em>not</em> replace a clinical CBC or
              Bilirubin lab test.
            </div>

            {/* Disease Target */}
            <div className="card">
              <div className="card-title">🎯 Disease Target</div>
              <div className="task-selector">
                <button
                  id="task-jaundice"
                  className={`task-btn ${task === "jaundice" ? "active" : ""}`}
                  onClick={() => { setTask("jaundice"); setResult(null); }}
                >
                  <span className="icon">🟡</span>
                  Jaundice
                  <span style={{ fontSize: "0.65rem", color: "inherit" }}>(Sclera)</span>
                </button>
                <button
                  id="task-anemia"
                  className={`task-btn ${task === "anemia" ? "active" : ""}`}
                  onClick={() => { setTask("anemia"); setResult(null); }}
                >
                  <span className="icon">🔴</span>
                  Anemia
                  <span style={{ fontSize: "0.65rem", color: "inherit" }}>(Eyelid)</span>
                </button>
              </div>
            </div>

            {/* Patient History */}
            <div className="card">
              <div className="card-title">📋 Patient History</div>
              <div className="form-group">
                <label className="form-label" htmlFor="onset-history">
                  Symptom Onset Duration
                </label>
                <select
                  id="onset-history"
                  className="form-select"
                  value={onsetHistory}
                  onChange={(e) => setOnsetHistory(e.target.value)}
                >
                  <option>N/A or Not Yellow</option>
                  <option>Recent (Days/Weeks - Acute)</option>
                  <option>Long-Term (Years/Since Birth - Chronic)</option>
                </select>
              </div>
            </div>

            {/* Upload */}
            <div className="card">
              <div className="card-title">📁 Upload Eye Scan</div>
              <div
                className={`upload-zone ${dragover ? "dragover" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDragover(true); }}
                onDragLeave={() => setDragover(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpg,image/jpeg,image/png"
                  style={{ display: "none" }}
                  onChange={(e) => handleFile(e.target.files[0])}
                  id="file-upload"
                />
                {previewUrl ? (
                  <>
                    <img src={previewUrl} alt="Preview" className="upload-preview" />
                    <div className="file-name">{file?.name}</div>
                  </>
                ) : (
                  <>
                    <div className="upload-icon">📸</div>
                    <div className="upload-text">
                      <strong>Click or drag &amp; drop</strong>
                      <br />to upload patient eye scan
                    </div>
                    <div className="upload-hint">JPG, JPEG, PNG · Max 10 MB</div>
                  </>
                )}
              </div>
            </div>

            {/* Analyze Button */}
            <button
              id="analyze-btn"
              className={`analyze-btn ${loading ? "loading" : ""}`}
              onClick={handleAnalyze}
              disabled={!file || loading}
            >
              {loading ? "⏳ Analyzing..." : "🔬 Run AI Diagnostic"}
            </button>

            {/* Performance Metrics */}
            <div className="card">
              <div className="card-title">📊 Model Performance</div>
              <table className="metrics-table">
                <thead>
                  <tr>
                    <th>Module</th>
                    <th>Algorithm</th>
                    <th>Target</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Segmentation</td>
                    <td>YOLOv11-Nano</td>
                    <td>mAP@50 &gt;0.85</td>
                  </tr>
                  <tr>
                    <td>Anemia</td>
                    <td>Random Forest</td>
                    <td>MAE &lt;1.2 g/dL</td>
                  </tr>
                  <tr>
                    <td>Jaundice</td>
                    <td>Logistic Reg.</td>
                    <td>Accuracy &gt;90%</td>
                  </tr>
                  <tr>
                    <td>Vision AI</td>
                    <td>Gemini 2.5 Flash</td>
                    <td>Multimodal</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </aside>

          {/* ── Content Area ── */}
          <div className="content-area">
            {/* Error */}
            {error && (
              <div
                className="card"
                style={{
                  background: "var(--risk-high-bg)",
                  borderColor: "rgba(239,68,68,0.3)",
                  padding: "16px 20px",
                  fontSize: "0.85rem",
                  color: "#fca5a5",
                }}
              >
                ⚠️ <strong>Inference Error:</strong> {error}
              </div>
            )}

            {/* Empty State */}
            {!file && !loading && !result && (
              <div className="empty-state">
                <span className="eye-icon">👁️</span>
                <h2>Ready for Diagnostic Analysis</h2>
                <p>
                  Upload a patient eye scan and select a disease target to begin
                  AI-assisted triage using multimodal scleral analysis.
                </p>
                <div className="empty-steps">
                  {[
                    ["1", "Select Target", "Jaundice or Anemia"],
                    ["2", "Upload Scan", "Clear eye photo"],
                    ["3", "Run Analysis", "AI diagnosis"],
                  ].map(([num, title, sub]) => (
                    <div className="empty-step" key={num}>
                      <div className="step-num">{num}</div>
                      <strong style={{ fontSize: "0.78rem" }}>{title}</strong>
                      <span>{sub}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Image Preview */}
            {file && !loading && (
              <div className="card image-card">
                <img src={previewUrl} alt="Uploaded Eye Scan" />
                <div className="image-label">📷 Uploaded Eye Scan — {task === "jaundice" ? "Jaundice" : "Anemia"} Mode</div>
              </div>
            )}

            {/* Loading */}
            {loading && <LoadingCard step={loadingStep} />}

            {/* Results */}
            {result && !loading && (
              <>
                {/* Show image alongside results */}
                {previewUrl && (
                  <div className="card image-card">
                    <img src={previewUrl} alt="Uploaded Eye Scan" />
                    <div className="image-label">📷 Analyzed Eye Scan</div>
                  </div>
                )}
                <div className="results-grid">
                  <ResultsPanel result={result} task={task} />
                </div>
              </>
            )}
          </div>
        </main>

        <footer className="footer">
          Hemo-Sclera AI Triage · Built with Next.js &amp; OpenRouter · For research &amp; screening purposes only
        </footer>
      </div>
    </>
  );
}

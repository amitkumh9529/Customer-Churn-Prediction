import { useState, useEffect } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  TrendingUp,
  TrendingDown,
  Info,
  RefreshCw,
} from "lucide-react";
import PredictionForm from "../components/PredictionForm";
import {
  predictChurn,
  checkHealth,
  type CustomerFeatures,
  type PredictionResponse,
  type HealthResponse,
} from "../services/api";

function RiskGauge({ probability }: { probability: number }) {
  const pct = Math.round(probability * 100);
  const angle = -135 + probability * 270; // –135° → +135°

  const color =
    pct >= 70
      ? "#ef4444"
      : pct >= 40
      ? "#f59e0b"
      : "#22c55e";

  return (
    <div className="flex flex-col items-center gap-2">
      <svg viewBox="0 0 120 80" className="w-40 h-auto">
        {/* Track */}
        <path
          d="M 15 70 A 50 50 0 0 1 105 70"
          fill="none"
          stroke="#1e293b"
          strokeWidth="10"
          strokeLinecap="round"
        />
        {/* Fill */}
        <path
          d="M 15 70 A 50 50 0 0 1 105 70"
          fill="none"
          stroke={color}
          strokeWidth="10"
          strokeLinecap="round"
          strokeDasharray={`${probability * 157} 157`}
          style={{ transition: "stroke-dasharray 0.8s cubic-bezier(.4,0,.2,1)" }}
        />
        {/* Needle */}
        <g transform={`rotate(${angle}, 60, 70)`}>
          <line x1="60" y1="70" x2="60" y2="28" stroke={color} strokeWidth="2.5" strokeLinecap="round" />
          <circle cx="60" cy="70" r="4" fill={color} />
        </g>
        {/* Label */}
        <text x="60" y="62" textAnchor="middle" fill={color} fontSize="14" fontWeight="700" fontFamily="JetBrains Mono, monospace">
          {pct}%
        </text>
      </svg>
      <p className="text-xs text-slate-500 tracking-widest uppercase">Churn Risk</p>
    </div>
  );
}

function ResultCard({ result }: { result: PredictionResponse }) {
  const isChurn = result.prediction === 1;
  const pct = Math.round(result.churn_probability * 100);

  const riskLevel =
    pct >= 70 ? "High Risk" : pct >= 40 ? "Medium Risk" : "Low Risk";
  const riskColor =
    pct >= 70
      ? "text-red-400 border-red-500/30 bg-red-500/5"
      : pct >= 40
      ? "text-amber-400 border-amber-500/30 bg-amber-500/5"
      : "text-emerald-400 border-emerald-500/30 bg-emerald-500/5";

  return (
    <div className="card p-6 animate-slide-up">
      {/* Header */}
      <div className="flex items-start justify-between mb-6">
        <div>
          <p className="text-xs uppercase tracking-widest text-slate-500 mb-1">
            Prediction Result
          </p>
          <div className="flex items-center gap-2">
            {isChurn ? (
              <AlertTriangle size={20} className="text-red-400" />
            ) : (
              <CheckCircle2 size={20} className="text-emerald-400" />
            )}
            <h2 className="text-2xl font-display text-white">
              {result.prediction_label}
            </h2>
          </div>
        </div>
        <span
          className={`text-xs font-semibold px-3 py-1.5 rounded-full border ${riskColor}`}
        >
          {riskLevel}
        </span>
      </div>

      {/* Gauge */}
      <div className="flex justify-center mb-6">
        <RiskGauge probability={result.churn_probability} />
      </div>

      {/* Probability Bars */}
      <div className="space-y-3 mb-6">
        <div>
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span className="flex items-center gap-1">
              <TrendingUp size={11} className="text-red-400" /> Churn Probability
            </span>
            <span className="font-mono font-semibold text-red-400">
              {(result.churn_probability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-amber-500 to-red-500 rounded-full transition-all duration-700"
              style={{ width: `${result.churn_probability * 100}%` }}
            />
          </div>
        </div>

        <div>
          <div className="flex justify-between text-xs text-slate-400 mb-1">
            <span className="flex items-center gap-1">
              <TrendingDown size={11} className="text-emerald-400" /> Retention Probability
            </span>
            <span className="font-mono font-semibold text-emerald-400">
              {(result.no_churn_probability * 100).toFixed(1)}%
            </span>
          </div>
          <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-teal-500 to-emerald-500 rounded-full transition-all duration-700"
              style={{ width: `${result.no_churn_probability * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Recommendation */}
      <div
        className={`rounded-xl border p-4 ${
          isChurn
            ? "bg-red-500/5 border-red-500/20"
            : "bg-emerald-500/5 border-emerald-500/20"
        }`}
      >
        <p className="text-xs font-semibold uppercase tracking-widest mb-1.5 text-slate-400">
          Recommendation
        </p>
        <p className="text-sm text-slate-300">
          {isChurn
            ? pct >= 70
              ? "🚨 Critical — Immediate intervention required. Contact this customer with a targeted retention offer, loyalty discount, or dedicated support call."
              : "⚠️ At Risk — Schedule a proactive outreach. Consider offering a contract upgrade or service bundle discount."
            : "✅ Healthy — Customer shows strong retention signals. Continue standard engagement and monitor for changes."}
        </p>
      </div>

      {/* Meta */}
      <div className="mt-4 flex items-center gap-1.5 text-xs text-slate-600">
        <Info size={11} />
        <span>Model v{result.model_version} · Threshold 0.50</span>
      </div>
    </div>
  );
}

function HealthBadge({ health }: { health: HealthResponse | null }) {
  if (!health) return null;
  return (
    <div
      className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded-full border ${
        health.model_loaded
          ? "text-emerald-400 bg-emerald-400/10 border-emerald-400/20"
          : "text-red-400 bg-red-400/10 border-red-400/20"
      }`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${
          health.model_loaded ? "bg-emerald-400 animate-pulse" : "bg-red-400"
        }`}
      />
      {health.model_loaded ? "Model Ready" : "Model Not Loaded"}
    </div>
  );
}

export default function Predict() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);

  useEffect(() => {
    checkHealth()
      .then(setHealth)
      .catch(() => setHealth(null));
  }, []);

  const handleSubmit = async (data: CustomerFeatures) => {
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await predictChurn(data);
      setResult(res);
    } catch (err: unknown) {
      const msg =
        err instanceof Error
          ? err.message
          : "Failed to reach the prediction API. Is the server running?";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 py-8">
      {/* Page header */}
      <div className="mb-8">
        <div className="flex items-center justify-between flex-wrap gap-3">
          <div>
            <h1 className="text-3xl font-display text-white mb-1">
              Customer Churn Predictor
            </h1>
            <p className="text-slate-400 text-sm">
              Enter customer attributes to predict churn probability in real time.
            </p>
          </div>
          <HealthBadge health={health} />
        </div>

        {/* Stats strip */}
        {/* <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: "Dataset", value: "Telco Churn" },
            { label: "Model", value: "XGBoost" },
            { label: "ROC-AUC", value: "≥ 0.84" },
            { label: "Pipeline", value: "Scikit-learn" },
          ].map(({ label, value }) => (
            <div key={label} className="card px-4 py-3">
              <p className="text-xs text-slate-500 uppercase tracking-wider mb-0.5">
                {label}
              </p>
              <p className="text-sm font-semibold font-mono text-brand-300">{value}</p>
            </div>
          ))}
        </div> */}
      </div>

      {/* Main layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Form — takes 2 cols */}
        <div className="lg:col-span-2">
          <PredictionForm onSubmit={handleSubmit} loading={loading} />
        </div>

        {/* Result panel */}
        <div className="lg:col-span-1 space-y-4">
          {error && (
            <div className="card p-5 border-red-500/30 bg-red-500/5 animate-fade-in">
              <div className="flex items-start gap-3">
                <AlertTriangle size={16} className="text-red-400 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-semibold text-red-400 mb-1">
                    Prediction Failed
                  </p>
                  <p className="text-xs text-slate-400">{error}</p>
                  <button
                    onClick={() => setError(null)}
                    className="mt-3 flex items-center gap-1 text-xs text-slate-500 hover:text-slate-300 transition-colors"
                  >
                    <RefreshCw size={11} /> Dismiss
                  </button>
                </div>
              </div>
            </div>
          )}

          {result ? (
            <ResultCard result={result} />
          ) : (
            !error && (
              <div className="card p-6 flex flex-col items-center justify-center text-center min-h-[320px] border-dashed">
                <div className="w-12 h-12 rounded-2xl bg-slate-800 flex items-center justify-center mb-4">
                  <TrendingUp size={22} className="text-slate-600" />
                </div>
                <p className="text-slate-500 text-sm font-medium mb-1">
                  No prediction yet
                </p>
                <p className="text-slate-600 text-xs max-w-[180px]">
                  Fill in the customer details and click{" "}
                  <span className="text-brand-400">Predict Churn Risk</span> to
                  get a result.
                </p>
              </div>
            )
          )}

          {/* How it works */}
          {/* <div className="card p-5">
            <p className="section-title">How It Works</p>
            <ol className="space-y-2.5">
              {[
                "Enter customer demographic & service data",
                "FastAPI sends the input to the ML pipeline",
                "XGBoost scores the preprocessed features",
                "Churn probability and label are returned",
              ].map((step, i) => (
                <li key={i} className="flex items-start gap-2.5 text-xs text-slate-400">
                  <span className="w-4 h-4 rounded-full bg-brand-600/20 text-brand-400 text-[10px] font-bold flex items-center justify-center flex-shrink-0 mt-0.5">
                    {i + 1}
                  </span>
                  {step}
                </li>
              ))}
            </ol>
          </div> */}
        </div>
      </div>
    </div>
  );
}

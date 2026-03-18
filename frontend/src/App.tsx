import { Routes, Route, NavLink } from "react-router-dom";
import { BarChart3, Github } from "lucide-react";
import Predict from "./pages/Predict";

function NavItem({
  to,
  children,
}: {
  to: string;
  children: React.ReactNode;
}) {
  return (
    <NavLink
      to={to}
      className={({ isActive }) =>
        `text-sm font-medium px-3 py-1.5 rounded-lg transition-all duration-150 ${
          isActive
            ? "bg-brand-600/20 text-brand-400"
            : "text-slate-400 hover:text-slate-200 hover:bg-slate-800"
        }`
      }
    >
      {children}
    </NavLink>
  );
}

export default function App() {
  return (
    <div className="min-h-screen bg-slate-950 flex flex-col">
      {/* ── Header ── */}
      <header className="border-b border-slate-800/60 backdrop-blur-sm sticky top-0 z-50 bg-slate-950/80">
        <div className="max-w-7xl mx-auto px-6 h-14 flex items-center justify-between">
          {/* Logo */}
          <NavLink to="/" className="flex items-center gap-2.5 group">
            <div className="w-7 h-7 rounded-lg bg-brand-600 flex items-center justify-center shadow-lg shadow-brand-600/30 group-hover:shadow-brand-500/40 transition-shadow">
              <BarChart3 size={14} className="text-white" />
            </div>
            <span className="font-display text-lg text-white tracking-tight">
              Churn<span className="text-brand-400">Scope</span>
            </span>
          </NavLink>

          {/* Nav */}
          <nav className="flex items-center gap-1">
            <NavItem to="/">Predict</NavItem>
            {/* <a
              href="http://localhost:5000"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-all duration-150"
            >
              MLflow
            </a>
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-medium px-3 py-1.5 rounded-lg text-slate-400 hover:text-slate-200 hover:bg-slate-800 transition-all duration-150"
            >
              API Docs
            </a> */}
          </nav>

          {/* Badge */}
          <div className="flex items-center gap-2">
            <span className="hidden sm:flex items-center gap-1.5 text-xs text-emerald-400 bg-emerald-400/10 border border-emerald-400/20 px-2.5 py-1 rounded-full">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
              Live
            </span>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-slate-500 hover:text-slate-300 transition-colors"
            >
              <Github size={18} />
            </a>
          </div>
        </div>
      </header>

      {/* ── Main ── */}
      <main className="flex-1">
        <Routes>
          <Route path="/" element={<Predict />} />
          <Route path="/predict" element={<Predict />} />
        </Routes>
      </main>

      {/* ── Footer ── */}
      <footer className="border-t border-slate-800/60 py-4">
        <div className="max-w-7xl mx-auto px-6 flex items-center justify-between">
          <p className="text-xs text-slate-600">
            ChurnScope — Production ML System v1.0.0
          </p>
          {/* <p className="text-xs text-slate-600 flex items-center gap-1">
            <Zap size={10} className="text-brand-600" />
            Powered by XGBoost · FastAPI · MLflow
          </p> */}
        </div>
      </footer>
    </div>
  );
}

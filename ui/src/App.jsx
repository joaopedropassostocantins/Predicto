import { useState, useCallback } from "react";
import ResultsPanel from "./ResultsPanel";

const TABS = ["Blend", "Elo", "Poisson", "XGBoost", "Manual Weights", "Clipping"];

const DEFAULT = {
  blend: { xgb: 0.55, poisson: 0.20, seed: 0.10, elo: 0.10, manual: 0.05 },
  elo: {
    k_factor: 25.0, initial_rating: 1500.0, carry_factor: 0.75,
    use_margin: true, margin_cap: 30.0,
  },
  poisson: {
    shrinkage: 0.20, alpha_ci: 0.10, max_points: 220,
    w_recent3: 0.35, w_recent5: 0.30, w_season: 0.35,
  },
  xgb: {
    n_estimators: 700, learning_rate: 0.02, max_depth: 4,
    subsample: 0.80, colsample_bytree: 0.80, min_child_weight: 5,
    reg_lambda: 2.0, reg_alpha: 0.2,
  },
  manual_weights: {
    seed_diff: 0.90, elo_diff: 1.10, season_margin_diff: 1.20,
    season_win_pct_diff: 0.80, recent3_margin_diff: 1.25,
    recent5_margin_diff: 1.00, matchup_diff: 1.10,
    poisson_win_prob_centered: 2.20, quality_diff: 1.00,
    sos_diff: 0.70, rank_diff_signed: 0.60, consistency_edge: 0.35,
  },
  clip: { pred_clip_min: 0.025, pred_clip_max: 0.975, manual_temperature: 10.0 },
};

const API_URL = import.meta.env.VITE_API_URL ?? "http://localhost:8787";

function deepCopy(o) { return JSON.parse(JSON.stringify(o)); }

function Slider({ label, value, min, max, step, onChange, fmt }) {
  const display = fmt ? fmt(value) : value;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 13, color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>{label}</span>
        <span style={{ fontSize: 13, fontWeight: 500, minWidth: 48, textAlign: "right" }}>{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%" }} />
    </div>
  );
}

function Toggle({ label, value, onChange }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 14 }}>
      <span style={{ fontSize: 13, color: "var(--color-text-secondary)", fontFamily: "var(--font-mono)" }}>{label}</span>
      <button onClick={() => onChange(!value)}
        style={{
          padding: "2px 14px", fontSize: 12, fontWeight: 500,
          borderRadius: "var(--border-radius-md)",
          border: "0.5px solid var(--color-border-secondary)",
          background: value ? "var(--color-background-info)" : "var(--color-background-secondary)",
          color: value ? "var(--color-text-info)" : "var(--color-text-secondary)",
          cursor: "pointer",
        }}>
        {value ? "true" : "false"}
      </button>
    </div>
  );
}

function BlendBar({ weights }) {
  const total = Object.values(weights).reduce((a, b) => a + b, 0);
  const norm = Object.entries(weights).map(([k, v]) => [k, v / total]);
  const colors = { xgb: "#5B8DEF", poisson: "#43B89C", seed: "#E6A23C", elo: "#9B73D4", manual: "#E08080" };
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display: "flex", height: 12, borderRadius: 6, overflow: "hidden", marginBottom: 6 }}>
        {norm.map(([k, v]) => (
          <div key={k} style={{ width: `${(v * 100).toFixed(1)}%`, background: colors[k] }} />
        ))}
      </div>
      <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
        {norm.map(([k, v]) => (
          <div key={k} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12 }}>
            <div style={{ width: 8, height: 8, borderRadius: 2, background: colors[k] }} />
            <span style={{ color: "var(--color-text-secondary)" }}>{k}</span>
            <span style={{ fontWeight: 500 }}>{(v * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 24 }}>
      <p style={{ fontSize: 11, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-tertiary)", marginBottom: 12 }}>{title}</p>
      {children}
    </div>
  );
}

export default function App() {
  const [tab, setTab] = useState(0);
  const [cfg, setCfg] = useState(deepCopy(DEFAULT));
  const [copied, setCopied] = useState(false);
  const [runState, setRunState] = useState({ status: "idle", results: null, error: null });

  const set = (group, key, val) => setCfg(c => {
    const n = deepCopy(c);
    n[group][key] = val;
    return n;
  });

  const reset = () => { setCfg(deepCopy(DEFAULT)); setRunState({ status: "idle", results: null, error: null }); };

  // ── Run backtest via FastAPI server ─────────────────────────────────────────
  const runBacktest = useCallback(async () => {
    setRunState({ status: "loading", results: null, error: null });
    try {
      const resp = await fetch(`${API_URL}/run-backtest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(cfg),
      });
      const data = await resp.json();
      if (data.ok) {
        setRunState({ status: "done", results: data.results, error: null });
      } else {
        setRunState({ status: "error", results: null, error: data.error ?? "Unknown error" });
      }
    } catch (e) {
      setRunState({ status: "error", results: null, error: e.message });
    }
  }, [cfg]);

  // ── Generate config.py text ─────────────────────────────────────────────────
  const generateConfig = useCallback(() => {
    const c = cfg;
    const bw = c.blend;
    const mw = c.manual_weights;
    return `# src/config.py — gerado pelo painel de calibração
import os
_ON_KAGGLE = os.path.exists("/kaggle/input")
CONFIG = {
    "data_dir": (
        "/kaggle/input/march-machine-learning-mania-2026"
        if _ON_KAGGLE
        else "/home/ubuntu/predicto_local/data"
    ),
    "target_season": 2026,
    "backtest_seasons": [2015, 2016, 2017, 2018, 2019, 2021, 2022, 2023, 2024, 2025],
    "recent_games_window": 3,
    "poisson_windows": [3, 5, "season"],
    "poisson_blend_weights": {
        "recent3": ${c.poisson.w_recent3.toFixed(2)},
        "recent5": ${c.poisson.w_recent5.toFixed(2)},
        "season":  ${c.poisson.w_season.toFixed(2)},
    },
    "alpha_ci":          ${c.poisson.alpha_ci.toFixed(2)},
    "max_points_poisson": ${c.poisson.max_points},
    "poisson_shrinkage": ${c.poisson.shrinkage.toFixed(2)},
    "pred_clip_min": ${c.clip.pred_clip_min.toFixed(3)},
    "pred_clip_max": ${c.clip.pred_clip_max.toFixed(3)},
    "elo_k_factor":       ${c.elo.k_factor.toFixed(1)},
    "elo_initial_rating": ${c.elo.initial_rating.toFixed(1)},
    "elo_carry_factor":   ${c.elo.carry_factor.toFixed(2)},
    "elo_use_margin":     ${c.elo.use_margin},
    "elo_margin_cap":     ${c.elo.margin_cap.toFixed(1)},
    "manual_temperature": ${c.clip.manual_temperature.toFixed(1)},
    "temperature_candidates": [0.70, 0.80, 0.90, 1.00, 1.25, 1.50, 2.00],
    "calibration_methods": ["identity", "temperature", "platt", "isotonic"],
    "fallback_points_for":     70.0,
    "fallback_points_against": 70.0,
    "fallback_seed":  8.5,
    "fallback_elo":   1500.0,
    "tabular_model": "xgb_or_hgb",
    "xgb_params": {
        "n_estimators":     ${c.xgb.n_estimators},
        "learning_rate":    ${c.xgb.learning_rate.toFixed(3)},
        "max_depth":        ${c.xgb.max_depth},
        "subsample":        ${c.xgb.subsample.toFixed(2)},
        "colsample_bytree": ${c.xgb.colsample_bytree.toFixed(2)},
        "min_child_weight": ${c.xgb.min_child_weight},
        "reg_lambda":       ${c.xgb.reg_lambda.toFixed(1)},
        "reg_alpha":        ${c.xgb.reg_alpha.toFixed(2)},
        "objective":    "binary:logistic",
        "eval_metric":  "logloss",
        "tree_method":  "hist",
        "random_state": 42,
        "verbosity":    0,
    },
    "hgb_params": {
        "learning_rate":     0.03,
        "max_depth":         4,
        "max_iter":          600,
        "min_samples_leaf":  15,
        "l2_regularization": 0.5,
        "random_state":      42,
    },
    "blend_weights": {
        "xgb":     ${bw.xgb.toFixed(2)},
        "poisson": ${bw.poisson.toFixed(2)},
        "seed":    ${bw.seed.toFixed(2)},
        "elo":     ${bw.elo.toFixed(2)},
        "manual":  ${bw.manual.toFixed(2)},
    },
    "manual_feature_weights": {
        "seed_diff":               ${mw.seed_diff.toFixed(2)},
        "elo_diff":                ${mw.elo_diff.toFixed(2)},
        "season_margin_diff":      ${mw.season_margin_diff.toFixed(2)},
        "season_win_pct_diff":     ${mw.season_win_pct_diff.toFixed(2)},
        "recent3_margin_diff":     ${mw.recent3_margin_diff.toFixed(2)},
        "recent5_margin_diff":     ${mw.recent5_margin_diff.toFixed(2)},
        "matchup_diff":            ${mw.matchup_diff.toFixed(2)},
        "poisson_win_prob_centered": ${mw.poisson_win_prob_centered.toFixed(2)},
        "quality_diff":            ${mw.quality_diff.toFixed(2)},
        "sos_diff":                ${mw.sos_diff.toFixed(2)},
        "rank_diff_signed":        ${mw.rank_diff_signed.toFixed(2)},
        "consistency_edge":        ${mw.consistency_edge.toFixed(2)},
    },
    "massey_system_priority": ["POM", "SAG", "MOR", "DOL", "WOL", "RTH", "COL"],
    "feature_cols": [
        "seed_diff", "elo_diff",
        "season_points_for_diff", "season_points_against_diff",
        "season_margin_diff", "season_win_pct_diff",
        "recent3_points_for_diff", "recent3_points_against_diff", "recent3_margin_diff",
        "recent5_points_for_diff", "recent5_points_against_diff", "recent5_margin_diff",
        "matchup_diff", "offense_vs_defense_low", "offense_vs_defense_high",
        "sos_diff", "quality_diff", "rank_diff_signed",
        "poisson_lambda_low", "poisson_lambda_high",
        "poisson_expected_margin", "poisson_win_prob", "poisson_win_prob_centered",
        "consistency_edge",
        "season_trajectory_diff", "quality_win_pct_diff",
        "blowout_pct_diff", "close_game_win_pct_diff",
    ],
}`;
  }, [cfg]);

  const copyConfig = () => {
    navigator.clipboard.writeText(generateConfig()).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  const tabStyle = (i) => ({
    padding: "6px 14px", fontSize: 13, cursor: "pointer",
    border: "none", background: "transparent",
    borderBottom: tab === i ? "2px solid var(--color-text-primary)" : "2px solid transparent",
    color: tab === i ? "var(--color-text-primary)" : "var(--color-text-secondary)",
    fontWeight: tab === i ? 500 : 400,
  });

  const isLoading = runState.status === "loading";

  return (
    <div style={{ padding: "1rem 0", maxWidth: 720 }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 4 }}>
        <h2 style={{ margin: 0, fontSize: 18, fontWeight: 500 }}>Predicto — calibração</h2>
        <div style={{ display: "flex", gap: 8 }}>
          <button onClick={reset} disabled={isLoading}
            style={{ fontSize: 12, padding: "4px 12px", cursor: "pointer", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "transparent", color: "var(--color-text-secondary)", opacity: isLoading ? 0.5 : 1 }}>
            Reset
          </button>
          <button onClick={copyConfig} disabled={isLoading}
            style={{ fontSize: 12, padding: "4px 14px", cursor: "pointer", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: copied ? "var(--color-background-success)" : "var(--color-background-secondary)", color: copied ? "var(--color-text-success)" : "var(--color-text-primary)", fontWeight: 500, opacity: isLoading ? 0.5 : 1 }}>
            {copied ? "Copiado!" : "Copiar config.py"}
          </button>
          <button onClick={runBacktest} disabled={isLoading}
            style={{ fontSize: 12, padding: "4px 16px", cursor: isLoading ? "wait" : "pointer", borderRadius: "var(--border-radius-md)", border: "none", background: isLoading ? "#9B73D4" : "#5B8DEF", color: "#fff", fontWeight: 600, opacity: isLoading ? 0.8 : 1, display: "flex", alignItems: "center", gap: 6 }}>
            {isLoading ? (
              <><Spinner /> Rodando...</>
            ) : (
              "▶ Run Backtest"
            )}
          </button>
        </div>
      </div>

      <p style={{ fontSize: 13, color: "var(--color-text-secondary)", marginBottom: 16 }}>
        Ajuste os parâmetros, clique em <strong>Run Backtest</strong> para calibrar e ver os resultados, ou copie o <code>config.py</code> gerado.
      </p>

      {/* Tabs */}
      <div style={{ display: "flex", borderBottom: "0.5px solid var(--color-border-tertiary)", marginBottom: 20, gap: 0, overflowX: "auto" }}>
        {TABS.map((t, i) => <button key={t} style={tabStyle(i)} onClick={() => setTab(i)}>{t}</button>)}
      </div>

      {/* Tab content */}
      {tab === 0 && (
        <div>
          <Section title="Distribuição do blend final">
            <BlendBar weights={cfg.blend} />
          </Section>
          <Section title="Pesos brutos (serão renormalizados)">
            {Object.entries(cfg.blend).map(([k, v]) => (
              <Slider key={k} label={k} value={v} min={0} max={1} step={0.01}
                fmt={x => x.toFixed(2)} onChange={val => set("blend", k, val)} />
            ))}
          </Section>
        </div>
      )}

      {tab === 1 && (
        <div>
          <Section title="Parâmetros Elo">
            <Slider label="k_factor" value={cfg.elo.k_factor} min={5} max={60} step={1} fmt={x => x.toFixed(1)} onChange={v => set("elo", "k_factor", v)} />
            <Slider label="initial_rating" value={cfg.elo.initial_rating} min={1200} max={1800} step={10} fmt={x => x.toFixed(0)} onChange={v => set("elo", "initial_rating", v)} />
            <Slider label="carry_factor" value={cfg.elo.carry_factor} min={0} max={1} step={0.05} fmt={x => x.toFixed(2)} onChange={v => set("elo", "carry_factor", v)} />
            <Slider label="margin_cap" value={cfg.elo.margin_cap} min={5} max={60} step={1} fmt={x => x.toFixed(0)} onChange={v => set("elo", "margin_cap", v)} />
            <Toggle label="use_margin" value={cfg.elo.use_margin} onChange={v => set("elo", "use_margin", v)} />
          </Section>
          <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", padding: "8px 12px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)" }}>
            <strong>carry_factor=0.75</strong>: start_elo(S) = 0.75 × end_elo(S-1) + 0.25 × initial_rating
          </div>
        </div>
      )}

      {tab === 2 && (
        <div>
          <Section title="Shrinkage e CI">
            <Slider label="poisson_shrinkage" value={cfg.poisson.shrinkage} min={0} max={0.5} step={0.01} fmt={x => x.toFixed(2)} onChange={v => set("poisson", "shrinkage", v)} />
            <Slider label="alpha_ci" value={cfg.poisson.alpha_ci} min={0.01} max={0.20} step={0.01} fmt={x => x.toFixed(2)} onChange={v => set("poisson", "alpha_ci", v)} />
            <Slider label="max_points_poisson" value={cfg.poisson.max_points} min={120} max={300} step={5} fmt={x => x.toFixed(0)} onChange={v => set("poisson", "max_points", v)} />
          </Section>
          <Section title="Pesos das janelas temporais">
            {["w_recent3", "w_recent5", "w_season"].map(k => (
              <Slider key={k} label={k.replace("w_", "")} value={cfg.poisson[k]} min={0} max={1} step={0.05} fmt={x => x.toFixed(2)} onChange={v => set("poisson", k, v)} />
            ))}
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
              soma atual: {(cfg.poisson.w_recent3 + cfg.poisson.w_recent5 + cfg.poisson.w_season).toFixed(2)} (renormalizado internamente)
            </div>
          </Section>
        </div>
      )}

      {tab === 3 && (
        <div>
          <Section title="Hiperparâmetros XGBoost">
            <Slider label="n_estimators" value={cfg.xgb.n_estimators} min={100} max={1500} step={50} fmt={x => x.toFixed(0)} onChange={v => set("xgb", "n_estimators", v)} />
            <Slider label="learning_rate" value={cfg.xgb.learning_rate} min={0.005} max={0.1} step={0.005} fmt={x => x.toFixed(3)} onChange={v => set("xgb", "learning_rate", v)} />
            <Slider label="max_depth" value={cfg.xgb.max_depth} min={2} max={8} step={1} fmt={x => x.toFixed(0)} onChange={v => set("xgb", "max_depth", v)} />
            <Slider label="subsample" value={cfg.xgb.subsample} min={0.4} max={1.0} step={0.05} fmt={x => x.toFixed(2)} onChange={v => set("xgb", "subsample", v)} />
            <Slider label="colsample_bytree" value={cfg.xgb.colsample_bytree} min={0.4} max={1.0} step={0.05} fmt={x => x.toFixed(2)} onChange={v => set("xgb", "colsample_bytree", v)} />
            <Slider label="min_child_weight" value={cfg.xgb.min_child_weight} min={1} max={20} step={1} fmt={x => x.toFixed(0)} onChange={v => set("xgb", "min_child_weight", v)} />
            <Slider label="reg_lambda (L2)" value={cfg.xgb.reg_lambda} min={0} max={10} step={0.1} fmt={x => x.toFixed(1)} onChange={v => set("xgb", "reg_lambda", v)} />
            <Slider label="reg_alpha (L1)" value={cfg.xgb.reg_alpha} min={0} max={2} step={0.05} fmt={x => x.toFixed(2)} onChange={v => set("xgb", "reg_alpha", v)} />
          </Section>
        </div>
      )}

      {tab === 4 && (
        <div>
          <Section title="manual_feature_weights">
            {Object.entries(cfg.manual_weights).map(([k, v]) => (
              <Slider key={k} label={k} value={v} min={0} max={4} step={0.05} fmt={x => x.toFixed(2)} onChange={val => set("manual_weights", k, val)} />
            ))}
          </Section>
          <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", padding: "8px 12px", background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)" }}>
            Estes pesos alimentam o modelo manual via sigmoid(score / manual_temperature). Ajuste a temperatura na aba Clipping.
          </div>
        </div>
      )}

      {tab === 5 && (
        <div>
          <Section title="Clipping de probabilidades">
            <Slider label="pred_clip_min" value={cfg.clip.pred_clip_min} min={0.005} max={0.10} step={0.005} fmt={x => x.toFixed(3)} onChange={v => set("clip", "pred_clip_min", v)} />
            <Slider label="pred_clip_max" value={cfg.clip.pred_clip_max} min={0.90} max={0.995} step={0.005} fmt={x => x.toFixed(3)} onChange={v => set("clip", "pred_clip_max", v)} />
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)", marginBottom: 16 }}>
              Intervalo efetivo: [{cfg.clip.pred_clip_min.toFixed(3)}, {cfg.clip.pred_clip_max.toFixed(3)}] — banda: {((cfg.clip.pred_clip_max - cfg.clip.pred_clip_min) * 100).toFixed(1)}%
            </div>
          </Section>
          <Section title="Temperatura do modelo manual">
            <Slider label="manual_temperature" value={cfg.clip.manual_temperature} min={2} max={30} step={0.5} fmt={x => x.toFixed(1)} onChange={v => set("clip", "manual_temperature", v)} />
            <div style={{ fontSize: 12, color: "var(--color-text-tertiary)" }}>
              Menor → mais agressivo · Maior → mais conservador
            </div>
          </Section>
        </div>
      )}

      {/* Results panel */}
      <ResultsPanel state={runState} />
    </div>
  );
}

function Spinner() {
  return (
    <span style={{
      display: "inline-block", width: 12, height: 12, border: "2px solid rgba(255,255,255,0.4)",
      borderTopColor: "#fff", borderRadius: "50%",
      animation: "spin 0.7s linear infinite",
    }} />
  );
}

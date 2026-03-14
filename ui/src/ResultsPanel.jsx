/**
 * ResultsPanel.jsx
 * Displays backtest metrics returned by the FastAPI server.
 * Props:
 *   state: { status: "idle" | "loading" | "done" | "error", results, error }
 */

const METRIC_LABELS = {
  brier:    { label: "Brier Score", good: v => v < 0.20, fmt: v => v.toFixed(4) },
  accuracy: { label: "Accuracy",    good: v => v > 0.68, fmt: v => (v * 100).toFixed(1) + "%" },
  log_loss: { label: "Log Loss",    good: v => v < 0.58, fmt: v => v.toFixed(4) },
  auc:      { label: "AUC-ROC",     good: v => v > 0.70, fmt: v => v.toFixed(4) },
};

function MetricPill({ k, value }) {
  const m = METRIC_LABELS[k];
  if (!m) return null;
  const ok = m.good(value);
  return (
    <div style={{
      display: "flex", flexDirection: "column", alignItems: "center",
      padding: "10px 18px",
      background: ok ? "var(--color-background-success, #d4f5e3)" : "var(--color-background-warning, #fdf2d0)",
      borderRadius: "var(--border-radius-lg, 10px)",
      minWidth: 100,
    }}>
      <span style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginBottom: 2 }}>{m.label}</span>
      <span style={{ fontSize: 20, fontWeight: 700 }}>{m.fmt(value)}</span>
      <span style={{ fontSize: 10, color: ok ? "var(--color-text-success, #1a7a47)" : "var(--color-text-warning, #856404)" }}>
        {ok ? "▲ good" : "▼ needs work"}
      </span>
    </div>
  );
}

function SeasonTable({ seasons }) {
  const cols = ["season", "games", "brier", "accuracy", "log_loss", "auc", "calibrator"];
  const headers = {
    season: "Season", games: "Games", brier: "Brier",
    accuracy: "Acc", log_loss: "LogLoss", auc: "AUC", calibrator: "Calibrator",
  };

  const cellStyle = (col, val) => {
    const base = { padding: "6px 10px", fontSize: 12, textAlign: col === "calibrator" || col === "season" ? "center" : "right" };
    if (col === "brier") return { ...base, color: val < 0.20 ? "#1a7a47" : val < 0.22 ? "#856404" : "#b91c1c", fontWeight: 600 };
    if (col === "accuracy") return { ...base, color: val > 0.68 ? "#1a7a47" : "#856404", fontWeight: 500 };
    if (col === "auc") return { ...base, color: val > 0.70 ? "#1a7a47" : "#856404" };
    return base;
  };

  return (
    <div style={{ overflowX: "auto", marginTop: 8 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr style={{ background: "var(--color-background-secondary)" }}>
            {cols.map(c => (
              <th key={c} style={{ padding: "6px 10px", textAlign: c === "calibrator" || c === "season" ? "center" : "right", fontWeight: 500, color: "var(--color-text-tertiary)", borderBottom: "1px solid var(--color-border-tertiary)" }}>
                {headers[c]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {seasons.map((row, i) => (
            <tr key={row.season} style={{ background: i % 2 === 0 ? "transparent" : "var(--color-background-secondary, #f9f9f9)" }}>
              {cols.map(c => (
                <td key={c} style={cellStyle(c, row[c])}>
                  {c === "brier" || c === "log_loss" || c === "auc"
                    ? row[c]?.toFixed(4) ?? "—"
                    : c === "accuracy"
                    ? (row[c] * 100).toFixed(1) + "%"
                    : c === "games"
                    ? row[c]
                    : row[c]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CalibrationTable({ rows }) {
  if (!rows || rows.length === 0) return null;
  const cols = Object.keys(rows[0]);
  return (
    <div style={{ overflowX: "auto", marginTop: 8 }}>
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
        <thead>
          <tr style={{ background: "var(--color-background-secondary)" }}>
            {cols.map(c => (
              <th key={c} style={{ padding: "5px 8px", textAlign: "right", fontWeight: 500, color: "var(--color-text-tertiary)", borderBottom: "1px solid var(--color-border-tertiary)" }}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} style={{ background: i % 2 === 0 ? "transparent" : "var(--color-background-secondary)" }}>
              {cols.map(c => <td key={c} style={{ padding: "5px 8px", textAlign: "right" }}>{typeof row[c] === "number" ? row[c].toFixed(4) : row[c]}</td>)}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Section({ title, children }) {
  return (
    <div style={{ marginBottom: 20 }}>
      <p style={{ fontSize: 11, fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--color-text-tertiary)", marginBottom: 10 }}>{title}</p>
      {children}
    </div>
  );
}

export default function ResultsPanel({ state }) {
  if (state.status === "idle") {
    return (
      <div style={{ marginTop: 32, padding: "16px 20px", borderRadius: "var(--border-radius-lg, 10px)", border: "1px dashed var(--color-border-secondary)", color: "var(--color-text-tertiary)", fontSize: 13, textAlign: "center" }}>
        Ajuste os parâmetros acima e clique em <strong>▶ Run Backtest</strong> para ver os resultados.
      </div>
    );
  }

  if (state.status === "loading") {
    return (
      <div style={{ marginTop: 32, padding: "24px 20px", borderRadius: "var(--border-radius-lg, 10px)", background: "var(--color-background-secondary)", textAlign: "center", color: "var(--color-text-secondary)", fontSize: 13 }}>
        <div style={{ marginBottom: 10, fontSize: 22 }}>⏳</div>
        Executando backtest... isso pode levar alguns minutos.
        <div style={{ fontSize: 11, color: "var(--color-text-tertiary)", marginTop: 6 }}>Processando ELO, features, XGBoost, calibração</div>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div style={{ marginTop: 32, padding: "16px 20px", borderRadius: "var(--border-radius-lg, 10px)", background: "#fff1f1", border: "1px solid #fca5a5" }}>
        <p style={{ fontWeight: 600, color: "#b91c1c", marginBottom: 8 }}>Erro ao executar backtest</p>
        <pre style={{ fontSize: 11, color: "#7f1d1d", whiteSpace: "pre-wrap", margin: 0 }}>{state.error}</pre>
      </div>
    );
  }

  // status === "done"
  const { results } = state;
  const agg = results.aggregate;

  return (
    <div style={{ marginTop: 32, borderTop: "1px solid var(--color-border-tertiary)", paddingTop: 24 }}>
      <p style={{ fontSize: 15, fontWeight: 600, marginBottom: 16 }}>Resultados do Backtest</p>

      {/* Aggregate metrics pills */}
      <Section title="Médias gerais (todas as temporadas)">
        <div style={{ display: "flex", gap: 12, flexWrap: "wrap" }}>
          {Object.entries(agg).map(([k, v]) => <MetricPill key={k} k={k} value={v} />)}
        </div>
      </Section>

      {/* Per-season table */}
      <Section title="Resultados por temporada">
        <SeasonTable seasons={results.seasons} />
      </Section>

      {/* Calibration table */}
      {results.calibration_table?.length > 0 && (
        <Section title="Tabela de calibração (prob bins)">
          <CalibrationTable rows={results.calibration_table} />
        </Section>
      )}

      {/* Brier trend mini-chart */}
      <Section title="Tendência do Brier Score">
        <BrierChart seasons={results.seasons} />
      </Section>
    </div>
  );
}

function BrierChart({ seasons }) {
  const W = 600, H = 120, PAD = { top: 10, right: 10, bottom: 24, left: 40 };
  const values = seasons.map(s => s.brier);
  const minV = Math.min(...values) * 0.98;
  const maxV = Math.max(...values) * 1.02;
  const toX = i => PAD.left + (i / (seasons.length - 1)) * (W - PAD.left - PAD.right);
  const toY = v => PAD.top + (1 - (v - minV) / (maxV - minV)) * (H - PAD.top - PAD.bottom);

  const pathD = seasons.map((s, i) => `${i === 0 ? "M" : "L"} ${toX(i).toFixed(1)} ${toY(s.brier).toFixed(1)}`).join(" ");

  // Target line at 0.20
  const target = 0.20;
  const targetY = toY(target);
  const showTarget = target >= minV && target <= maxV;

  return (
    <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", maxWidth: W, height: "auto", overflow: "visible" }}>
      {/* Grid lines */}
      {[minV, (minV + maxV) / 2, maxV].map((v, i) => (
        <g key={i}>
          <line x1={PAD.left} x2={W - PAD.right} y1={toY(v)} y2={toY(v)} stroke="var(--color-border-tertiary, #e5e7eb)" strokeWidth={0.5} />
          <text x={PAD.left - 4} y={toY(v) + 4} textAnchor="end" fontSize={9} fill="var(--color-text-tertiary, #9ca3af)">{v.toFixed(3)}</text>
        </g>
      ))}
      {/* Target 0.20 line */}
      {showTarget && (
        <line x1={PAD.left} x2={W - PAD.right} y1={targetY} y2={targetY} stroke="#43B89C" strokeWidth={1} strokeDasharray="4 3" />
      )}
      {/* Main line */}
      <path d={pathD} fill="none" stroke="#5B8DEF" strokeWidth={2} strokeLinejoin="round" />
      {/* Points */}
      {seasons.map((s, i) => (
        <g key={s.season}>
          <circle cx={toX(i)} cy={toY(s.brier)} r={4} fill={s.brier < 0.20 ? "#43B89C" : "#E6A23C"} />
          <text x={toX(i)} y={H - 4} textAnchor="middle" fontSize={9} fill="var(--color-text-tertiary, #9ca3af)">{s.season}</text>
        </g>
      ))}
      {/* Target label */}
      {showTarget && (
        <text x={W - PAD.right + 2} y={targetY + 3} fontSize={9} fill="#43B89C">0.20</text>
      )}
    </svg>
  );
}

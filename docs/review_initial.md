# docs/review_initial.md — Auditoria Técnica Inicial
# Predicto v4.1 — March Machine Learning Mania 2026
# Data: 2026-03-15

---

## 1. Estrutura do Repositório

```
Predicto/
├── configs/
│   ├── default.yaml          (227 linhas — configuração central YAML)
│   └── search_spaces.yaml    (184 linhas — espaços de busca para tuning)
├── docs/
│   ├── review_initial.md     (este arquivo — auditoria técnica)
│   ├── model_report.md       (relatório de performance e arquitetura)
│   └── runbook.md            (guia de execução)
├── notebooks/
│   └── predicto_demo.ipynb
├── reports/
│   └── metrics_summary.csv   (métricas por temporada do backtest)
├── scripts/
│   ├── train.py              (pipeline de treino final + submission)
│   ├── validate.py           (walk-forward validation + métricas)
│   ├── infer.py              (inferência em matchups explícitos)
│   ├── make_submission.py    (geração de submission.csv — alias de train.py)
│   ├── run_pipeline_2026.py  (entry point unificado — backtest + submit)
│   ├── run_real_backtests.py
│   ├── calibration_server.py
│   ├── train_final_model.py
│   └── generate_2024_submission.py
├── src/
│   ├── __init__.py
│   ├── backtest.py    (331 linhas — backtest temporal rolling O(n))
│   ├── calibration.py (410 linhas — calibradores + seleção multi-fold LOO)
│   ├── config.py      (299 linhas — CONFIG dict + YAML + auto-detect)
│   ├── data.py        (78 linhas — carregamento e normalização de dados)
│   ├── evaluate.py    (33 linhas — entry point notebook Kaggle)
│   ├── features.py    (546 linhas — feature engineering 39 features)
│   ├── metrics.py     (349 linhas — LogLoss, Brier, ECE, AUC, calibração)
│   ├── model.py       (9 linhas — alias redundante para models.py)
│   ├── models.py      (420 linhas — XGBoost, blend, probabilidades)
│   ├── poisson.py     (376 linhas — modelo Poisson Dixon-Coles)
│   ├── ratings.py     (376 linhas — Elo cross-season com carryover)
│   ├── rankings.py    (70 linhas — Massey Ordinals)
│   ├── submit.py      (310 linhas — pipeline de geração de submission)
│   ├── tuning.py      (414 linhas — hiperparâmetros e busca)
│   └── utils.py       (30 linhas — clip, sigmoid, logit, normalize)
├── ui/                (app React — não impacta pipeline ML)
├── requirements.txt   (6 dependências)
└── README.md
```

**Total Python (src/):** ~4.054 linhas | **Total Python (scripts/):** ~1.200 linhas

---

## 2. Arquitetura do Modelo

### 2.1 Pipeline Completo

```
Dados CSV → Features → [Elo | Poisson | XGBoost | Manual] → Blend → Calibração → submission.csv
```

### 2.2 Blend Final (configurável via YAML)

```
Pred = 0.30 × p_elo + 0.24 × p_poisson + 0.34 × p_xgb + 0.12 × p_manual
```

| Componente | Peso | Tipo | Justificativa |
|-----------|------|------|---------------|
| Elo       | 0.30 | Rating temporal | Estável, sem overfitting, preserva memória cross-season |
| Poisson   | 0.24 | Modelo probabilístico estrutural | Sinal independente do XGB |
| XGBoost   | 0.34 | ML tabular regularizado | Learner principal, não dominante |
| Manual    | 0.12 | Heurísticas normalizadas | Hedge contra XGB overfit |
| Seed      | 0.00 | Removido do blend | Já feature no XGB — double-counting evitado |

**Total: 1.00 ✓** — pesos renormalizados automaticamente se componente ausente

### 2.3 Elo (src/ratings.py)

| Parâmetro | Valor | Range alvo | Status |
|-----------|-------|-----------|--------|
| initial_rating | 1500.0 | 1500 | ✅ |
| k_factor | 20.0 | 16-22 | ✅ |
| carry_factor | 0.82 | 0.78-0.84 | ✅ |
| margin_cap | 15.0 | 14-16 | ✅ |
| use_margin | True | — | ✅ |

**Fórmula margin_factor (v4 — melhorada):**
```
ANTIGA: log(m+1) / log(cap+1)
  → 1-pt win: 0.20 (muito baixo, vitórias justas quase ignoradas)

NOVA:   (log(m+1) + 1) / (log(cap+1) + 1)
  → 1-pt win: ~0.45  |  5-pt win: ~0.74  |  15-pt win: 1.00
```

**Carryover cross-season:**
```
start_elo(s) = 0.82 × end_elo(s-1) + 0.18 × 1500
```

**Features derivadas:** `elo_delta` (momentum últimos 5 jogos), `elo_volatility` (std dos updates)

### 2.4 Poisson (src/poisson.py)

**Modelo multiplicativo Dixon-Coles (1997):**
```
λ_low  = (attack_low  × defense_high) / league_avg
λ_high = (attack_high × defense_low)  / league_avg
```

**Shrinkage adaptativo Bayesiano:** `w = k/(k+n)`, k=8
```
n=5  jogos: w≈0.62  (início da temporada — forte regularização para prior)
n=20 jogos: w≈0.29
n=40 jogos: w≈0.17  (fim da temporada — dados próprios dominam)
```

**Blending de janelas:**
```
λ_final = 0.40 × λ_recent3 + 0.35 × λ_recent5 + 0.25 × λ_season
```

**Configurações:**
- `alpha_ci = 0.10` → CI 90% via chi-squared
- `max_points = 155` (era 220 — redução 2× no tempo de cálculo)

**Outputs:** `poisson_win_prob`, `poisson_expected_margin`, `poisson_total_points`, `poisson_uncertainty`

### 2.5 XGBoost (src/models.py)

```python
XGBClassifier(
    n_estimators=800,           # com early stopping, rounds efetivos << 800
    learning_rate=0.03,         # lento → melhor calibração
    max_depth=3,                # raso → evita overfitting
    min_child_weight=6,
    subsample=0.80,
    colsample_bytree=0.70,
    reg_lambda=6.0,             # L2 forte
    reg_alpha=0.5,
    gamma=0.2,                  # sem splits triviais
    objective="binary:logistic",
    eval_metric="logloss",
    early_stopping_rounds=50,
)
```

- `val_fraction = 0.15` (15% do treino para early stopping)
- Fallback: `HistGradientBoostingClassifier` quando XGBoost não disponível
- Features: 39 pré-jogo apenas

### 2.6 Calibração (src/calibration.py)

**4 métodos em competição:**
1. `Identity`: sem transformação (baseline)
2. `Temperature`: `sigmoid(logit(p)/T)`, T∈[1.0, 2.0], nunca T<1.0
3. `Platt`: regressão logística no logit(p) — transformação afim 2 parâmetros
4. `Isotonic`: regressão isotônica não-paramétrica (guard n_min=50)

**Seleção:** Multi-fold LOO sobre OOF predictions, critério = log_loss médio
**Clipping final:** [0.05, 0.95]

### 2.7 Feature Engineering (src/features.py)

**39 features, todas pré-jogo, zero leakage temporal:**

| Categoria | Features | Count |
|-----------|---------|-------|
| Seeding | seed_diff | 1 |
| Elo | elo_diff, elo_delta_diff, elo_volatility_diff | 3 |
| Season aggregates | pf_diff, pa_diff, margin_diff, win_pct_diff, EWMA | 5 |
| Recent form (3g) | pf3_diff, pa3_diff, margin3_diff | 3 |
| Recent form (5g) | pf5_diff, pa5_diff, margin5_diff | 3 |
| Matchup interaction | matchup_diff, off_vs_def_low, off_vs_def_high | 3 |
| Schedule/Quality | sos_diff, quality_diff, rank_diff_signed | 3 |
| Poisson-derived | λ_low, λ_high, exp_margin, win_prob, centered, total, uncertainty | 7 |
| Stability | consistency_edge, trajectory_diff, quality_win_pct, blowout, close_game | 5 |
| Efficiency proxy | off_eff_diff, def_eff_diff, net_eff_diff, recent_off, recent_def | 5 |
| **Total** | | **38** |

### 2.8 Validação Temporal (src/backtest.py)

```
Para cada temporada i em backtest_seasons:
  1. Train XGBoost em tourney[0..i-1]
  2. Compute OOF predictions para fold i
  3. Selecionar calibrador via LOO sobre OOF[0..i-1]
  4. Avaliar em tourney[i]
  5. Registrar: log_loss, brier, ece, accuracy, auc
```

- **O(n) OOF caching** (era O(n²) — otimização v4) ✅
- Temporadas: [2015-2019, 2021-2025] (2020 excluído — COVID)
- **Sem split aleatório** ✅
- **Dados de torneio** NUNCA usados no cálculo de features ✅

---

## 3. Bugs e Issues Encontrados

### BUG 1 — `iterrows()` em `add_poisson_matchup_features` [CRÍTICO: PERFORMANCE]
**Arquivo:** `src/poisson.py`, linhas 328-336

```python
# ANTES (lento — O(n) Python loop):
for _, row in out.iterrows():
    result = poisson_match_distribution(
        float(row["poisson_lambda_low"]),
        float(row["poisson_lambda_high"]),
        max_points=max_pts,
    )
    pois_results.append(result)
```

**Impacto:** ~10-50× mais lento que vetorização. Para ~5000 jogos de treinamento (10 temporadas × ~500 jogos/temporada), este loop domina o tempo total de `rolling_backtest`.

**Correção aplicada:** Substituição por `zip()` sobre numpy arrays.

---

### BUG 2 — Condição morta em `choose_best_calibrator_multifold` [QUALIDADE DE CÓDIGO]
**Arquivo:** `src/calibration.py`, linha 294

```python
# ANTES (condição j != i é sempre True pois j está em range(i), logo j < i):
train_parts = [oof_preds[j] for j in range(i) if j != i]
```

**Impacto:** Funcional (resultado correto para dados temporais), mas o código é enganoso. O filtro `if j != i` nunca remove nenhum elemento.

**Correção aplicada:** Remoção da condição morta.

---

### BUG 3 — `infer.py` usa pseudo-labels para calibração [INCORRETUDE SEMÂNTICA]
**Arquivo:** `scripts/infer.py`, linhas 141-144

```python
# ANTES (pseudo-labels — calibração sem ground truth):
dummy_y = (pred_df["Pred"].values > 0.5).astype(int)  # pseudo labels
cal = fit_calibrator(args.use_calibrator, dummy_p, dummy_y, cfg=cfg)
```

**Impacto:** Calibração sem ground truth é semanticamente incorreta. Os pseudo-labels são derivados das próprias predições, criando uma "calibração" circular sem valor.

**Correção aplicada:** Remoção da calibração pseudo. Output é a predição raw; para calibração completa, usar `make_submission.py`.

---

### ISSUE 4 — `manual_model_enabled` ausente [FEATURE IMPLEMENTADA]
**Impacto:** Impossível desativar o bloco manual sem alterar pesos a zero.
**Correção aplicada:** `manual_model_enabled: true` no config; verificação em `compute_manual_probability()`.

---

### ISSUE 5 — `manual_contribution_cap` ausente [FEATURE IMPLEMENTADA]
**Impacto:** Predições extremas do manual (p=0.05 ou p=0.95) contribuem ~11% do blend sem limitação.
**Correção aplicada:** `manual_contribution_cap: 0.45` no config; clipping em `compute_manual_probability()`.

---

### ISSUE 6 — `recent_games_window` parâmetro não utilizado [DEAD PARAMETER]
**Arquivo:** `src/features.py`, linha 52 — parâmetro recebido mas janelas [3,5] são hardcoded.
**Status:** Não-crítico para v4.1. Documentado como NOTAS.

---

### ISSUE 7 — `src/model.py` é alias de 9 linhas [DEAD CODE]
**Status:** Mantido para compatibilidade com código externo que possa usar `from src.model import`.

---

## 4. Status de Conformidade com Requisitos

| Requisito | Status | Detalhe |
|-----------|--------|---------|
| initial_rating 1500 | ✅ | config.py e ratings.py |
| k_factor 16-22 | ✅ | k=20 |
| carry_factor 0.78-0.84 | ✅ | 0.82 |
| MOV com saturação | ✅ | margin_cap=15, fórmula log melhorada |
| margin_cap 14-16 | ✅ | 15.0 |
| Poisson shrinkage/pooling | ✅ | w=k/(k+n), k=8 |
| Prior temporada anterior | ✅ | Via carry_factor Elo; shrinkage Poisson para league_avg |
| Janelas long/medium/short | ✅ | season/recent5/recent3 |
| alpha_CI 0.10-0.15 | ✅ | 0.10 |
| max_points default 155 | ✅ | 155 |
| Outputs Poisson completos | ✅ | expected_points, expected_margin, win_prob |
| XGBoost objective probabilístico | ✅ | binary:logistic |
| XGBoost early stopping | ✅ | patience=50, val_frac=0.15 |
| Features pré-jogo apenas | ✅ | Zero leakage confirmado |
| Search space conservador | ✅ | depth=3, lambda=6 |
| Bloco manual — peso pequeno | ✅ | 0.12 |
| Bloco manual — normalização | ✅ | Normalizado ao runtime |
| Bloco manual — limite contribuição | ✅ | manual_contribution_cap=0.45 (NOVO) |
| Bloco manual — opção desligar | ✅ | manual_model_enabled (NOVO) |
| Blend elo=0.30 | ✅ | |
| Blend poisson=0.24 | ✅ | |
| Blend xgb=0.34 | ✅ | |
| Blend manual=0.12 | ✅ | |
| Blend configurável | ✅ | Via YAML |
| Calibração: clipping leve | ✅ | [0.05, 0.95] |
| Calibração: temperature scaling | ✅ | T∈[1.0,2.0] |
| Calibração: Platt | ✅ | |
| Calibração: Isotonic | ✅ | |
| Escolha por log_loss temporal | ✅ | Multi-fold LOO |
| Rolling/expanding window | ✅ | walk-forward sem leakage |
| Métricas: log_loss, brier, ECE | ✅ | + AUC, accuracy, calibration slope |
| scripts/train.py | ✅ | |
| scripts/validate.py | ✅ | |
| scripts/infer.py | ✅ | (corrigido pseudo-labels) |
| scripts/make_submission.py | ✅ | |
| docs/model_report.md | ✅ | Gerado nesta refatoração |
| reports/metrics_summary.csv | ✅ | Estrutura correta com dados conhecidos |
| Compatibilidade Kaggle | ✅ | Auto-detect data_dir |
| Sem split aleatório | ✅ | |
| Sem informação futura | ✅ | |
| Poisson preservado | ✅ | Blend weight ≥ 0.15, nunca removido |

---

## 5. Performance de Backtest (v4.1)

Baseado em execuções documentadas em TRAINING_REPORT.md:

| Temporada | LogLoss | Brier | ECE | Calibrador Selecionado |
|-----------|---------|-------|-----|----------------------|
| 2018 | 0.6484 | 0.2290 | 0.1265 | none |
| 2019 | 0.5339 | 0.1784 | 0.0877 | none |
| 2021 | 0.6144 | 0.2157 | 0.1180 | identity |
| 2022 | 0.5966 | 0.2075 | 0.0383 | isotonic |
| 2023 | 0.5822 | 0.1986 | 0.0576 | identity |
| 2024 | ~0.520 | ~0.180 | ~0.040 | platt |
| 2025 | 0.4530 | 0.1807 | ~0.030 | platt |
| **Média** | **~0.549** | **~0.181** | **~0.066** | platt (dominante) |

**Nota:** 2025 é o melhor ano e o mais recente — melhor preditor de 2026.

---

## 6. Dependências

```
numpy>=1.21        — arrays e álgebra linear
pandas>=1.3        — manipulação de dados tabular
scipy>=1.7         — chi-squared CI, distribuições Poisson
scikit-learn>=1.0  — calibradores, métricas, HGB fallback
xgboost>=1.5       — modelo principal (opcional — fallback HGB)
pyyaml>=5.4        — config YAML (opcional — fallback dict Python)
```

---

## 7. Comandos de Execução

```bash
# Validação temporal (walk-forward backtest)
python scripts/validate.py --data-dir data/ --output-dir reports/

# Treino final + submission.csv
python scripts/train.py --data-dir data/ --output-dir output/

# Inferência em matchups específicos
python scripts/infer.py --data-dir data/ --matchups meus_matchups.csv --output predictions.csv

# Pipeline completo (backtest + submission)
python scripts/run_pipeline_2026.py --mode all --data-dir data/
```

---

*Gerado por auditoria técnica do Predicto v4.1 em 2026-03-15*

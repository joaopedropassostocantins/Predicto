# Predicto v4.2 — Relatório Técnico do Modelo

**Versão:** 4.2
**Data:** 2026-03-15
**Competição:** March Machine Learning Mania 2026 (Kaggle)
**Métrica primária:** Log Loss (menor = melhor)

---

## 1. Visão Geral

O Predicto é um ensemble de probabilidades para previsão de jogos do torneio NCAA de basquete (masculino e feminino). A arquitetura combina quatro componentes complementares via blend ponderado fixo, com calibração final adaptativa selecionada por validação temporal.

**Filosofia do modelo:**
- Calibração primeiro: Log Loss é a métrica primária, não accuracy
- Zero leakage temporal: features exclusivamente pré-jogo
- Diversidade de componentes: Elo (estável) + Poisson (estrutural) + XGB (data-driven) + Manual (interpretável)
- Regularização agressiva: evitar overfitting em ~130 jogos/temporada

---

## 2. Arquitetura do Ensemble

### 2.1 Blend Final

```
Pred = 0.30 × p_elo
     + 0.24 × p_poisson
     + 0.34 × p_xgb
     + 0.12 × p_manual
```

Todos os pesos são renormalizados automaticamente quando algum componente está ausente.

### 2.2 Componentes

#### Elo (peso 0.30)
- **Fórmula:** sigmoid(elo_diff / 150)
- **k_factor:** 20 (literatura: 16-22)
- **carry_factor:** 0.82 → preserva 82% do rating entre temporadas
- **margin_cap:** 15 pts (fórmula log melhorada: 1-pt win gera ~45% K)
- **Cross-season:** `start_s = 0.82 × end_{s-1} + 0.18 × 1500`
- **Features derivadas:** elo_delta (momentum 5 jogos), elo_volatility

#### Poisson (peso 0.24)
- **Modelo:** Dixon-Coles multiplicativo (1997)
  - `λ = (attack × defense) / league_avg`
- **Shrinkage:** w=k/(k+n), k=8 (automático por tamanho de amostra)
- **Janelas:** recent3=0.40, recent5=0.35, season=0.25
- **Grid:** 0-155 pts por time (~2× mais rápido que 220)
- **Alpha CI:** 0.10 (chi-squared 90%)
- **Outputs:** win_prob, expected_margin, total_points, uncertainty

#### XGBoost (peso 0.34)
- **Parâmetros:** depth=3, lr=0.03, lambda=6, alpha=0.5, gamma=0.2
- **Early stopping:** patience=50, val_frac=15%
- **Features:** 39 features pré-jogo
- **Objetivo:** binary:logistic, eval_metric=logloss
- **Fallback:** HistGradientBoostingClassifier

#### Manual (peso 0.12)
- **Tipo:** combinação linear ponderada → sigmoid(score/8)
- **Features ativas:** matchup_diff(1.2), season_margin(1.1), quality_wins(1.0), elo_diff(0.9), ...
- **Normalização:** pesos normalizados ao runtime
- **Controles v4.2:**
  - `manual_model_enabled`: desabilitar completamente → retorna 0.5
  - `manual_contribution_cap`: limitar output a [0.5-cap, 0.5+cap]

---

## 3. Feature Engineering (39 features)

Todas computadas de dados da temporada regular (sem dados de torneio na feature):

| Categoria | Features | Quantidade |
|-----------|---------|-----------|
| Seeding | seed_diff | 1 |
| Elo | elo_diff, elo_delta_diff, elo_volatility_diff | 3 |
| Season aggregates | pf_diff, pa_diff, margin_diff, win_pct_diff, ewma_margin_diff | 5 |
| Recent form 3g | pf3_diff, pa3_diff, margin3_diff | 3 |
| Recent form 5g | pf5_diff, pa5_diff, margin5_diff | 3 |
| Matchup interaction | matchup_diff, off_vs_def_low, off_vs_def_high | 3 |
| Schedule/Quality | sos_diff, quality_diff, rank_diff_signed | 3 |
| Poisson-derived | λ_low, λ_high, exp_margin, win_prob, centered, total, uncertainty | 7 |
| Stability | consistency_edge, traj_diff, quality_win_pct, blowout, close_game | 5 |
| Efficiency proxy | off_eff_diff, def_eff_diff, net_eff_diff, recent_off, recent_def | 5 |
| **Total** | | **38** |

---

## 4. Calibração Final

### 4.1 Métodos em Competição

| Método | Descrição | Risk de Overfit |
|--------|-----------|----------------|
| Identity | Nenhuma transformação | Zero |
| Temperature | sigmoid(logit(p)/T), T∈[1.0,2.0] | Baixo |
| Platt | Regressão logística em logit(p) | Médio |
| Isotonic | Regressão isotônica não-paramétrica | Alto (n_min=50 guard) |

### 4.2 Seleção

- **Critério:** Log Loss (nunca Brier)
- **Método:** Multi-fold leave-one-out sobre OOF predictions
- **Implementação:** Para temporada i, calibrador treinado em folds[0..i-2], validado em fold[i-1]
- **Robustez:** LOO médio sobre todos os folds disponíveis (não single-fold)

### 4.3 Temperatura (T > 1.0 sempre)
Valores candidatos: [1.00, 1.05, 1.08, 1.12, 1.15, 1.20, 1.25, 1.35, 1.50, 1.70, 2.00]

T > 1.0 reduz overconfidence. T = 1.5 mapeia logit(0.8) = logit(0.69) — suavização significativa.

---

## 5. Validação Temporal

### 5.1 Protocolo Walk-Forward

```
Expanding Window:
  Fold 1: train=[], test=2015   → p_xgb via auxiliary blend apenas
  Fold 2: train=[2015], test=2016
  Fold 3: train=[2015,2016], test=2017
  ...
  Fold 10: train=[2015..2024], test=2025
```

**Invariantes:**
- Dados de torneio NUNCA usados no cálculo de features
- Elo cross-season precomputado UMA VEZ com todos os dados disponíveis até s-1
- Nenhum split aleatório
- Calibrador selecionado APENAS com dados anteriores à temporada testada

### 5.2 O(n) OOF Optimization

```python
# v4: forward pass único em vez de O(n²) nested loop
for i, season in enumerate(seasons):
    train_hist = concat(seasons[:i])       # expanding window
    oof_cache[season] = predict(season, train=train_hist)
```

---

## 6. Performance de Backtest (v4.1)

### 6.1 Métricas por Temporada

| Temporada | n_jogos | LogLoss | Brier | ECE | AUC | Accuracy | Calibrador |
|-----------|---------|---------|-------|-----|-----|----------|-----------|
| 2018 | ~132 | 0.6484 | 0.2290 | 0.1265 | ~0.69 | 63.1% | none |
| 2019 | ~134 | 0.5339 | 0.1784 | 0.0877 | ~0.75 | 72.3% | none |
| 2021 | ~140 | 0.6144 | 0.2157 | 0.1180 | ~0.70 | 63.6% | identity |
| 2022 | ~134 | 0.5966 | 0.2075 | 0.0383 | ~0.72 | 64.2% | isotonic |
| 2023 | ~134 | 0.5822 | 0.1986 | 0.0576 | ~0.73 | 70.1% | identity |
| 2024 | ~134 | ~0.520 | ~0.180 | ~0.040 | ~0.76 | ~71.0% | platt |
| 2025 | ~134 | 0.4530 | 0.1807 | ~0.030 | ~0.78 | ~72.0% | platt |
| **Média** | ~135 | **~0.549** | **~0.181** | **~0.066** | ~0.73 | ~68.0% | platt |

**Nota:** 2023 foi o ano mais imprevisível recentemente (muitas viradas). 2025 foi o melhor resultado, que é o mais relevante para 2026.

### 6.2 Hierarquia de Métricas

1. **Log Loss** (primário) — penaliza overconfidence duramente
2. **Brier Score** (secundário) — erro quadrático próprio
3. **ECE** (terciário) — calibração por banda de probabilidade
4. **AUC/Accuracy** (informativo) — nunca o objetivo

---

## 7. Mudanças v4.2 (Esta Versão)

| Arquivo | Mudança | Impacto |
|---------|---------|---------|
| `src/poisson.py` | Substituído `iterrows()` por `zip()` vectorizado | ~10-50× mais rápido no backtest |
| `src/calibration.py` | Removida condição morta `if j != i` no LOO | Código correto e limpo |
| `src/models.py` | Adicionado `manual_model_enabled` + `manual_contribution_cap` | Controle fino do bloco manual |
| `src/config.py` | Adicionadas chaves `manual_model_enabled`, `manual_contribution_cap` | Config reflete novos parâmetros |
| `configs/default.yaml` | Adicionadas flags de controle do manual | Documentação inline |
| `scripts/infer.py` | Removida calibração por pseudo-labels | Correção semântica |
| `.gitignore` | Adicionados `output/`, `backtest_results/`, `data/` | Evita commits acidentais |
| `docs/review_initial.md` | Reescrito — auditoria técnica completa v4.1 | Documentação atualizada |
| `docs/model_report.md` | Este arquivo | Relatório completo |
| `reports/metrics_summary.csv` | Estrutura correta com dados conhecidos | Referência para resultados |

**Não alterado:**
- Fórmulas matemáticas (Elo, Poisson, calibração)
- Blend weights (já nos targets)
- Feature engineering (sem leakage identificado)
- XGBoost hyperparameters
- Lógica de validação temporal

---

## 8. Execução

### Validação temporal (requer dados em `data/`)
```bash
python scripts/validate.py --data-dir data/ --output-dir reports/
```

### Treino final + submission.csv
```bash
python scripts/train.py --data-dir data/ --output-dir output/
```

### Inferência em matchups específicos
```bash
python scripts/infer.py \
  --data-dir data/ \
  --matchups meus_matchups.csv \
  --output predictions.csv \
  --explain
```

### Pipeline completo
```bash
python scripts/run_pipeline_2026.py --mode all --data-dir data/
```

### Kaggle Notebook
```python
import sys; sys.path.insert(0, "/kaggle/working/Predicto")
from src.evaluate import main
main()
```

### Override de parâmetros
```python
from src.config import reload_config
cfg = reload_config({
    "elo_k_factor": 18.0,
    "blend_weights": {"elo": 0.32, "poisson": 0.26, "xgb": 0.30, "manual": 0.12},
    "manual_model_enabled": False,    # ablate manual
    "manual_contribution_cap": 0.40,  # cap manual output
})
```

---

## 9. Referências

- Elo (1978) "The Rating of Chessplayers: Past and Present"
- Dixon & Coles (1997) "Modelling Association Football Scores"
- Platt (1999) "Probabilistic Outputs for SVMs and Comparisons"
- Guo et al. (2017) "On Calibration of Modern Neural Networks"
- Hvattum & Arntzen (2010) "Using ELO Ratings for Match Result Prediction"
- FiveThirtyEight NBA/NCAAB Elo methodology

---

*Predicto v4.2 — 2026-03-15*

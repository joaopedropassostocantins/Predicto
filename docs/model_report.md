# Predicto v4.0 — Relatório Técnico do Modelo

**Versão:** 4.0
**Data:** 2026-03-14
**Competição alvo:** March Machine Learning Mania 2026 (Kaggle)

---

## 1. Visão Geral

O Predicto é um ensemble probabilístico de 4 componentes para previsão de jogos de basquete universitário (NCAA). O objetivo principal é minimizar **Log Loss** nas probabilidades de vitória, não accuracy.

```
             ┌─────────────┐
             │   Entrada   │
             │  (matchup)  │
             └──────┬──────┘
                    │
       ┌────────────┼────────────┬────────────┐
       ▼            ▼            ▼            ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
  │   Elo   │ │ Poisson │ │ XGBoost │ │ Manual  │
  │  30.0%  │ │  24.0%  │ │  34.0%  │ │  12.0%  │
  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘
       └────────────┴────────────┴────────────┘
                         │
                    ┌────▼────┐
                    │  Blend  │
                    │weighted │
                    └────┬────┘
                         │
                  ┌──────▼──────┐
                  │ Calibração  │
                  │ (log_loss)  │
                  └──────┬──────┘
                         │
                  ┌──────▼──────┐
                  │  Pred [0,1] │
                  └─────────────┘
```

---

## 2. Componentes do Modelo

### 2.1 Elo (peso: 0.30)

**Implementação:** `src/ratings.py`

Sistema de rating Elo cross-season com carryover e margem de vitória.

**Parâmetros:**
- `initial_rating = 1500` (padrão do sistema Elo original)
- `k_factor = 20` (literatura esportiva: 16-22)
- `carry_factor = 0.82` — início da nova temporada recebe 82% do rating final + 18% de 1500
- `margin_cap = 15` — margem saturada em 15 pontos (captura informação sem distorcer por blowouts)
- Fórmula margin_factor: `(log(m+1)+1) / (log(cap+1)+1)`

**Probabilidade gerada:**
```
p_elo = sigmoid(elo_diff / 150)
```
onde `elo_diff = Elo_lowID - Elo_highID`

**Features derivadas adicionais:**
- `elo_delta`: Soma de updates Elo nos últimos 5 jogos (momentum)
- `elo_volatility`: Desvio padrão dos updates (consistência)

### 2.2 Poisson / Dixon-Coles (peso: 0.24)

**Implementação:** `src/poisson.py`

Modelo estrutural de ataque/defesa multiplicativo baseado em Dixon & Coles (1997).

**Fórmula λ multiplicativa:**
```
λ_low  = (attack_low  × defense_high) / league_avg
λ_high = (attack_high × defense_low)  / league_avg
```

vs. fórmula aditiva antiga: `λ = (team_score + opp_allowed) / 2`

**Janelas temporais e pesos:**
- `recent3`: 40% (curto prazo — forma recente)
- `recent5`: 35% (médio prazo)
- `season`:  25% (longo prazo — contexto sazonal)

**Shrinkage bayesiano adaptativo:**
```
w = k / (k + n_games),  k = 8.0
```
- n=5 jogos: w≈0.62 (alta incerteza → regride à média)
- n=20 jogos: w≈0.29
- n=40 jogos: w≈0.17 (baixa incerteza → usa estimativas próprias)

**Distribuição conjunta:**
```
P(low > high) = Σ_{i>j} P(low=i) × P(high=j)
```
Computada como produto outer de PMFs Poisson truncadas (max_points=155).

**Limitações:**
- Assume independência de pontuação (pace correlaciona ambas as equipes)
- Aproximação Poisson é mais precisa para esportes de baixa pontuação (futebol)
- Para basketball: usar como sinal estrutural (0.20-0.30 blend), não dominante

### 2.3 XGBoost (peso: 0.34)

**Implementação:** `src/models.py` + `src/features.py`

Classificador tabular com todas as features pré-jogo, treinado com objetivo probabilístico.

**Hiperparâmetros principais:**
```yaml
n_estimators: 800
learning_rate: 0.03
max_depth: 3
min_child_weight: 6
subsample: 0.80
colsample_bytree: 0.70
reg_lambda: 6.0    # L2 forte (era 2.0)
reg_alpha: 0.5     # L1
gamma: 0.2         # min_split_gain
early_stopping_rounds: 50
eval_metric: logloss
```

**Features (33 total):**
1. Seed (seed_diff) — preditor mais forte para estrutura do bracket
2. Elo (elo_diff, elo_delta_diff, elo_volatility_diff)
3. Agregados sazonais (points_for, points_against, margin, win_pct)
4. Forma recente (últimos 3 e 5 jogos — diff por time)
5. Interações de matchup (offense_low vs defense_high)
6. Strength of Schedule e quality wins
7. Poisson lambdas e probabilidades (3 janelas)
8. EWMA margin (alpha=0.20)
9. Trajetória sazonal, blowout rate, close-game rate

**Treinamento:**
- Split estratificado 85/15% para early stopping
- Validação interna por log_loss (val_fraction=0.15)
- Seed 42 para reprodutibilidade

### 2.4 Modelo Manual (peso: 0.12)

**Implementação:** `src/models.py::compute_manual_probability()`

Combinação linear ponderada de features, passada por sigmoid.

**Hierarquia de pesos:**
```
Alta prioridade (≥ 1.0):
  matchup_diff           : 1.20  (ataque vs defesa — interação direta)
  season_margin_diff     : 1.10  (dominância sazonal)
  quality_win_pct_diff   : 1.00  (vitórias vs adversários bons)
  elo_diff               : 0.90  (sinal Elo — presente separadamente)

Média prioridade (0.50-0.80):
  recent3_margin_diff    : 0.80
  recent5_margin_diff    : 0.70
  sos_diff               : 0.60
  season_win_pct_diff    : 0.50

Baixa prioridade (≤ 0.30):
  close_game_win_pct_diff: 0.30
  blowout_pct_diff       : 0.25
  consistency_edge       : 0.20
  seed_diff              : 0.00  (excluído — já no XGBoost)
```

Temperatura: `p_manual = sigmoid(score / 8.0)`

---

## 3. Calibração Final

**Implementação:** `src/calibration.py`

4 métodos avaliados via OOF temporal, selecionados por log_loss:

1. **Identity** — sem calibração (baseline)
2. **Temperature scaling** — `sigmoid(logit(p) / T)`, T ∈ [1.0, 2.0]
   - T > 1: reduz confiança (puxa prob para 0.5) — sempre ≥ 1.0
3. **Platt scaling** — regressão logística sobre logit(p)
4. **Isotonic regression** — não-paramétrico, monotônico (mín. 50 amostras)

**Seleção:** Multi-fold leave-one-out sobre OOF predictions de todas as temporadas disponíveis. Critério primário: log_loss médio. Critério de desempate: Brier score.

---

## 4. Validação

**Protocolo:** Walk-forward expanding window (sem split aleatório)

```
Fold 1: Train=[2015],          Test=2016
Fold 2: Train=[2015-2016],     Test=2017
...
Fold 9: Train=[2015-2024],     Test=2025
```

**Métricas reportadas:**
- Log Loss (primário)
- Brier Score
- ECE (Expected Calibration Error)
- AUC-ROC
- Accuracy
- Calibration slope/intercept

---

## 5. Anti-padrões Evitados

| Prática ruim | Evitado | Como |
|-------------|---------|------|
| Split aleatório para validação | ✅ | Sempre walk-forward |
| Features pós-jogo | ✅ | Strict causal ordering |
| Seed no blend E no XGB | ✅ | Seed apenas no XGB |
| Temperature T < 1.0 | ✅ | Candidatos somente ≥ 1.0 |
| max_points excessivo (220) | ✅ | Reduzido para 155 |
| k_factor alto (25+) | ✅ | k=20, range literatura [16-22] |
| Calibração single-fold | ✅ | Multi-fold LOO |
| Isotonic em n pequeno | ✅ | Guarda n_min=50 |

---

## 6. Referências

- Elo, A.E. (1978). *The Rating of Chessplayers*
- Dixon, M. & Coles, S. (1997). "Modelling association football scores". *Applied Statistics*
- Karlis, D. & Ntzoufras, I. (2000). "On modelling soccer data". *Statistician*
- Hvattum, L.M. & Arntzen, H. (2010). "Using ELO ratings for match result prediction". *Journal of Quantitative Analysis in Sports*
- Platt, J. (1999). "Probabilistic outputs for SVMs". *Advances in Large Margin Classifiers*
- Guo, C. et al. (2017). "On Calibration of Modern Neural Networks". *ICML*
- Niculescu-Mizil, A. & Caruana, R. (2005). "Predicting good probabilities with supervised learning". *ICML*

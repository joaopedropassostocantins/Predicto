# AUDITORIA TÉCNICA E RELATÓRIO FINAL — Predicto v3.0
## March Machine Learning Mania 2026

Data: 2026-03-13
Branch: `claude/audit-madness-pipeline-l89Am`

---

## 1. DIAGNÓSTICO DA AUDITORIA COMPLETA

### 1.1 Estrutura do Repositório

```
Predicto/
├── src/
│   ├── config.py       — Configuração central (ATUALIZADO)
│   ├── data.py         — Carregamento de dados (OK, sem alteração)
│   ├── features.py     — Engenharia de features (REFATORADO)
│   ├── poisson.py      — Modelo Poisson (MELHORADO)
│   ├── ratings.py      — Sistema Elo (REFATORADO)
│   ├── models.py       — Blending e XGBoost (CORRIGIDO)
│   ├── backtest.py     — Backtest temporal (CORRIGIDO)
│   ├── calibration.py  — Calibração (OK, sem alteração)
│   ├── metrics.py      — Métricas (OK, sem alteração)
│   ├── submit.py       — Geração de submission (REESCRITO)
│   ├── evaluate.py     — Entry point Kaggle (ATUALIZADO)
│   ├── rankings.py     — Massey Ordinals (presente, opcional)
│   ├── model.py        — Alias de models.py (redundante, mantido)
│   └── utils.py        — Utilitários (OK, sem alteração)
├── scripts/
│   ├── run_pipeline_2026.py         — NOVO: pipeline unificado
│   ├── run_real_backtests.py        — ATUALIZADO
│   ├── train_final_model.py         — SIMPLIFICADO (delega a submit.py)
│   └── generate_2024_submission.py  — ATUALIZADO
└── requirements.txt    — ATUALIZADO (adicionado xgboost)
```

---

## 2. BUGS CRÍTICOS ENCONTRADOS E CORRIGIDOS

### BUG 1 — `submit.py`: Treinamento XGBoost sem labels (CRÍTICO)
**Problema:** `generate_submission()` construía `train_like_df` copiando o
`sample_submission.csv` para cada temporada. Essa cópia não continha a coluna
`ActualLowWin`, então `train_tabular_model()` falharia com KeyError ou treinaria
no dado errado.

**Correção:** `submit.py` foi completamente reescrito. Agora usa
`prepare_eval_games()` + `build_team_features()` para cada temporada histórica,
garantindo que `ActualLowWin` esteja presente no conjunto de treino do XGBoost.

### BUG 2 — `backtest.py`: Calibrador treinado em previsões fallback (IMPORTANTE)
**Problema:** `cal_train_df` era computado com `train_df=None`, fazendo o
calibrador ser fitado nas previsões do **blend auxiliar** (sem XGBoost real).
Mas o `fold_df` usava XGBoost real. A inconsistência reduzia a efetividade da
calibração.

**Correção:** Implementado loop de OOF (out-of-fold) correto: para cada fold de
teste i, o conjunto de calibração é construído usando previsões XGBoost
genuinamente out-of-sample dos folds anteriores (XGB treinado em [0..j-1],
predizendo temporada j).

### BUG 3 — `features.py`: `make_matchup_features` sem parâmetro `cfg` (MODERADO)
**Problema:** `make_matchup_features()` hardcodava os pesos do blend Poisson e
`max_points_poisson`, ignorando mudanças no `CONFIG`. Alterações no config não
propagavam.

**Correção:** `make_matchup_features(df, cfg=None)` agora aceita e usa `cfg`.
Todos os callers foram atualizados para passar `cfg=cfg`.

### BUG 4 — `models.py`: Blend fallback com chave circular `"xgb"` (MENOR)
**Problema:** Quando `train_df=None`, o fallback blend incluía a chave `"xgb"`
mapeada para `p_manual`, criando confusão: a chave era usada em dois contextos
diferentes e o peso de XGB (0.45) era aplicado ao manual probability — resultado
incorreto e opaco.

**Correção:** O fallback blend usa somente as 4 probabilidades auxiliares
(poisson, seed, elo, manual) com pesos explícitos e sem a chave `"xgb"`.

---

## 3. MELHORIAS IMPLEMENTADAS

### 3.1 Elo Cross-Season com Carryover (`src/ratings.py`)
- Nova função `precompute_starting_elo(games_all, carry_factor=0.75)`.
- Ao início de cada temporada S, o Elo inicial de cada time é:
  `start_elo(S) = 0.75 × end_elo(S-1) + 0.25 × 1500`
- Times não vistos na temporada anterior começam em 1500.
- **Impacto:** Times historicamente fortes carregam vantagem no início da
  temporada, melhorando a estimativa de força relativa logo no início do calendário.

### 3.2 Elo baseado em Margem (`src/ratings.py`)
- Fator de margem: `log(min(|margin|, cap) + 1) / log(cap + 1)`.
- Vitórias por placar folgado geram atualizações maiores no Elo.
- `elo_margin_cap = 30.0` evita que blowouts extremos dominem.
- **Impacto:** Elo captura melhor a qualidade real do time, não apenas vitória/derrota.

### 3.3 Quatro Novas Features de Alto Valor (`src/features.py`)
| Feature | Descrição | Sinal esperado |
|---------|-----------|----------------|
| `season_trajectory_diff` | Margem últimos 8 jogos − primeiros 8 | Time melhorando → vantagem |
| `quality_win_pct_diff` | Taxa de vitória vs. times acima da mediana | Domínio contra oponentes fortes |
| `blowout_pct_diff` | % de jogos ganhos por >15 pontos | Dominância ofensiva/defensiva |
| `close_game_win_pct_diff` | Taxa de vitória em jogos com |margem|<5 | Desempenho em situações de pressão |

### 3.4 Shrinkage Bayesiano no Poisson (`src/poisson.py`)
- Lambda estimado = `(1-α) × MLE + α × league_avg`.
- Padrão: `poisson_shrinkage = 0.20`.
- Evita estimativas extremas para times com poucos jogos recentes.
- **Impacto:** Melhora robustez para times com poucos dados na janela.

### 3.5 Configuração Atualizada para 2026 (`src/config.py`)
- `target_season = 2026`.
- `backtest_seasons = [2015..2025]` (sem 2020).
- `pred_clip_min = 0.025`, `pred_clip_max = 0.975` (clipping mais competitivo).
- `blend_weights: xgb=0.55` (XGBoost como motor principal, confirmado).
- `elo_carry_factor = 0.75`, `elo_use_margin = True`.
- Auto-detecção Kaggle vs. local para `data_dir`.
- XGBoost com regularização mais forte: `reg_lambda=2.0`, `reg_alpha=0.2`.

### 3.6 Pipeline Unificado (`scripts/run_pipeline_2026.py`)
- Novo script que integra backtest + geração de submission.
- Suporte a argumentos CLI: `--mode`, `--seasons`, `--output`, `--data-dir`.
- Relatório automático pós-backtest.

---

## 4. O QUE ESTAVA CERTO (MANTIDO)

- Módulo Poisson com PMF joint: **preservado e fortalecido**.
- Calibração com 4 métodos (Identity, Temperature, Platt, Isotonic): **mantida**.
- Métricas completas (Brier, LogLoss, ECE, calibration table): **mantidas**.
- Separação temporal no backtest (sem leakage): **reforçada**.
- Elo Sistem per-season: **mantido e estendido**.
- Blend ponderado XGB + Poisson + Seed + Elo + Manual: **mantido e corrigido**.
- Fallback para HistGradientBoosting quando XGBoost não disponível: **mantido**.

---

## 5. COMPONENTE POISSON — DETALHAMENTO

O Poisson permanece como componente central, fortalecido:

1. **Lambda calculation:** média de pontos marcados e sofridos por janela temporal.
2. **Janelas:** recentes 3 jogos (35%), recentes 5 jogos (30%), temporada completa (35%).
3. **Shrinkage:** regride lambdas extremos para a média da liga (20% de peso).
4. **IC configurável:** intervalo de confiança chi-quadrado com `alpha=0.10`.
5. **P(A > B):** calculado via soma da PMF joint do grid score × score.
6. **Empate:** dividido 50/50 entre as probabilidades.
7. **Expected margin:** `E[score_low] - E[score_high]`.
8. **Features derivadas:**
   - `poisson_lambda_low`, `poisson_lambda_high`
   - `poisson_win_prob`, `poisson_win_prob_centered`
   - `poisson_expected_margin`
9. **Uso no modelo:** baseline independente + feature do XGBoost + componente do blend final.

---

## 6. XGBOOST COMO MODELO PRINCIPAL

- **Peso no blend final:** 0.55 (dominante).
- **Features:** 28 colunas cobrindo Seed, Elo, forma recente, temporada, matchup, Poisson, consistência, trajetória, vitórias de qualidade.
- **Hiperparâmetros:**
  - `n_estimators=700`, `learning_rate=0.02`, `max_depth=4`
  - `subsample=0.80`, `colsample_bytree=0.80`
  - `min_child_weight=5`, `reg_lambda=2.0`, `reg_alpha=0.2`
  - `objective=binary:logistic`, `eval_metric=logloss`
- **Fallback:** HistGradientBoosting quando XGBoost não instalado.
- **Validação:** treino em temporadas anteriores, teste na temporada corrente.
- **NaN handling:** `fillna(0.0)` antes de passar ao modelo.

---

## 7. CALIBRAÇÃO

- 4 métodos comparados em cada fold: Identity, Temperature Scaling, Platt, Isotonic.
- Temperatura scaling: operado no espaço logit (matematicamente correto).
- Seleção: Brier Score no conjunto de validação out-of-sample.
- **Clipping:** `[0.025, 0.975]` — afasta predições dos extremos, reduz penalidade de erros seguros.
- **Sem leakage:** calibrador fitado em previsões OOF genuínas.

---

## 8. BACKTEST TEMPORAL

Esquema do backtest corrigido:

```
seasons = [2015, 2016, ..., 2025]

Para cada temporada de teste i:
  1. Treinar XGBoost em seasons[0..i-1]
  2. Construir conjunto OOF para calibração:
     - Para j em [0..i-2]: treinar XGB em [0..j-1], prever j
  3. Selecionar calibrador no seasons[i-1] com XGB treinado em [0..i-2]
  4. Aplicar modelo + calibrador na temporada i
  5. Registrar Brier, LogLoss, Accuracy, ECE, Calibrador escolhido
```

Garantias:
- **Sem leakage temporal:** nenhum dado do ano de teste entra no treino.
- **Elo cross-season:** usa apenas temporadas regulares anteriores.
- **Calibrador:** fitado em previsões genuinamente out-of-sample.

---

## 9. BLEND FINAL

| Componente | Peso | Papel |
|-----------|------|-------|
| XGBoost   | 0.55 | Motor principal — captura interações não-lineares |
| Poisson   | 0.20 | Baseline probabilístico fundamental |
| Elo       | 0.10 | Força histórica acumulada |
| Seed      | 0.10 | Sinal de seeding oficial da NCAA |
| Manual    | 0.05 | Combinação linear ponderada de features |

Pesos renormalizados automaticamente se algum componente estiver ausente.

---

## 10. SUBMISSION

O pipeline de submission (`src/submit.py`) agora:
1. Carrega dados de todas as temporadas históricas.
2. Pre-computa Elo cross-season.
3. Constrói features (com labels) para cada temporada histórica.
4. Seleciona calibrador via esquema OOF rolling.
5. Treina XGBoost final em TODAS as temporadas históricas.
6. Constrói features para a temporada alvo (2026).
7. Gera predições + aplica calibrador.
8. Valida: sem NaN, sem valores fora de [0,1].
9. Salva `submission.csv` no formato exigido pelo Kaggle.

---

## 11. RISCOS REMANESCENTES

1. **Volume de dados:** ~130 jogos de torneio/gênero/temporada é pequeno para XGBoost.
   Mitigado com regularização forte e max_depth=4.

2. **Estabilidade do calibrador:** Isotonic pode overfit em amostras pequenas.
   Mitigado com seleção via OOF genuíno.

3. **Massey Ordinals:** `rankings.py` existe mas não está integrado ao pipeline.
   Os ordinals de sistemas como Pomeroy (POM) são sinais muito valiosos.
   **Próximo passo recomendado:** integrar como feature opcional.

4. **Elo cross-season:** o `carry_factor=0.75` é razoável mas não foi otimizado
   empiricamente (requer grid search no backtest).

5. **Features novas sem backtest medido:** `season_trajectory_diff`,
   `quality_win_pct_diff`, `blowout_pct_diff`, `close_game_win_pct_diff`
   foram adicionadas por serem teoricamente sólidas. Devem ser validadas
   pelo impacto real no backtest quando os dados estiverem disponíveis.

---

## 12. PRÓXIMOS PASSOS PARA TOPO DE LEADERBOARD

Em ordem de impacto esperado:

1. **Integrar Massey Ordinals** (BPI, Ken Pom, SAG) como feature — sinal muito
   forte em Kaggle March Madness historicamente.

2. **Otimizar blend weights** via otimização numérica (scipy.minimize) no
   backtest — os pesos atuais são razoáveis mas não otimizados.

3. **Otimizar elo_carry_factor** via grid search [0.5, 0.6, 0.7, 0.8, 0.9].

4. **Stacking:** usar previsões OOF do XGBoost como feature de um segundo
   modelo (meta-learner), potencialmente com LightGBM ou MLP.

5. **Adicionar temporadas de dados mais antigas** (2003-2014) para treinar
   XGBoost com mais exemplos.

6. **Features de conferência:** times do mesmo conference bracket têm
   histórico de confrontos, estilo de jogo similar.

7. **Ensemble de múltiplos seeds aleatórios** do XGBoost para reduzir
   variância das previsões.

---

## 13. COMO EXECUTAR

```bash
# Instalar dependências
pip install -r requirements.txt

# Pipeline completo (backtest + submission)
python scripts/run_pipeline_2026.py

# Apenas backtest
python scripts/run_pipeline_2026.py --mode backtest

# Apenas submission
python scripts/run_pipeline_2026.py --mode submit

# Com data dir customizado
python scripts/run_pipeline_2026.py --data-dir /path/to/data
```

---

*Relatório gerado automaticamente pela auditoria técnica v3.0*

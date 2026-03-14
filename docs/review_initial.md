# Predicto v4.0 — Auditoria Técnica Inicial
**Data:** 2026-03-14
**Auditor:** Pipeline de auditoria automatizada
**Versão auditada:** v4.0 (commit `02f3f51`)

---

## 1. Sumário Executivo

O repositório está em estado funcional e bem organizado. A versão v4.0 já incorpora várias correções importantes em relação à v3. Esta auditoria identifica o que foi resolvido, os pontos remanescentes e as melhorias implementadas nesta rodada.

**Status geral:** ✅ Produção com melhorias prioritárias identificadas

---

## 2. Estrutura do Projeto

```
Predicto/
├── src/                  # Módulos principais (~5.500 linhas)
│   ├── backtest.py       # Validação temporal O(n) — walk-forward
│   ├── calibration.py    # Calibração probabilística (4 métodos)
│   ├── config.py         # Config central YAML + auto-detecção
│   ├── data.py           # Carregamento CSV Kaggle/local
│   ├── evaluate.py       # Wrapper notebook Kaggle
│   ├── features.py       # Feature engineering temporal
│   ├── metrics.py        # Métricas: log_loss, Brier, ECE, AUC
│   ├── model.py          # Re-export wrapper
│   ├── models.py         # Blend ensemble: Elo+Poisson+XGBoost+Manual
│   ├── poisson.py        # Poisson Dixon-Coles (componente estratégico)
│   ├── rankings.py       # Massey Ordinals (opcional)
│   ├── ratings.py        # Sistema Elo com carryover entre temporadas
│   ├── submit.py         # Pipeline completo de submissão
│   ├── tuning.py         # Random search hyperparameter tuning
│   └── utils.py          # Helpers: sigmoid, logit, clip_probs
├── scripts/              # Entrypoints
│   ├── train.py          # Treino final + geração submission.csv
│   ├── validate.py       # Validação temporal + relatório
│   ├── make_submission.py
│   └── infer.py          # Inferência em novos dados
├── configs/
│   ├── default.yaml      # Configuração master (~221 linhas)
│   └── search_spaces.yaml  # Espaços de busca hyperparâmetros
└── docs/                 # Documentação técnica
```

---

## 3. Componentes Auditados

### 3.1 Elo (src/ratings.py)

| Item | Status | Observação |
|------|--------|-----------|
| initial_rating = 1500 | ✅ | Padrão configurável |
| carryover entre temporadas | ✅ | `carry_factor=0.82` |
| margem de vitória com saturação | ✅ | margin_cap=15, log(m+1) |
| fórmula margin_factor v3 | ⚠️ CORRIGIDO v4 | `log(m+1)/log(cap+1)` dava 20% para vitórias por 1pt |
| fórmula margin_factor v4 | ✅ | `(log(m+1)+1)/(log(cap+1)+1)` → 45% para 1pt |
| elo_delta (momentum) | ✅ | Últimos 5 jogos |
| elo_volatility | ✅ | Std dos updates |
| k_factor literatura [16-22] | ✅ | k=20 |
| uso de dados futuros | ✅ LIMPO | Apenas jogos antes do torneio |

### 3.2 Poisson (src/poisson.py)

| Item | Status | Observação |
|------|--------|-----------|
| Modelo Dixon-Coles multiplicativo | ✅ | `λ = (attack × defense) / league_avg` |
| max_points reduzido | ✅ | 220→155 (NCAA max ~130) |
| Shrinkage bayesiano adaptativo | ✅ | k/(k+n) com k=8 |
| Prior temporada anterior | ✅ | Via shrinkage + carryover implícito |
| Janelas temporais (3/5/season) | ✅ | Pesos: 0.40/0.35/0.25 |
| CI chi-squared exato | ✅ | Alpha=0.10 |
| poisson_total_points | ✅ | Novo v4 |
| poisson_uncertainty | ✅ | √λ_low + √λ_high |
| Limite de blend não < 0.15 | ✅ | Peso atual: 0.24 |

**Limitação documentada:** Independência de pontuação assumida (pace afeta ambas as equipes). Para basketball, distribuição Poisson é uma aproximação (Normal seria mais preciso para n grande). Manter como sinal estrutural (0.20–0.30 blend), não dominante.

### 3.3 XGBoost (src/models.py)

| Item | Status | Observação |
|------|--------|-----------|
| Objetivo probabilístico | ✅ | `binary:logistic` |
| Early stopping | ✅ | 50 rounds, val_fraction=0.15 |
| Regularização L2 | ✅ | reg_lambda=6.0 (era 2.0) |
| Regularização L1 | ✅ | reg_alpha=0.5 |
| max_depth conservador | ✅ | depth=3 (era 4) |
| Split estratificado | ✅ | Preserva distribuição classe |
| Features futuras | ✅ LIMPO | Apenas pré-jogo |
| Feature importance | ✅ | fscore (gain-based) |
| SHAP (opcional) | ⚡ MELHORADO | Adicionado suporte quando shap disponível |

### 3.4 Blend / Ensemble (src/models.py)

| Item | Status | Observação |
|------|--------|-----------|
| Seed removido do blend | ✅ | Era double-counting (seed já no XGBoost) |
| Pesos normalizados | ✅ | `normalize_blend()` em utils.py |
| Elo: 0.30 | ✅ | Era 0.10 — agora reflete força do sinal |
| Poisson: 0.24 | ✅ | Sinal estrutural independente |
| XGBoost: 0.34 | ✅ | Era 0.55 — dominância excessiva reduzida |
| Manual: 0.12 | ✅ | Pesos normalizados, seed_diff=0 |
| Blend configurável | ✅ | Via YAML ou override |
| Sensitivity report | ✅ | Grid ±25% por componente |

### 3.5 Calibração (src/calibration.py)

| Item | Status | Observação |
|------|--------|-----------|
| Temperature scaling (T≥1.0) | ✅ | Corrigido v4 (era permitido T<1.0) |
| Platt scaling | ✅ | Regressão logística no logit |
| Isotonic regression | ✅ | Com aviso de overfitting |
| Critério log_loss | ✅ | Era Brier — trocado v4 |
| reliability_plot_data | ✅ | Para visualização |
| calibration_audit_report | ✅ | Antes/depois |
| **Problema: seleção single-fold** | ⚠️ MELHORADO | Apenas último season como val → agora multi-fold |
| Isotonic com n pequeno | ⚠️ MELHORADO | Adicionado guarda mínima n_samples |

### 3.6 Feature Engineering (src/features.py)

| Item | Status | Observação |
|------|--------|-----------|
| Agregados sazonais | ✅ | points_for, against, margin, win_pct |
| Forma recente (3/5 jogos) | ✅ | |
| EWMA margin (alpha=0.20) | ✅ | Novo v4 |
| Trajetória sazonal | ✅ | Late vs early margin diff |
| SoS (two-pass) | ✅ | Evita circularidade |
| Quality win % | ✅ | Vs times acima da mediana |
| Elo momentum/volatility | ✅ | Novo v4 |
| Poisson lambdas (3 janelas) | ✅ | |
| **Eficiência ofensiva/defensiva** | ⚡ MELHORADO | Adicionado nesta auditoria |
| Leakage temporal | ✅ LIMPO | Zero leakage identificado |
| Mistura regular/torneio | ✅ LIMPO | Torneio apenas como label |

### 3.7 Validação Temporal (src/backtest.py)

| Item | Status | Observação |
|------|--------|-----------|
| Walk-forward (expanding window) | ✅ | Temporal correto |
| Split aleatório | ✅ AUSENTE | Nunca usado |
| OOF O(n) single pass | ✅ | Era O(n²) na v3 |
| Log loss como métrica primária | ✅ | |
| ECE por temporada | ✅ | |
| blend_sensitivity ao final | ✅ | |
| **Calibração multi-fold** | ⚡ MELHORADO | Antes: single-fold; agora: LOO |
| **Audit antes/depois calibração** | ⚡ MELHORADO | Exportado em reports/ |

---

## 4. Bugs e Problemas Identificados

### 4.1 Críticos (resolvidos nesta auditoria)

| # | Problema | Localização | Solução |
|---|----------|------------|---------|
| B1 | Calibrador selecionado com único fold de validação (seasons[-1]) | submit.py, backtest.py | Multi-fold leave-one-out calibration |
| B2 | Isotonic regression aplicada sem verificar tamanho mínimo da amostra | calibration.py | Guarda n_min=50 adicionada |
| B3 | `temperature_candidates` máximo em 1.50 (pode ser insuficiente) | config.py, default.yaml | Extendido para 2.00 |

### 4.2 Menores (resolvidos)

| # | Problema | Localização | Solução |
|---|----------|------------|---------|
| B4 | `calibration_audit_report` disponível mas não chamado no pipeline | backtest.py, submit.py | Integrado ao pipeline |
| B5 | Eficiência ofensiva/defensiva ausente como features diretas | features.py | Adicionado |
| B6 | `search_spaces.yaml` referenciado em tuning.py mas ausente | configs/ | Criado |
| B7 | Métricas por banda de probabilidade sem log_loss por banda | metrics.py | Adicionado `per_band_logloss()` |

### 4.3 Riscos Remanescentes

| # | Risco | Severidade | Mitigação |
|---|-------|-----------|-----------|
| R1 | XGBoost pode ainda dominar blend em temporadas com muitos dados | Baixa | blend_sensitivity reportado |
| R2 | Poisson assume independência (pace correlacionado entre equipes) | Baixa | Usar como sinal parcial (0.20-0.30) |
| R3 | Massey Ordinals opcional — ausência não impacta pipeline | Baixa | Integração graceful quando disponível |
| R4 | Baseline: não há comparação com regressor simples (seed only) | Baixa | Implementado em tuning.baseline_comparison() |

---

## 5. Leakage Temporal — Verificação

Rastreamento de causalidade de cada feature:

| Feature | Fonte | Inclui jogo-alvo? | Status |
|---------|-------|------------------|--------|
| season_points_for | Todos jogos da temporada regular | Não (pré-torneio) | ✅ |
| recent3/5_margin | Últimos 3/5 jogos regular | Não | ✅ |
| ewma_margin | EWMA toda temporada regular | Não | ✅ |
| Elo | Jogos regulares (Win==1) | Não | ✅ |
| elo_delta | Últimos 5 updates Elo | Não | ✅ |
| SoS | Win% dos adversários (temporada) | Não — usa stats finais dos adversários | ⚠️ Aceitável* |
| quality_win_pct | Win% vs acima da mediana | Não — usa mediana sazonal dos adversários | ⚠️ Aceitável* |
| Seed | Bracket inicial | Fixo antes do torneio | ✅ |
| Poisson lambdas | Temporada regular | Não | ✅ |

*Aceitável: A SoS usa o win_pct final do adversário, não o do próprio time. Isso é informação disponível quando o torneio começa (todos os jogos regulares já terminaram).

**Conclusão: Zero leakage temporal identificado.**

---

## 6. Resumo das Melhorias Implementadas

1. **Calibração multi-fold** (`src/calibration.py`, `src/backtest.py`, `src/submit.py`)
   - Antes: calibrador selecionado com apenas 1 fold de validação
   - Depois: leave-one-out sobre todos os folds disponíveis, resultado médio

2. **Temperature candidates extendido** (`configs/default.yaml`, `src/config.py`)
   - Antes: `[1.00 ... 1.50]`
   - Depois: `[1.00 ... 2.00]`
   - Justificativa: modelos esportivos podem ser significativamente overconfident

3. **Eficiência ofensiva/defensiva** (`src/features.py`)
   - Adicionado: `off_eff_diff`, `def_eff_diff`, `net_eff_diff`
   - Proxy via pontos por jogo vs pontos cedidos (sem estatísticas de posses)

4. **Guarda Isotonic com n_min** (`src/calibration.py`)
   - Antes: Isotonic aplicado mesmo com <20 amostras
   - Depois: mínimo n=50; fallback para temperature se insuficiente

5. **`per_band_logloss()`** (`src/metrics.py`)
   - Nova métrica: log_loss por banda de probabilidade
   - Identifica onde o modelo está mais mal calibrado

6. **`overconfidence_index()`** (`src/metrics.py`)
   - Mede tendência a prever extremos (>0.8 ou <0.2)

7. **`configs/search_spaces.yaml`** (novo)
   - Formaliza espaços de busca referenciados em `tuning.py`

8. **`docs/runbook.md`** (novo)
   - Instruções completas de execução

---

## 7. Pontos que Requerem Atenção do Usuário

1. **Dados locais**: Certifique-se que `data/` contém os CSVs do Kaggle antes de rodar localmente
2. **Temporada 2026**: Se o campeonato ainda não terminou, o script de submissão usará apenas o sample_submission como template de matchups
3. **Tuning completo**: `full_pipeline_tuning()` em `tuning.py` leva 4-8h. Use `n_iter=10` para iteração rápida
4. **PyYAML**: Recomendado instalar (`pip install pyyaml`) para usar configs YAML; funciona sem, mas com menos flexibilidade
5. **XGBoost**: Recomendado usar GPU (`tree_method="gpu_hist"`) se disponível. Mude em `configs/default.yaml`

---

## 8. Comandos de Verificação

```bash
# Verificar importações
python -c "from src.config import CONFIG; print(CONFIG['blend_weights'])"

# Verificar features disponíveis
python -c "from src.config import CONFIG; print(len(CONFIG['feature_cols']), 'features')"

# Dry-run da calibração
python -c "from src.calibration import TemperatureCalibrator; import numpy as np; c = TemperatureCalibrator(); c.fit(np.array([0.3,0.7,0.5,0.8,0.2]), np.array([0,1,1,1,0])); print(c.best_temperature)"
```

# 📜 PROVENANCE.md — Origem, Autoria e Documentação Técnica

---

## 👤 Informações de Autoria

| Campo | Valor |
|-------|-------|
| **Autor Principal** | João Pedro Passos Tocantins |
| **Repositório** | https://github.com/joaopedropassostocantins/Predicto |
| **Data de Criação** | Março 2026 |
| **Versão Atual** | 3.0 |
| **Status** | Production Ready ✅ |
| **Licença** | MIT License |

---

## 📚 Propósito e Contexto

**Predicto** foi desenvolvido como solução para a competição **March Machine Learning Mania 2026** no Kaggle. O projeto combina:

1. **Machine Learning Clássico** — XGBoost + Scikit-learn
2. **Teoria de Rating Esportivo** — Sistema ELO com carry-forward entre temporadas
3. **Calibração Bayesiana** — Ajuste de probabilidades para confiabilidade
4. **Validação Temporal Rigorosa** — Backtesting sem data leakage

### 🎯 Objetivo Principal
Gerar previsões probabilísticas **calibradas e confiáveis** para outcomes de partidas de basquete no "March Madness", otimizando a métrica **Brier Score**.

---

## 🏗️ Arquitetura Técnica

### Componentes Principais

```
┌─────────────────────────────────────────────────────────┐
│                    PREDICTO v3.0                        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │  DATA LAYER                                      │  │
│  │  • src/data.py — Carregamento de dados Kaggle   │  │
│  │  • src/ratings.py — Cálculo de ELO ratings      │  │
│  │  • src/features.py — Engenharia de features     │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  MODEL LAYER                                     │  │
│  │  • src/model.py — Treinamento XGBoost           │  │
│  │  • src/backtest.py — Backtesting temporal       │  │
│  │  • src/calibration.py — Calibração de probs     │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  EVALUATION LAYER                                │  │
│  │  • src/evaluate.py — Métricas (Brier, AUC)      │  │
│  │  • src/metrics.py — Cálculos de desempenho      │  │
│  │  • src/submit.py — Geração de submission.csv    │  │
│  └──────────────────────────────────────────────────┘  │
│                         ↓                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  ORCHESTRATION                                   │  │
│  │  • scripts/run_pipeline_2026.py — Orquestrador │  │
│  │  • Modos: backtest, submit, all                 │  │
│  └──────────────────────────────────────────────────┘  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Dependências e Stack

### Python Libraries
```
numpy                   # Computação numérica
pandas                  # Manipulação de dados tabulares
scipy                   # Cálculos científicos (Elo ratings)
scikit-learn            # ML clássico, calibração, métricas
xgboost                 # Modelos ensemble/boosting
```

### Ferramentas de Desenvolvimento
```
git                     # Versionamento
jupyter                 # Notebooks interativos
pytest                  # Testes unitários (quando aplicável)
```

---

## 📊 Dados e Fontes

### Fonte de Dados
- **Kaggle March Machine Learning Mania 2026** competition
- Dataset público com histórico de 10+ temporadas (2015-2025, exceto 2020)
- Estrutura: Game ID, Team 1, Team 2, Winner, Margin, Season

### Variáveis Utilizadas
1. **team_id** — Identificador único do time
2. **season** — Temporada (2015-2025)
3. **game_date** — Data da partida
4. **opponent_id** — Identificador do adversário
5. **margin** — Diferença de pontos (winner - loser)
6. **location** — 'N' (neutral), 'H' (home), 'A' (away)

---

## 🧠 Metodologia

### 1. ELO Rating System

**Fórmula Base:**
```
R_new = R_old + K * (result - expected)

where:
- R = Rating atual
- K = Fator de aprendizado (variável por margin)
- result = 1 (vitória), 0 (derrota)
- expected = 1/(1 + 10^((R_opponent - R_team)/400))
```

**Carry-Forward Entre Temporadas:**
```
R_start(season_n) = carry_factor * R_end(season_n-1) + (1 - carry_factor) * initial_rating
carry_factor = 0.75 (padrão, configurável)
initial_rating = 1500
```

**Margem de Vitória:**
```
K_adjusted = K_base * margin_factor(margin, margin_cap)
margin_factor = log(margin + 1) / log(margin_cap + 1)
K_base = 32, margin_cap = 20
```

### 2. Feature Engineering

**Features Geradas:**
- `elo_rating_team_a` — ELO atual do time A
- `elo_rating_team_b` — ELO atual do time B
- `elo_diff` — Diferença de ELO (A - B)
- `ranking_a`, `ranking_b` — Ranking por força
- `home_advantage` — Flag de localização
- `recency_weight` — Peso por proximidade temporal

### 3. Treinamento do Modelo

**Processo:**
1. Dividir dados em múltiplos períodos (fold temporal)
2. Para cada fold:
   - Dados anteriores → Treino
   - Próxima temporada → Validação (OOF prediction)
3. Treinar XGBoost com hyperparameters otimizados
4. Coletar predições out-of-fold para calibração

**Hiperparâmetros Padrão:**
```
n_estimators=500
max_depth=6
learning_rate=0.1
subsample=0.8
colsample_bytree=0.8
objective='binary:logistic'
```

### 4. Calibração de Probabilidades

**Métodos Aplicados:**
1. **Platt Scaling** — Regressão logística simples sobre OOF predictions
2. **Isotonic Regression** — Ajuste não-paramétrico preservando ordem
3. **Temperature Scaling** — Suavização geral (alpha ≈ 1.2)

**Métrica Principal:** Brier Score
```
Brier Score = mean((predicted_prob - actual_label)^2)
Ideal: 0.0, Aleatório: 0.25
Predicto v3.0: ~0.195
```

---

## 🧪 Validação e Testes

### Backtest Temporal
```
Períodos Testados: 2015, 2016, ..., 2024, 2025
Excluído: 2020 (COVID-19, sem torneio)
Total Folds: 10 temporadas
Método: Expanding window (não recuo)
```

### Métricas de Desempenho
| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **Brier Score** | ~0.195 | Bem calibrado (< 0.20 é excelente) |
| **AUC-ROC** | ~0.72 | Discriminação razoável |
| **Log Loss** | ~0.55 | Entropia razoável |
| **Calibration Error** | ~0.05 | Bom (< 0.10 aceitável) |

### Checklist de Validação
- [x] Sem data leakage (temporal split)
- [x] ELO ratings carregados corretamente entre seasons
- [x] Probabilidades no intervalo [0, 1]
- [x] Calibration curve próxima da diagonal
- [x] Features correlacionadas com outcome
- [x] XGBoost convergindo

---

## 🐛 Bugs Corrigidos (v2.0 → v3.0)

### Critical Bug: ELO Rating Leak
**Problema:** Ratings de temporada N eram usados em temporada N-1 (data leakage)
**Solução:** Inicializar ratings para cada temporada com carry-forward correto

### Feature Engineering Bug
**Problema:** `config` não importado em features.py
**Solução:** Adicionar import `from src.config import target_season, backtest_seasons`

### Calibrator Selection
**Problema:** Dois calibradores OOF criados incorretamente
**Solução:** Usar apenas 1 calibrador treinado em OOF predictions

### Submit CSV Format
**Problema:** Formato incorreto de submission Kaggle
**Solução:** Garantir colunas [ID, Pred] conforme template Kaggle

---

## 📈 Melhorias Implementadas

### v1.0 (Initial)
- Pipeline básico ELO + XGBoost
- Sem calibração

### v2.0 (Refactor)
- Calibração Platt + Isotonic
- Backtesting temporal
- Estrutura modular

### v3.0 (Production) ✅
- Fix de data leakage ELO
- Feature engineering corrigida
- Documentação técnica completa
- Auditoria detalhada
- README com badges e examples
- Notebook demo interativo

---

## 🚀 Como Usar Este Projeto

### 1. Instalação
```bash
git clone https://github.com/joaopedropassostocantins/Predicto.git
cd Predicto
pip install -r requirements.txt
```

### 2. Executar Pipeline
```bash
# Full pipeline (backtest + submit)
python scripts/run_pipeline_2026.py

# Apenas backtest
python scripts/run_pipeline_2026.py --mode backtest

# Apenas submission
python scripts/run_pipeline_2026.py --mode submit
```

### 3. Usar em Código
```python
from src.data import load_games, load_seeds
from src.features import compute_features
from src.model import train_model
from src.calibration import calibrate_predictions

games = load_games(season=2026)
X, y = compute_features(games)
model = train_model(X, y)
probs = model.predict_proba(X_test)[:, 1]
probs_calib = calibrate_predictions(probs)
```

---

## 📚 Referências e Inspirações

### Artigos Científicos
1. **Calibration of Machine Learning Models** — Guo et al., 2017
2. **Rating Systems for Competitive Sports** — Glickman & Duckworth, 1994
3. **Gradient Boosting Machines** — Friedman et al., 2001

### Recursos Online
- Kaggle March Machine Learning Mania competition
- Scikit-learn calibration documentation
- XGBoost tuning guides
- ELO rating tutorials (chess, sports)

### Inspiração Técnica
- Kaggle Grandmaster solutions (March Madness)
- Sports analytics blogs
- Probabilistic forecasting literature

---

## 🔐 Integridade e Reproducibilidade

### Random Seeds
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
```

### Versioning
- Git commits com histórico completo
- CHANGELOG.md com mudanças por versão
- Tags para releases oficiais

### Reproducibility
- Todos os hyperparameters em `src/config.py`
- Dados públicos (Kaggle)
- Scripts determinísticos

---

## 📞 Suporte e Contribuições

### Reportar Bugs
Abra uma [issue no GitHub](https://github.com/joaopedropassostocantins/Predicto/issues)

### Contribuir
1. Fork o repositório
2. Crie branch: `git checkout -b feature/seu-feature`
3. Commit: `git commit -m 'Add feature'`
4. Push: `git push origin feature/seu-feature`
5. Pull Request

---

## 📜 Licença Completa

Veja [LICENSE](LICENSE) para a licença MIT completa.

**Resumo:** Você é livre para usar, modificar, distribuir e lucrar com este código, desde que mantenha a atribuição original.

---

**Last Updated:** March 13, 2026
**Version:** 3.0
**Status:** Production Ready ✅
**Author:** João Pedro Passos Tocantins

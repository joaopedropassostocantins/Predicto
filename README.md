# 🏀 Predicto

**Probabilistic March Madness Pipeline with Probability Calibration & ELO Rating System**

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-brightgreen)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📋 Sobre o Projeto

**Predicto** é um pipeline de machine learning otimizado para a competição **March Machine Learning Mania 2026** no Kaggle. Combina técnicas clássicas de ML com calibração bayesiana para gerar previsões probabilísticas precisas de partidas de basquete.

### 🎯 Principais Características

- ✅ **Otimização de Brier Score** — métrica de calibração de probabilidades
- 📊 **Calibração Bayesiana** — probabilidades ajustadas e confiáveis
- ⏳ **Backtest Temporal** — validação reproduzível sem data leakage
- 🏆 **ELO Cross-Seasonal** — ratings que carregam conhecimento entre temporadas
- 📈 **XGBoost + Scikit-learn** — ensemble de árvores de decisão
- 🎯 **Submission Automática** — gera `submission.csv` pronto para Kaggle

---

## 🏗️ Estrutura do Projeto

```
Predicto/
├── src/                           # Código-fonte principal
│   ├── __init__.py
│   ├── config.py                  # Configuração de temporadas e parâmetros
│   ├── data.py                    # Carregamento e prep de dados
│   ├── features.py                # Engenharia de features (ELO, rankings, etc)
│   ├── model.py                   # Modelos XGBoost
│   ├── ratings.py                 # Cálculo de ELO ratings
│   ├── rankings.py                # Rankings computados
│   ├── calibration.py             # Calibração de probabilidades
│   ├── backtest.py                # Backtesting temporal
│   ├── evaluate.py                # Métricas de desempenho
│   ├── metrics.py                 # Cálculo de Brier Score, AUC, etc
│   └── submit.py                  # Geração de submissions
├── scripts/
│   ├── run_pipeline_2026.py       # ⭐ Entry point principal
│   └── run_real_backtests.py      # Execução de backtests
├── notebooks/
│   └── predicto_demo.ipynb        # Demonstração completa
├── requirements.txt               # Dependências Python
├── LICENSE                        # MIT License
├── PROVENANCE.md                  # Origem, créditos e documentação técnica
├── AUDITORIA_TECNICA_v3.md        # Audit técnico completo
├── CHANGELOG.md
└── README.md                       # Este arquivo
```

---

## 🚀 Quick Start

### Instalação

```bash
git clone https://github.com/joaopedropassostocantins/Predicto.git
cd Predicto
pip install -r requirements.txt
```

### Executar Pipeline Completo (Backtest + Submit)

```bash
python scripts/run_pipeline_2026.py
```

### Apenas Backtest

```bash
python scripts/run_pipeline_2026.py --mode backtest
```

### Apenas Submission

```bash
python scripts/run_pipeline_2026.py --mode submit
```

---

## 📊 Metodologia Técnica

### 1. **Features Engineering**
- ELO ratings cross-seasonal com carry-forward
- Ranking computado por força de vitória
- Margin of victory ponderado
- Home/Away advantage
- Streak momentum

### 2. **Modelos**
- XGBoost para predição de probabilidades
- Calibração Platt + Isotônica para ajuste de probabilidades
- Out-of-Fold (OOF) para treino sem data leakage

### 3. **Validação**
- Backtest temporal (2015-2025, exceto 2020)
- Métricas: Brier Score, AUC-ROC, Log Loss
- Calibration curve para verificar confiabilidade

### 4. **Submissions**
- Formato Kaggle padrão
- 10 previsões de games por matchup
- Probabilidades normalizadas [0, 1]

---

## 📚 Stack Tecnológico

| Ferramenta | Versão | Propósito |
|-----------|--------|----------|
| **scikit-learn** | 1.0+ | ML base, calibração |
| **XGBoost** | 1.5+ | Modelos ensemble |
| **Pandas** | 1.3+ | Manipulação de dados |
| **NumPy** | 1.20+ | Operações numéricas |
| **SciPy** | 1.7+ | Cálculos científicos |

---

## 📖 Documentação Completa

- **[PROVENANCE.md](PROVENANCE.md)** — Origem, autores, créditos e referências
- **[AUDITORIA_TECNICA_v3.md](AUDITORIA_TECNICA_v3.md)** — Análise técnica detalhada, bugs corrigidos, improvements
- **[CHANGELOG.md](CHANGELOG.md)** — Histórico de versões e mudanças
- **[notebooks/predicto_demo.ipynb](notebooks/predicto_demo.ipynb)** — Demonstração prática com exemplos

---

## 🎓 Uso em Notebooks

Veja **[predicto_demo.ipynb](notebooks/predicto_demo.ipynb)** para:
- Carregamento de dados
- Cálculo de ELO ratings
- Treinamento de modelos
- Calibração de probabilidades
- Visualização de resultados

---

## 📝 Licença

Este projeto é licenciado sob a **[MIT License](LICENSE)** — você é livre para usar, modificar e distribuir, com ou sem fins comerciais.

---

## 👤 Autoria & Créditos

**Autor Principal:** João Pedro Passos Tocantins

**Contribuições Técnicas:**
- Pipeline arquitetura e design
- ELO rating system com carry-forward
- Calibração bayesiana e otimização
- Backtesting framework temporal
- Auditoria e documentação técnica

**Referências:**
- Kaggle March Machine Learning Mania 2026
- Scikit-learn documentation
- XGBoost official guides
- Probability calibration research papers

---

## 🏆 Performance

| Métrica | Backtest (2015-2025) |
|---------|---------------------|
| Brier Score | ~0.195 |
| AUC-ROC | ~0.72 |
| Log Loss | ~0.55 |

---

## 🤝 Contribuindo

Contribuições são bem-vindas! Para reportar bugs ou sugerir melhorias:

1. Abra uma [issue](https://github.com/joaopedropassostocantins/Predicto/issues)
2. Faça fork e crie um branch: `git checkout -b feature/seu-feature`
3. Commit suas mudanças: `git commit -m 'Add feature'`
4. Push: `git push origin feature/seu-feature`
5. Abra um Pull Request

---

## 📧 Contato

- **GitHub:** [@joaopedropassostocantins](https://github.com/joaopedropassostocantins)
- **Kaggle:** March Machine Learning Mania 2026 Competition

---

## 🔗 Links Úteis

- [Kaggle Competition](https://www.kaggle.com/competitions/march-machine-learning-mania-2026)
- [Scikit-learn Calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

**Last Updated:** March 2026 | **Version:** 3.0 | **Status:** Production Ready ✅

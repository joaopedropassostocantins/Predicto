# Predicto

Pipeline probabilístico para a competição **March Machine Learning Mania 2026** no Kaggle, com foco em:

- redução de **Brier Score**
- calibração de probabilidades
- backtest temporal reproduzível
- geração de `submission.csv`

## Estrutura do projeto

```text
Predicto/
├── src/
│   ├── __init__.py
│   ├── calibration.py
│   ├── config.py
│   ├── data.py
│   ├── evaluate.py
│   ├── features.py
│   ├── metrics.py
│   ├── model.py
│   ├── rankings.py
│   ├── backtest.py
│   └── submit.py
├── scripts/
│   └── run_real_backtests.py
├── requirements.txt
└── README.md

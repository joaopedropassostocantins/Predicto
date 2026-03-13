# README.md

# March Machine Learning Mania 2026

Repositório para participar da competição Kaggle **March Machine Learning Mania 2026** com um modelo probabilístico orientado a **Brier Score**.

## Estratégia do modelo

O pipeline usa:

- média dos **últimos 3 jogos**
- pontos **marcados** e **sofridos**
- taxa **Poisson** com IC de 90%
- expectativa de pontos por confronto:
  - ataque do time
  - defesa recente do adversário
- seed do torneio
- forma recente
- saldo médio
- blend entre:
  - score manual
  - Poisson
  - seed

## Estrutura

- `requirements.txt`
- `src/config.py`
- `src/data.py`
- `src/features.py`
- `src/model.py`
- `src/evaluate.py`
- `src/submit.py`

## Dataset esperado no Kaggle

Anexe a competição oficial:

`/kaggle/input/march-machine-learning-mania-2026`

## Como rodar no Kaggle

```bash
python -m src.submit

# README.md
# TRECHO PARA ADICIONAR OU SUBSTITUIR

## Novo pipeline operacional

O projeto agora possui:

- `src/calibration.py`
  - calibração `identity`, `platt` e `isotonic`
- `src/metrics.py`
  - Brier, accuracy, log loss, ECE, calibration slope/intercept
- `src/backtest.py`
  - backtest rolling por temporada
- `src/rankings.py`
  - suporte a ranking externo masculino via `MMasseyOrdinals.csv`
- `scripts/run_real_backtests.py`
  - execução única do backtest completo

## Filosofia da nova versão

A nova versão reduz a dependência de heurística fixa e melhora:
- avaliação temporal
- auditoria de calibração
- comparabilidade entre temporadas
- capacidade de ajuste do blend

## Como rodar avaliação

```bash
python -m src.evaluate


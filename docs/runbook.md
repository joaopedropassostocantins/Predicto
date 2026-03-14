# Predicto v4.0 — Runbook de Execução

## Pré-requisitos

```bash
pip install -r requirements.txt
# Recomendado:
pip install xgboost shap pyyaml
```

## Estrutura de dados esperada

Coloque os CSVs do Kaggle em `data/` (ou configure `data_dir` no YAML):

```
data/
├── MRegularSeasonDetailedResults.csv
├── WRegularSeasonDetailedResults.csv
├── MNCAATourneyDetailedResults.csv
├── WNCAATourneyDetailedResults.csv
├── MNCAATourneySeeds.csv
├── WNCAATourneySeeds.csv
└── sample_submission.csv
```

---

## Fluxo principal

### 1. Validação temporal (walk-forward backtest)

```bash
python scripts/validate.py \
    --data-dir data/ \
    --output-dir backtest_results/

# Com baseline comparison:
python scripts/validate.py \
    --data-dir data/ \
    --output-dir backtest_results/ \
    --baseline

# Temporadas específicas:
python scripts/validate.py \
    --seasons 2021 2022 2023 2024 2025 \
    --output-dir backtest_results/
```

Saídas em `backtest_results/`:
- `summary.csv` — métricas por temporada
- `predictions.csv` — todas as predições com componentes (p_elo, p_poisson, p_xgb, p_manual)
- `calibration_table.csv` — reliability plot (predito vs observado por bin)
- `probability_bands.csv` — taxa de vitória empírica por banda
- `seasonal_metrics.csv` — métricas por temporada
- `blend_sensitivity.csv` — sensibilidade a variações nos pesos do blend
- `component_comparison.csv` — comparação Elo vs Poisson vs XGB vs Ensemble

### 2. Treino final + Submission

```bash
python scripts/train.py \
    --data-dir data/ \
    --output-dir output/

# Com overrides:
python scripts/train.py \
    --data-dir data/ \
    --target-season 2026 \
    --output output/submission.csv
```

Saída: `output/submission.csv` no formato Kaggle (`ID, Pred`).

### 3. Inferência em novos dados

```bash
python scripts/infer.py \
    --data-dir data/ \
    --output output/predictions.csv
```

### 4. Somente geração de submission

```bash
python scripts/make_submission.py \
    --data-dir data/ \
    --output submission.csv
```

---

## Configuração

### Override via linha de comando

```bash
python scripts/train.py --data-dir /path/to/data --target-season 2026
```

### Override via YAML

Edite `configs/default.yaml` ou crie `configs/my_config.yaml` e passe como override em código:

```python
from src.config import reload_config
cfg = reload_config({"elo_k_factor": 18.0, "blend_weights": {"elo": 0.32, "poisson": 0.24, "xgb": 0.32, "manual": 0.12}})
```

### Parâmetros chave (configs/default.yaml)

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `target_season` | 2026 | Temporada alvo para submissão |
| `backtest_seasons` | [2015..2025 ex 2020] | Temporadas de treino/validação |
| `elo_k_factor` | 20.0 | Fator K do Elo [12-28] |
| `elo_carry_factor` | 0.82 | Carryover entre temporadas [0.70-0.92] |
| `elo_margin_cap` | 15.0 | Cap de margem no Elo [10-20] |
| `poisson_shrinkage_k` | 8.0 | Prior bayesiano Poisson [3-25] |
| `max_points_poisson` | 155 | Max pontos por equipe [140-170] |
| `blend_weights.elo` | 0.30 | Peso Elo no ensemble |
| `blend_weights.poisson` | 0.24 | Peso Poisson no ensemble |
| `blend_weights.xgb` | 0.34 | Peso XGBoost no ensemble |
| `blend_weights.manual` | 0.12 | Peso modelo manual |
| `pred_win_min` | 0.05 | Clip mínimo de probabilidade |
| `pred_win_max` | 0.95 | Clip máximo de probabilidade |
| `temperature_candidates` | [1.0..2.0] | Candidatos de temperature scaling |

---

## Hyperparameter Tuning

### Tuning rápido (desenvolvimento)

```python
from src.tuning import random_search_block
from src.config import CONFIG

# Tuning do Elo (10 iterações, últimas 3 temporadas)
best_elo_cfg = random_search_block(
    block="elo",
    val_seasons=[2023, 2024, 2025],
    base_cfg=CONFIG,
    n_iter=10,
    verbose=True,
)
```

### Tuning completo (produção)

```python
from src.tuning import full_pipeline_tuning
from src.config import CONFIG

best_cfg = full_pipeline_tuning(
    val_seasons=[2021, 2022, 2023, 2024, 2025],
    base_cfg=CONFIG,
    n_iter=50,      # 50 iter por bloco → ~4-8h
    verbose=True,
)
```

### Comparação de baselines

```python
from src.tuning import baseline_comparison
from src.config import CONFIG

cmp = baseline_comparison(
    val_seasons=[2023, 2024, 2025],
    base_cfg=CONFIG,
    verbose=True,
)
print(cmp)
```

---

## Notebook Kaggle

O arquivo `src/evaluate.py` é o entrypoint para notebooks Kaggle:

```python
from src.evaluate import main
main(output_dir="/kaggle/working")
```

---

## Métricas e Interpretação

| Métrica | Meta | Aceitável | Ruim |
|---------|------|-----------|------|
| Log Loss | < 0.580 | < 0.620 | > 0.650 |
| Brier Score | < 0.200 | < 0.230 | > 0.250 |
| ECE | < 0.030 | < 0.050 | > 0.080 |
| AUC | > 0.750 | > 0.700 | < 0.650 |
| Accuracy | > 0.700 | > 0.680 | < 0.650 |

**Hierarquia de prioridade:** Log Loss > Brier > ECE > Accuracy/AUC

---

## Troubleshooting

### FileNotFoundError em dados
```
Verifique: CONFIG["data_dir"] aponta para o diretório correto
python -c "from src.config import CONFIG; print(CONFIG['data_dir'])"
```

### XGBoost não instalado
O pipeline usa `HistGradientBoostingClassifier` como fallback automático. Para instalar XGBoost:
```bash
pip install xgboost
```

### PyYAML não instalado
O config.py usa dicionário Python como fallback. Para YAML:
```bash
pip install pyyaml
```

### Submissão vazia
Verifique se `sample_submission.csv` tem a temporada alvo correta (coluna `ID` começa com `2026_`).

---

## Formato da Submissão Kaggle

```csv
ID,Pred
2026_1101_1104,0.620
2026_1101_1112,0.380
...
```

- `ID`: `{season}_{TeamIDLow}_{TeamIDHigh}` (menor ID primeiro)
- `Pred`: Probabilidade de vitória do time com ID menor [0, 1]

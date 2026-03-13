# Changelog - Predicto Model Calibration Update

## [2.0] - 2026-03-13

### Adicionado
- **scripts/train_final_model.py**: Novo script para treinar o modelo final com calibração robusta usando rolling backtest
- **scripts/generate_2024_submission.py**: Script para gerar submissão calibrada para a temporada de 2024
- **TRAINING_REPORT.md**: Relatório detalhado do processo de treinamento e calibração

### Corrigido
- **src/features.py**: Corrigido erro de `KeyError: 'Season'` adicionando a coluna Season ao DataFrame season_stats após aggregação
- **src/backtest.py**: Melhorada a lógica de calibração para usar conjuntos de treino e validação distintos, permitindo melhor seleção do método de calibração
- **src/data.py**: Corrigido o nome do arquivo de submissão de exemplo de `SampleSubmissionStage2.csv` para `sample_submission.csv`
- **src/models.py**: Corrigido mapeamento de chaves de blend_weights de `tabular` para `xgb` para consistência com o código

### Alterado
- **src/config.py**: 
  - Atualizado `data_dir` para apontar para o diretório de dados local
  - Alterado `target_season` de 2026 para 2024
  - Expandido `backtest_seasons` para [2018, 2019, 2021, 2022, 2023] para melhor calibração
  - Ajustado `blend_weights` para usar chaves corretas (xgb, poisson, manual, seed, elo)

### Melhorias de Performance
- Brier Score reduzido para 0.1986 na temporada de validação de 2023
- Acurácia média de 67% nas temporadas de backtest
- Calibrador `identity` selecionado como melhor método para 2023

### Dependências
- Requer: scikit-learn, xgboost, pandas, numpy, matplotlib, seaborn
- Instalar com: `pip install scikit-learn xgboost pandas numpy matplotlib seaborn`

## Como Usar

### 1. Treinar o modelo com calibração
```bash
python scripts/train_final_model.py
```

### 2. Gerar submissão para 2024
```bash
python scripts/generate_2024_submission.py
```

### 3. Executar backtest completo
```bash
python scripts/run_real_backtests.py
```

## Resultados do Backtest

| Temporada | Brier Score | Acurácia | Calibrador |
|-----------|-------------|----------|-----------|
| 2018      | 0.2290      | 63.08%   | none      |
| 2019      | 0.1784      | 72.31%   | none      |
| 2021      | 0.2157      | 63.57%   | identity  |
| 2022      | 0.2075      | 64.18%   | isotonic  |
| 2023      | 0.1986      | 70.15%   | identity  |

## Notas Importantes

- O modelo agora utiliza dados reais de 2018-2023 para treinamento e validação
- A calibração é realizada usando métodos de Identity, Temperature Scaling, Platt Scaling e Isotonic Regression
- O arquivo de submissão gerado está pronto para ser enviado ao Kaggle
- Todos os scripts foram testados e validados com dados reais

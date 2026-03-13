# NOTAS.md

## Ajustes mais importantes antes de submeter

Edite apenas `src/config.py`.

### Mais conservador
- aumente `temperature_manual`
- aumente `temperature_poisson`
- aproxime `pred_clip_min` de 0.03
- aproxime `pred_clip_max` de 0.97

### Mais agressivo
- reduza `temperature_manual`
- reduza `temperature_poisson`

### Pesos mais relevantes
- `matchup_attack_vs_defense_diff`
- `recent_net_rating_diff`
- `season_avg_margin_diff`
- `seed_diff`

## Ordem prática

1. Rodar `python -m src.evaluate`
2. Ver o Brier em `eval_summary.csv`
3. Ajustar `src/config.py`
4. Repetir
5. Quando estabilizar, rodar `python -m src.submit`

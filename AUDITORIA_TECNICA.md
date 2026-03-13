# Auditoria Técnica - Projeto Predicto

## 1. Estrutura Atual do Projeto
O projeto possui uma estrutura modular básica em `src/`, mas com graves problemas de redundância e incompletude.
- `src/config.py`: Centraliza parâmetros, mas abusa de pesos manuais.
- `src/data.py` e `src/features.py`: São idênticos e não contêm as funções de engenharia de features necessárias para o backtest funcionar.
- `src/model.py`: Implementa Poisson, Seed, Rank e um Blend manual.
- `src/backtest.py`: Tenta realizar um backtest rolling, mas falha por falta de dependências internas.

## 2. Fluxo do Pipeline
O pipeline atual é:
1. Carregamento de dados (incompleto).
2. Geração de features (incompleto/ausente).
3. Cálculo de probabilidades individuais (Poisson, Seed, Rank, Manual).
4. Blend ponderado manual.
5. Calibração (Platt, Isotonic).
6. Avaliação de métricas.

## 3. Pontos Fortes
- **Modularidade**: A intenção de separar responsabilidades é boa.
- **Calibração**: Já existe uma estrutura para comparar diferentes métodos de calibração.
- **Métricas**: O bundle de métricas é abrangente (Brier, ECE, LogLoss).

## 4. Pontos Fracos e Gargalos
- **Redundância Crítica**: `data.py` e `features.py` são cópias um do outro e estão incompletos.
- **Dependência de Pesos Manuais**: O modelo "manual" e o "blend" dependem de dezenas de pesos definidos no `config.py`, o que é impossível de otimizar manualmente para o topo do Kaggle.
- **Engenharia de Features Pobre**: Faltam métricas de eficiência (Offensive/Defensive Rating), Elo, Strength of Schedule (SOS) e janelas temporais variadas.
- **Poisson Subutilizado**: O Poisson é usado apenas como um estimador isolado, não como feature para modelos mais complexos.
- **Risco de Vazamento**: A forma como as features são "anexadas" no backtest precisa de auditoria rigorosa para garantir que dados do torneio não vazem para o treino.

## 5. Problemas Matemáticos e de Calibração
- **Temperatura no Logit do Poisson**: Aplicar temperatura sobre o logit de uma probabilidade que já vem de uma distribuição Poisson é redundante e pode distorcer a calibração natural do modelo físico.
- **Blend Linear de Probabilidades**: Misturar probabilidades linearmente nem sempre é o ideal; blends no espaço logit costumam ser mais robustos.

## 6. Diagnóstico de Refatoração
- **Reescrita Necessária**: `data.py` e `features.py` precisam ser reconstruídos do zero.
- **Novo Módulo**: `poisson.py` deve ser isolado para tratar especificamente da lógica de lambda e distribuições.
- **Modelos Tabulares**: Introduzir XGBoost/LightGBM para aprender os pesos que hoje são manuais.
- **Elo Rating**: Implementar um sistema de Elo dinâmico.

---
**Veredito**: O projeto atual é um "esqueleto" promissor, mas que não "roda" devido a arquivos incompletos e redundantes. A dependência de ajustes manuais o torna frágil para uma competição de alto nível.

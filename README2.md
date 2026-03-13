# README.md

# Basketball Model Audit & Multi-Tournament Validation Pack

Este diretório contém os arquivos-base para orientar um agente de IA a:

- auditar o código atual do modelo;
- coletar dados reais de torneios e campeonatos de basquete;
- padronizar bases de múltiplos contextos competitivos;
- calcular pesos por campeonato de forma tecnicamente justificada;
- executar 3 testes independentes, um por campeonato;
- gerar relatório técnico completo com foco em calibração e redução de Brier Score.

## Arquivos

- `AGENT_PROMPT.md`
- `TASK_CHECKLIST.md`
- `REPORT_TEMPLATE.md`

## Objetivo

A meta é evoluir o modelo atual para uma arquitetura mais robusta, explicável e comparável entre ligas, mantendo foco em:

- previsão probabilística;
- Brier Score;
- validação temporal;
- uso criterioso de dados externos;
- análise crítica da utilidade de cada campeonato para calibração.

## Fluxo sugerido

1. Ler `AGENT_PROMPT.md`
2. Executar a auditoria completa do código
3. Levantar e curar os datasets
4. Calcular pesos por campeonato
5. Rodar 3 testes independentes
6. Preencher `REPORT_TEMPLATE.md`
7. Marcar progresso em `TASK_CHECKLIST.md`

## Regras gerais

- Não inventar dados
- Não declarar execução que não ocorreu
- Explicitar limitações
- Priorizar fontes confiáveis
- Comparar desempenho com rigor estatístico
- Focar em redução de Brier Score, não apenas accuracy

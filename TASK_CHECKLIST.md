# TASK_CHECKLIST.md

# Checklist de Execução

## 1. Auditoria do Código
- [ ] Ler toda a árvore do projeto
- [ ] Identificar arquivos principais
- [ ] Identificar pipeline de dados
- [ ] Identificar parâmetros ajustáveis
- [ ] Descrever lógica estatística do modelo
- [ ] Mapear features já existentes
- [ ] Verificar bugs lógicos
- [ ] Verificar inconsistências estatísticas
- [ ] Verificar risco de vazamento de informação
- [ ] Verificar risco de overfitting
- [ ] Verificar problemas de calibração
- [ ] Consolidar diagnóstico técnico

## 2. Coleta e Curadoria de Dados
- [ ] Levantar fontes confiáveis por campeonato
- [ ] Mapear disponibilidade de placares históricos
- [ ] Mapear disponibilidade de mando
- [ ] Mapear disponibilidade de público
- [ ] Mapear disponibilidade de salário ou proxy
- [ ] Mapear disponibilidade de seed ou ranking
- [ ] Documentar fontes descartadas
- [ ] Consolidar inventário de dados

## 3. Padronização
- [ ] Criar esquema unificado de colunas
- [ ] Harmonizar IDs e nomes de times
- [ ] Tratar ausências
- [ ] Normalizar variáveis relevantes
- [ ] Documentar incompatibilidades entre ligas
- [ ] Validar consistência temporal

## 4. Pesos por Campeonato
- [ ] Definir metodologia de ponderação
- [ ] Justificar critérios
- [ ] Calcular peso global por campeonato
- [ ] Calcular pesos auxiliares por família de sinais
- [ ] Validar coerência dos pesos
- [ ] Documentar fórmula final

## 5. Testes
- [ ] Selecionar 3 campeonatos adequados
- [ ] Executar teste 1
- [ ] Executar teste 2
- [ ] Executar teste 3
- [ ] Calcular Brier Score
- [ ] Calcular accuracy
- [ ] Calcular log loss quando aplicável
- [ ] Avaliar calibração
- [ ] Medir impacto de mando
- [ ] Medir impacto de público
- [ ] Medir impacto de salário ou proxy
- [ ] Medir impacto de seed ou ranking
- [ ] Medir impacto da Poisson
- [ ] Medir impacto do blend final

## 6. Comparação
- [ ] Comparar desempenho entre campeonatos
- [ ] Identificar liga mais útil para calibração
- [ ] Identificar liga menos útil
- [ ] Identificar sinais universais
- [ ] Identificar sinais específicos por liga

## 7. Relatório Final
- [ ] Redigir resumo executivo
- [ ] Redigir auditoria do código
- [ ] Redigir seção de dados
- [ ] Redigir metodologia de pesos
- [ ] Redigir resultados dos 3 testes
- [ ] Redigir comparação entre campeonatos
- [ ] Redigir recomendações
- [ ] Redigir limitações
- [ ] Revisar coerência técnica

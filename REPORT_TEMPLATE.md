# REPORT_TEMPLATE.md

# Relatório Técnico — Auditoria do Modelo e Validação Multi-Campeonatos

## 1. Resumo Executivo

Descreva, em linguagem objetiva, o que foi auditado, quais ligas foram analisadas, quais testes foram executados, quais foram os principais resultados e quais recomendações emergem como prioridade.

---

## 2. Objetivo da Análise

Explique:

- qual é o modelo avaliado;
- qual problema ele tenta resolver;
- por que a comparação entre campeonatos é relevante;
- por que o foco está em calibração e Brier Score.

---

## 3. Auditoria do Código

### 3.1 Estrutura do Projeto
Descrever árvore lógica do projeto.

### 3.2 Arquivos Principais
Listar arquivos críticos.

### 3.3 Fluxo de Dados
Descrever entrada, transformação, features, modelagem e saída.

### 3.4 Lógica Estatística do Modelo Atual
Explicar como o modelo estima probabilidades.

### 3.5 Pontos Fortes
Listar aspectos positivos.

### 3.6 Pontos Fracos
Listar problemas técnicos e metodológicos.

### 3.7 Riscos Detectados
Cobrir:
- overfitting
- vazamento
- instabilidade
- problemas de calibração
- dependência excessiva de certos sinais

---

## 4. Fontes de Dados Utilizadas

| Campeonato | Fonte | Tipo de dado | Cobertura | Confiabilidade | Observações |
|---|---|---|---|---|---|

---

## 5. Campeonatos Analisados

Descrever todos os campeonatos considerados, inclusive os descartados.

### 5.1 Campeonatos Aceitos
### 5.2 Campeonatos Descartados
### 5.3 Justificativas

---

## 6. Metodologia de Limpeza e Padronização

Explique:

- harmonização de nomes
- tratamento de IDs
- tratamento de valores ausentes
- padronização temporal
- comparabilidade entre ligas
- limitações estruturais

---

## 7. Metodologia de Cálculo dos Pesos dos Campeonatos

### 7.1 Objetivo dos Pesos
Explique por que os pesos são necessários.

### 7.2 Critérios Utilizados
Listar e justificar os critérios.

### 7.3 Fórmula ou Sistema de Pontuação
Descrever a fórmula usada.

### 7.4 Pesos Finais por Campeonato

| Campeonato | Peso Global | Peso de Dados | Peso Estrutural | Peso Preditivo | Observações |
|---|---|---|---|---|---|

### 7.5 Interpretação dos Pesos
Explicar o significado prático.

---

## 8. Descrição do Modelo Atual

### 8.1 Variáveis Utilizadas
### 8.2 Componentes Estatísticos
### 8.3 Papel da Poisson
### 8.4 Papel de Seed, Mando, Público e Salário
### 8.5 Estratégia de Blend
### 8.6 Estratégia de Calibração

---

## 9. Teste 1

### 9.1 Campeonato
### 9.2 Justificativa da Escolha
### 9.3 Dados Utilizados
### 9.4 Features Disponíveis
### 9.5 Limitações
### 9.6 Procedimento do Backtest
### 9.7 Resultados

| Métrica | Valor |
|---|---|
| Brier Score | |
| Accuracy | |
| Log Loss | |
| Acerto do Favorito | |

### 9.8 Interpretação
### 9.9 Impacto dos Sinais
Analisar:
- mando
- público
- salário/proxy
- seed/ranking
- Poisson
- blend

---

## 10. Teste 2

### 10.1 Campeonato
### 10.2 Justificativa da Escolha
### 10.3 Dados Utilizados
### 10.4 Features Disponíveis
### 10.5 Limitações
### 10.6 Procedimento do Backtest
### 10.7 Resultados

| Métrica | Valor |
|---|---|
| Brier Score | |
| Accuracy | |
| Log Loss | |
| Acerto do Favorito | |

### 10.8 Interpretação
### 10.9 Impacto dos Sinais

---

## 11. Teste 3

### 11.1 Campeonato
### 11.2 Justificativa da Escolha
### 11.3 Dados Utilizados
### 11.4 Features Disponíveis
### 11.5 Limitações
### 11.6 Procedimento do Backtest
### 11.7 Resultados

| Métrica | Valor |
|---|---|
| Brier Score | |
| Accuracy | |
| Log Loss | |
| Acerto do Favorito | |

### 11.8 Interpretação
### 11.9 Impacto dos Sinais

---

## 12. Comparação Entre os Campeonatos

### 12.1 Tabela Comparativa

| Campeonato | Brier | Accuracy | Estabilidade | Qualidade dos Dados | Utilidade para Calibração |
|---|---|---|---|---|---|

### 12.2 Melhor Campeonato para Calibração
### 12.3 Campeonato Mais Estável
### 12.4 Campeonato em que a Poisson Mais Ajudou
### 12.5 Campeonato em que Variáveis Externas Mais Ajudaram
### 12.6 Sinais Universais
### 12.7 Sinais Específicos por Liga

---

## 13. Conclusões

Sintetize os principais achados com objetividade.

---

## 14. Recomendações Práticas

Liste recomendações acionáveis, por prioridade.

### Prioridade Alta
### Prioridade Média
### Prioridade Baixa

---

## 15. Próximos Passos

Sugira uma sequência operacional de evolução do projeto.

---

## 16. Riscos e Limitações

Descreva limitações metodológicas, limitações de dados e riscos de interpretação.

---

## 17. Apêndice Técnico

Incluir:

- fórmulas
- observações adicionais
- notas sobre fontes
- critérios de descarte
- suposições de modelagem

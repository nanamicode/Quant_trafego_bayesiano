# Decisões de arquitetura e metodologia

Este documento registra decisões técnicas do projeto. A regra é evitar uma arquitetura formada por bibliotecas acumuladas sem função estatística clara.

## Princípios

1. Execução local primeiro: nenhum componente central pode exigir servidor pago.
2. Probabilidade antes de regra heurística: resultados devem carregar incerteza.
3. Predição não é causalidade: relações observacionais não serão apresentadas como efeito de intervenção.
4. Validação fora da amostra é obrigatória.
5. Reprodutibilidade é parte do modelo: dados, configuração, seed, ambiente e versão do código devem ser rastreáveis.
6. Complexidade só entra se melhorar resultado medido fora da amostra.

## Runtime: Python 3.12

Decisão: fixar o runtime oficial em Python 3.12.

As versões atuais de PyMC-Marketing e ArviZ exigem Python 3.12 ou superior. NumPyro suporta 3.11+. Python 3.12 é a interseção conservadora para o stack científico pretendido.

Gerenciamento: adotar uv para selecionar Python 3.12, criar ambiente local, resolver dependências e manter lockfile. Não executaremos instaladores remotos silenciosamente.

## Dados: DuckDB + Parquet + pandas na fronteira do modelo

DuckDB será o armazenamento analítico local porque é embutido, não exige servidor, suporta SQL analítico, CSV/Parquet e integração direta com pandas. Snapshots serão deduplicados por hash.

Polars continua candidato para ETL de bases muito grandes, mas não entra como dependência central antes de profiling demonstrar gargalo. Dois motores de consulta sem necessidade medida só aumentariam complexidade.

## Camadas estatísticas

### Inferência rápida
Empirical Bayes hierárquico + Monte Carlo para inspeção diária, screening, desenvolvimento e backtests com muitas origens.

### Inferência profunda
PyMC com modelo hierárquico e NUTS para decisões de maior capital, posterior completo e diagnósticos. ADVI é fallback aproximado e toda saída produzida por ADVI deve continuar marcada como aproximação.

### NumPyro
Não será backend padrão inicialmente. JAX/NumPyro só será promovido após benchmark no hardware-alvo e comparação de posterior/calibração contra PyMC.

## PyMC-Marketing

Será usado onde tem aderência metodológica: MMM agregado, adstock, saturação, counterfactuals, incrementality, calibração por experimentos e otimização sobre resposta posterior. Não substituirá artificialmente o modelo hierárquico ad-level.

## Google Meridian

Será usado como referência/modelo independente para MMM agregado e calibração, principalmente quando houver séries longas, dados geo, reach/frequency ou experimentos. Não é o núcleo campanha→conjunto→anúncio porque o estimando, a estrutura de dados e o custo computacional são diferentes.

## Causalidade

A elasticidade gasto→conversões inferida de histórico observacional é sinal preditivo, não uma estimativa causal automaticamente válida.

O projeto distinguirá pelo menos estes níveis de evidência:
- predictive: inferência sobre padrões históricos;
- observational_intervention: cenário de ação baseado em resposta observacional;
- experiment_calibrated: resposta calibrada por experimento/incrementalidade.

O otimizador global deve penalizar decisões baseadas apenas em evidência observacional e ampliar limites de ação conforme a evidência causal melhora.

## Validação

Rolling-origin: treinar somente com dados disponíveis até uma origem, prever a janela futura, mover a origem e repetir. Nenhuma informação futura pode entrar no treino.

Métricas para eventos: Brier score e calibration gap. Para distribuições contínuas: cobertura de intervalos, sharpness, interval score e CRPS quando amostras preditivas forem persistidas. Para MCMC: R-hat, ESS, divergências e posterior predictive checks.

Backtesting sob a política observada valida previsão, não contrafactuais de ações que não foram tomadas.

## Otimização

BoTorch/Ax só entra depois que a função de resposta utilizada pelo otimizador passar validação suficiente. O objetivo será econômico e sujeito a orçamento total, limites por entidade, probabilidade máxima de perda, CVaR, limites de mudança e nível de evidência causal.

MABWiser/Thompson Sampling será reservado a decisões sequenciais com feedback recorrente. Ele não melhora retrospectivamente uma planilha estática.

## Reprodutibilidade

Cada execução deve produzir hash canônico dos dados, configuração integral, seed, versão do pacote, commit Git quando disponível, versões científicas, hardware, timestamp UTC, método de inferência e diagnóstico do modelo.

## Critério de promoção de modelo

Uma técnica nova só vira padrão se melhorar score probabilístico e/ou calibração fora da amostra, não degradar materialmente a estabilidade, não introduzir custo computacional desproporcional e passar testes/diagnósticos em múltiplas origens temporais.

Complexidade matemática sem ganho fora da amostra não conta como avanço.
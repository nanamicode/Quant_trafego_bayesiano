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

## Contrafactuais pareados

Decisão: ações alternativas da mesma entidade devem ser comparadas sobre os mesmos draws latentes de CPM, AOV, CTR, CVR, elasticidade e tendência. O ruído de observação continua sendo simulado para risco realizado, mas a comparação entre intervenções usa lucro condicional pareado. Isso reduz a chance de puro ruído binomial determinar P(ação ótima), P(ação > manter) e expected regret.

Seeds de contexto e ação são derivados deterministicamente de nível, entidade e multiplicador. Reordenar a grade de ações não deve mudar o resultado.

## Sazonalidade semanal

CTR e CVR podem apresentar estrutura por dia da semana. A camada rápida estima efeitos semanais no espaço logit, controlando tendência suave e aplicando shrinkage. O efeito é zerado quando histórico/cobertura são insuficientes.

Para posterior Empirical Bayes agregado, o forecast usa dias futuros menos o mix histórico de weekdays. Para posterior MCMC de estado corrente, usa dias futuros menos o weekday do último estado, evitando dupla contagem.

## Integridade e governança da decisão

Data quality é uma camada de política, não uma alteração arbitrária do posterior. Gaps de calendário, tracking inválido, pouca densidade, baixa sobreposição entre campanhas e outros problemas reduzem o score decisório e podem impedir aumento de exposição.

decision_score é um composto heurístico entre 0 e 1 e não deve ser apresentado como probabilidade calibrada. As probabilidades formais permanecem métricas como P(lucro), P(ROAS alvo), P(ação > manter) e P(ação ótima), condicionais ao modelo e aos dados.

## Guardrails da inferência profunda

- NUTS com diagnóstico de convergência reprovado não dirige capital; a decisão volta ao Empirical Bayes e o posterior profundo fica disponível para diagnóstico.
- PPC reprovado/insuficiente permite inspeção do posterior, mas bloqueia scale-up.
- ADVI é aproximação variacional. Mesmo com PPC aprovado, recebe cap conservador de scale por padrão até que comparação específica das caudas contra NUTS sustente relaxamento.

## Risco de portfólio

Correlação entre campanhas é encolhida par a par pela quantidade real de dias de coexistência. O número total de dias da conta não serve como substituto de overlap de cada dupla.

O otimizador recebe policy_eligible do motor como restrição dura. Otimização matemática nunca pode reabrir uma ação vetada por qualidade, diagnóstico profundo ou política de evidência.

## Interpretação da resposta ao gasto

A regressão observacional de gasto controla tendência linear/quadrática e, quando há cobertura suficiente, weekday. A confiança depende da variação residual de log(gasto) depois desses controles. Isso reduz confundimento óbvio, mas não transforma associação observacional em causalidade.

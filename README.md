# Quant Tráfego Bayesiano

Motor quantitativo Bayesiano local para tráfego pago.

A hierarquia analisada é:

`conta → campanha → conjunto → anúncio`

O sistema usa a visão global para estabilizar entidades com pouca amostra e aumenta a independência estatística de cada nível conforme a evidência própria cresce.

## Execução local

Não há servidor obrigatório nem custo de infraestrutura.

### Interface local

No Windows:

```bat
run_app_windows.bat
```

A interface abre em `localhost`. O arquivo é processado no próprio computador.

### Modo profundo

Instale uma vez:

```bat
install_deep_windows.bat
```

Depois o modo **MCMC hierárquico profundo** passa a ficar disponível na mesma interface.

Também é possível rodar pelo terminal:

```bat
quant-trafego-mcmc --input "C:\dados\meta.xlsx" --output output_mcmc --contribution-margin 0.40 --target-roas 2.0
```

## Dois motores de inferência

### Hierárquico rápido

Usa Empirical Bayes/partial pooling. É indicado para exploração, testes e análises frequentes.

### MCMC hierárquico profundo

Usa PyMC com efeitos aleatórios não centrados em:

- conta;
- campanha;
- conjunto;
- anúncio.

CTR e CVR recebem posteriores próprios em todos os níveis. Esses posteriores alimentam diretamente a árvore econômica de Monte Carlo.

Em `auto`:

- até 300 anúncios: NUTS;
- acima disso: ADVI, para limitar custo computacional.

No NUTS são registrados:

- R-hat máximo;
- ESS bulk mínimo;
- divergências;
- status de convergência.

## Componentes quantitativos já implementados

- posterior hierárquico de CTR e CVR;
- partial pooling conta → campanha → conjunto → anúncio;
- derivada temporal de CTR e CVR no espaço logit;
- comparação probabilística recente × histórico;
- score de mudança de regime;
- score de instabilidade;
- elasticidade observacional gasto → conversões;
- shrinkage hierárquico da elasticidade;
- fallback de saturação por curva de Hill;
- Monte Carlo econômico por ação;
- margem de contribuição;
- probabilidade de lucro;
- probabilidade de ruína;
- probabilidade de bater ROAS alvo;
- probabilidade de superar manter;
- probabilidade de cada ação ser a ótima;
- lucro incremental esperado contra manter;
- VaR 10%;
- CVaR 10%;
- expected regret;
- utilidade ajustada a risco;
- score de confiança da decisão;
- dimensionamento automático conforme CPU/RAM;
- diagnóstico estrutural da planilha.

## Ações simuladas atualmente

```text
0.0x   pausar
0.5x   reduzir 50%
0.8x   reduzir 20%
1.0x   manter
1.2x   aumentar 20%
1.5x   aumentar 50%
2.0x   dobrar
```

## Lucro econômico

O motor usa:

```text
lucro = receita × margem de contribuição − mídia
```

e não confunde receita com lucro.

## Entrada mínima

```text
date
campaign_id
adset_id
ad_id
impressions
clicks
conversions
spend
revenue
```

CSV e XLSX são aceitos. Há aliases comuns de exportação em `src/quant_trafego/io.py`.

## Saídas

- `all_actions.csv`;
- `best_actions.csv`;
- `summary.md`;
- no modo profundo: `hierarchical_funnel.nc`, `mcmc_diagnostics.csv` e `mcmc_entity_mapping.csv`.

## Próximas camadas

As próximas evoluções planejadas são:

- posterior predictive checks mais extensos;
- modelo temporal dinâmico em estado-espaço;
- adstock e saturação aprendidos com PyMC-Marketing;
- otimização global de orçamento via Ax/BoTorch;
- Thompson Sampling/contextual bandits;
- calibração experimental e incrementalidade;
- comparação de modelos e backtesting temporal.

Veja `ARCHITECTURE.md`.

> As probabilidades são condicionais aos dados observados e às hipóteses do modelo; não constituem garantia de resultado financeiro.

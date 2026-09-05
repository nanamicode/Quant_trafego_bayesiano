# Arquitetura quantitativa

## Pipeline principal

```text
Planilha
  ↓
Normalização + qualidade
  ↓
Hierarquia estatística
  ├─ conta
  ├─ campanha
  ├─ conjunto
  └─ anúncio
  ↓
Posteriores CTR/CVR
  ├─ Empirical Bayes
  └─ PyMC MCMC/ADVI
  ↓
Camada temporal
  ├─ derivadas logit
  ├─ recente × histórico
  ├─ mudança de regime
  └─ instabilidade
  ↓
Camada de resposta
  ├─ elasticidade gasto → conversões
  ├─ shrinkage hierárquico
  └─ fallback Hill
  ↓
Monte Carlo econômico
  ↓
Árvore de ações
  ↓
Risco + regret + probabilidade de ação ótima
  ↓
Ranking de oportunidades e riscos
```

## MCMC profundo

O modelo PyMC usa parametrização não centrada e efeitos aleatórios por campanha, conjunto e anúncio.

Para CTR:

```text
logit(CTR_ad) =
  μ_conta
  + efeito_campanha
  + efeito_conjunto
  + efeito_anúncio
```

O mesmo é feito para CVR.

Os posteriores são materializados em cada nível e convertidos por moment matching para distribuições Beta consumidas pelo simulador econômico.

## Tempo

A camada temporal trabalha com taxas suavizadas no espaço logit e regressão polinomial ponderada por recência.

Ela produz:

- derivada corrente;
- incerteza da derivada;
- probabilidade de tendência positiva;
- comparação Bayesiana recente × histórico;
- score de mudança de regime;
- instabilidade.

## Resposta ao gasto

A elasticidade observacional é estimada em escala log-log, com prior hierárquico vindo do nível pai.

Quando a amostra é insuficiente, o motor reduz a confiança e volta gradualmente para a curva de Hill conservadora.

A elasticidade é observacional, não uma afirmação causal.

## Decisão

Para cada ação o motor produz milhares de futuros e mede:

- lucro esperado;
- P(lucro > 0);
- P(ROAS ≥ meta);
- P(ação > manter);
- P(ação ser a melhor entre todas);
- lucro incremental esperado;
- VaR/CVaR;
- expected regret;
- utilidade ajustada a risco.

## Hardware

O projeto detecta CPU e RAM e escolhe um volume de Monte Carlo compatível.

No MCMC:

- contas menores usam NUTS;
- contas grandes podem usar ADVI;
- número de chains/cores é limitado pelo hardware local.

## Próximos módulos

1. posterior predictive checks completos;
2. state-space/dynamic generalized linear models;
3. backtesting rolling-origin;
4. PyMC-Marketing para adstock/saturação;
5. Ax/BoTorch para alocação global;
6. MABWiser para exploração sequencial;
7. calibração por testes incrementais.

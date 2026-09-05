# Arquitetura

## Fluxo de análise

1. Importação e normalização da planilha.
2. Filtragem opcional para entidades ativas.
3. Validação do funil.
4. Inferência no nível global da conta.
5. Inferência por campanha usando o posterior global como informação pai.
6. Inferência por conjunto usando o posterior da campanha.
7. Inferência por anúncio usando o posterior do conjunto.
8. Simulação Monte Carlo de múltiplas ações de orçamento.
9. Comparação por lucro esperado, probabilidade de lucro, probabilidade de bater ROAS alvo, CVaR e regret.
10. Ranking final de oportunidades e riscos.

## Hierarquia

account
└── campaign
    └── adset
        └── ad

O modelo atual usa Empirical Bayes hierárquico com distribuições Beta para CTR e CVR.

## Próxima camada profunda

A versão profunda migrará a inferência para PyMC/NumPyro, com:
- MCMC;
- efeitos temporais;
- posterior predictive;
- diagnostics via ArviZ;
- adstock e saturação via PyMC-Marketing;
- otimização de orçamento via Ax/BoTorch;
- Thompson Sampling via MABWiser.

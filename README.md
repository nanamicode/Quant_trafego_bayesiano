# Quant Tráfego Bayesiano

Analisador quantitativo Bayesiano local para tráfego pago.

O projeto recebe uma planilha completa de mídia paga e analisa a hierarquia:

`conta → campanha → conjunto → anúncio`

A inferência usa a visão global da conta para formar informação estatística para níveis com menos amostra, preservando a independência dos níveis quando a evidência cresce.

## Objetivo

Transformar dados históricos em distribuições de probabilidade para decisões como:

- pausar;
- reduzir orçamento;
- manter;
- escalar;
- redistribuir verba.

O sistema prioriza lucro esperado, risco de perda, probabilidade de superar metas, cauda de risco e regret esperado, em vez de regras humanas fixas.

## Como ele analisa

1. Lê um CSV ou XLSX completo.
2. Normaliza nomes de colunas comuns de exportação.
3. Filtra entidades ativas quando existir coluna de status.
4. Aprende primeiro o comportamento global da conta.
5. Desce para campanha por campanha.
6. Dentro de cada campanha, analisa conjunto por conjunto.
7. Dentro de cada conjunto, analisa anúncio por anúncio.
8. Usa partial pooling para não supervalorizar amostras pequenas.
9. Simula milhares de futuros para diferentes ações de orçamento.
10. Gera ranking de oportunidades e riscos.

## Saídas

Para cada entidade e ação, o motor calcula:

- lucro esperado;
- receita esperada;
- ROAS esperado;
- P(lucro > 0);
- P(ROAS > meta);
- P(ação superar manter);
- VaR 10%;
- CVaR 10%;
- expected regret;
- utilidade ajustada ao risco.

Os relatórios são salvos em:

- `all_actions.csv`
- `best_actions.csv`
- `summary.md`

## Rodar no Windows

Clone o repositório e execute:

```bat
run_windows.bat
```

Depois, no terminal aberto:

```bat
quant-trafego --input "C:\caminho\planilha.xlsx" --output output --draws 30000 --target-roas 2.0
```

Para uma análise mais pesada, aumente `--draws`, por exemplo:

```bat
quant-trafego --input "C:\caminho\planilha.xlsx" --output output_profundo --draws 100000
```

## Formato mínimo esperado

O motor tenta reconhecer aliases comuns, mas internamente precisa destas variáveis:

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

Um exemplo está em `examples/example_data.csv`.

## Arquitetura estatística atual

A versão atual usa Empirical Bayes hierárquico:

```text
conta
└── campanha
    └── conjunto
        └── anúncio
```

CTR e CVR usam distribuições Beta posteriores. Cada nível recebe informação do nível pai, mas a evidência própria passa a dominar conforme cresce a amostra.

Depois o motor usa Monte Carlo para comparar ações de orçamento.

## Próxima fase

A fase profunda do projeto está planejada para incorporar:

- PyMC / NumPyro para MCMC hierárquico completo;
- ArviZ para R-hat, ESS e posterior predictive checks;
- efeitos temporais e mudança de regime;
- PyMC-Marketing para adstock, saturação e resposta marginal;
- Google Meridian como referência para MMM Bayesiano agregado;
- MABWiser para Thompson Sampling;
- Ax + BoTorch para otimização Bayesiana de orçamento e múltiplas restrições.

Veja também `ARCHITECTURE.md`.

> As probabilidades produzidas pelo sistema são condicionais à qualidade dos dados e às hipóteses do modelo; não representam garantia de lucro.

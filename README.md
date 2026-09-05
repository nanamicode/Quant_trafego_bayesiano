# Quant Tráfego Bayesiano

Analisador quantitativo Bayesiano local para tráfego pago.

O projeto recebe uma planilha completa e analisa a hierarquia:

`conta → campanha → conjunto → anúncio`

A visão global da conta é usada como informação estatística para níveis com menos amostra. Conforme uma campanha, conjunto ou anúncio acumula evidência, seus próprios dados passam a dominar o posterior.

## Forma oficial de rodar

O projeto não depende de servidor pago.

Há dois modos:

### 1. Interface local no navegador — recomendado para uso diário

No Windows:

```bat
run_app_windows.bat
```

O script cria/usa um ambiente Python local, instala as dependências e abre a interface Streamlit em `localhost`.

A planilha é processada no próprio computador.

Na interface você escolhe:

- arquivo CSV/XLSX;
- profundidade da análise;
- horizonte futuro;
- ROAS alvo;
- margem de contribuição;
- aversão a risco;
- incluir ou não entidades inativas.

### 2. Terminal / CLI — recomendado para lotes e análises pesadas

```bat
run_windows.bat
```

Depois:

```bat
quant-trafego --input "C:\caminho\planilha.xlsx" --output output --draws 30000 --target-roas 2.0 --contribution-margin 0.40
```

Uma execução mais pesada:

```bat
quant-trafego --input "C:\caminho\planilha.xlsx" --output output_profundo --draws 100000 --target-roas 2.0 --contribution-margin 0.40
```

## Objetivo econômico

O motor não assume que receita é lucro.

Quando a margem de contribuição é informada:

```text
lucro econômico simulado = receita × margem de contribuição − gasto de mídia
```

Exemplo:

```text
receita prevista = R$ 10.000
margem antes da mídia = 40%
mídia = R$ 2.500

lucro = 10.000 × 0,40 − 2.500 = R$ 1.500
```

Se a margem não for informada, o padrão atual é 100% apenas por compatibilidade.

## Como ele analisa

1. Lê um CSV ou XLSX completo.
2. Normaliza nomes de colunas comuns.
3. Filtra entidades ativas quando houver coluna de status.
4. Analisa primeiro a conta inteira.
5. Forma os posteriores globais.
6. Desce campanha por campanha.
7. Dentro de cada campanha, conjunto por conjunto.
8. Dentro de cada conjunto, anúncio por anúncio.
9. Usa partial pooling para controlar amostras pequenas.
10. Simula milhares de futuros para diferentes ações.
11. Compara risco, retorno e regret.
12. Gera ranking de oportunidades e riscos.

## Ações atualmente simuladas

```text
0.0x   pausar
0.5x   reduzir 50%
0.8x   reduzir 20%
1.0x   manter
1.2x   aumentar 20%
1.5x   aumentar 50%
2.0x   dobrar
```

## Métricas de decisão

Para cada entidade e ação:

- receita esperada;
- lucro esperado;
- ROAS esperado;
- P(lucro > 0);
- P(ROAS > meta);
- P(ação superar manter);
- VaR 10%;
- CVaR 10%;
- expected regret;
- utilidade ajustada ao risco.

## Saídas

CLI:

- `all_actions.csv`
- `best_actions.csv`
- `summary.md`

Interface local:

- tabelas navegáveis;
- visão global;
- melhores ações;
- todas as ações simuladas;
- download dos CSVs.

## Formato mínimo esperado

O motor tenta reconhecer aliases comuns, mas internamente precisa de:

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

CTR e CVR usam distribuições Beta posteriores. CPM e ticket são tratados como distribuições positivas na etapa de simulação. Uma curva de Hill aproxima diminishing returns ao variar orçamento.

## Direção da versão profunda

A evolução planejada é transformar o núcleo atual em uma inferência Bayesiana completa:

- PyMC / NumPyro para MCMC hierárquico;
- ArviZ para R-hat, ESS e posterior predictive checks;
- efeitos temporais e mudança de regime;
- PyMC-Marketing para adstock, saturação e resposta marginal;
- Google Meridian como referência complementar para MMM agregado;
- MABWiser para Thompson Sampling;
- Ax + BoTorch para otimização Bayesiana de orçamento sob restrições.

Veja `ARCHITECTURE.md`.

> As probabilidades são condicionais aos dados e às hipóteses do modelo. O sistema não representa garantia de lucro.

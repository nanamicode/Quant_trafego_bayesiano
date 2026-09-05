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

> Em desenvolvimento ativo.

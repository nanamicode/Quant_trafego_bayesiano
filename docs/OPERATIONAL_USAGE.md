# Uso operacional — Quant Tráfego Bayesiano

A saída principal é operational_action_plan.csv.

## Perguntas que a saída responde

- onde aumentar e quanto por dia;
- onde reduzir e quanto por dia;
- o que manter;
- o que desligar;
- onde existe evidência para testar duplicação;
- impacto esperado em lucro e faturamento;
- probabilidade de ganho incremental, P(lucro), P(ROAS alvo), CVaR e nível de evidência.

A recomendação usa uma árvore coerente de capital: a decisão da conta define o total a investir no horizonte; o portfólio distribui esse total entre campanhas; depois os conjuntos são reconciliados para não ultrapassar o valor aprovado da campanha. Anúncios não recebem orçamento próprio e são tratados como manter, desligar, priorizar ou candidato a duplicação.

## Objetivo econômico

1. maximizar lucro esperado ajustado a risco;
2. entre soluções próximas do ótimo de lucro, favorecer maior faturamento;
3. faturamento adicional não pode justificar perda material de lucro;
4. respeitar orçamento, qualidade da base, evidência, PPC/MCMC e limites de scale.

A margem de contribuição é obrigatória.
Lucro econômico = receita × margem de contribuição − mídia.

## Planilha mínima

A base deve estar granularizada por dia. Campos obrigatórios:

- date
- campaign_id
- adset_id
- ad_id
- impressions
- clicks
- conversions
- spend
- revenue

O leitor reconhece aliases comuns de exports do Meta e nomes equivalentes em português.

## Campos fortemente recomendados

- campaign_name
- adset_name
- ad_name
- status
- campaign_daily_budget
- adset_daily_budget
- reach
- frequency
- landing_page_views
- adds_to_cart
- checkouts

Se orçamento diário não existir, a referência monetária operacional usa spend diário recente. O histórico completo continua sendo usado para inferência.

## Sobre duplicação

Duplicar é uma intervenção diferente de aumentar orçamento. Sem histórico experimental que identifique o efeito de clones, o sistema não afirma que duplicar é superior a escalar o ativo existente.

Nesses casos uma entidade forte aparece como TESTAR_DUPLICACAO. DUPLICAR é reservado a resposta experiment_calibrated.

## Execução no Windows

1. Instale uv uma vez: winget install --id=astral-sh.uv -e
2. Execute run_app_windows.bat na pasta do projeto.
3. A interface abre em localhost.
4. Envie CSV ou XLSX.
5. Informe margem de contribuição real, ROAS alvo e horizonte.
6. Execute a análise.
7. A primeira aba é Plano operacional.
8. Baixe operational_action_plan.csv.

Para o modo profundo, execute install_deep_windows.bat uma vez e depois selecione MCMC hierárquico profundo na interface.

Exemplo terminal profundo:
uv run quant-trafego-mcmc --input C:\dados\meta.xlsx --output output_mcmc --contribution-margin 0.40 --target-roas 2.0

## Rotina recomendada

- exportar dados diariamente ou em ciclos consistentes;
- manter a mesma definição de conversão e receita;
- informar margem real;
- analisar campanhas ativas;
- executar o plano;
- registrar as ações aplicadas;
- reanalisar com os novos resultados.

Registrar as ações tomadas é essencial para evoluir de associação observacional para estimativas de intervenção melhores.

A ferramenta não garante lucro futuro. Ela organiza a decisão para maximizar valor esperado e controlar risco de forma auditável.
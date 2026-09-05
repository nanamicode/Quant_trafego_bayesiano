# Changelog

## 0.9.0

- camada operacional responde aumentar, reduzir, manter, desligar e candidatos a duplicação;
- operational_action_plan.csv e operational_action_plan.md como saídas principais;
- alocação global de campanha substitui ótimo independente no plano operacional;
- valores diários usam orçamento explícito quando disponível e spend recente como fallback;
- campanha/conjunto/anúncio e respectivos nomes são preservados nas decisões;
- lucro continua objetivo primário; faturamento desempata apenas soluções próximas do ótimo;
- desempate por faturamento implementado também nos otimizadores MILP e CVaR;
- receita incremental vs manter passa a ser reportada;
- duplicação é separada de scale: TESTAR_DUPLICACAO para evidência observacional e DUPLICAR para evidência experimental calibrada;
- margem de contribuição passa a ser obrigatória na UI e nos CLIs de análise;
- template CSV operacional e guia de uso em docs/OPERATIONAL_USAGE.md;
- interface coloca Plano operacional como primeira aba.

## 0.8.0

- contrafactuais de orçamento passam a compartilhar os mesmos mundos latentes posteriores;
- P(ação > manter), P(ação ótima) e regret usam lucro condicional pareado, separado do ruído realizado usado em risco;
- seeds estáveis tornam resultados invariantes à ordem da grade de ações;
- elasticidade observacional gasto→conversões passa a controlar tendência linear/quadrática e dia da semana;
- confiança da resposta depende da variação de gasto identificável após os controles;
- sazonalidade semanal encolhida de CTR/CVR integrada ao horizonte futuro;
- baseline sazonal distinto para posterior agregado e estado corrente MCMC;
- qualidade da base passa a auditar calendário, gaps, tracking, numéricos, densidade e sobreposição de campanhas;
- qualidade da base reduz o score decisório e pode bloquear scale-up;
- correlações de portfólio passam a usar shrinkage por sobreposição temporal de cada par de campanhas;
- otimizadores MILP/CVaR passam a respeitar policy_eligible como restrição dura;
- NUTS não convergido cai para Empirical Bayes para decisões;
- PPC reprovado preserva posterior apenas para diagnóstico e bloqueia scale-up;
- ADVI aprovado continua explicitamente aproximado e recebe cap padrão de 1.2x;
- decision_confidence é mantido apenas como alias retrocompatível; a UI usa decision_score, explicitamente não calibrado;
- promoção de modelo temporal usa mediana robusta de melhorias, maioria de métricas e veto a regressão material;
- CI passa a testar o commit exato, compilar todo src e executar smoke dos CLIs.

## 0.7.0

- MCMC profundo migrado para painel diário;
- efeito temporal global Bayesiano via GaussianRandomWalk;
- posterior corrente por conta/campanha/conjunto/anúncio;
- posterior predictive checks em observações diárias;
- state-space Bayesiano rápido como candidato temporal;
- promoção de modelo governada por rolling-origin backtesting;
- Brier, ECE, reliability tables, cobertura e interval score;
- recomendação oficial separada do ótimo irrestrito;
- política de evidência limita scale-up sem sustentação suficiente;
- MILP exato para alocação discreta;
- portfólio com cenários correlacionados, cópula Gaussiana e CVaR;
- diagnóstico probabilístico opcional de funil completo;
- suporte a reach, frequency, LPV, ATC e checkout;
- core CI e deep-smoke PyMC separados;
- comandos de comparação de modelos e otimização.

## 0.6.0

- runtime oficial fixado em Python 3.12;
- gerenciamento de ambiente migrado para uv;
- DuckDB + Parquet como warehouse local embutido;
- hash canônico dos dados e run_manifest para reprodutibilidade;
- rolling-origin backtesting probabilístico;
- Brier score, calibration gap, cobertura e interval score;
- intervalos preditivos persistidos no Monte Carlo;
- decisões de arquitetura e causalidade documentadas;
- CLI específico de backtesting.

## 0.5.0

- MCMC hierárquico conectado diretamente à árvore de decisões;
- posteriores materializados em conta, campanha, conjunto e anúncio;
- moment matching das amostras MCMC para o simulador econômico;
- modo profundo disponível na interface local;
- NUTS/ADVI automático conforme tamanho da estrutura;
- diagnóstico de R-hat, ESS e divergências;
- semente temporal determinística entre execuções;
- CLI profundo fim a fim.

## 0.4.0

- análise hierárquica conta → campanha → conjunto → anúncio;
- derivadas temporais de CTR e CVR;
- comparação recente vs. histórico e score de mudança de regime;
- estimativa hierárquica de elasticidade observacional do gasto;
- Monte Carlo com probabilidade de cada ação ser ótima;
- lucro econômico com margem de contribuição;
- detecção de hardware e dimensionamento automático;
- modelo hierárquico profundo opcional em PyMC (NUTS/ADVI);
- diagnóstico estrutural da planilha.

# Changelog

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

# Quant Tráfego Bayesiano

Motor quantitativo Bayesiano local para tráfego pago.

Hierarquia principal:

`conta → campanha → conjunto → anúncio`

O objetivo é transformar planilhas históricas de mídia em inferência probabilística auditável, validação fora da amostra e decisões econômicas sob incerteza — sem servidor pago.

## Saída operacional principal

A v0.9 transforma a inferência em um plano executável. A primeira saída a consultar é `operational_action_plan.csv`. O capital é reconciliado de cima para baixo: a ação da conta define o envelope total, o solver distribui entre campanhas e um segundo solver distribui cada campanha entre seus conjuntos.

Ela responde por campanha, conjunto e anúncio:

- aumentar quanto;
- reduzir quanto;
- manter;
- desligar;
- quando existe evidência para testar/usar duplicação;
- impacto esperado em lucro e faturamento;
- probabilidades e risco da decisão.

Lucro é o objetivo primário. Entre soluções próximas do ótimo de lucro ajustado a risco, o motor favorece maior faturamento.

A margem de contribuição é obrigatória. O sistema não assume 100% silenciosamente.

Guia completo: `docs/OPERATIONAL_USAGE.md`.
Template de entrada: `examples/operational_input_template.csv`.

## Ambiente oficial

O runtime é **Python 3.12**.

O projeto usa **uv** para criar o ambiente, selecionar a versão correta do Python e manter dependências reproduzíveis.

No Windows, se uv ainda não estiver instalado:

```bat
winget install --id=astral-sh.uv -e
```

Depois:

```bat
run_app_windows.bat
```

A interface roda somente em `localhost`.

## Modos de inferência

### Hierárquico rápido

Empirical Bayes + partial pooling + Monte Carlo.

Uso principal:

- screening;
- análise diária;
- backtesting com muitas origens;
- diagnóstico rápido;
- desenvolvimento.

### MCMC hierárquico profundo

Instale uma vez:

```bat
install_deep_windows.bat
```

Depois selecione **MCMC hierárquico profundo** na interface ou rode:

```bat
uv run quant-trafego-mcmc --input "C:\dados\meta.xlsx" --output output_mcmc --contribution-margin 0.40 --target-roas 2.0
```

O modelo PyMC produz posteriores por conta, campanha, conjunto e anúncio. NUTS registra R-hat, ESS e divergências. NUTS não convergido não dirige capital: a decisão volta ao Empirical Bayes. ADVI permanece aproximação explícita e recebe cap conservador de scale. PPC reprovado preserva o posterior para diagnóstico, mas impede aumento de exposição.

## Validação probabilística

A v0.8 separa três perguntas diferentes:

- **o modelo converge numericamente?** → R-hat, ESS e divergências;
- **o modelo consegue reproduzir dados plausíveis?** → posterior predictive checks;
- **o modelo prevê melhor fora da amostra?** → rolling-origin + proper scores.


Backtesting temporal é parte da arquitetura, não um relatório opcional.

Exemplo:

```bat
uv run quant-trafego-backtest --input "C:\dados\meta.xlsx" --output backtest_output --horizon-days 7 --min-train-days 21
```

O protocolo rolling-origin:

1. treina somente com o passado disponível;
2. prevê a próxima janela;
3. compara a distribuição prevista com o que realmente ocorreu;
4. avança a origem;
5. repete.

Métricas atuais:

- Brier score de lucro;
- Brier score de ROAS alvo;
- calibration gap;
- Expected Calibration Error (ECE);
- reliability tables por faixa de probabilidade;
- cobertura do intervalo de 90%;
- largura/sharpness do intervalo;
- interval score;
- bias de lucro;
- MAE de lucro.

Esse backtest valida previsão sob a política observada. Ele **não** revela o contrafactual de uma ação de orçamento que não foi tomada.

## Causalidade

A elasticidade histórica gasto → conversões é tratada como **observacional**. A v0.8 controla tendência linear/quadrática e dia da semana e mede quanta variação de gasto ainda é identificável depois desses controles.

O sistema não deve transformar automaticamente correlação histórica em efeito causal de aumentar orçamento.

A arquitetura distingue níveis de evidência:

- `predictive`;
- `observational_intervention`;
- `experiment_calibrated`.

Decisões agressivas de alocação devem receber restrições mais conservadoras quando a resposta ao gasto não tiver calibração experimental.

## Modelos temporais

Existem dois modelos temporais no motor rápido:

- `derivative`: regressão derivativa local ponderada por recência;
- `state_space`: estado local-linear Bayesiano aproximado por filtro de Kalman no espaço logit.

O state-space é tratado como **candidato**. Para comparar:

```bat
uv run quant-trafego-compare-models --input "C:\dados\meta.xlsx"
```

Ele só deve substituir o modelo de referência quando passar os gates de backtesting, calibração e cobertura.

No modo profundo, o PyMC preserva observações diárias e adiciona um efeito temporal global `GaussianRandomWalk`. A decisão usa o posterior do estado temporal corrente.

CTR e CVR também recebem sazonalidade semanal encolhida quando há histórico suficiente. O forecast diferencia posterior agregado de posterior MCMC corrente para não contar duas vezes o efeito do último weekday.

## Política de decisão

O sistema mantém duas respostas:

- `unconstrained_best_multiplier`: ótimo matemático sem política;
- `action_multiplier`: recomendação permitida pelo nível de evidência.

Scale-ups são limitados quando a resposta ao gasto é apenas preditiva/observacional e exigem probabilidades mínimas de lucro e ganho incremental. Qualidade insuficiente da base também pode limitar ou bloquear aumento de exposição.

As ações da mesma entidade são comparadas em **mundos latentes compartilhados** de CTR, CVR, CPM, AOV, elasticidade e tendência. P(ação > manter), P(ação ótima) e regret usam lucro condicional pareado; CVaR e P(lucro) continuam usando risco realizado.

`decision_score` é um score composto de governança entre 0 e 1. **Não é uma probabilidade calibrada.** O antigo campo `decision_confidence` permanece apenas como alias retrocompatível.

## Otimização global

A grade discreta atual é resolvida exatamente por MILP:

```bat
uv run quant-trafego-optimize --actions output\all_actions.csv
```

Quando o histórico original também está disponível, o modo preferido usa cenários correlacionados e CVaR:

```bat
uv run quant-trafego-portfolio --actions output\all_actions.csv --history "C:\dados\meta.xlsx" --contribution-margin 0.40
```

Esse portfólio usa correlação histórica encolhida par a par pela quantidade real de dias de coexistência + cópula Gaussiana para dependência estatística. O otimizador respeita `policy_eligible` como restrição dura e não pode reabrir uma ação bloqueada pelo motor. Isso melhora risco conjunto, mas **não é uma estimativa causal de canibalização de leilão**.

## Funil opcional

Quando a planilha contém colunas como reach, frequency, landing page views, add-to-cart e checkout, elas são preservadas. O sistema constrói transições probabilísticas e intervalos posteriores por conta, campanha, conjunto e anúncio, além de apontar violações de tracking.

## Warehouse local e auditoria

Cada execução pode ser persistida localmente em:

```text
workspace/
├── quant_trafego.duckdb
├── snapshots/
│   └── <sha256>.parquet
└── runs/
    └── <run_id>/
        ├── run_manifest.json
        ├── all_actions.csv
        └── best_actions.csv
```

O snapshot da planilha é deduplicado por SHA-256.

O `run_manifest.json` registra:

- hash dos dados;
- configuração;
- seed;
- versão do pacote;
- commit Git quando disponível;
- Python;
- versões científicas;
- hardware;
- método de inferência;
- timestamp UTC;
- diagnósticos adicionais.

## Componentes quantitativos implementados

- posterior hierárquico diário de CTR e CVR;
- partial pooling conta → campanha → conjunto → anúncio;
- derivadas temporais no espaço logit;
- comparação recente × histórico;
- mudança de regime;
- instabilidade;
- elasticidade observacional de resposta ao gasto controlada por tendência/weekday;
- shrinkage hierárquico da elasticidade;
- fallback Hill;
- Monte Carlo econômico com contextos latentes compartilhados entre ações;
- contrafactuais pareados para ação ótima, incremental e regret;
- sazonalidade semanal encolhida de CTR/CVR;
- intervalos preditivos;
- margem de contribuição;
- P(lucro);
- P(ruína);
- P(ROAS ≥ meta);
- P(ação > manter);
- P(ação ótima entre alternativas);
- lucro incremental esperado;
- VaR;
- CVaR;
- expected regret;
- utilidade ajustada a risco;
- decision score explicitamente não calibrado como probabilidade;
- guardrails de qualidade da base;
- fallback profundo quando NUTS/PPC não sustentam ação;
- autodetecção de CPU/RAM;
- qualidade estrutural da base;
- posterior predictive checks;
- state-space temporal candidato;
- rolling-origin backtesting;
- reliability tables e ECE;
- política de decisão baseada em evidência;
- alocação MILP exata;
- portfólio correlacionado com CVaR e shrinkage por overlap par a par;
- diagnóstico probabilístico de funil opcional;
- persistência DuckDB/Parquet;
- manifestos reproduzíveis.

## Lucro econômico

```text
lucro = receita × margem de contribuição − gasto de mídia
```

O motor não usa faturamento como sinônimo de lucro.

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

CSV e XLSX são aceitos.

## Ferramentas e papéis

- **PyMC**: inferência Bayesiana profunda;
- **ArviZ**: diagnósticos;
- **DuckDB/Parquet**: armazenamento analítico local;
- **PyMC-Marketing**: MMM, adstock, saturação, incrementality e calibração quando o nível agregado for apropriado;
- **Meridian**: modelo independente de MMM agregado/calibração, não núcleo ad-level;
- **BoTorch/Ax**: futura otimização global sob restrições, somente depois de validar a função de resposta;
- **MABWiser**: futura exploração sequencial quando existir feedback recorrente.

As decisões metodológicas e alternativas rejeitadas ficam registradas em `docs/ARCHITECTURE_DECISIONS.md`.

> Uma técnica só deve virar padrão se melhorar calibração/score fora da amostra de forma consistente e com custo computacional justificável.

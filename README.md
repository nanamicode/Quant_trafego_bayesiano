# Quant Tráfego Bayesiano

Motor quantitativo Bayesiano local para tráfego pago.

Hierarquia principal:

`conta → campanha → conjunto → anúncio`

O objetivo é transformar planilhas históricas de mídia em inferência probabilística auditável, validação fora da amostra e decisões econômicas sob incerteza — sem servidor pago.

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

O modelo PyMC produz posteriores por conta, campanha, conjunto e anúncio. NUTS registra R-hat, ESS e divergências. ADVI é usado apenas como fallback aproximado para estruturas grandes e permanece marcado como aproximação.

## Validação probabilística

A v0.7 separa três perguntas diferentes:

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

A elasticidade histórica gasto → conversões é tratada como **observacional**.

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

## Política de decisão

O sistema mantém duas respostas:

- `unconstrained_best_multiplier`: ótimo matemático sem política;
- `action_multiplier`: recomendação permitida pelo nível de evidência.

Scale-ups são limitados quando a resposta ao gasto é apenas preditiva/observacional e exigem probabilidades mínimas de lucro e ganho incremental.

## Otimização global

A grade discreta atual é resolvida exatamente por MILP:

```bat
uv run quant-trafego-optimize --actions output\all_actions.csv
```

Quando o histórico original também está disponível, o modo preferido usa cenários correlacionados e CVaR:

```bat
uv run quant-trafego-portfolio --actions output\all_actions.csv --history "C:\dados\meta.xlsx" --contribution-margin 0.40
```

Esse portfólio usa correlação histórica encolhida + cópula Gaussiana para dependência estatística. Isso melhora risco conjunto, mas **não é uma estimativa causal de canibalização de leilão**.

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
- elasticidade observacional de resposta ao gasto;
- shrinkage hierárquico da elasticidade;
- fallback Hill;
- Monte Carlo econômico;
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
- decisão com score de confiança;
- autodetecção de CPU/RAM;
- qualidade estrutural da base;
- posterior predictive checks;
- state-space temporal candidato;
- rolling-origin backtesting;
- reliability tables e ECE;
- política de decisão baseada em evidência;
- alocação MILP exata;
- portfólio correlacionado com CVaR;
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

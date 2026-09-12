# i-ai-project

Projeto sobre o dataset **Student Performance Factors** com quatro partes:

- **Análise de correlação (Pearson)**: conversão de colunas categóricas em
  valores numéricos, construção da matriz de correlação e geração de gráficos
  (barras, heatmap e dispersão com regressão linear).
- **Treino de modelo**: pré-processamento dos dados (split, imputação,
  encoding ordinal/one-hot, normalização z-score) e regressão linear
  implementada pela Equação Normal, com avaliação por RMSE e R² e gráficos de
  resultado.
- **Análise de peso (weights)**: gráfico de barras com os pesos finais
  aprendidos pela regressão linear e gráficos da evolução dos pesos e da
  convergência do custo (MSE) no gradiente descendente.
- **Análise de desbalanceamento**: estatísticas e distribuição da
  variável-alvo `Exam_Score` (histograma com KDE, média, mediana e faixa
  interquartil).

## Pré-requisitos

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes usado pelo projeto)

## Instalação

Na raiz do projeto:

```bash
uv sync
```

Isso cria o `.venv` e instala as dependências do `pyproject.toml` (numpy,
pandas, scikit-learn e matplotlib).

## Como rodar

### Análise de correlação

A partir da raiz do projeto:

```bash
uv run python src/i_ai_project/main.py
```

Ou, a partir da pasta do pacote:

```bash
cd src/i_ai_project
uv run python main.py
```

O script imprime as informações do DataFrame no terminal e abre janelas do
matplotlib com os gráficos. Feche cada janela para que a execução continue.

### Treino do modelo

O script `traine` (registrado em `[project.scripts]` no `pyproject.toml`) treina
um modelo e imprime as métricas no terminal:

```bash
uv run traine linear_regression
```

O tipo de treino é o primeiro argumento; se omitido, usa `linear_regression`.
Tipos desconhecidos são rejeitados com a lista de opções válidas.

O fluxo é: `prepare_data()` em `utils/data_prep.py` faz o pré-processamento
(split 80/20 com `random_state=42`, imputação de nulos pela mediana do treino,
encoding ordinal manual, one-hot nas nominais e normalização z-score com
estatísticas do treino) e retorna `X_train`, `X_test`, `y_train`, `y_test`.
O modelo (`LinearRegressionEquation`) é treinado pela **Equação Normal** com
pseudoinversa e avaliado com RMSE e R² calculados manualmente (sem sklearn).
Por fim, são gerados dois gráficos (real vs predito e curva de aprendizado),
salvos em `figures/resultado_<tipo>.png`.

### Análise de peso (weights)

Dois gráficos analisam os pesos aprendidos pelos modelos:

- **Pesos finais da regressão linear** (`analysis/linear_regression_analysis.py`):
  treina o modelo pela Equação Normal, ordena as features por magnitude dos
  pesos (`get_feature_weights`) e plota um gráfico de barras horizontais com
  a linha de referência zero (`plot_feature_weights`), salvo em
  `figures/feature_weights.png`.
- **Evolução dos pesos no gradiente descendente** (`analysis/gradient_descent_analysis.py`):
  `plot_weights_evolution` mostra como cada peso varia ao longo das épocas de
  treino e `plot_cost_convergence` compara a convergência do MSE para
  diferentes learning rates. Os gráficos são salvos em
  `figures/weights_evolution.png` e `figures/cost_convergence.png`.

```bash
uv run python src/i_ai_project/analysis/linear_regression_analysis.py
uv run python src/i_ai_project/main_gradient.py
```

### Análise de desbalanceamento

`analysis/imbalance.py` analisa a variável-alvo `Exam_Score`: calcula
estatísticas (valores válidos/ausentes, média, mediana, desvio-padrão,
assimetria, mínimo, máximo e quartis), imprime a frequência de cada nota e
plota a distribuição — histograma com curva KDE, linhas de média e mediana e
faixa dos 50% centrais. O gráfico é exibido em janela interativa do
matplotlib (não é salvo automaticamente).

```bash
uv run python src/i_ai_project/analysis/imbalance.py
```

### Em ambiente sem interface gráfica (opcional)

Se estiver em servidor/SSH sem display, rode com o backend não interativo:

```bash
MPLBACKEND=Agg uv run python src/i_ai_project/main.py
# ou, para o treino:
MPLBACKEND=Agg uv run traine linear_regression
# para as análises de peso e desbalanceamento:
MPLBACKEND=Agg uv run python src/i_ai_project/analysis/linear_regression_analysis.py
MPLBACKEND=Agg uv run python src/i_ai_project/main_gradient.py
MPLBACKEND=Agg uv run python src/i_ai_project/analysis/imbalance.py
```

## Arquivos de dados

O código resolve os caminhos subindo até a **raiz do projeto** (`utils/paths.py`),
portanto estes arquivos precisam estar lá:

- `Student Performance Factors.csv` — dataset de desempenho estudantil
- `encodings.json` — mapeamento dos valores categóricos para números

## Estrutura

```
i-ai-project/
├── Student Performance Factors.csv   # dataset
├── encodings.json                    # mapeamento categórico → numérico
├── pyproject.toml
├── src/i_ai_project/
│   ├── main.py                       # ponto de entrada (análise de correlação)
│   ├── main_gradient.py              # ponto de entrada (gráficos do gradiente descendente)
│   ├── data/                         # leitura de CSV/JSON e conversão numérica
│   ├── analysis/                     # correlação, pesos, gradiente descendente e desbalanceamento
│   │   ├── imbalance.py              # análise de desbalanceamento de Exam_Score
│   │   ├── linear_regression_analysis.py  # gráfico de pesos finais (Equação Normal)
│   │   └── gradient_descent_analysis.py   # evolução dos pesos e convergência do custo
│   ├── traine/                       # treino de modelos
│   │   ├── run_traine.py             # script: prepara dados, treina, avalia e plota
│   │   └── linear_regression.py      # regressão linear pela Equação Normal
│   └── utils/
│       ├── paths.py                  # resolução de caminhos
│       └── data_prep.py              # pré-processamento (split, encoding, z-score)
└── .venv/                            # criado pelo uv sync
```

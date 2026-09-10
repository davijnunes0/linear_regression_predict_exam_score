# i-ai-project

Projeto sobre o dataset **Student Performance Factors** com duas partes:

- **Análise de correlação (Pearson)**: conversão de colunas categóricas em
  valores numéricos, construção da matriz de correlação e geração de gráficos
  (barras, heatmap e dispersão com regressão linear).
- **Treino de modelo**: pré-processamento dos dados (split, imputação,
  encoding ordinal/one-hot, normalização z-score) e regressão linear
  implementada pela Equação Normal, com avaliação por RMSE e R² e gráficos de
  resultado.

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
salvos em `resultado_<tipo>.png`.

### Em ambiente sem interface gráfica (opcional)

Se estiver em servidor/SSH sem display, rode com o backend não interativo:

```bash
MPLBACKEND=Agg uv run python src/i_ai_project/main.py
# ou, para o treino:
MPLBACKEND=Agg uv run traine linear_regression
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
│   ├── data/                         # leitura de CSV/JSON e conversão numérica
│   ├── analysis/                     # correlação de Pearson, matriz e gráficos
│   ├── traine/                       # treino de modelos
│   │   ├── run_traine.py             # script: prepara dados, treina, avalia e plota
│   │   └── linear_regression.py      # regressão linear pela Equação Normal
│   └── utils/
│       ├── paths.py                  # resolução de caminhos
│       └── data_prep.py              # pré-processamento (split, encoding, z-score)
└── .venv/                            # criado pelo uv sync
```

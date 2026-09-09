# i-ai-project

Análise de correlação (Pearson) sobre o dataset **Student Performance Factors**:
conversão de colunas categóricas em valores numéricos, construção da matriz de
correlação e geração de gráficos (barras, heatmap e dispersão com regressão linear).

## Pré-requisitos

- Python **3.13+**
- [uv](https://docs.astral.sh/uv/) (gerenciador de pacotes usado pelo projeto)

## Instalação

Na raiz do projeto:

```bash
uv sync
```

Isso cria o `.venv` e instala as dependências do `pyproject.toml` (numpy, pandas).

> **Atenção:** o `matplotlib` (usado nos gráficos) ainda **não está listado** no
> `pyproject.toml`. Instale-o antes de rodar:
>
> ```bash
> uv pip install matplotlib
> ```
>
> (ou `uv add matplotlib` para adicioná-lo permanentemente ao projeto).

## Como rodar

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

### Em ambiente sem interface gráfica (opcional)

Se estiver em servidor/SSH sem display, rode com o backend não interativo:

```bash
MPLBACKEND=Agg uv run python src/i_ai_project/main.py
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
│   ├── main.py                       # ponto de entrada
│   ├── data/                         # leitura de CSV/JSON e conversão numérica
│   ├── analysis/                     # correlação de Pearson, matriz e gráficos
│   └── utils/paths.py                # resolução de caminhos
└── .venv/                            # criado pelo uv sync
```

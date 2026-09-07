from pathlib import Path
import json
import csv
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

def main():

    DATA_PATH = path_resolve(2, "Student Performance Factors.csv")
    data_frame = pd.read_csv(DATA_PATH)
    data_frame.info()

    print()

    encodings = load_encodings(path_resolve(2, "encodings.json"))
    print()


    rows = read_csv(DATA_PATH)
    numeric_columns = convert_to_numeric(
        rows,
        encodings
    )

    print(numeric_columns)



def path_resolve(parent_number: int, file: str) -> Path:
    # Sobe `parent_number` níveis a partir deste arquivo e anexa o nome do arquivo alvo
    return Path(__file__).resolve().parents[parent_number] / file


def generation_of_correlation_matrix(encodings: dict[str, Any], file: str):
    pass

def load_encodings(path: Path) -> dict[str, Any]:
    # encoding="utf-8" garante que caracteres acentuados do JSON sejam lidos corretamente
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)

def read_csv(path: str) -> list[dict[str,str]]:
    """
    Lê um arquivo CSV e retorna as linhas em forma de dicionário.

    Args:
        path: Caminho para o arquivo CSV.

    Returns:
        Uma lista onde cada item representa uma linha do CSV.
        Cada dicionário tem chaves que correspondem aos nomes das colunas.
    """

    # utf-8-sig descarta o BOM, caso o arquivo tenha sido salvo com ele
    with open(path, mode="r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader)



def to_numeric(column: str, value: str, encodings: dict[str, dict[str, int | float]]) -> float:
    """
    Transforma qualquer valor lido do CSV em número, para que ele possa
    entrar no cálculo da correlação.

    Args:
        column: Nome da coluna, por exemplo "Family_Income".
        value: Valor lido do CSV, por exemplo "High".
        encodings: Dicionário de mapeamento carregado do JSON.

    Exemplo de encodings:
        {
            "Family_Income": {
                "Low": 0,
                "Medium": 1,
                "High": 2
            }
        }
    """
    value = value.strip()

    # Célula vazia vira NaN para não quebrar o cálculo da correlação
    if value == "":
        return np.nan

    if column in encodings:
        # Valor categórico: usa o mapeamento; categorias desconhecidas viram NaN
        return float(
            encodings[column].get(
                value,
                np.nan
            )
        )

    try:
        return float(value)
    except ValueError:
        # Nem número nem categoria conhecida → tratado como dado ausente
        return np.nan

def convert_to_numeric(rows: list[dict[str,str]], encodings: dict[str,dict[str, int | float]]) -> dict[str, np.ndarray]:
    """
    Converte todo o CSV para arrays com números inteiros e/ou flutuantes.

    Args:
        rows: Linhas lidas do arquivo CSV.
        encodings: Mapeia categorias para valores numéricos.

    Returns:
        Um dicionário em que cada chave é o nome de uma coluna e cada valor
        é um array NumPy contendo os valores numéricos dessa coluna.
    """

    columns = rows[0].keys()

    numeric_columns = {}

    for column in columns:
        values = []
        for row in rows:
            numeric_value = to_numeric(
                column,
                row[column],
                encodings
            )

            values.append(numeric_value)

        # dtype=float permite que o NaN conviva com os números no mesmo array
        numeric_columns[column] = np.asarray(
            values,
            dtype=float
        )

    # Retorno fora do loop: só devolve o resultado após converter todas as colunas
    return numeric_columns

if __name__ == "__main__":
    main()



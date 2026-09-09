import numpy as np


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


def convert_to_numeric(rows: list[dict[str, str]], encodings: dict[str, dict[str, int | float]]) -> dict[str, np.ndarray]:
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
        # print(f"{column=}")
        # print()
        values = []
        for row in rows:
           #  print(f"{row=}")
           # print()
            numeric_value = to_numeric(
                column,
                row[column],
                encodings
            )

            # print(f"{numeric_value=}")
            # print()
            values.append(numeric_value)

        # dtype=float permite que o NaN conviva com os números no mesmo array
        numeric_columns[column] = np.asarray(
            values,
            dtype=float
        )

        # print(f"{numeric_columns[column]=}")
        # print()
    # Retorno fora do loop: só devolve o resultado após converter todas as colunas
    return numeric_columns

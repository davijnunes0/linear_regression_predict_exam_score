from pathlib import Path
import pandas as pd


def path_resolve(parent_number: int, path: str) -> Path:
    """ Sobe 'parent_number' níveis a partir deste arquivo e anexa 'path'

        Args:
            parent_number: Quantidade de niveis a subir.
            path: Caminho de arquivo alvo.

        Returns:
            Caminho absoluto resolvido ('Path').

    """
    return Path(__file__).resolve().parents[parent_number + 1] / path


def read_dataset(path: str) -> pd.DataFrame:
    """Lê um CSV e retorna um DataFrame.

    Args:
        path: Caminho do Arquivo CSV.
    """
    return pd.read_csv(path)

import csv
import json
from pathlib import Path
from typing import Any


def load_encodings(path: Path) -> dict[str, Any]:
    # encoding="utf-8" garante que caracteres acentuados do JSON sejam lidos corretamente
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def read_csv(path: str) -> list[dict[str, str]]:
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

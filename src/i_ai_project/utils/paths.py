from pathlib import Path


def path_resolve(parent_number: int, file: str) -> Path:
    # Sobe `parent_number` níveis a partir deste arquivo e anexa o nome do arquivo alvo
    return Path(__file__).resolve().parents[parent_number + 1] / file

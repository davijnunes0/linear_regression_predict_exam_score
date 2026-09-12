from typing import Any

import numpy as np


def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """
        Calcule o coeficiente de correlação de Pearson entre dois arrays númericos.

        Args:
            x: Valores númericos da primeira variável.
            y: Valores númericos  da segunda variável.


        Returns:
            Coeficiente de correlação de Pearson entre x e y
            Retorna 'valor ausente' caso a correlação não possa ser calculada.
    """

    # Cria uma máscara booleana que mantém apenas posições onde x e y são números finitos.
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]

    # Verificar se há dados suficiente
    n = len(x_valid)
    if n < 2:
        return np.nan

    # Calcular as médias
    x_mean = np.sum(x_valid) / n
    y_mean = np.sum(y_valid) / n


    # Calcular os desvios
    x_deviation = x_valid - x_mean
    y_deviation = y_valid - y_mean

    numerator = np.sum(x_deviation * y_deviation)

    denominator = np.sqrt(np.sum(x_deviation ** 2) * np.sum(y_deviation ** 2))

    if denominator == 0:
        raise ZeroDivisionError("Divisão por zero é inválido.")

    return float(numerator / denominator)

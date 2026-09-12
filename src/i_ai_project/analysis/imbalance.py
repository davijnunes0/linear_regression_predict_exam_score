from dataclasses import dataclass

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from i_ai_project.utils.paths import path_resolve


TARGET_COLUMN = "Exam_Score"
DATASET_NAME = "Student Performance Factors.csv"


@dataclass(frozen=True)
class TargetStatistics:
    """
    Armazena as estatísticas calculadas para a variável-alvo.

    Attributes:
        count: Quantidade de valores numéricos válidos.
        missing_values: Quantidade de valores ausentes ou inválidos.
        average: Média aritmética das notas.
        median: Valor central da distribuição.
        standard_deviation: Desvio-padrão das notas.
        asymmetry: Coeficiente de assimetria da distribuição.
        minimum: Menor nota encontrada.
        maximum: Maior nota encontrada.
        first_quartile: Valor abaixo do qual estão 25% das notas.
        third_quartile: Valor abaixo do qual estão 75% das notas.
    """

    count: int
    missing_values: int
    average: float
    median: float
    standard_deviation: float
    asymmetry: float
    minimum: float
    maximum: float
    first_quartile: float
    third_quartile: float


def load_dataset() -> pd.DataFrame:
    """
    Carrega o conjunto de dados de desempenho dos estudantes.

    O caminho é construído pela função path_resolve(), considerando
    o arquivo definido em DATASET_NAME.

    Returns:
        pd.DataFrame: Tabela com todas as linhas e colunas do dataset.

    Raises:
        FileNotFoundError: Se o arquivo CSV não for encontrado.
        pd.errors.ParserError: Se o arquivo não puder ser interpretado.
    """
    dataset_path = path_resolve(2, DATASET_NAME)
    return pd.read_csv(dataset_path)


def show_dataset_overview(df: pd.DataFrame) -> None:
    """
    Exibe uma visão geral do conjunto de dados.

    Args:
        df: DataFrame que será examinado.

    Returns:
        None: A função apenas imprime informações no terminal.
    """
    print("Dimensão do dataset:", df.shape)

    print("\nColunas disponíveis:")
    print(df.columns.tolist())

    print("\nPrimeiras linhas:")
    print(df.head())


def get_target(
    df: pd.DataFrame,
    column: str = TARGET_COLUMN,
) -> pd.Series:
    """
    Extrai e prepara a variável-alvo da regressão.

    Os valores são convertidos para números. Valores ausentes ou textos
    que não possam ser convertidos são removidos.

    Args:
        df: DataFrame que contém os dados.
        column: Nome da coluna-alvo. Por padrão, utiliza Exam_Score.

    Returns:
        pd.Series: Série contendo somente notas numéricas válidas.

    Raises:
        ValueError: Se a coluna não existir ou não possuir valores
            numéricos válidos.
    """
    if column not in df.columns:
        raise ValueError(
            f"A coluna '{column}' não foi encontrada no dataset."
        )

    numeric_target = pd.to_numeric(
        df[column],
        errors="coerce",
    )

    target = numeric_target.dropna()

    if target.empty:
        raise ValueError(
            f"A coluna '{column}' não possui valores numéricos válidos."
        )

    return target


def get_statistics(
    target: pd.Series,
    total_records: int,
) -> TargetStatistics:
    """
    Calcula as estatísticas da variável-alvo.

    Args:
        target: Série contendo somente valores numéricos válidos.
        total_records: Quantidade total de registros existentes no
            DataFrame antes da remoção dos valores inválidos.

    Returns:
        TargetStatistics: Objeto contendo contagem, valores ausentes,
        média, mediana, desvio-padrão, assimetria, mínimo, máximo e
        quartis.
    """
    return TargetStatistics(
        count=int(target.count()),
        missing_values=int(total_records - target.count()),
        average=float(target.mean()),
        median=float(target.median()),
        standard_deviation=float(target.std()),
        asymmetry=float(target.skew()),
        minimum=float(target.min()),
        maximum=float(target.max()),
        first_quartile=float(target.quantile(0.25)),
        third_quartile=float(target.quantile(0.75)),
    )


def show_target_statistics(
    target: pd.Series,
    statistics: TargetStatistics,
) -> None:
    """
    Exibe as estatísticas e frequências da variável-alvo.

    Args:
        target: Série numérica com as notas válidas.
        statistics: Objeto retornado por get_statistics().

    Returns:
        None: Os resultados são apenas impressos no terminal.
    """
    print("\nResumo retornado por describe():")
    print(target.describe())

    print("\nEstatísticas calculadas:")
    print(f"Valores válidos: {statistics.count}")
    print(f"Valores ausentes: {statistics.missing_values}")
    print(f"Média: {statistics.average:.4f}")
    print(f"Mediana: {statistics.median:.4f}")
    print(
        "Desvio-padrão: "
        f"{statistics.standard_deviation:.4f}"
    )
    print(f"Assimetria: {statistics.asymmetry:.4f}")
    print(f"Mínimo: {statistics.minimum:.2f}")
    print(f"Máximo: {statistics.maximum:.2f}")
    print(
        "Primeiro quartil: "
        f"{statistics.first_quartile:.2f}"
    )
    print(
        "Terceiro quartil: "
        f"{statistics.third_quartile:.2f}"
    )

    print("\nFrequência de cada nota:")
    print(target.value_counts().sort_index())


def plot_target_distribution(
    target: pd.Series,
    statistics: TargetStatistics,
    column: str = TARGET_COLUMN,
) -> tuple[Figure, Axes]:
    """
    Constrói o gráfico de distribuição da variável-alvo.

    Args:
        target: Série contendo as notas numéricas válidas.
        statistics: Estatísticas calculadas por get_statistics().
        column: Nome da variável apresentado no título do gráfico.

    Returns:
        tuple[Figure, Axes]:
            Figure é a figura completa criada pelo Matplotlib.
            Axes é a área do gráfico, útil para alterações posteriores.
    """
    figure, axis = plt.subplots(figsize=(12, 6))

    axis.axvspan(
        statistics.first_quartile,
        statistics.third_quartile,
        color="green",
        alpha=0.10,
        zorder=0,
    )

    sns.histplot(
        target,
        bins=20,
        kde=True,
        stat="count",
        color="royalblue",
        edgecolor="black",
        alpha=0.50,
        line_kws={
            "linewidth": 2.2,
            "color": "blue",
        },
        ax=axis,
    )

    axis.axvline(
        statistics.average,
        color="red",
        linestyle="--",
        linewidth=2,
    )

    axis.axvline(
        statistics.median,
        color="orange",
        linestyle="--",
        linewidth=2,
    )

    legend_elements = [
        Patch(
            facecolor="royalblue",
            edgecolor="black",
            alpha=0.50,
            label="Barras: frequência das notas",
        ),
        Line2D(
            [0],
            [0],
            color="blue",
            linewidth=2.2,
            label="Curva: formato da distribuição",
        ),
        Line2D(
            [0],
            [0],
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Média: {statistics.average:.2f}",
        ),
        Line2D(
            [0],
            [0],
            color="orange",
            linestyle="--",
            linewidth=2,
            label=f"Mediana: {statistics.median:.2f}",
        ),
        Patch(
            facecolor="green",
            alpha=0.10,
            label=(
                "50% central: "
                f"{statistics.first_quartile:.2f} a "
                f"{statistics.third_quartile:.2f}"
            ),
        ),
    ]

    axis.legend(
        handles=legend_elements,
        title="Elementos do gráfico",
        loc="upper right",
        fontsize=9,
    )

    statistics_text = (
        f"Observações: {statistics.count}\n"
        f"Mínimo: {statistics.minimum:.2f}\n"
        f"Máximo: {statistics.maximum:.2f}\n"
        f"Desvio-padrão: "
        f"{statistics.standard_deviation:.2f}\n"
        f"Assimetria: {statistics.asymmetry:.4f}"
    )

    axis.text(
        0.98,
        0.60,
        statistics_text,
        transform=axis.transAxes,
        horizontalalignment="right",
        verticalalignment="top",
        fontsize=9,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "gray",
            "alpha": 0.90,
        },
    )

    axis.set_title(
        f"Distribuição da variável {column}",
        fontsize=14,
        fontweight="bold",
    )
    axis.set_xlabel("Nota no exame")
    axis.set_ylabel("Quantidade de estudantes")
    axis.grid(axis="y", linestyle=":", alpha=0.30)

    figure.tight_layout()
    plt.show()

    return figure, axis


def imbalance_apply(
) -> tuple[pd.DataFrame, pd.Series, TargetStatistics]:
    """
    Executa a análise de desbalanceamento de Exam_Score.

    A execução contempla:

    1. leitura do dataset;
    2. apresentação das informações gerais;
    3. validação e preparação da variável-alvo;
    4. cálculo das estatísticas;
    5. impressão dos resultados;
    6. criação do gráfico.

    Returns:
        tuple:
            pd.DataFrame: Dataset completo.
            pd.Series: Variável Exam_Score sem valores inválidos.
            TargetStatistics: Estatísticas calculadas para Exam_Score.
    """
    df = load_dataset()

    show_dataset_overview(df)

    target = get_target(df)

    statistics = get_statistics(
        target=target,
        total_records=len(df),
    )

    show_target_statistics(
        target=target,
        statistics=statistics,
    )

    plot_target_distribution(
        target=target,
        statistics=statistics,
    )

    return df, target, statistics


if __name__ == "__main__":
    dataframe, target_values, target_statistics = imbalance_apply()

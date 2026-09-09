import numpy as np
from i_ai_project.analysis.statistics import pearson_correlation
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

def build_correlation_matrix(numeric_columns: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    """
        Constroi uma matriz de correlação usando numeric_columns.

        Args:
            numeric_columns:
                Dictionary where each key is a column name and each
                value is a NumPy Array containing the numeric values
                of that column.

        Returns:
            A tuple containing:
            - A NumPy Matrix with the Pearson correlation coefficientes.
            - A list containing the column names in the same order.
            used by the matrix.
    """

    columns = list(numeric_columns.keys())

    number_of_columns = len(columns)

    correlation_matrix = np.empty(
        (
            number_of_columns,
            number_of_columns
        ),
    dtype=float
    )

    for i, column_x in enumerate(columns):
        for j, column_y in enumerate(columns):
            correlation_matrix[i,j] = pearson_correlation(
                numeric_columns[column_x],
                numeric_columns[column_y]
            )


    return correlation_matrix, columns


def get_target_correlations(correlation_matrix: np.ndarray, columns: list[str], target: str) -> dict[str, float]:
    """
        Extract and sort correlations with a target variable.

        Args:
            correlation_matrix:
                Perason correlation matrix.

            columns:
                columns name in the same order as the amtrix.

            target:
                Name of the target variable.

        Returns:
            Correlations sorted by absolute magnitude,
            from strongest to weakest.
    """
    # Encontra a posição alvo

    target_index = columns.index(target)

    correlations = {}


    # Itera sobre as colunas
    # Pega o valor da célula [i, target_index] -- ou seja, a correlação entre a coluna atual e o alvo
    for i, column in enumerate(columns):
        if column == target:
            continue


        correlations[column] = float(
            correlation_matrix[i, target_index]
        )



    return dict(
        sorted(
            correlations.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )
    )


def plot_target_correlations(correlations: dict[str,float], target: str) -> None:

    """
        Plot the correlation values between each attribute
        and a target variable.

        Args:
            correlations:
                Dictionary containing attribute names and their
                Pearson correlation values with the target
            target:
                Name of the target variable.

        Returns:
            None.
    """

    # Extrai os nomes dos atributos e seus respectivos valores de correlação em duas listas separadas, para usar no gráfico.
    attributes = list(correlations.keys())
    values = list(correlations.values())

    # Configuração da figura
    # Cria uma nova figura com dimensão 10x7 polegadas (largura x altura). Isso garante espaço suficiente para os rótulos dos atributos.
    plt.figure(figsize=(10,7))


    # Gráfico de barras horizontais
    # barh = bar horizontal (barras horizontaias);
    # Cada atributo fica no eixo Y (vertical)
    # O valor da correlação fica no eixo X (horizontal).
    plt.barh(attributes,values)

    # Linha de referência zero
    # Barras para a direita -> correlações positivas
    # Barras para a esquerda -> correlações negativas
    plt.axvline(x=0, linewidth=1)

    plt.xlabel("Pearson correlation coefficient")   # legenda do eixo X
    plt.ylabel("Attributes")                        # legenda do eixo Y
    plt.title(f"Correlation with {target}")         # título dinâmico

    # Ajusta automaticamente os espaçamentos para que nad afique cortado (título, rótulos, etc).
    plt.tight_layout()
    # renderiza e exibe o gráfico na tela.
    plt.show()


def plot_correlation_matrix(correlation_matrix: np.ndarray, columns: list[str], target: str) -> None:
    """
        Plot a hetmap of the Pearson correlation matrix.

        Args:
            correlation_matrix:
                Matrix containing Pearson correlation coefficients.

            columns:
                Columns name in the same order used by the matrix.

            target:
                Name of the target variable to highligh.

        Returns:
            None
    """

    number_of_columns = len(columns)

    norm = TwoSlopeNorm(
        vmin=-1.0,
        vcenter=0.0,
        vmax=1.0
    )

    fig, ax = plt.subplots(
        figsize=(16,14)
    )

    image = ax.imshow(
        correlation_matrix,
        cmap="coolwarm",
        norm=norm,
        aspect="equal"
    )

    ax.set_xticks(
        np.arange(number_of_columns)
    )

    ax.set_yticks(
        np.arange(number_of_columns)
    )

    ax.set_xticklabels(
        columns,
        rotation=55,
        ha="right",
        fontsize=9
    )

    ax.set_yticklabels(
        columns,
        fontsize=9
    )

    ax.set_title(
        "Correlation Matrix",
        fontsize=16,
        pad=18
    )


    # Add correlation values inside cells
    for i in range(number_of_columns):

        for j in range(number_of_columns):

            value = correlation_matrix[i, j]

            if np.isfinite(value):

                text_color = (
                    "white"
                    if abs(value) >= 0.5
                    else "black"
                )

                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color=text_color
                )

    # Highlight the target variable
    target_index = columns.index(target)

    ax.add_patch(
        Rectangle(
            (-0.5, target_index - 0.5),
            number_of_columns,
            1,
            fill=False,
            linewidth=2
        )
    )

    ax.add_patch(
        Rectangle(
            (target_index - 0.5, -0.5),
            1,
            number_of_columns,
            fill=False,
            linewidth=2
        )
    )

    colorbar = fig.colorbar(
        image,
        ax=ax
    )

    colorbar.set_label(
        "Pearson correlation coefficient"
    )

    fig.tight_layout()

    plt.show()


def plot_feature_vs_target(x: np.ndarray, y: np.ndarray , feature_name: str, target_name: str, max_points: int = 10_000, random_state: int | None = 42):
    """
        Plot a 2D scatter plot between one feature and the target,
        together with the fitted simple linear regression line.

        Args:
            x:
                Numeric values of the explanatory variable.
            y:
                Numeric values of the target variable.

            feature_name:
                Name of the explanatory variable.

            target_name:
                Name of the target variable.

            max_points:
                Maximum number of points drawn in the scatter plot.
                If the data has more points, a random sample is used
                (the regression is still fit on all valid points).

            random_state:
                Seed for the sampling. Use None for no seeding.


            Returns:
                A matplotlib figure containing the scatter plot and the regression line, with the fitted equantion and R² show in the legend.
    """

    # 1. Normalização e validação da entrada
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()

    if x.shape != y.shape:
        raise ValueError(f"x and y must have the same shape, got {x.shape} and {y.shape}.")

    # 2. Removação de valores inválidos
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]

    if len(x_valid) < 2:
        raise ValueError("At leat two valid points are required.")

    # 3.  Regressão linear simples (mínimos quadrados via SVD)
    # np.polyfit retorna [inclinação, intercepto] (grau mais alto primeiro)
    theta_1, theta_0 = np.polyfit(x_valid, y_valid, 1)

    # 4. R² - qualidade de ajuste
    y_mean = np.mean(y_valid)
    y_hat = theta_0 + theta_1 * x_valid

    ss_res = np.sum((y_valid - y_hat) ** 2)
    ss_tot = np.sum((y_valid - y_mean) ** 2)


    if ss_tot < 1e-12:
        raise ValueError(
            "The target has near-zero variance; R² is undefined."
        )

    r_squared = 1 - ss_res / ss_tot

    # 5. Amostragem para o scatter (regressão usa todos esses pontos)
    if len(x_valid) >= max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(x_valid), size=max_points, replace=False)
        x_plot, y_plot = x_valid[idx], y_valid[idx]
    else:
        x_plot, y_plot = x_valid, y_valid


    # 6. Figura
    fig, ax = plt.subplots(figsize=(10,6))

    ax.scatter(
        x_plot,
        y_plot,
        s=15,
        alpha=0.4,
        edgecolors="none",
    )

    # 7. Linha de regressão com apenas 2 pontos
    x_line = np.array([x_valid.min(), x_valid.max()])
    y_line = theta_0 + theta_1 * x_line

    ax.plot(
            x_line,
            y_line,
            linewidth=2,
            color="tab:red",
            label=f"ŷ = {theta_0:.2f} + {theta_1:.2f}x   (R² = {r_squared:.3f})",
        )

    # 8. Rótulos, legenda e layout
    ax.set_xlabel(feature_name)
    ax.set_ylabel(target_name)
    ax.set_title(f"{feature_name} vs {target_name}")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plt.show()


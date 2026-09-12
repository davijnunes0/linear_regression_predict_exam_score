import matplotlib.pyplot as plt
from i_ai_project.traine.linear_regression import LinearRegressionEquation
from i_ai_project.utils.paths import path_resolve


def get_feature_weights(model: LinearRegressionEquation, feature_names: list[str]) -> dict[str, float]:
    """
        Extract and sort the final weights learned by a trained
        linear regression model.

        Args:
            model:
                A trained instance of LinearRegressionEquation containing coef_.

            feature_names:
                List with the name of each feature, in the same
                order used to train the model.

        Returns:
            Weights sorted by absolute magnitude,
            from strongest to weakest.
    """

    weights = {}

    for name, value in zip(feature_names, model.coef_):
        weights[name] = float(value)

    return dict(
        sorted(
            weights.items(),
            key=lambda item: abs(item[1]),
            reverse=True
        )
    )


def plot_feature_weights(weights: dict[str, float], model_name: str = "Linear Regression") -> None:
    """
        Plot the final weights learned by a linear regression model
        after training.

        Args:
            weights:
                Dictionary containing feature names and their
                learned weights.

            model_name:
                Name of the trained model, used in the plot title.

        Returns:
            None
    """

    # Extrai os nomes das features e seus respectivos pesos em duas listas separadas, para usar no gráfico.
    features = list(weights.keys())
    values = list(weights.values())

    plt.figure(figsize=(10, 7))

    # Gráfico de barras horizontais
    # barh = bar horizontal (barras horizontais);
    # Cada feature fica no eixo Y (vertical)
    # O valor do peso fica no eixo X (horizontal).
    plt.barh(features, values)

    # Linha de referência zero
    # Barras para a direita -> pesos positivos
    # Barras para a esquerda -> pesos negativos
    plt.axvline(x=0, linewidth=1)

    plt.xlabel("Weight value")          # legenda do eixo X
    plt.ylabel("Features")              # legenda do eixo Y
    plt.title(f"Feature weights after training ({model_name})")

    # Ajusta automaticamente os espaçamentos para que nada fique cortado (título, rótulos, etc).
    plt.tight_layout()

    plt.savefig(path_resolve(2, "figures/feature_weights.png"), dpi=120)
    plt.show()


if __name__ == "__main__":
    from i_ai_project.utils.data_prep import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()

    print("Treinando Regressão Linear (Equação Normal)")
    model = LinearRegressionEquation()
    model.fit(X_train, y_train)

    print("Gerando Gráfico: Pesos Finais Após o Treinamento")
    weights = get_feature_weights(model, X_train.columns.tolist())
    plot_feature_weights(weights, model_name="Normal Equation")

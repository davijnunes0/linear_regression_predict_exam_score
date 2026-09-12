import matplotlib.pyplot as plt
import numpy as np
from i_ai_project.traine.gradient_descent import GradientDescent
from i_ai_project.utils.paths import path_resolve

def plot_weights_evolution(model: GradientDescent, max_epochs: int = 100) -> None:
    """
    Plot the evolution of the weights (thetas) during Gradient Descent training.

    Args:
        model:
            A trained instance of LinearRegressionGradientDescent containing theta_history_.
        max_epochs:
            Maximum number of epochs to display on the x-axis for better visibility.

    Returns:
        None
    """
    thetas_hist = np.array(model.theta_history_)
    epochs_to_plot = min(max_epochs, len(thetas_hist))

    fig, ax = plt.subplots(figsize=(10, 6))

    # Start from index 1 to ignore the intercept (theta_0) and focus on features
    for i in range(1, thetas_hist.shape[1]):
        ax.plot(range(epochs_to_plot), thetas_hist[:epochs_to_plot, i], linewidth=2)

    ax.set_title("Weights (Thetas) Evolution during Training", fontsize=14)
    ax.set_xlabel("Epochs (Iterations)")
    ax.set_ylabel("Weight Value")
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()

    plt.savefig(path_resolve(2, "figures/weights_evolution.png"), dpi=120)
    plt.show()


def plot_cost_convergence(x_train: np.ndarray, y_train: np.ndarray, learning_rates: list[float], epochs: int = 500) -> None:
    """
    Plot the cost (MSE) convergence for different learning rates.

    Args:
        x_train: Numeric values of the explanatory variables.
        y_train: Numeric values of the target variable.
        learning_rates: A list of learning rates to test and compare.
        epochs: Number of epochs to train each model.

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for lr in learning_rates:
        model = GradientDescent(learning_rate=lr, epochs=epochs)
        model.fit(x_train, y_train)
        ax.plot(range(epochs), model.cost_history_, label=f"LR = {lr}", linewidth=2)

    ax.set_title("Cost (MSE) Convergence by Learning Rate", fontsize=14)
    ax.set_xlabel("Epochs (Iterations)")
    ax.set_ylabel("Error (MSE)")
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()

    plt.savefig(path_resolve(2, "figures/cost_convergence.png"), dpi=120)
    plt.show()

if __name__ == "__main__":
    from i_ai_project.utils.data_prep import prepare_data

    X_train, X_test, y_train, y_test = prepare_data()

    print("Gerando Gráfico 1: Evolução dos Pesos")
    model = GradientDescent(learning_rate=0.1, epochs=200)
    model.fit(X_train, y_train)
    plot_weights_evolution(model, max_epochs=100)

    print("Gerando Gráfico 2: Convergência do Custo (MSE)")
    taxas = [0.1, 0.01, 0.001]
    plot_cost_convergence(X_train, y_train, learning_rates=taxas, epochs=500)
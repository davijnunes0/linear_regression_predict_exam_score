from i_ai_project.utils.data_prep import prepare_data
from i_ai_project.traine.linear_regression import GradientDescent
from i_ai_project.analysis.gradient_descent_analysis import plot_weights_evolution, plot_cost_convergence

def main_gradiente():
    X_train, X_test, y_train, y_test = prepare_data()

    print("Gerando Gráfico 1: Evolução dos Pesos")
    model = GradientDescent(learning_rate=0.1, epochs=200)
    model.fit(X_train, y_train)
    plot_weights_evolution(model, max_epochs=100)

    print("Gerando Gráfico 2: Convergência do Custo (MSE)")
    taxas = [0.1, 0.01, 0.001]
    plot_cost_convergence(X_train, y_train, learning_rates=taxas, epochs=500)

if __name__ == "__main__":
    main_gradiente()
import sys

import matplotlib.pyplot as plt
import numpy as np

from i_ai_project.traine.linear_regression import LinearRegressionEquation
from i_ai_project.traine.gradient_descent import GradientDescent
from i_ai_project.utils.data_prep import prepare_data

TRAINES = {
    "linear_regression": LinearRegressionEquation,
    "gradient_descent": GradientDescent,
}

# Apelidos aceitos na linha de comando (ex.: regressão linear, descida do gradiente)
ALIASES = {
    "regressao_linear": "linear_regression",
    "regressão_linear": "linear_regression",
    "regressao": "linear_regression",
    "linear": "linear_regression",
    "descida_do_gradiente": "gradient_descent",
    "descida_gradiente": "gradient_descent",
    "gradiente": "gradient_descent",
    "gd": "gradient_descent",
}


def run_traine():
    traine_type = sys.argv[1] if len(sys.argv) > 1 else "linear_regression"
    traine_type = ALIASES.get(traine_type, traine_type)

    if traine_type not in TRAINES:
        print(f"Tipo de treino desconhecido: '{sys.argv[1]}'")
        print(f"Opções válidas: {', '.join(TRAINES)}")
        sys.exit(1)

    # 1. Carrega e pré-processa os dados (split, imputação, encoding, z-score)
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. Treina o modelo escolhido
    model = TRAINES[traine_type]()
    model.fit(X_train, y_train)

    # 3. Prediz no conjunto de teste
    y_pred = model.predict(X_test)

    # 4. Avalia com RMSE e R² (calculados na mão, sem sklearn)
    mse = np.mean((y_test - y_pred) ** 2)
    rmse = np.sqrt(mse)

    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    print(f"Modelo: {traine_type}")
    print(f"Intercepto: {model.intercept_:.4f}")
    print(f"Número de coeficientes: {len(model.coef_)}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MSE:  {mse:.4f}")
    print(f"R²:   {r2:.4f}")

    # 5. Compara predições com valores reais (primeiras 5 amostras)
    print("\nComparação (real vs predito):")
    for real, pred in zip(y_test[:10], y_pred[:10]):
        print(f"  real: {real:6.2f} | predito: {pred:6.2f}")

    # 6. Gráficos: real vs predito + gráfico específico do método
    plot_results(traine_type, model, TRAINES[traine_type], X_train, y_train, X_test, y_test, y_pred)


def plot_results(name, model, model_class, X_train, y_train, X_test, y_test, y_pred):
    """Gera dois gráficos e salva como PNG.

    - Real vs Predito: cada ponto é uma amostra do teste; quanto mais
      colado na linha pontilhada (y = x), melhor o modelo. Vale para
      qualquer método.
    - Direita: depende do método. Se o modelo registrou histórico de
      custo por época (Gradient Descent), plota a convergência do custo.
      Caso contrário (Equação Normal), como não há épocas/iterações, o
      mais próximo é medir o RMSE no teste treinando com frações
      crescentes dos dados de treino.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # esquerda: real vs predito
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.3, s=15)
    lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    ax.plot(lims, lims, "r--", label="perfeito (y = x)")
    ax.set_xlabel("Real")
    ax.set_ylabel("Predito")
    ax.set_title("Real vs Predito (teste)")
    ax.legend()

    # direita: convergência do custo (GD) ou curva de aprendizado (Equação Normal)
    ax = axes[1]
    cost_history = getattr(model, "cost_history_", None)
    if cost_history:
        ax.plot(np.arange(1, len(cost_history) + 1), cost_history)
        ax.set_xlabel("Época")
        ax.set_ylabel("Custo (MSE)")
        ax.set_title("Convergência do Gradient Descent")
    else:
        fractions = np.linspace(0.05, 1.0, 20)
        rmse_curve = []
        n = len(X_train)
        for f in fractions:
            k = max(1, int(f * n))
            m = model_class()
            m.fit(X_train[:k], y_train[:k])
            pred = m.predict(X_test)
            rmse_curve.append(np.sqrt(np.mean((np.asarray(y_test) - pred) ** 2)))
        ax.plot(fractions * 100, rmse_curve, marker="o")
        ax.set_xlabel("% dos dados de treino usados")
        ax.set_ylabel("RMSE (teste)")
        ax.set_title("Curva de aprendizado")

    fig.tight_layout()
    out = f"resultado_{name}.png"
    fig.savefig(out, dpi=120)
    plt.show()


if __name__ == "__main__":
    run_traine()

import numpy as np


class GradientDescent:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.theta = None
        self.intercept_ = None
        self.coef_ = None
        # Atributos extras para gerar os gráficos
        self.cost_history_ = []
        self.theta_history_ = []

    def fit(self, X, Y):
        """ Treina utilizando Gradient Descent. """

        x = np.asarray(X, dtype=float)
        y = np.asarray(Y, dtype=float)

        m = x.shape[0]

        # Adiciona a coluna de uns, idêntico à Equação Normal
        X_b = np.c_[
            np.ones((m, 1)),
            x
        ]

        self.theta = np.zeros(X_b.shape[1])
        self.cost_history_ = []
        self.theta_history_ = []

        for _ in range(self.epochs):
            pred = X_b @ self.theta
            errors = pred - y
            gradient = (2/m) * (X_b.T @ errors)
            self.theta = self.theta - self.learning_rate * gradient

            self.cost_history_.append(np.mean(errors ** 2))
            self.theta_history_.append(self.theta.copy())

        self.intercept_ = self.theta[0]
        self.coef_ = self.theta[1:]

        return self

    def predict(self, x):
        """ Realiza a predição utilizando os parâmetros aprendidos. """
        x = np.asarray(x, dtype=float)
        m = x.shape[0]

        X_b = np.c_[
            np.ones((m,1)),
            x
        ]

        return X_b @ self.theta

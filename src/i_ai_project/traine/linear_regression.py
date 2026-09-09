import numpy as np

class LinearRegressionEquation:
    def __init__(self):
        self.theta = None
        self.intercept_ = None
        self.coef_ = None

    def fit(self, X, Y):
        """  Treina a regressão linear utilizando a Equação Normal
             theta = (X^T X) ^ -1 X ^ T y

             (Utiliza pseudoinversa para maior instabilidade númerica).
        """

        # converte listas/tuplas para arrays NumPy sem copiar se já for um.
        x = np.asarray(X, dtype = float)
        y = np.asarray(Y, dtype = float)


        # Retorna uma tupla com as dimensões (linhas, colunas) . [0] pega número de amostras (m)
        m = X.shape[0]

        # np.ones((m,1)) cria uma coluna de uns, (m x 1)
        # np.c[a, b] é uma "Index  trick" do NumPy que concatena colunas (empilha horizontalmente).
        # Porque a coluna de uns? Luiz é para que o intercepto entre na equação como um coeficiente comum:
        # (y estimativa) = theta_0 * 1
        X_b = np.c_[
            np.ones((m, 1)),
            x
        ]

        # np.linalg.pinv(X_b) computa a pseudoinversa: X+=(XTX)−1XT  (calculada internamente via SVD, sem inversão explícita).
        # @ é o operador de multiplicação de matrizes do Python 3.5+ (equivalente a np.dot ou np.matmul). Para vetores 1D, funciona como produto interno.
        self.theta = np.linalg.pinv(X_b) @ y

        # theta[0] = intercepto
        self.intercept_ = self.theta[0]

        # theta[1:] = peso dos atributos
        # (Começa do index (1) vai até o final)
        self.coef_ = self.theta[1:]

        return self

    def predict(self, x):
        """
        Realiza  a predição utilizando
        os parâmetros aprendidos no fit.
        """

        x = np.asarray(x, dtype=float)

        m = x.shape[0]

        X_b = np.c_[
            np.ones((m,1)),
            x
        ]

        return X_b @ self.theta

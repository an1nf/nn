import numpy as np


class Adaline:
    def __init__(self, dimension, tasa=0.01):
        self.pesos = np.zeros(dimension)
        self.sesgo = 0.0
        self.tasa = tasa

    def salida_lineal(self, X):
        return X @ self.pesos + self.sesgo

    def predecir(self, X):
        return np.where(self.salida_lineal(X) >= 0, 1, -1)

    def entrenar(self, X, y, epocas=40, rng=None):
        rng = rng or np.random.default_rng(0)
        for _ in range(epocas):
            for i in rng.permutation(len(X)):
                # Regla delta (LMS): gradiente estocástico de ½(y − wᵀx − b)².
                error = y[i] - self.salida_lineal(X[i])
                self.pesos += self.tasa * error * X[i]
                self.sesgo += self.tasa * error
        return np.mean((y - self.salida_lineal(X)) ** 2)


if __name__ == "__main__":
    rng = np.random.default_rng(11)
    X = np.vstack([rng.normal(1.0, 0.9, size=(100, 2)), rng.normal(-1.0, 0.9, size=(100, 2))])
    y = np.concatenate([np.ones(100), -np.ones(100)])
    modelo = Adaline(2, tasa=0.005)
    ecm = modelo.entrenar(X, y)
    print(f"ECM {ecm:.3f}; exactitud {np.mean(modelo.predecir(X) == y):.2f}")

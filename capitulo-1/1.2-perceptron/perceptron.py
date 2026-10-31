import numpy as np


class Perceptron:
    def __init__(self, dimension, tasa=1.0):
        self.pesos = np.zeros(dimension)
        self.sesgo = 0.0
        self.tasa = tasa

    def predecir(self, X):
        return np.where(X @ self.pesos + self.sesgo >= 0, 1, -1)

    def entrenar(self, X, y, epocas=100, rng=None):
        rng = rng or np.random.default_rng(0)
        for epoca in range(1, epocas + 1):
            errores = 0
            for i in rng.permutation(len(X)):
                if y[i] * (X[i] @ self.pesos + self.sesgo) <= 0:
                    self.pesos += self.tasa * y[i] * X[i]
                    self.sesgo += self.tasa * y[i]
                    errores += 1
            if errores == 0:
                return epoca
        return epocas


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    X = rng.uniform(-3, 3, size=(200, 2))
    y = np.where(X @ np.array([1.0, -0.8]) > 0.3, 1, -1)
    modelo = Perceptron(2)
    epocas = modelo.entrenar(X, y, rng=rng)
    print(f"Convergió en {epocas} épocas; exactitud {np.mean(modelo.predecir(X) == y):.2f}")

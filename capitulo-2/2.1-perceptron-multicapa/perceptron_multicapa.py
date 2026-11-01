import numpy as np


class Densa:
    def __init__(self, entrada, salida, rng):
        self.W = rng.normal(0, np.sqrt(1.0 / entrada), (entrada, salida))
        self.b = np.zeros(salida)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def adelante(self, X):
        self.X = X
        return X @ self.W + self.b

    def atras(self, grad):
        self.dW[...] = self.X.T @ grad
        self.db[...] = grad.sum(axis=0)
        return grad @ self.W.T

    def parametros(self):
        return [(self.W, self.dW), (self.b, self.db)]


class Tanh:
    def adelante(self, X):
        self.Y = np.tanh(X)
        return self.Y

    def atras(self, grad):
        return grad * (1 - self.Y ** 2)

    def parametros(self):
        return []


class Red:
    def __init__(self, tamanos, rng):
        self.capas = []
        for entrada, salida in zip(tamanos[:-2], tamanos[1:-1]):
            self.capas += [Densa(entrada, salida, rng), Tanh()]
        self.capas.append(Densa(tamanos[-2], tamanos[-1], rng))

    def adelante(self, X):
        for capa in self.capas:
            X = capa.adelante(X)
        return X

    def perdida(self, X, y):
        z = self.adelante(X)[:, 0]
        p = 1 / (1 + np.exp(-z))
        perdida = np.mean(np.logaddexp(0, z) - y * z)
        # ∂L/∂z de sigmoide + entropía cruzada; luego retropropagación capa a capa.
        grad = ((p - y) / len(y))[:, None]
        for capa in reversed(self.capas):
            grad = capa.atras(grad)
        return perdida

    def parametros(self):
        return [par for capa in self.capas for par in capa.parametros()]

    def probabilidad(self, X):
        return 1 / (1 + np.exp(-self.adelante(X)[:, 0]))


class Adam:
    def __init__(self, red, tasa=0.01, beta1=0.9, beta2=0.999):
        self.red, self.tasa, self.beta1, self.beta2 = red, tasa, beta1, beta2
        self.m = [np.zeros_like(p) for p, _ in red.parametros()]
        self.v = [np.zeros_like(p) for p, _ in red.parametros()]
        self.t = 0

    def paso(self):
        self.t += 1
        for k, (p, g) in enumerate(self.red.parametros()):
            self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * g ** 2
            m = self.m[k] / (1 - self.beta1 ** self.t)
            v = self.v[k] / (1 - self.beta2 ** self.t)
            p -= self.tasa * m / (np.sqrt(v) + 1e-8)


if __name__ == "__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)
    red = Red([2, 2, 1], np.random.default_rng(5))
    optimizador = Adam(red, tasa=0.05)
    for _ in range(1500):
        perdida = red.perdida(X, y)
        optimizador.paso()
    print(f"XOR: pérdida {perdida:.4f}, salidas {np.round(red.probabilidad(X), 3)}")

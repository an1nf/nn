import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


class Convolucion:
    def __init__(self, canales, filtros, tamano, rng):
        self.k = tamano
        self.W = rng.normal(0, np.sqrt(2.0 / (canales * tamano * tamano)), (canales * tamano * tamano, filtros))
        self.b = np.zeros(filtros)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def adelante(self, X):
        self.forma = X.shape
        relleno = self.k // 2
        Xp = np.pad(X, ((0, 0), (0, 0), (relleno, relleno), (relleno, relleno)))
        ventanas = sliding_window_view(Xp, (self.k, self.k), axis=(2, 3))
        n, c, h, w = X.shape
        # im2col: la convolución se reduce a un producto matricial.
        self.columnas = ventanas.transpose(0, 2, 3, 1, 4, 5).reshape(n, h, w, c * self.k * self.k)
        return (self.columnas @ self.W + self.b).transpose(0, 3, 1, 2)

    def atras(self, grad):
        n, c, h, w = self.forma
        grad = grad.transpose(0, 2, 3, 1)
        self.dW[...] = self.columnas.reshape(-1, self.W.shape[0]).T @ grad.reshape(-1, self.W.shape[1])
        self.db[...] = grad.sum(axis=(0, 1, 2))
        dcolumnas = (grad @ self.W.T).reshape(n, h, w, c, self.k, self.k)
        relleno = self.k // 2
        dXp = np.zeros((n, c, h + 2 * relleno, w + 2 * relleno))
        for i in range(self.k):
            for j in range(self.k):
                dXp[:, :, i:i + h, j:j + w] += dcolumnas[:, :, :, :, i, j].transpose(0, 3, 1, 2)
        return dXp[:, :, relleno:relleno + h, relleno:relleno + w]

    def parametros(self):
        return [(self.W, self.dW), (self.b, self.db)]


class ReLU:
    def adelante(self, X):
        self.mascara = X > 0
        return X * self.mascara

    def atras(self, grad):
        return grad * self.mascara

    def parametros(self):
        return []


class MaxPool:
    def adelante(self, X):
        n, c, h, w = X.shape
        bloques = X.reshape(n, c, h // 2, 2, w // 2, 2)
        salida = bloques.max(axis=(3, 5))
        self.mascara = bloques == salida[:, :, :, None, :, None]
        return salida

    def atras(self, grad):
        n, c, h, w = grad.shape
        expandido = self.mascara * grad[:, :, :, None, :, None]
        return expandido.reshape(n, c, 2 * h, 2 * w)

    def parametros(self):
        return []


class Aplanar:
    def adelante(self, X):
        self.forma = X.shape
        return X.reshape(len(X), -1)

    def atras(self, grad):
        return grad.reshape(self.forma)

    def parametros(self):
        return []


class Densa:
    def __init__(self, entrada, salida, rng):
        self.W = rng.normal(0, np.sqrt(2.0 / entrada), (entrada, salida))
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


class Secuencial:
    def __init__(self, capas):
        self.capas = capas

    def adelante(self, X):
        for capa in self.capas:
            X = capa.adelante(X)
        return X

    def perdida(self, X, y):
        logits = self.adelante(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        probabilidades = np.exp(logits)
        probabilidades /= probabilidades.sum(axis=1, keepdims=True)
        perdida = -np.mean(np.log(probabilidades[np.arange(len(y)), y] + 1e-12))
        grad = probabilidades.copy()
        grad[np.arange(len(y)), y] -= 1
        grad /= len(y)
        for capa in reversed(self.capas):
            grad = capa.atras(grad)
        return perdida

    def parametros(self):
        return [par for capa in self.capas for par in capa.parametros()]

    def exactitud(self, X, y):
        return np.mean(self.adelante(X).argmax(axis=1) == y)


class Adam:
    def __init__(self, modelo, tasa=0.003):
        self.modelo, self.tasa, self.t = modelo, tasa, 0
        self.m = [np.zeros_like(p) for p, _ in modelo.parametros()]
        self.v = [np.zeros_like(p) for p, _ in modelo.parametros()]

    def paso(self):
        self.t += 1
        for k, (p, g) in enumerate(self.modelo.parametros()):
            self.m[k] = 0.9 * self.m[k] + 0.1 * g
            self.v[k] = 0.999 * self.v[k] + 0.001 * g ** 2
            p -= self.tasa * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)


def red_convolucional(rng):
    return Secuencial([Convolucion(1, 8, 3, rng), ReLU(), MaxPool(), Aplanar(), Densa(8 * 4 * 4, 10, rng)])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    digitos = load_digits()
    X = (digitos.images / 16.0)[:, None, :, :]
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, digitos.target, test_size=0.3, random_state=0)
    modelo = red_convolucional(rng)
    optimizador = Adam(modelo)
    for _ in range(30):
        orden = rng.permutation(len(X_ent))
        for inicio in range(0, len(X_ent), 32):
            lote = orden[inicio:inicio + 32]
            modelo.perdida(X_ent[lote], y_ent[lote])
            optimizador.paso()
    print(f"Exactitud de prueba: {modelo.exactitud(X_pru, y_pru):.3f}")

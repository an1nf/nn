import numpy as np


class RNNElman:
    def __init__(self, entrada, oculta, rng):
        self.oculta = oculta
        self.p = {
            "U": rng.normal(0, 1 / np.sqrt(entrada), (entrada, oculta)),
            "W": rng.normal(0, 1 / np.sqrt(oculta), (oculta, oculta)),
            "b": np.zeros(oculta),
            "v": rng.normal(0, 1 / np.sqrt(oculta), oculta),
            "c": np.zeros(1),
        }

    def adelante(self, X):
        h = np.zeros((len(X), self.oculta))
        self.estados = [h]
        for t in range(X.shape[1]):
            h = np.tanh(X[:, t] @ self.p["U"] + h @ self.p["W"] + self.p["b"])
            self.estados.append(h)
        return h @ self.p["v"] + self.p["c"]

    def perdida(self, X, y):
        z = self.adelante(X)
        perdida = np.mean(np.logaddexp(0, z) - y * z)
        dz = (1 / (1 + np.exp(-z)) - y) / len(y)
        g = {nombre: np.zeros_like(valor) for nombre, valor in self.p.items()}
        g["v"] = self.estados[-1].T @ dz
        g["c"] = np.array([dz.sum()])
        dh = np.outer(dz, self.p["v"])
        # BPTT: cada paso hacia atrás multiplica por diag(1 − h²) Wᵀ.
        for t in reversed(range(X.shape[1])):
            da = dh * (1 - self.estados[t + 1] ** 2)
            g["U"] += X[:, t].T @ da
            g["W"] += self.estados[t].T @ da
            g["b"] += da.sum(axis=0)
            dh = da @ self.p["W"].T
        return perdida, g


class SGDConMomento:
    def __init__(self, parametros, tasa=0.05, momento=0.9, recorte=1.0):
        self.parametros, self.tasa, self.momento, self.recorte = parametros, tasa, momento, recorte
        self.velocidad = {k: np.zeros_like(v) for k, v in parametros.items()}

    def paso(self, gradientes):
        norma = np.sqrt(sum(np.sum(g ** 2) for g in gradientes.values()))
        escala = min(1.0, self.recorte / (norma + 1e-12))
        for k, g in gradientes.items():
            self.velocidad[k] = self.momento * self.velocidad[k] - self.tasa * escala * g
            self.parametros[k] += self.velocidad[k]


def tarea_primer_bit(n, longitud, rng):
    bits = rng.choice([-1.0, 1.0], size=(n, longitud, 1))
    return bits, (bits[:, 0, 0] > 0).astype(float)


def entrenar(modelo, longitud, pasos, rng, lote=64):
    optimizador = SGDConMomento(modelo.p)
    for _ in range(pasos):
        optimizador.paso(modelo.perdida(*tarea_primer_bit(lote, longitud, rng))[1])
    X, y = tarea_primer_bit(1000, longitud, rng)
    return np.mean((modelo.adelante(X) > 0) == (y > 0.5))


if __name__ == "__main__":
    for longitud in (10, 40):
        rng = np.random.default_rng(0)
        print(f"T = {longitud}: exactitud {entrenar(RNNElman(1, 32, rng), longitud, 2000, rng):.3f}")

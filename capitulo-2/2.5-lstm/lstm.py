import importlib.util
from pathlib import Path

import numpy as np

_especificacion = importlib.util.spec_from_file_location(
    "rnn_elman", Path(__file__).resolve().parents[1] / "2.4-rnn-elman" / "rnn_elman.py")
rnn_elman = importlib.util.module_from_spec(_especificacion)
_especificacion.loader.exec_module(rnn_elman)


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


class LSTM:
    def __init__(self, entrada, oculta, rng, sesgo_de_olvido=3.0):
        self.oculta = H = oculta
        b = np.zeros(4 * H)
        b[H:2 * H] = sesgo_de_olvido
        self.p = {
            "Wx": rng.normal(0, 1 / np.sqrt(entrada), (entrada, 4 * H)),
            "Wh": rng.normal(0, 1 / np.sqrt(H), (H, 4 * H)),
            "b": b,
            "v": rng.normal(0, 1 / np.sqrt(H), H),
            "c": np.zeros(1),
        }

    def adelante(self, X):
        n, H = len(X), self.oculta
        h, c = np.zeros((n, H)), np.zeros((n, H))
        self.cache = []
        for t in range(X.shape[1]):
            a = X[:, t] @ self.p["Wx"] + h @ self.p["Wh"] + self.p["b"]
            i, f, o = sigmoide(a[:, :H]), sigmoide(a[:, H:2 * H]), sigmoide(a[:, 2 * H:3 * H])
            g = np.tanh(a[:, 3 * H:])
            self.cache.append((X[:, t], h, c, i, f, o, g))
            c = f * c + i * g
            h = o * np.tanh(c)
        self.h_final, self.c_final = h, c
        return h @ self.p["v"] + self.p["c"]

    def perdida(self, X, y):
        z = self.adelante(X)
        perdida = np.mean(np.logaddexp(0, z) - y * z)
        dz = (sigmoide(z) - y) / len(y)
        g = {nombre: np.zeros_like(valor) for nombre, valor in self.p.items()}
        g["v"] = self.h_final.T @ dz
        g["c"] = np.array([dz.sum()])
        dh = np.outer(dz, self.p["v"])
        dc_siguiente = np.zeros_like(dh)
        c = self.c_final
        for x, h_previo, c_previo, i, f, o, gg in reversed(self.cache):
            tc = np.tanh(c)
            # Camino aditivo de la celda: ∂c_t/∂c_{t−1} = f_t, sin multiplicar por W.
            dc = dc_siguiente + dh * o * (1 - tc ** 2)
            da = np.concatenate([
                dc * gg * i * (1 - i),
                dc * c_previo * f * (1 - f),
                dh * tc * o * (1 - o),
                dc * i * (1 - gg ** 2),
            ], axis=1)
            g["Wx"] += x.T @ da
            g["Wh"] += h_previo.T @ da
            g["b"] += da.sum(axis=0)
            dh = da @ self.p["Wh"].T
            dc_siguiente = dc * f
            c = c_previo
        return perdida, g


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    print(f"T = 60: exactitud {rnn_elman.entrenar(LSTM(1, 32, rng), 60, 2000, rng):.3f}")

import numpy as np


class Hopfield:
    def __init__(self, patrones):
        # Regla de Hebb: W = (1/N) Σ ξ ξᵀ, sin autoconexiones.
        self.W = patrones.T @ patrones / patrones.shape[1]
        np.fill_diagonal(self.W, 0)

    def energia(self, s):
        return -0.5 * s @ self.W @ s

    def recordar(self, s, rng, barridos=5):
        s = s.copy()
        for _ in range(barridos):
            for i in rng.permutation(len(s)):
                s[i] = 1 if self.W[i] @ s >= 0 else -1
        return s


def recuperacion_moderna(patrones, s, beta=8.0):
    # Hopfield moderna: ξ ← Xᵀ softmax(β X ξ), la misma operación que la atención.
    X = patrones / np.sqrt(patrones.shape[1])
    puntajes = beta * X @ (s / np.sqrt(patrones.shape[1]))
    pesos = np.exp(puntajes - puntajes.max())
    return np.where(X.T @ (pesos / pesos.sum()) >= 0, 1, -1)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    patrones = rng.choice([-1, 1], size=(10, 100))
    ruidoso = patrones[0] * np.where(rng.random(100) < 0.1, -1, 1)
    red = Hopfield(patrones)
    clasica = red.recordar(ruidoso, rng)
    print(f"Energía: {red.energia(ruidoso):.2f} → {red.energia(clasica):.2f}")
    print(f"Píxeles correctos: clásica {np.mean(clasica == patrones[0]):.0%}, "
          f"moderna {np.mean(recuperacion_moderna(patrones, ruidoso) == patrones[0]):.0%}")

import itertools

import numpy as np


def separables(n, epocas=300):
    # Un perceptrón por cada una de las 2^(2^n) funciones, en paralelo.
    vertices = np.array(list(itertools.product([0, 1], repeat=n)), dtype=float)
    vertices = np.column_stack([vertices, np.ones(len(vertices))])
    total = 2 ** len(vertices)
    etiquetas = ((np.arange(total)[:, None] >> np.arange(len(vertices))[None, :]) & 1) * 2 - 1
    pesos = np.zeros((total, n + 1))
    for _ in range(epocas):
        cambios = np.zeros(total, dtype=bool)
        for k, x in enumerate(vertices):
            error = etiquetas[:, k] * (pesos @ x) <= 0
            pesos[error] += etiquetas[error, k, None] * x
            cambios |= error
        if not cambios.any():
            break
    return np.all(etiquetas * (pesos @ vertices.T) > 0, axis=1)


if __name__ == "__main__":
    for n in range(1, 5):
        print(f"n = {n}: {separables(n).sum()} de {2 ** 2 ** n} funciones son linealmente separables")

import itertools

import numpy as np

PUERTAS = {
    "AND": (np.array([1.0, 1.0]), 1.5),
    "OR": (np.array([1.0, 1.0]), 0.5),
    "NAND": (np.array([-1.0, -1.0]), -1.5),
    "NOR": (np.array([-1.0, -1.0]), -0.5),
}


def neurona(x, pesos, umbral):
    return (x @ pesos >= umbral).astype(int)


def xor(x):
    oculta = np.stack([neurona(x, *PUERTAS["OR"]), neurona(x, *PUERTAS["NAND"])], axis=-1)
    return neurona(oculta, *PUERTAS["AND"])


if __name__ == "__main__":
    x = np.array(list(itertools.product([0, 1], repeat=2)), dtype=float)
    for nombre, parametros in PUERTAS.items():
        print(f"{nombre:>4}: {neurona(x, *parametros)}")
    print(f" XOR: {xor(x)}")

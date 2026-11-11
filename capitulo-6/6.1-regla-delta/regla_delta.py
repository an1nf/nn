import numpy as np

REGLAS = {
    "lineal": lambda S, k, v, alfa, beta: S + np.outer(v, k),
    "con olvido": lambda S, k, v, alfa, beta: alfa * S + np.outer(v, k),
    # Un paso de SGD sobre ½‖S k − v‖² (Widrow-Hoff): se escribe solo lo que falta.
    "delta": lambda S, k, v, alfa, beta: S + beta * np.outer(v - S @ k, k),
    "delta con compuerta": lambda S, k, v, alfa, beta: alfa * S + beta * np.outer(v - alfa * S @ k, k),
}


def escribir(claves, valores, regla, alfa=0.97, beta=1.0):
    S = np.zeros((valores.shape[1], claves.shape[1]))
    for k, v in zip(claves, valores):
        S = REGLAS[regla](S, k, v, alfa, beta)
    return S


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    claves = rng.normal(size=(12, 32))
    claves /= np.linalg.norm(claves, axis=1, keepdims=True)
    valores = rng.normal(size=(12, 32))
    nuevo = rng.normal(size=32)
    coseno = lambda a, b: a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    for regla in REGLAS:
        leido = escribir(np.vstack([claves, claves[:1]]), np.vstack([valores, nuevo]), regla) @ claves[0]
        print(f"{regla:>19}: similitud con el valor nuevo {coseno(leido, nuevo):.2f}, "
              f"con el viejo {coseno(leido, valores[0]):.2f}")

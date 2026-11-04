import torch
import torch.nn as nn
from sklearn.datasets import load_digits

INICIALIZACIONES = {
    "glorot": nn.init.xavier_normal_,
    "he": lambda w: nn.init.kaiming_normal_(w, nonlinearity="relu"),
    "unitaria": lambda w: nn.init.normal_(w, 0.0, 1.0),
}


def construir(activacion, inicializacion, profundidad=20, ancho=64, entrada=64, salida=10):
    capas, dimension = [], entrada
    for _ in range(profundidad):
        lineal = nn.Linear(dimension, ancho)
        # Glorot: Var(w) = 2/(n_ent + n_sal); He: Var(w) = 2/n_ent.
        INICIALIZACIONES[inicializacion](lineal.weight)
        nn.init.zeros_(lineal.bias)
        capas += [lineal, activacion()]
        dimension = ancho
    return nn.Sequential(*capas, nn.Linear(dimension, salida))


def datos():
    digitos = load_digits()
    X = (digitos.data - digitos.data.mean(0)) / (digitos.data.std(0) + 1e-8)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(digitos.target)


def entrenar(modelo, X, y, epocas=30, tasa=0.005):
    optimizador = torch.optim.SGD(modelo.parameters(), lr=tasa, momentum=0.9)
    generador = torch.Generator().manual_seed(0)
    for _ in range(epocas):
        for lote in torch.randperm(len(X), generator=generador).split(64):
            perdida = nn.functional.cross_entropy(modelo(X[lote]), y[lote])
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
    with torch.no_grad():
        return nn.functional.cross_entropy(modelo(X), y).item()


if __name__ == "__main__":
    X, y = datos()
    for nombre, activacion, inicializacion in [("sigmoide + Glorot", nn.Sigmoid, "glorot"),
                                               ("ReLU + He", nn.ReLU, "he")]:
        torch.manual_seed(0)
        print(f"{nombre:>17}: pérdida final {entrenar(construir(activacion, inicializacion), X, y):.3f}")

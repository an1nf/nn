import torch
import torch.nn as nn
from sklearn.datasets import load_digits

ANCHO = 64


class Bloque(nn.Module):
    def __init__(self, residual):
        super().__init__()
        self.residual = residual
        self.f = nn.Sequential(
            nn.Linear(ANCHO, ANCHO), nn.BatchNorm1d(ANCHO), nn.ReLU(),
            nn.Linear(ANCHO, ANCHO), nn.BatchNorm1d(ANCHO),
        )
        if residual:
            # γ = 0 en la última BatchNorm: cada bloque empieza siendo la identidad.
            nn.init.zeros_(self.f[-1].weight)

    def forward(self, x):
        return torch.relu(x + self.f(x)) if self.residual else torch.relu(self.f(x))


class Red(nn.Module):
    def __init__(self, bloques, residual, entrada=64, salida=10):
        super().__init__()
        self.entrada = nn.Sequential(nn.Linear(entrada, ANCHO), nn.BatchNorm1d(ANCHO), nn.ReLU())
        self.bloques = nn.ModuleList(Bloque(residual) for _ in range(bloques))
        self.salida = nn.Linear(ANCHO, salida)

    def forward(self, x):
        x = self.entrada(x)
        for bloque in self.bloques:
            x = bloque(x)
        return self.salida(x)


def datos():
    digitos = load_digits()
    X = (digitos.data - digitos.data.mean(0)) / (digitos.data.std(0) + 1e-8)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(digitos.target)


def entrenar(modelo, X, y, epocas=25, tasa=0.05):
    optimizador = torch.optim.SGD(modelo.parameters(), lr=tasa, momentum=0.9)
    generador = torch.Generator().manual_seed(0)
    for _ in range(epocas):
        modelo.train()
        for lote in torch.randperm(len(X), generator=generador).split(64):
            perdida = nn.functional.cross_entropy(modelo(X[lote]), y[lote])
            optimizador.zero_grad()
            perdida.backward()
            optimizador.step()
    modelo.eval()
    with torch.no_grad():
        return nn.functional.cross_entropy(modelo(X), y).item()


if __name__ == "__main__":
    X, y = datos()
    for tipo in ("plana", "residual"):
        torch.manual_seed(0)
        print(f"{tipo:>8}, 66 capas: pérdida final {entrenar(Red(32, residual=tipo == 'residual'), X, y):.3f}")

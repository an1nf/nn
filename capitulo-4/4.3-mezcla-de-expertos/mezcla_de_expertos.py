import math

import torch
import torch.nn as nn


def experto(oculta):
    return nn.Sequential(nn.Linear(2, oculta), nn.Tanh(), nn.Linear(oculta, oculta), nn.Tanh(), nn.Linear(oculta, 1))


class MezclaDeExpertos(nn.Module):
    def __init__(self, expertos=4, oculta=16):
        super().__init__()
        self.enrutador = nn.Linear(2, expertos)
        self.expertos = nn.ModuleList(experto(oculta) for _ in range(expertos))

    def forward(self, x):
        probabilidades = torch.softmax(self.enrutador(x), dim=-1)
        elegido = probabilidades.argmax(-1)
        salidas = torch.zeros(len(x))
        # Top-1; escalar por p da gradiente al enrutador.
        for i, e in enumerate(self.expertos):
            seleccion = elegido == i
            if seleccion.any():
                salidas[seleccion] = e(x[seleccion]).squeeze(-1) * probabilidades[seleccion, i]
        # Pérdida de balanceo de Switch: E · Σ_i f_i P_i, mínima con carga uniforme.
        fraccion = torch.bincount(elegido, minlength=len(self.expertos)).float() / len(x)
        auxiliar = len(self.expertos) * torch.sum(fraccion * probabilidades.mean(0))
        return salidas, elegido, auxiliar


def funcion_objetivo(x):
    regiones = (x[:, 0] > 0).long() * 2 + (x[:, 1] > 0).long()
    candidatos = torch.stack([
        torch.sin(4 * math.pi * x[:, 0]),
        4 * (x[:, 0] ** 2 + x[:, 1] ** 2) - 2,
        torch.sign(torch.sin(5 * math.pi * x[:, 1])),
        3 * x[:, 0] * x[:, 1] - 1,
    ], dim=1)
    return candidatos.gather(1, regiones[:, None]).squeeze(1)


def entrenar(modelo, peso_auxiliar=0.05, pasos=4000):
    generador = torch.Generator().manual_seed(0)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=3e-3)
    for _ in range(pasos):
        X = torch.rand(256, 2, generator=generador) * 2 - 1
        prediccion, _, auxiliar = modelo(X)
        perdida = nn.functional.mse_loss(prediccion, funcion_objetivo(X)) + peso_auxiliar * auxiliar
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()


if __name__ == "__main__":
    torch.manual_seed(1)
    modelo = MezclaDeExpertos()
    entrenar(modelo)
    X = torch.rand(4000, 2, generator=torch.Generator().manual_seed(1)) * 2 - 1
    with torch.no_grad():
        prediccion, elegido, _ = modelo(X)
    print(f"ECM de prueba {nn.functional.mse_loss(prediccion, funcion_objetivo(X)).item():.3f}; "
          f"uso de expertos {(torch.bincount(elegido, minlength=4) / len(X)).numpy().round(2)}")

import math

import torch
import torch.nn as nn


class RecurrenciaDiagonal(nn.Module):
    def __init__(self, variante, clases, canales=8):
        super().__init__()
        self.variante = variante
        if variante.startswith("real con signo"):
            self.magnitud = nn.Parameter(torch.randn(2, canales) * 1.5)
        else:
            self.magnitud = nn.Parameter(torch.randn(2, canales) + 2.0)
        self.angulo = nn.Parameter(torch.randn(2, canales) * 0.5)
        self.inicial = nn.Parameter(torch.ones(canales))
        self.lectura = nn.Sequential(nn.Linear(2 * canales, 64), nn.GELU(), nn.Linear(64, clases))

    def transicion(self):
        if self.variante.startswith("real positivo"):
            return torch.sigmoid(self.magnitud).to(torch.cfloat)
        if self.variante.startswith("real con signo"):
            return torch.tanh(self.magnitud).to(torch.cfloat)
        # λ = e^{iθ}: rotaciones de módulo 1; contar módulo m equivale a girar 2π/m por cada 1.
        return torch.exp(1j * math.pi * torch.tanh(self.angulo))

    def forward(self, bits):
        lambdas = self.transicion()
        h = self.inicial.to(torch.cfloat).expand(len(bits), -1)
        for t in range(bits.shape[1]):
            h = lambdas[bits[:, t]] * h
        return self.lectura(torch.cat([h.real, h.imag], dim=-1))


def lote(n, longitud, modulo, generador):
    bits = torch.randint(0, 2, (n, longitud), generator=generador)
    return bits, bits.sum(1) % modulo


def entrenar(modelo, modulo, pasos=2000, longitudes=(8, 64)):
    generador = torch.Generator().manual_seed(0)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=3e-3)
    for _ in range(pasos):
        longitud = int(torch.randint(longitudes[0], longitudes[1] + 1, (1,), generator=generador))
        bits, y = lote(128, longitud, modulo, generador)
        perdida = nn.functional.cross_entropy(modelo(bits), y)
        optimizador.zero_grad()
        perdida.backward()
        optimizador.step()


@torch.no_grad()
def exactitud(modelo, modulo, longitud):
    bits, y = lote(1000, longitud, modulo, torch.Generator().manual_seed(longitud))
    return (modelo(bits).argmax(-1) == y).float().mean().item()


if __name__ == "__main__":
    for variante in ("real positivo", "real con signo", "complejo unitario"):
        torch.manual_seed(0)
        modelo = RecurrenciaDiagonal(variante, 3)
        entrenar(modelo, 3)
        print(f"{variante:>17}: exactitud en conteo módulo 3 con T = 96: {exactitud(modelo, 3, 96):.2f}")

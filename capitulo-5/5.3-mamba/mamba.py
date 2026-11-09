import math

import torch
import torch.nn as nn
import torch.nn.functional as F

RUIDO, MARCADOR, PRIMER_DATO, VOCABULARIO = 0, 1, 2, 10
CONTEXTO, DATOS = 32, 6


def copia_selectiva(n, generador):
    tokens = torch.full((n, CONTEXTO + DATOS), RUIDO)
    tokens[:, CONTEXTO:] = MARCADOR
    objetivos = torch.randint(PRIMER_DATO, VOCABULARIO, (n, DATOS), generator=generador)
    for i in range(n):
        posiciones = torch.randperm(CONTEXTO, generator=generador)[:DATOS].sort().values
        tokens[i, posiciones] = objetivos[i]
    return tokens, objetivos


class SSMSelectivo(nn.Module):
    def __init__(self, canales, estados=16, selectivo=True):
        super().__init__()
        self.selectivo = selectivo
        self.A_log = nn.Parameter(torch.log(torch.arange(1, estados + 1).float()).repeat(canales, 1))
        self.D = nn.Parameter(torch.ones(canales))
        paso_inicial = torch.exp(torch.rand(canales) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        # Inversa de softplus: Δ arranca log-uniforme en [0.001, 0.1].
        sesgo_de_paso = paso_inicial + torch.log(-torch.expm1(-paso_inicial))
        if selectivo:
            self.proyeccion_paso = nn.Linear(canales, canales)
            with torch.no_grad():
                self.proyeccion_paso.weight.mul_(0.1)
                self.proyeccion_paso.bias.copy_(sesgo_de_paso)
            self.proyeccion_B = nn.Linear(canales, estados, bias=False)
            self.proyeccion_C = nn.Linear(canales, estados, bias=False)
        else:
            self.sesgo_de_paso = nn.Parameter(sesgo_de_paso)
            self.B = nn.Parameter(torch.randn(canales, estados) / math.sqrt(estados))
            self.C = nn.Parameter(torch.randn(canales, estados) / math.sqrt(estados))

    def forward(self, x):
        b, L, canales = x.shape
        A = -torch.exp(self.A_log)
        # Selectividad: Δ, B y C dependen de x_t.
        if self.selectivo:
            paso = F.softplus(self.proyeccion_paso(x))
            B = self.proyeccion_B(x)[:, :, None, :]
            C = self.proyeccion_C(x)[:, :, None, :]
        else:
            paso = F.softplus(self.sesgo_de_paso).expand(b, L, canales)
            B = self.B.expand(b, L, canales, -1)
            C = self.C.expand(b, L, canales, -1)
        # Discretización ZOH con A diagonal.
        A_barra = torch.exp(paso[..., None] * A)
        Bx = paso[..., None] * B * x[..., None]
        h = torch.zeros(b, canales, A.shape[1])
        salidas = []
        for t in range(L):
            h = A_barra[:, t] * h + Bx[:, t]
            salidas.append((h * C[:, t]).sum(-1))
        return torch.stack(salidas, dim=1) + self.D * x


class BloqueMamba(nn.Module):
    def __init__(self, d, selectivo, expansion=2, nucleo=4):
        super().__init__()
        interna = expansion * d
        self.norma = nn.LayerNorm(d)
        self.entrada = nn.Linear(d, 2 * interna)
        self.convolucion = nn.Conv1d(interna, interna, nucleo, groups=interna, padding=nucleo - 1)
        self.ssm = SSMSelectivo(interna, selectivo=selectivo)
        self.salida = nn.Linear(interna, d)

    def forward(self, u):
        x, z = self.entrada(self.norma(u)).chunk(2, dim=-1)
        x = self.convolucion(x.transpose(1, 2))[..., :u.shape[1]].transpose(1, 2)
        y = self.ssm(F.silu(x)) * F.silu(z)
        return u + self.salida(y)


class Modelo(nn.Module):
    def __init__(self, selectivo, d=32, capas=2):
        super().__init__()
        self.incrustacion = nn.Embedding(VOCABULARIO, d)
        self.bloques = nn.ModuleList(BloqueMamba(d, selectivo) for _ in range(capas))
        self.norma = nn.LayerNorm(d)
        self.cabeza = nn.Linear(d, VOCABULARIO)

    def forward(self, tokens):
        x = self.incrustacion(tokens)
        for bloque in self.bloques:
            x = bloque(x)
        return self.cabeza(self.norma(x))


def entrenar(modelo, pasos=1500):
    generador = torch.Generator().manual_seed(0)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=0.0)
    planificador = torch.optim.lr_scheduler.OneCycleLR(optimizador, max_lr=2e-3, total_steps=pasos, pct_start=0.1)
    for _ in range(pasos):
        tokens, objetivos = copia_selectiva(32, generador)
        logits = modelo(tokens)[:, CONTEXTO:]
        perdida = F.cross_entropy(logits.reshape(-1, VOCABULARIO), objetivos.reshape(-1))
        optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        optimizador.step()
        planificador.step()
    X, y = copia_selectiva(256, generador)
    with torch.no_grad():
        return (modelo(X)[:, CONTEXTO:].argmax(-1) == y).float().mean().item()


if __name__ == "__main__":
    for nombre, selectivo in [("invariante en el tiempo (S4)", False), ("selectivo (Mamba)", True)]:
        torch.manual_seed(0)
        print(f"SSM {nombre}: exactitud en copia selectiva {entrenar(Modelo(selectivo)):.3f}")

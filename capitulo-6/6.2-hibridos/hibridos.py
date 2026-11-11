import torch
import torch.nn as nn
import torch.nn.functional as F

CLAVES, VALORES = 64, 64
VOCABULARIO = CLAVES + VALORES
PARES = [4, 8, 16, 24, 32]


def mqar(n, pares, generador):
    claves = torch.stack([torch.randperm(CLAVES, generator=generador)[:pares] for _ in range(n)])
    valores = torch.randint(CLAVES, VOCABULARIO, (n, pares), generator=generador)
    orden = torch.stack([torch.randperm(pares, generator=generador) for _ in range(n)])
    contexto = torch.stack([claves, valores], dim=2).reshape(n, 2 * pares)
    consultas = claves.gather(1, orden)
    objetivos = valores.gather(1, orden)
    return torch.cat([contexto, consultas], dim=1), objetivos


class AtencionSoftmax(nn.Module):
    def __init__(self, d, cabezas):
        super().__init__()
        self.cabezas = cabezas
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.salida = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, L, d = x.shape
        q, k, v = self.qkv(x).view(b, L, 3, self.cabezas, d // self.cabezas).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.salida(y.transpose(1, 2).reshape(b, L, d))


class ConvolucionCorta(nn.Module):
    # Convolución causal por canal, de núcleo corto.
    def __init__(self, canales, nucleo=3):
        super().__init__()
        self.conv = nn.Conv1d(canales, canales, nucleo, groups=canales, padding=nucleo - 1)

    def forward(self, x):
        return F.silu(self.conv(x.transpose(1, 2))[..., :x.shape[1]].transpose(1, 2))


class AtencionLineal(nn.Module):
    def __init__(self, d, cabezas):
        super().__init__()
        self.cabezas = cabezas
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.corta = ConvolucionCorta(3 * d)
        self.salida = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, L, d = x.shape
        q, k, v = self.corta(self.qkv(x)).view(b, L, 3, self.cabezas, d // self.cabezas).permute(2, 0, 3, 1, 4)
        q, k = F.elu(q) + 1, F.elu(k) + 1
        pesos = (q @ k.transpose(-1, -2)).tril()
        y = (pesos @ v) / (pesos.sum(-1, keepdim=True) + 1e-6)
        return self.salida(y.transpose(1, 2).reshape(b, L, d))


class GatedDeltaNet(nn.Module):
    def __init__(self, d, cabezas):
        super().__init__()
        self.cabezas, self.dk = cabezas, d // cabezas
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.corta = ConvolucionCorta(3 * d)
        self.compuertas = nn.Linear(d, 2 * cabezas)
        with torch.no_grad():
            self.compuertas.bias[:cabezas] = 3.0
        self.salida = nn.Linear(d, d, bias=False)

    def forward(self, x):
        b, L, d = x.shape
        q, k, v = self.corta(self.qkv(x)).view(b, L, 3, self.cabezas, self.dk).unbind(2)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
        alfa, beta = torch.sigmoid(self.compuertas(x)).chunk(2, dim=-1)
        S = torch.zeros(b, self.cabezas, self.dk, self.dk)
        salidas = []
        # S_t = α_t S_{t−1} + β_t (v_t − α_t S_{t−1} k_t) k_tᵀ
        for t in range(L):
            kt, vt = k[:, t], v[:, t]
            at, bt = alfa[:, t, :, None, None], beta[:, t, :, None]
            lectura = torch.einsum("bhij,bhj->bhi", at * S, kt)
            S = at * S + torch.einsum("bhi,bhj->bhij", bt * (vt - lectura), kt)
            salidas.append(torch.einsum("bhij,bhj->bhi", S, q[:, t]))
        return self.salida(torch.stack(salidas, dim=1).reshape(b, L, d))


MEZCLADORES = {"A": AtencionSoftmax, "L": AtencionLineal, "D": GatedDeltaNet}


class Bloque(nn.Module):
    def __init__(self, tipo, d, cabezas):
        super().__init__()
        self.norma_1, self.norma_2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.mezclador = MEZCLADORES[tipo](d, cabezas)
        self.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x):
        x = x + self.mezclador(self.norma_1(x))
        return x + self.mlp(self.norma_2(x))


class Modelo(nn.Module):
    def __init__(self, patron, d=64, cabezas=4, contexto=3 * max(PARES)):
        super().__init__()
        self.incrustacion = nn.Embedding(VOCABULARIO, d)
        self.posicion = nn.Embedding(contexto, d)
        self.bloques = nn.ModuleList(Bloque(tipo, d, cabezas) for tipo in patron)
        self.norma = nn.LayerNorm(d)
        self.cabeza = nn.Linear(d, VOCABULARIO)

    def forward(self, tokens):
        x = self.incrustacion(tokens) + self.posicion(torch.arange(tokens.shape[1]))
        for bloque in self.bloques:
            x = bloque(x)
        return self.cabeza(self.norma(x))


def entrenar(modelo, pasos):
    generador = torch.Generator().manual_seed(0)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=2e-3, weight_decay=0.01)
    planificador = torch.optim.lr_scheduler.OneCycleLR(optimizador, max_lr=2e-3, total_steps=pasos, pct_start=0.1)
    for paso in range(pasos):
        pares = PARES[paso % len(PARES)]
        tokens, objetivos = mqar(32, pares, generador)
        logits = modelo(tokens)[:, 2 * pares:]
        perdida = F.cross_entropy(logits.reshape(-1, VOCABULARIO), objetivos.reshape(-1))
        optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        optimizador.step()
        planificador.step()


@torch.no_grad()
def evaluar(modelo):
    generador = torch.Generator().manual_seed(99)
    resultados = []
    for pares in PARES:
        tokens, objetivos = mqar(256, pares, generador)
        prediccion = modelo(tokens)[:, 2 * pares:].argmax(-1)
        resultados.append((prediccion == objetivos).float().mean().item())
    return resultados


if __name__ == "__main__":
    for patron in ("LL", "DA"):
        torch.manual_seed(0)
        modelo = Modelo(patron)
        entrenar(modelo, pasos=3000)
        print(f"{patron}: exactitud MQAR con " + ", ".join(f"{p}:{r:.2f}" for p, r in zip(PARES, evaluar(modelo))) + " pares")

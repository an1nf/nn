import math

import torch
import torch.nn as nn

VOCABULARIO, LONGITUD, SEGMENTO_MINIMO, SEGMENTO_MAXIMO = 24, 32, 5, 12


class AtencionMulticabeza(nn.Module):
    def __init__(self, d, cabezas):
        super().__init__()
        self.cabezas, self.dk = cabezas, d // cabezas
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proyeccion = nn.Linear(d, d, bias=False)

    def forward(self, x, cache=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.cabezas, self.dk).permute(2, 0, 3, 1, 4)
        # Con caché KV, en la generación solo se calculan q, k, v del símbolo nuevo.
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], dim=2)
                v = torch.cat([cache["v"], v], dim=2)
            cache["k"], cache["v"] = k, v
        puntajes = q @ k.transpose(-1, -2) / math.sqrt(self.dk)
        m = k.shape[2]
        # Máscara causal desplazada m − n posiciones, válida también con caché.
        causal = torch.ones(n, m, dtype=torch.bool).tril(diagonal=m - n)
        pesos = torch.softmax(puntajes.masked_fill(~causal, float("-inf")), dim=-1)
        salida = (pesos @ v).transpose(1, 2).reshape(b, n, d)
        return self.proyeccion(salida)


class Bloque(nn.Module):
    def __init__(self, d, cabezas):
        super().__init__()
        self.norma_1, self.norma_2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.atencion = AtencionMulticabeza(d, cabezas)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, cache=None):
        x = x + self.atencion(self.norma_1(x), cache)
        return x + self.mlp(self.norma_2(x))


class Transformer(nn.Module):
    def __init__(self, vocabulario=VOCABULARIO, d=64, cabezas=2, capas=2, contexto=LONGITUD):
        super().__init__()
        self.incrustacion = nn.Embedding(vocabulario, d)
        self.posicion = nn.Embedding(contexto, d)
        self.bloques = nn.ModuleList(Bloque(d, cabezas) for _ in range(capas))
        self.norma = nn.LayerNorm(d)
        self.cabeza = nn.Linear(d, vocabulario, bias=False)

    def forward(self, tokens, caches=None, desde=0):
        posiciones = torch.arange(desde, desde + tokens.shape[1])
        x = self.incrustacion(tokens) + self.posicion(posiciones)
        for i, bloque in enumerate(self.bloques):
            x = bloque(x, None if caches is None else caches[i])
        return self.cabeza(self.norma(x))


def lote_de_copia(n, generador):
    tokens = torch.randint(0, VOCABULARIO, (n, LONGITUD), generator=generador)
    segmentos = torch.randint(SEGMENTO_MINIMO, SEGMENTO_MAXIMO + 1, (n,), generator=generador)
    inicios = torch.zeros(n, dtype=torch.long)
    mascara = torch.zeros(n, LONGITUD - 1, dtype=torch.bool)
    for i, s in enumerate(segmentos.tolist()):
        a = int(torch.randint(0, LONGITUD - 2 * s + 1, (1,), generator=generador))
        inicios[i] = a
        tokens[i, a + s:a + 2 * s] = tokens[i, a:a + s]
        mascara[i, a + s:a + 2 * s - 1] = True
    return tokens, mascara, inicios, segmentos


def entrenar(modelo, pasos=4000):
    generador = torch.Generator().manual_seed(0)
    optimizador = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=0.01)
    for _ in range(pasos):
        tokens, mascara, _, _ = lote_de_copia(64, generador)
        logits = modelo(tokens[:, :-1])
        objetivo = tokens[:, 1:]
        perdida = nn.functional.cross_entropy(logits.reshape(-1, VOCABULARIO), objetivo.reshape(-1),
                                              reduction="none").view(objetivo.shape)
        optimizador.zero_grad()
        perdida.mean().backward()
        optimizador.step()
    return (logits.argmax(-1) == objetivo)[mascara].float().mean().item()


if __name__ == "__main__":
    torch.manual_seed(0)
    print(f"Exactitud en la segunda aparición del segmento: {entrenar(Transformer()):.3f}")

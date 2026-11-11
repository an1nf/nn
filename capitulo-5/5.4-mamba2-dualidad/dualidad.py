import torch


def recurrente(a, q, k, v):
    S = torch.zeros(v.shape[1], k.shape[1], dtype=v.dtype)
    salidas = []
    for t in range(len(a)):
        S = a[t] * S + torch.outer(v[t], k[t])
        salidas.append(S @ q[t])
    return torch.stack(salidas)


def mascara_semiseparable(a):
    acumulado = torch.cumsum(torch.log(a), dim=0)
    # L[i, j] = a[j+1] ··· a[i] para j ≤ i, calculado en espacio logarítmico.
    diferencias = acumulado[:, None] - acumulado[None, :]
    return torch.exp(diferencias.masked_fill(~torch.ones(len(a), len(a), dtype=torch.bool).tril(),
                                             float("-inf")))


def cuadratica(a, q, k, v):
    # Dualidad SSD: y = (L ∘ Q Kᵀ) V, atención con máscara semiseparable.
    return (mascara_semiseparable(a) * (q @ k.T)) @ v


def por_bloques(a, q, k, v, bloque=64):
    T = len(a)
    S = torch.zeros(v.shape[1], k.shape[1], dtype=v.dtype)
    salidas = []
    for inicio in range(0, T, bloque):
        fin = min(inicio + bloque, T)
        ab, qb, kb, vb = a[inicio:fin], q[inicio:fin], k[inicio:fin], v[inicio:fin]
        acumulado = torch.cumsum(torch.log(ab), dim=0)
        # Dentro del bloque, forma cuadrática; entre bloques, el estado S.
        intra = (mascara_semiseparable(ab) * (qb @ kb.T)) @ vb
        inter = torch.exp(acumulado)[:, None] * (qb @ S.T)
        salidas.append(intra + inter)
        decaimiento = torch.exp(acumulado[-1] - acumulado)
        S = torch.exp(acumulado[-1]) * S + (vb * decaimiento[:, None]).T @ kb
    return torch.cat(salidas)


if __name__ == "__main__":
    generador = torch.Generator().manual_seed(0)
    a = torch.sigmoid(torch.randn(1000, generator=generador, dtype=torch.float64) + 2.5)
    q, k, v = (torch.randn(1000, 16, generator=generador, dtype=torch.float64) / 2 for _ in range(3))
    referencia = recurrente(a, q, k, v)
    for nombre, forma in [("cuadrática", cuadratica), ("por bloques", por_bloques)]:
        print(f"Recurrente frente a {nombre}: diferencia máxima {(forma(a, q, k, v) - referencia).abs().max():.2e}")

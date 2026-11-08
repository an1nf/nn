import torch


def caracteristicas(x):
    return torch.nn.functional.elu(x) + 1


def lineal_paralela(q, k, v):
    fq, fk = caracteristicas(q), caracteristicas(k)
    pesos = (fq @ fk.T) * torch.ones(len(q), len(q)).tril()
    return (pesos @ v) / pesos.sum(-1, keepdim=True)


def lineal_recurrente(q, k, v):
    fq, fk = caracteristicas(q), caracteristicas(k)
    S = torch.zeros(v.shape[1], k.shape[1], dtype=v.dtype)
    z = torch.zeros(k.shape[1], dtype=v.dtype)
    salidas = []
    for t in range(len(q)):
        # S_t = S_{t-1} + v_t φ(k_t)ᵀ: estado de tamaño fijo d × d, costo constante por símbolo.
        S = S + torch.outer(v[t], fk[t])
        z = z + fk[t]
        salidas.append(S @ fq[t] / (z @ fq[t]))
    return torch.stack(salidas)


if __name__ == "__main__":
    torch.manual_seed(0)
    q, k, v = (torch.randn(200, 16, dtype=torch.float64) for _ in range(3))
    diferencia = (lineal_paralela(q, k, v) - lineal_recurrente(q, k, v)).abs().max().item()
    print(f"Forma paralela frente a recurrente: diferencia máxima {diferencia:.2e}")

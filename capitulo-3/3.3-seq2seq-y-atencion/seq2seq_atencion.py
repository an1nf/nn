import torch
import torch.nn as nn

RELLENO, INICIO, FIN = 0, 1, 2
VOCABULARIO = 13
LONGITUD_MINIMA, LONGITUD_MAXIMA = 4, 24


def lote_de_inversion(n, generador, longitud=None):
    if longitud is None:
        longitudes = torch.randint(LONGITUD_MINIMA, LONGITUD_MAXIMA + 1, (n,), generator=generador)
    else:
        longitudes = torch.full((n,), longitud)
    maximo = int(longitudes.max())
    fuente = torch.full((n, maximo), RELLENO)
    entrada_objetivo = torch.full((n, maximo + 1), RELLENO)
    salida_objetivo = torch.full((n, maximo + 1), RELLENO)
    for i, largo in enumerate(longitudes.tolist()):
        simbolos = torch.randint(3, VOCABULARIO, (largo,), generator=generador)
        invertidos = simbolos.flip(0)
        fuente[i, :largo] = simbolos
        entrada_objetivo[i, 0] = INICIO
        entrada_objetivo[i, 1:largo + 1] = invertidos
        salida_objetivo[i, :largo] = invertidos
        salida_objetivo[i, largo] = FIN
    return fuente, longitudes, entrada_objetivo, salida_objetivo


class Seq2Seq(nn.Module):
    def __init__(self, atencion, incrustacion=32, oculta=128):
        super().__init__()
        self.atencion = atencion
        self.incrustar = nn.Embedding(VOCABULARIO, incrustacion, padding_idx=RELLENO)
        self.codificador = nn.GRU(incrustacion, oculta, batch_first=True)
        if atencion:
            self.W_estado = nn.Linear(oculta, oculta, bias=False)
            self.W_memoria = nn.Linear(oculta, oculta)
            self.v = nn.Linear(oculta, 1, bias=False)
            self.salida = nn.Linear(2 * oculta, VOCABULARIO)
        else:
            self.salida = nn.Linear(oculta, VOCABULARIO)
        self.celda = nn.GRUCell(incrustacion, oculta)

    def codificar(self, fuente, longitudes):
        empaquetado = nn.utils.rnn.pack_padded_sequence(self.incrustar(fuente), longitudes, batch_first=True,
                                                        enforce_sorted=False)
        memoria, estado = self.codificador(empaquetado)
        memoria, _ = nn.utils.rnn.pad_packed_sequence(memoria, batch_first=True, total_length=fuente.shape[1])
        return memoria, estado[0], fuente != RELLENO

    def paso(self, simbolo, estado, memoria, claves, mascara):
        estado = self.celda(self.incrustar(simbolo), estado)
        if not self.atencion:
            return self.salida(estado), estado, None
        # Atención aditiva (Bahdanau): e_j = vᵀ tanh(W_s s + W_h h_j); el contexto es Σ α_j h_j.
        puntajes = self.v(torch.tanh(self.W_estado(estado)[:, None, :] + claves)).squeeze(-1)
        alfa = torch.softmax(puntajes.masked_fill(~mascara, float("-inf")), dim=-1)
        contexto = torch.bmm(alfa[:, None, :], memoria).squeeze(1)
        return self.salida(torch.cat([estado, contexto], dim=-1)), estado, alfa

    def forward(self, fuente, longitudes, entrada_objetivo):
        memoria, estado, mascara = self.codificar(fuente, longitudes)
        claves = self.W_memoria(memoria) if self.atencion else None
        salidas = []
        for t in range(entrada_objetivo.shape[1]):
            logits, estado, _ = self.paso(entrada_objetivo[:, t], estado, memoria, claves, mascara)
            salidas.append(logits)
        return torch.stack(salidas, dim=1)

    @torch.no_grad()
    def decodificar(self, fuente, longitudes, pasos):
        memoria, estado, mascara = self.codificar(fuente, longitudes)
        claves = self.W_memoria(memoria) if self.atencion else None
        simbolo = torch.full((len(fuente),), INICIO)
        generados = []
        for _ in range(pasos):
            logits, estado, _ = self.paso(simbolo, estado, memoria, claves, mascara)
            simbolo = logits.argmax(-1)
            generados.append(simbolo)
        return torch.stack(generados, dim=1)


def entrenar(modelo, pasos=3000):
    generador = torch.Generator().manual_seed(1)
    optimizador = torch.optim.Adam(modelo.parameters(), lr=2e-3)
    for _ in range(pasos):
        fuente, longitudes, entrada_objetivo, salida_objetivo = lote_de_inversion(64, generador)
        logits = modelo(fuente, longitudes, entrada_objetivo)
        perdida = nn.functional.cross_entropy(logits.reshape(-1, VOCABULARIO), salida_objetivo.reshape(-1),
                                              ignore_index=RELLENO)
        optimizador.zero_grad()
        perdida.backward()
        nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
        optimizador.step()


def exactitud_por_longitud(modelo, longitudes):
    generador = torch.Generator().manual_seed(123)
    resultados = []
    for largo in longitudes:
        fuente, lens, _, salida_objetivo = lote_de_inversion(300, generador, largo)
        generados = modelo.decodificar(fuente, lens, largo)
        resultados.append((generados == salida_objetivo[:, :largo]).float().mean().item())
    return resultados


if __name__ == "__main__":
    for nombre, atencion in [("sin atención", False), ("con atención", True)]:
        torch.manual_seed(0)
        modelo = Seq2Seq(atencion)
        entrenar(modelo)
        exactitudes = exactitud_por_longitud(modelo, [8, 16, 24])
        print(f"{nombre:>12}: exactitud por símbolo con longitud 8, 16, 24 = "
              + ", ".join(f"{e:.2f}" for e in exactitudes))

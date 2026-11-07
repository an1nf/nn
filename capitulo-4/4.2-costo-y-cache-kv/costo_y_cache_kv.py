import importlib.util
import time
from pathlib import Path

import torch

_especificacion = importlib.util.spec_from_file_location(
    "transformer", Path(__file__).resolve().parents[1] / "4.1-autoatencion-y-transformer" / "transformer.py")
transformer = importlib.util.module_from_spec(_especificacion)
_especificacion.loader.exec_module(transformer)


@torch.no_grad()
def generar(modelo, inicio, nuevos, con_cache):
    if not con_cache:
        secuencia = inicio
        for _ in range(nuevos):
            secuencia = torch.cat([secuencia, modelo(secuencia)[:, -1].argmax(-1, keepdim=True)], dim=1)
        return secuencia
    caches = [{} for _ in modelo.bloques]
    generados = [inicio]
    for posicion in range(nuevos):
        generados.append(modelo(generados[-1], caches, desde=posicion)[:, -1].argmax(-1, keepdim=True))
    return torch.cat(generados, dim=1)


def bytes_por_token(capas, cabezas_kv, dimension_cabeza, bytes_por_valor=2, latente=None):
    # K y V por capa y cabeza; MLA guarda en su lugar un único vector latente comprimido.
    if latente is not None:
        return capas * latente * bytes_por_valor
    return 2 * capas * cabezas_kv * dimension_cabeza * bytes_por_valor


if __name__ == "__main__":
    torch.manual_seed(0)
    modelo = transformer.Transformer(vocabulario=256, d=256, cabezas=4, capas=4, contexto=385).eval()
    inicio = torch.zeros(1, 1, dtype=torch.long)
    resultados = {}
    for con_cache in (False, True):
        t = time.perf_counter()
        resultados[con_cache] = generar(modelo, inicio, 384, con_cache)
        print(f"{'con' if con_cache else 'sin'} caché: {time.perf_counter() - t:.2f} s")
    print(f"Secuencias idénticas: {torch.equal(resultados[False], resultados[True])}")
    for nombre, b in [("MHA", bytes_por_token(32, 32, 128)), ("GQA", bytes_por_token(32, 8, 128)),
                      ("MQA", bytes_por_token(32, 1, 128)), ("MLA", bytes_por_token(32, None, None, latente=576))]:
        print(f"{nombre}: {b / 1024:.0f} KiB por token, {b * 1e6 / 2 ** 30:.0f} GiB a 1 M tokens")

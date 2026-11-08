import numpy as np
from numpy.polynomial import legendre
from scipy.linalg import expm


class SSMDiagonal:
    def __init__(self, estados, paso, rng):
        self.A = -0.5 + 1j * np.pi * np.arange(estados)
        self.B = np.ones(estados, dtype=complex)
        self.C = (rng.normal(size=estados) + 1j * rng.normal(size=estados)) * np.sqrt(0.5)
        # Discretización ZOH con A diagonal: Ā = exp(ΔA), B̄ = (Ā − 1)/A · B.
        self.A_barra = np.exp(paso * self.A)
        self.B_barra = (self.A_barra - 1) / self.A * self.B

    def nucleo(self, longitud):
        potencias = self.A_barra[None, :] ** np.arange(longitud)[:, None]
        return 2 * np.real(potencias @ (self.C * self.B_barra))

    def recurrente(self, u):
        h = np.zeros_like(self.A)
        salidas = []
        for valor in u:
            h = self.A_barra * h + self.B_barra * valor
            salidas.append(2 * np.real(self.C @ h))
        return np.array(salidas)

    def convolucional(self, u):
        L = len(u)
        return np.fft.irfft(np.fft.rfft(u, 2 * L) * np.fft.rfft(self.nucleo(L), 2 * L), 2 * L)[:L]


def hippo_legt(estados):
    i = np.arange(estados)[:, None]
    j = np.arange(estados)[None, :]
    A = (2 * i + 1) * np.where(i < j, -1.0, (-1.0) ** (i - j + 1))
    B = (2 * np.arange(estados) + 1) * (-1.0) ** np.arange(estados)
    return A, B


def reconstruir(estados, ventana, senal):
    # El estado son los coeficientes de Legendre de la última ventana.
    A, B = hippo_legt(estados)
    A_c, B_c = A / ventana, B / ventana
    A_barra = expm(A_c)
    B_barra = np.linalg.solve(A_c, (A_barra - np.eye(estados)) @ B_c)
    m = np.zeros(estados)
    for valor in senal:
        m = A_barra @ m + B_barra * valor
    r = 2 * np.arange(ventana) / ventana - 1
    base = np.array([legendre.legval(r, np.eye(estados)[g]) for g in range(estados)])
    return (m @ base)[::-1]


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ssm = SSMDiagonal(16, 0.02, rng)
    u = np.cumsum(rng.normal(size=600)) * 0.1
    print(f"Recurrencia frente a convolución: diferencia máxima {np.abs(ssm.recurrente(u) - ssm.convolucional(u)).max():.2e}")
    t = np.arange(1200)
    senal = np.sin(2 * np.pi * t / 140) + 0.6 * np.sin(2 * np.pi * t / 47 + 1.0)
    for estados in (8, 32):
        error = reconstruir(estados, 400, senal) - senal[-400:]
        print(f"HiPPO-LegT con {estados} estados: error relativo {np.sqrt(np.mean(error ** 2)) / np.std(senal[-400:]):.3f}")

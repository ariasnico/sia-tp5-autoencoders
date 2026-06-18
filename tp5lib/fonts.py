"""Dataset font.h: 32 caracteres de 7x5 binarios. Carga, exploración y ruido bit-flip.

font.h trae 32 patrones de 7 bytes; de cada byte se usan los 5 bits bajos (columnas).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

H, W = 7, 5
D = H * W  # 35 píxeles por patrón
# 0x60='`', 0x61..0x7e = a..~ (31 chars), 0x7f = DEL
LABELS = [chr(0x60 + i) for i in range(31)] + ["DEL"]


def load_font(path) -> np.ndarray:
    """Parsea font.h -> array (32, 35) de floats {0,1}. Fila-mayor: 7 filas x 5 cols."""
    txt = Path(path).read_text()
    pats = []
    for row in re.findall(r"\{([^{}]*)\}", txt):
        hexs = re.findall(r"0x[0-9a-fA-F]+", row)
        if len(hexs) == 7:
            pats.append([int(x, 16) for x in hexs])
    return np.array(
        [[(b >> (4 - c)) & 1 for b in p for c in range(W)] for p in pats],
        dtype=np.float64,
    )


def bitflip(X: np.ndarray, p: float, rng: np.random.Generator) -> np.ndarray:
    """Voltea cada bit con probabilidad p. Ruido natural para imágenes binarias."""
    mask = rng.random(X.shape) < p
    return np.where(mask, 1.0 - X, X)


def hamming_matrix(X: np.ndarray) -> np.ndarray:
    """Matriz (N,N) de distancia Hamming (# píxeles distintos) entre patrones."""
    N = len(X)
    M = np.zeros((N, N), dtype=int)
    for i in range(N):
        M[i] = (X != X[i]).sum(axis=1)
    return M


def pixel_density(X: np.ndarray) -> np.ndarray:
    """# de píxeles encendidos por patrón."""
    return X.sum(axis=1).astype(int)

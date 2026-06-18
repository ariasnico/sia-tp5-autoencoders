"""Autoencoder = MLP simétrico sobre la librería mlp/ (TP3, numpy puro).

Un AE es un MLP con target = input y un cuello angosto. No se modifica nada de mlp/:
solo se orquestan sus primitivas y se agregan helpers de encode/decode que leen el
cache del forward, más métricas de error en píxeles.
"""
from __future__ import annotations

import numpy as np

from mlp.activations import ACTIVATIONS
from mlp.network import MLP
from mlp.optimizers import Adam


def build_ae(D, hidden=(20,), latent=2, act_hidden="tanh", act_latent="identity",
             act_out="sigmoid", loss="bce", optimizer=None, seed=0) -> MLP:
    """AE simétrico D-[hidden]-latent-[hidden invertido]-D.

    act_latent va en el cuello, act_out en la salida, act_hidden en el resto.
    Con hidden=() queda D-latent-D (útil para el AE lineal ≈ PCA).
    """
    enc = [D] + list(hidden) + [latent]
    sizes = enc + list(hidden)[::-1] + [D]
    nt = len(sizes) - 1               # nº de transiciones (capas de pesos)
    i_latent = len(enc) - 2           # transición cuya salida es el cuello
    acts = []
    for i in range(nt):
        if i == i_latent:
            acts.append(act_latent)
        elif i == nt - 1:
            acts.append(act_out)
        else:
            acts.append(act_hidden)
    return MLP(sizes, acts, loss=loss, optimizer=optimizer or Adam(lr=0.01), seed=seed)


def bottleneck_index(net: MLP) -> int:
    """Índice de la transición de menor tamaño de salida = el cuello latente."""
    return int(np.argmin(net.layer_sizes[1:]))


def encode(net: MLP, X: np.ndarray) -> np.ndarray:
    """Activación en el cuello (representación latente)."""
    _, cache = net.forward(X)
    return cache[bottleneck_index(net)][1]


def decode(net: MLP, Z: np.ndarray) -> np.ndarray:
    """Decodifica desde el latente hasta la salida (recorre el decoder)."""
    a = np.atleast_2d(Z).astype(np.float64)
    for i in range(bottleneck_index(net) + 1, len(net.weights)):
        act_fn = ACTIVATIONS[net.activations[i]][0]
        a = act_fn(np.column_stack([np.ones(len(a)), a]) @ net.weights[i].T)
    return a


def reconstruct_binary(net: MLP, X: np.ndarray) -> np.ndarray:
    """Reconstrucción binarizada a umbral 0.5."""
    return (net.predict_proba(X) > 0.5).astype(np.float64)


def px_err(net: MLP, X: np.ndarray) -> np.ndarray:
    """# píxeles incorrectos por muestra (reconstrucción binarizada vs X)."""
    return (reconstruct_binary(net, X) != X).sum(axis=1).astype(int)


def px_err_clean(net: MLP, noisy: np.ndarray, clean: np.ndarray) -> np.ndarray:
    """# píxeles incorrectos al limpiar `noisy`, comparado con el `clean` original."""
    return (reconstruct_binary(net, noisy) != clean).sum(axis=1).astype(int)


def train_denoising(net: MLP, X: np.ndarray, p_train: float, epochs: int,
                    rng: np.random.Generator) -> np.ndarray:
    """Entrena el AE como denoiser: entrada con bit-flip a prob p_train (ruido FRESCO
    cada época), target = X limpio. Full-batch. Devuelve la curva de loss.

    No usa fit(): el denoising necesita ruido nuevo por época y un target distinto de la
    entrada, así que se orquestan forward/backward/optimizer.step directo.
    """
    from mlp.losses import LOSSES
    from tp5lib.fonts import bitflip
    loss_fn, _ = LOSSES[net.loss_name]
    curve = []
    for _ in range(epochs):
        Xn = bitflip(X, p_train, rng)
        _, cache = net.forward(Xn, training=True)
        grads = net.backward(Xn, X, cache)
        net.optimizer.step(net.weights, grads)
        curve.append(loss_fn(X, net.predict_proba(Xn)))
    return np.array(curve)

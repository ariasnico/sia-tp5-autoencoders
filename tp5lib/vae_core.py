"""
TP5 - 2) VAE from scratch en numpy. Reusa Adam y activaciones de la libreria TP3.
Este archivo define la clase VAE y un chequeo numerico de gradientes (clave para
confiar en el backprop derivado a mano: reparam trick + KL).
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from mlp.optimizers import Adam

def _sigmoid(z):
    out = np.empty_like(z); p = z >= 0; n = ~p
    out[p] = 1/(1+np.exp(-z[p])); e = np.exp(z[n]); out[n] = e/(1+e)
    return out

class VAE:
    """Encoder D->He->(mu,logvar); z=mu+std*eps; Decoder Z->Hd->D (sigmoid)."""
    def __init__(self, D, He, Z, Hd, beta=1.0, lr=1e-3, seed=0):
        rng = np.random.default_rng(seed)
        def he_init(o, i): return rng.normal(0, np.sqrt(2.0/i), size=(o, i))
        self.D, self.He, self.Z, self.Hd, self.beta = D, He, Z, Hd, beta
        self.P = {
            "W_enc": he_init(He, D), "b_enc": np.zeros(He),
            "W_mu":  he_init(Z, He), "b_mu":  np.zeros(Z),
            "W_lv":  np.zeros((Z, He)), "b_lv": np.zeros(Z),   # logvar empieza en 0
            "W_dec": he_init(Hd, Z), "b_dec": np.zeros(Hd),
            "W_out": he_init(D, Hd), "b_out": np.zeros(D),
        }
        self.opt = Adam(lr=lr)
        self._keys = list(self.P.keys())

    def _enc(self, X):
        he = np.tanh(X @ self.P["W_enc"].T + self.P["b_enc"])
        mu = he @ self.P["W_mu"].T + self.P["b_mu"]
        lv = he @ self.P["W_lv"].T + self.P["b_lv"]
        return he, mu, lv

    def _dec(self, z):
        hd = np.tanh(z @ self.P["W_dec"].T + self.P["b_dec"])
        logits = hd @ self.P["W_out"].T + self.P["b_out"]
        return hd, logits

    def loss_and_grads(self, X, eps=None):
        B = len(X)
        he, mu, lv = self._enc(X)
        if eps is None:
            eps = np.random.standard_normal(mu.shape)
        std = np.exp(0.5 * lv)
        z = mu + std * eps
        hd, logits = self._dec(z)
        xhat = _sigmoid(logits)
        # --- loss ---
        recon = np.sum(np.maximum(logits, 0) - logits * X + np.log1p(np.exp(-np.abs(logits)))) / B
        kl = -0.5 * np.sum(1 + lv - mu**2 - np.exp(lv)) / B
        loss = recon + self.beta * kl
        # --- backward ---
        g = {}
        dlogits = (xhat - X) / B
        g["W_out"] = dlogits.T @ hd; g["b_out"] = dlogits.sum(0)
        dhd = dlogits @ self.P["W_out"]
        dpre_d = dhd * (1 - hd**2)
        g["W_dec"] = dpre_d.T @ z; g["b_dec"] = dpre_d.sum(0)
        dz = dpre_d @ self.P["W_dec"]
        # reparam split
        dmu = dz + self.beta * mu / B
        dlv = dz * eps * 0.5 * std + self.beta * 0.5 * (np.exp(lv) - 1) / B
        g["W_mu"] = dmu.T @ he; g["b_mu"] = dmu.sum(0)
        g["W_lv"] = dlv.T @ he; g["b_lv"] = dlv.sum(0)
        dhe = dmu @ self.P["W_mu"] + dlv @ self.P["W_lv"]
        dpre_e = dhe * (1 - he**2)
        g["W_enc"] = dpre_e.T @ X; g["b_enc"] = dpre_e.sum(0)
        return loss, g, dict(recon=recon, kl=kl)

    def step(self, X):
        loss, g, parts = self.loss_and_grads(X)
        ws = [self.P[k] for k in self._keys]
        gs = [g[k] for k in self._keys]
        self.opt.step(ws, gs)
        for k, w in zip(self._keys, ws): self.P[k] = w
        return loss, parts

    def reconstruct(self, X):
        _, mu, _ = self._enc(X)              # usa la media (sin ruido) para reconstruir
        _, logits = self._dec(mu)
        return _sigmoid(logits)

    def generate(self, zs):
        _, logits = self._dec(np.atleast_2d(zs))
        return _sigmoid(logits)

    def encode_mean(self, X):
        _, mu, _ = self._enc(X); return mu


def gradient_check():
    rng = np.random.default_rng(1)
    D, He, Z, Hd = 6, 5, 2, 4
    vae = VAE(D, He, Z, Hd, beta=0.7, seed=2)
    # romper simetria de W_lv para testear sus grads
    vae.P["W_lv"] = rng.normal(0, 0.3, size=(Z, He)); vae.P["b_lv"] = rng.normal(0,0.3,size=Z)
    X = (rng.random((4, D)) > 0.5).astype(float)
    eps = rng.standard_normal((4, Z))          # FIJO para que la funcion sea deterministica
    loss0, g, _ = vae.loss_and_grads(X, eps=eps)
    h = 1e-6; max_rel = 0
    for k in vae._keys:
        W = vae.P[k]; flat = W.ravel()
        gnum = np.zeros_like(flat)
        for i in range(flat.size):
            old = flat[i]
            flat[i] = old + h; lp,_,_ = vae.loss_and_grads(X, eps=eps)
            flat[i] = old - h; lm,_,_ = vae.loss_and_grads(X, eps=eps)
            flat[i] = old
            gnum[i] = (lp - lm) / (2*h)
        gana = g[k].ravel()
        rel = np.abs(gana - gnum) / (np.abs(gana) + np.abs(gnum) + 1e-12)
        print(f"  {k:6s} max rel err = {rel.max():.2e}")
        max_rel = max(max_rel, rel.max())
    print(f"\n>>> MAX REL ERR GLOBAL = {max_rel:.2e}  ({'OK' if max_rel < 1e-5 else 'REVISAR'})")

if __name__ == "__main__":
    gradient_check()

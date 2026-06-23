"""Genera tablas-imagen de setup/resultados para las slides (diseño limpio, listas para pegar).

Una sola función de render reutilizable. Cada celda se auto-ajusta: si el texto no entra en su
columna, baja el tamaño de fuente automáticamente -> nada se sale del recuadro. Salida en tablas/.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle  # noqa: E402

HEADER, VALUE, PARAM, ZEBRA, TEXT = "#1E3A5F", "#C2410C", "#0F172A", "#EEF2F7", "#475569"
OUT = Path(__file__).resolve().parents[1] / "tablas"; OUT.mkdir(exist_ok=True)
PAD = 0.008


def render_table(title, headers, rows, cols, fname, fs=13, w=15):
    """cols: lista de (x0, align, style); style in {'param','value','text'}."""
    n = len(rows)
    fig, ax = plt.subplots(figsize=(w, 1.25 + 0.46 * n))
    ax.axis("off"); ax.set_xlim(0, 1)
    top = n
    ax.set_ylim(-0.55, top + 1.7)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    edges = [c[0] for c in cols] + [1.0]

    def cell(x0, al, s, max_w, y, **kw):
        dx = PAD if al == "left" else (-PAD if al == "right" else 0)
        t = ax.text(x0 + dx, y, s, ha=al, va="center", **kw)
        if s:
            bb = t.get_window_extent(renderer=rend)
            inv = ax.transData.inverted()
            wt = inv.transform((bb.x1, 0))[0] - inv.transform((bb.x0, 0))[0]
            if wt > max_w > 0:
                t.set_fontsize(t.get_fontsize() * max_w / wt * 0.96)
        return t

    cell(0.004, "left", title, 1.0, top + 1.25, fontsize=fs + 8, fontweight="bold", color=HEADER)
    ax.plot([0, 1], [top + 0.72, top + 0.72], color=HEADER, lw=2.4)

    ax.add_patch(Rectangle((0, top + 0.05), 1, 0.9, color=HEADER, ec="none"))
    for i, ((x0, al, _), h) in enumerate(zip(cols, headers)):
        cell(x0, al, h, edges[i + 1] - x0 - 2 * PAD, top + 0.5, fontsize=fs, fontweight="bold", color="white")

    for r, row in enumerate(rows):
        yy = top - 1 - r + 0.5
        if r % 2 == 0:
            ax.add_patch(Rectangle((0, yy - 0.5), 1, 1.0, color=ZEBRA, ec="none"))
        for i, ((x0, al, st), val) in enumerate(zip(cols, row)):
            kw = dict(fontsize=fs - 1)
            if st == "param":
                kw.update(color=PARAM, fontweight="bold")
            elif st == "value":
                kw.update(color=VALUE, family="monospace", fontsize=fs - 1.5)
            else:
                kw.update(color=TEXT)
            cell(x0, al, str(val), edges[i + 1] - x0 - 2 * PAD, yy, **kw)

    fig.savefig(OUT / fname, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig); print("  ", fname)


CFG = [(0.0, "left", "param"), (0.21, "left", "value"), (0.55, "left", "text")]
RES = [(0.0, "left", "param"), (0.06, "left", "text"), (0.38, "left", "value"), (0.66, "left", "text")]

render_table(
    "1a · Autoencoder — configuración",
    ["Parámetro", "Valor", "Por qué"],
    [
        ["Arquitectura", "35-20-2-20-35", "MLP simétrico; cuello 2D (enunciado)"],
        ["Activaciones", "tanh / identity / sigmoid", "identity en latente; sigmoid → [0,1]"],
        ["Loss", "BCE", "pixeles binarios; MSE deja 2px (E7)"],
        ["Optimizador", "Adam (lr=0.01)", "converge donde SGD se estanca (E4)"],
        ["Épocas", "6000 (sin early-stop)", "de sobra: ya a ~1500 da 0px"],
        ["Batch", "32 (full-batch)", "son solo 32 muestras"],
        ["Inicialización", "auto (Xavier / He)", "según la activación de cada capa"],
        ["Dataset", "32 letras 7x5 · sin split", "memorizar las 32 (no generalizar)"],
        ["Métrica", "px incorrectos (umbral 0.5)", "objetivo: máximo ≤ 1"],
        ["Seed", "0", "reproducible"],
    ], CFG, "tabla_1a_config.png")

render_table(
    "1a · Autoencoder — experimentos: qué, para qué y resultado",
    ["Exp", "Para qué (la pregunta)", "Qué variamos", "Resultado / lectura"],
    [
        ["E1", "¿hace falta no-linealidad?", "modelo: lineal/PCA/no-lineal", "lineal=PCA=7.19px → no-lineal 0px"],
        ["E2", "¿cuántas dim. latentes?", "latente: 1,2,3,5,8", "1→18/32;  ≥2→0px  (codo)"],
        ["E3", "¿cuánta capacidad de red?", "oculta: (),(10),(20),(30),(20,20)", "()→15px máx (encoder lineal≈PCA); ≥20→0px"],
        ["E4", "¿qué optimizador converge?", "SGD/Momentum/Adam", "Adam 0 · Mom 2 · SGD 5 px (máx)"],
        ["E5", "¿sensibilidad al learning rate?", "lr: 0.0003/0.01/0.3", "solo 0.01 llega a 0px (extremos: 5 y 28px)"],
        ["E6", "¿qué activación oculta?", "tanh/relu/sigmoid", "las 3 → 0px (distinta velocidad)"],
        ["E7", "¿qué pérdida para binario?", "BCE/MSE", "BCE 0px · MSE 2px"],
    ], RES, "tabla_1a_resultados.png")

render_table(
    "1b · Denoising — configuración",
    ["Parámetro", "Valor", "Por qué"],
    [
        ["Arquitectura", "35-30-{cuello}-30-35", "cuello 10 (E9); capa oculta 30 (E9b)"],
        ["Entrenamiento", "bit-flip(p) → limpio", "entrada ruidosa, target limpio"],
        ["p_train (ganador)", "0.15", "compromiso del trade-off (E10)"],
        ["Loss / Opt", "BCE / Adam(0.01)", "igual que 1a"],
        ["Épocas", "6000 · ganador 15000", "el ganador se refuerza"],
        ["Evaluación", "30-50 realiz. x 32 letras", "robustez estadística"],
        ["Seed", "0", "reproducible"],
    ], CFG, "tabla_1b_config.png")

render_table(
    "1b · Denoising — experimentos: qué, para qué y resultado",
    ["Exp", "Para qué (la pregunta)", "Qué variamos", "Resultado / lectura"],
    [
        ["E9", "¿qué cuello para limpiar?", "cuello: 2,5,10,20", "2→44% · 10→54% · 20 satura"],
        ["E9b", "¿qué ancho de capa oculta?", "oculta: 20,30,35,40", "20→54% · 30→63% · plateau → 30 (parsimonia)"],
        ["E10", "¿cuánto ruido en el train?", "p_train: 0.05/0.15/0.30", "curvas se cruzan: trade-off"],
    ], RES, "tabla_1b_resultados.png")

render_table(
    "VAE · configuración",
    ["Parámetro", "Valor", "Por qué"],
    [
        ["Arquitectura", "D=400·He=128·Z=2·Hd=128", "z=μ+σ·ε, decoder → sigmoid"],
        ["Loss", "recon(BCE) + beta·KL", "reparam + KL a mano (grad check 5e-08)"],
        ["β (ganador)", "1", "equilibrio: ordena el mapa sin perder calidad (E12)"],
        ["Optimizador", "Adam (lr=1e-3)", ""],
        ["Épocas / batch", "3500 / 128", ""],
        ["Dataset", "5 emojis · 700 · 20x20", "canal alpha (siluetas); acc 0.95"],
        ["Augmentación", "rot ±15° · trasl ±2 · ruido 0.03", "variedad intra-clase (seedeada)"],
        ["Sampleo", "semilla 26", "elegida por cubrir las 5 clases"],
        ["Seed", "0", "reproducible"],
    ], CFG, "tabla_vae_config.png")

render_table(
    "VAE · experimento: qué, para qué y resultado",
    ["Exp", "Para qué (la pregunta)", "Qué variamos", "Resultado / lectura"],
    [
        ["E12", "¿cuánto pesa el KL?", "beta: 0/0.5/1/4", "β=0 ruido; β=1 equilibrio; β=4 sobre-reg."],
        ["—", "ídem, visto por reconstrucción", "recon % por β: 0/1/4", "2.6% · 3.2% · 3.7%  (peor con β↑)"],
    ], RES, "tabla_vae_resultados.png")

print("OK ->", OUT)

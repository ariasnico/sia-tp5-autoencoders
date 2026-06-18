"""Estilo de figuras consistente para la entrega del TP5.

Todos los make_figures.py llaman apply_style() al inicio para fuentes legibles, paleta
consistente y DPI adecuado para slides. Colores nombrados para reusar entre ejercicios.
"""
import matplotlib as mpl

PRIMARY = "#C2410C"     # naranja — serie principal / destaque
SECONDARY = "#1E3A5F"   # azul oscuro — serie secundaria
ACCENT = "#16A34A"      # verde — "justo"/ok
WARN = "#DC2626"        # rojo — caso que falla
# paleta cualitativa para clases (emojis / categorías), alto contraste
PALETTE = ["#2563EB", "#F59E0B", "#16A34A", "#DC2626", "#7C3AED", "#0891B2", "#DB2777", "#65A30D"]


def apply_style():
    mpl.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 140,
        "savefig.bbox": "tight",
        "font.family": "DejaVu Sans",
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.titleweight": "bold",
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "figure.titlesize": 16,
        "figure.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "image.cmap": "binary",
    })

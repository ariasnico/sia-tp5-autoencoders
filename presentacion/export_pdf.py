"""Exporta la presentación a PDF — una página por diapo, WYSIWYG.

En vez de usar el print-pdf de reveal (que recorta las diapos por el uso de unidades vh),
captura cada diapo como screenshot a 1280×720 con Chrome headless (exactamente como se ve)
y arma el PDF con Pillow. Sirve el dir localmente; usa index.html (reveal desde CDN → necesita
internet, igual que abrir la presentación).

Uso:
    python3 export_pdf.py             # -> presentacion.pdf
    python3 export_pdf.py salida.pdf  # nombre custom
"""
import http.server
import re
import shutil
import socketserver
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.image as mpimg  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "presentacion.pdf"
SRC = "index.html"
W, H, SCALE = 1280, 720, 2          # resolución de captura (×2 = más nítido)
PORT = 8123

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "google-chrome", "chromium", "chromium-browser",
]


def find_chrome():
    for c in CHROME_CANDIDATES:
        if c.startswith("/") and Path(c).exists():
            return c
        if not c.startswith("/") and shutil.which(c):
            return shutil.which(c)
    return None


def count_slides():
    html = (HERE / SRC).read_text(encoding="utf-8")
    body = html[html.find('class="slides"'):]
    return len(re.findall(r"<section\b", body))


def serve():
    handler = lambda *a, **k: http.server.SimpleHTTPRequestHandler(*a, directory=str(HERE), **k)
    httpd = socketserver.TCPServer(("127.0.0.1", PORT), handler)
    httpd.allow_reuse_address = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def shoot(chrome, url, out_png):
    cmd = [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
           f"--force-device-scale-factor={SCALE}",
           f"--window-size={W},{H}", "--virtual-time-budget=4000",
           f"--screenshot={out_png}", url]
    subprocess.run(cmd, capture_output=True, text=True, timeout=60)


def main():
    chrome = find_chrome()
    if not chrome:
        sys.exit("No encontré Chrome/Chromium/Edge/Brave.")
    n = count_slides()
    print(f"Diapos: {n} · Chrome: {chrome}")
    httpd = serve()
    tmp = Path(tempfile.mkdtemp(prefix="slides_"))
    pngs = []
    try:
        for i in range(n):
            png = tmp / f"s{i:03d}.png"
            url = f"http://127.0.0.1:{PORT}/{SRC}?pdf#/{i}"
            shoot(chrome, url, png)
            if not png.exists():
                print(f"  [aviso] no se capturó la diapo {i}"); continue
            pngs.append(png)
            print(f"  diapo {i+1}/{n} ✓", end="\r")
    finally:
        httpd.shutdown()

    if not pngs:
        sys.exit("No se capturó ninguna diapo.")
    with PdfPages(OUT) as pdf:
        for p in pngs:
            img = mpimg.imread(str(p))
            h, w = img.shape[:2]
            fig = plt.figure(figsize=(w / 150, h / 150), dpi=150)
            ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
            ax.imshow(img)
            pdf.savefig(fig); plt.close(fig)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"\nOK -> {OUT} ({OUT.stat().st_size/1024/1024:.2f} MB, {len(pngs)} páginas)")


if __name__ == "__main__":
    main()

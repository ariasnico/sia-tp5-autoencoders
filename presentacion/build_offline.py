import base64, re, os, subprocess, sys, urllib.request

# Sincroniza presentacion/figs/ con las figuras fuente ANTES de embeber,
# así el offline nunca queda con copias viejas. (Paso instantáneo.)
print("Sincronizando figuras (sync_figs.py)...")
subprocess.run([sys.executable, "sync_figs.py"], check=True,
               cwd=os.path.dirname(os.path.abspath(__file__)))

SRC = "index.html"
OUT = "presentacion_offline.html"
h = open(SRC, encoding="utf-8").read()

CDN = [
    ("css", "reveal.css",  "https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.css"),
    ("css", "white.css",   "https://cdn.jsdelivr.net/npm/reveal.js@5/dist/theme/white.css"),
    ("js",  "reveal.js",   "https://cdn.jsdelivr.net/npm/reveal.js@5/dist/reveal.js"),
    ("js",  "notes.js",    "https://cdn.jsdelivr.net/npm/reveal.js@5/plugin/notes/notes.js"),
]

def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60).read().decode("utf-8")

for kind, name, url in CDN:
    body = fetch(url)
    if kind == "css":
        body = re.sub(r"@import[^;]+;", "", body)   # quita @import de fuentes (relativos, no resuelven offline)
        block = f"<style>\n/* {name} (inline) */\n{body}\n</style>"
        link = f'<link rel="stylesheet" href="{url}">'
    else:
        block = f"<script>\n/* {name} (inline) */\n{body}\n</script>"
        link = f'<script src="{url}"></script>'
    assert link in h, f"no encontre el tag para {name}"
    h = h.replace(link, block)
    print(f"  inline {name}: {len(body)/1024:.0f} KB")

def img_data(m):
    path = m.group(1)
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f'src="data:image/png;base64,{b}"'

h, n = re.subn(r'src="(figs/[^"]+\.png)"', img_data, h)
print(f"  imagenes embebidas: {n}")

open(OUT, "w", encoding="utf-8").write(h)

left_https = len(re.findall(r'(?:href|src)="https://', h))
left_rel = len(re.findall(r'src="(?:\.\./|figs/)', h))
print(f"\nOK -> {OUT}: {os.path.getsize(OUT)/1024/1024:.2f} MB")
print(f"refs https restantes: {left_https} | refs a archivos restantes: {left_rel}")

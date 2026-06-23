import base64, re, os

SRC = "experimentos.html"
OUT = "experimentos_offline.html"
h = open(SRC, encoding="utf-8").read()

def img_data(m):
    path = m.group(1)
    with open(path, "rb") as f:
        b = base64.b64encode(f.read()).decode("ascii")
    return f'src="data:image/png;base64,{b}"'

h, n = re.subn(r'src="(figs/[^"]+\.png)"', img_data, h)
open(OUT, "w", encoding="utf-8").write(h)

left = len(re.findall(r'src="(?:\.\./|figs/)', h))
print(f"imagenes embebidas: {n}")
print(f"refs relativas '../' restantes: {left}")
print(f"tamano {OUT}: {os.path.getsize(OUT)/1024/1024:.2f} MB")

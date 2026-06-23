"""Sincroniza las figuras USADAS por la presentación a presentacion/figs/{ej1a,ej1b,ej2}/.

La presentación referencia sus imágenes localmente (figs/ejXX/...), no a las carpetas de
cada ejercicio. Este script copia, para cada PNG que los HTML realmente usan, la mejor fuente
disponible. Correr DESPUÉS de regenerar cualquier figura:

    python3 sync_figs.py

PRIORIDAD DE FUENTE (de mayor a menor):
  1. figs_manual/<bucket>/<archivo>  → override MANUAL. Ningún generador escribe acá; lo que
     pongas queda "pineado" y nunca se pisa al regenerar. Usalo para ediciones a mano
     (p.ej. recuadros amarillos hechos en un editor, o una versión retocada).
  2. carpetas fuente del ejercicio (ej1a_secuencial/figs, ej1b_denoising/figs, ...).

Es decir: si existe figs_manual/ej1a/fig_X.png, se usa ESA aunque make_figures.py regenere
la de la carpeta fuente. Para "soltar" un override, borrás el archivo de figs_manual/.

Qué copiar se determina parseando index.html y experimentos.html (refs figs/ejXX/<archivo>).
"""
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HTMLS = [HERE / "index.html", HERE / "experimentos.html"]
MANUAL_DIR = HERE / "figs_manual"   # overrides manuales: figs_manual/<bucket>/<archivo>

# Cada bucket (ej1a/ej1b/ej2) se nutre de una o más carpetas generadoras.
SOURCE_DIRS = {
    "ej1a": [ROOT / "ej1a_secuencial" / "figs"],
    "ej1b": [ROOT / "ej1b_denoising" / "figs", ROOT / "tablas"],
    "ej2":  [ROOT / "ej2_vae" / "figs", ROOT / "tablas"],
}

REF_RE = re.compile(r'src="figs/(ej1a|ej1b|ej2)/([^"]+\.png)"')


def needed_files():
    """Devuelve {bucket: set(nombres de archivo)} según lo que usan los HTML."""
    needed = {b: set() for b in SOURCE_DIRS}
    for html in HTMLS:
        if not html.exists():
            continue
        for bucket, fname in REF_RE.findall(html.read_text(encoding="utf-8")):
            needed[bucket].add(fname)
    return needed


def find_source(bucket, fname):
    """Devuelve (path, es_manual). El override manual tiene prioridad sobre la fuente."""
    man = MANUAL_DIR / bucket / fname
    if man.exists():
        return man, True
    for d in SOURCE_DIRS[bucket]:
        p = d / fname
        if p.exists():
            return p, False
    return None, False


def main():
    needed = needed_files()
    total, n_manual, missing, manual_list = 0, 0, [], []
    for bucket, files in needed.items():
        dest = HERE / "figs" / bucket
        dest.mkdir(parents=True, exist_ok=True)
        for fname in sorted(files):
            src, is_manual = find_source(bucket, fname)
            if src is None:
                missing.append(f"{bucket}/{fname}")
                continue
            shutil.copy2(src, dest / fname)
            total += 1
            if is_manual:
                n_manual += 1
                manual_list.append(f"{bucket}/{fname}")
        print(f"  {bucket}: {len(files)} usadas")
    print(f"\nOK -> {total} figuras copiadas a presentacion/figs/ ({n_manual} desde figs_manual/)")
    if manual_list:
        print("  overrides manuales (no se regeneran):")
        for m in manual_list:
            print("   *", m)
    if missing:
        print(f"\n[FALTAN {len(missing)} fuentes — regenerá la figura primero]:")
        for m in missing:
            print("  -", m)
        sys.exit(1)


if __name__ == "__main__":
    main()

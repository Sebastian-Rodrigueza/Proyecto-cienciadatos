"""
Descarga CAMELS-COL v3 desde Zenodo (DOI 10.5281/zenodo.18794895) a la carpeta `Camels col/`.

Por defecto descarga solo lo que usa el proyecto (unos 200 MB): series diarias, tablas de atributos y límites de
cuenca. Con --todo descarga también la red de drenaje, las figuras y el índice topográfico de humedad (3,2 GB).

Uso (entorno `proyecto`):
    python descargar_datos.py          # datos del proyecto
    python descargar_datos.py --todo   # conjunto completo
"""

import hashlib
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/18794895"
DESTINO = Path(__file__).resolve().parent / "Camels col"
OPCIONALES = ("12_", "13_", "14_")


def md5(ruta):
    h = hashlib.md5()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def descargar(url, ruta):
    with urllib.request.urlopen(url) as r, open(ruta, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        leido = 0
        for bloque in iter(lambda: r.read(1 << 20), b""):
            f.write(bloque)
            leido += len(bloque)
            if total:
                print(f"\r  {leido / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
    print()


def main(todo=False):
    DESTINO.mkdir(exist_ok=True)
    with urllib.request.urlopen(RECORD) as r:
        archivos = json.load(r)["files"]

    for a in archivos:
        nombre, esperado = a["key"], a["checksum"].split(":")[1]
        if nombre.startswith(OPCIONALES) and not todo:
            continue
        ruta = DESTINO / nombre
        carpeta = DESTINO / Path(nombre).stem
        if nombre.endswith(".zip") and carpeta.exists():
            print(f"Ya existe: {carpeta.name}")
            continue
        if not nombre.endswith(".zip") and ruta.exists() and md5(ruta) == esperado:
            print(f"Ya existe: {nombre}")
            continue

        print(f"Descargando {nombre} ({a['size'] / 1e6:.1f} MB)")
        descargar(a["links"]["self"], ruta)
        if md5(ruta) != esperado:
            ruta.unlink()
            sys.exit(f"Error: la suma MD5 de {nombre} no coincide. Vuelva a ejecutar el script.")

        if nombre.endswith(".zip"):
            with zipfile.ZipFile(ruta) as z:
                z.extractall(carpeta)
            ruta.unlink()

    print(f"Datos listos en {DESTINO}")


if __name__ == "__main__":
    main(todo="--todo" in sys.argv)

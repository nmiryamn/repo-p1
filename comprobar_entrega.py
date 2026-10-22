#!/usr/bin/env python3
"""
Comprueba que la entrega de la Practica 1 esta completa.

Uso:   python comprobar_entrega.py

No evalua la CALIDAD del trabajo, solo que no falta nada y que no se han
subido ficheros que no deben estar. Pasar esta comprobacion es condicion
necesaria para que la entrega se corrija, no suficiente para aprobarla.
"""

import csv
import json
import pathlib
import sys

RAIZ = pathlib.Path(__file__).resolve().parent
OK, FALLO, AVISO = "[ OK ]", "[FALTA]", "[AVISO]"
fallos = 0
avisos = 0


def exige(ruta, descripcion):
    global fallos
    p = RAIZ / ruta
    if p.exists():
        print(f"{OK} {descripcion}")
        return p
    fallos += 1
    print(f"{FALLO} {descripcion}  ->  {ruta}")
    return None


def sugiere(cond, mensaje):
    global avisos
    if not cond:
        avisos += 1
        print(f"{AVISO} {mensaje}")


print("=" * 62)
print("COMPROBACION DE LA ENTREGA - Practica 1")
print("=" * 62)

print("\n--- Documentos ---")
exige("informe/informe.pdf", "Informe en PDF")
exige("p1_plantilla.ipynb", "Cuaderno de trabajo")

print("\n--- Corpus ---")
frases = exige("frases.tsv", "Corpus de 20 frases")
if frases:
    with open(frases, encoding="utf-8") as f:
        filas = list(csv.DictReader(f, delimiter="\t"))
    faltan = {"id", "categoria", "texto", "leida"} - set(filas[0] if filas else {})
    if faltan:
        fallos += 1
        print(f"{FALLO} faltan columnas en frases.tsv: {sorted(faltan)}")
    else:
        from collections import Counter

        cuenta = Counter(f["categoria"] for f in filas)
        exigido = {"decl": 6, "int_abs": 4, "int_parc": 4, "cifras": 3, "siglas": 3}
        print(f"{OK} {len(filas)} frases, columnas correctas")
        for cat, n in exigido.items():
            if cuenta.get(cat, 0) < n:
                fallos += 1
                print(f"{FALLO} categoria '{cat}': {cuenta.get(cat, 0)} de {n}")
        vacias = [f["id"] for f in filas if not f["leida"].strip()]
        if vacias:
            fallos += 1
            print(f"{FALLO} columna 'leida' vacia en: {vacias[:5]}")

print("\n--- Resultados ---")
res = RAIZ / "resultados"
csvs = sorted(res.glob("*.csv")) if res.is_dir() else []
pngs = sorted(res.glob("*.png")) if res.is_dir() else []
if csvs:
    print(f"{OK} {len(csvs)} fichero(s) .csv: {', '.join(p.name for p in csvs[:6])}")
else:
    fallos += 1
    print(f"{FALLO} no hay ningun .csv en resultados/")
sugiere(pngs, "no hay figuras (.png) en resultados/: el informe las necesita")
for esperado in ("rtf.csv", "inteligibilidad.csv"):
    sugiere((res / esperado).exists(), f"no encuentro resultados/{esperado}")

print("\n--- Panel MOS ---")
clave = exige("mos_clave.json", "Clave del panel MOS")
if clave:
    datos = json.load(open(clave, encoding="utf-8"))
    print(f"{OK} {len(datos)} muestras codificadas")
muestras = sorted((RAIZ / "muestras").glob("*.wav")) if (RAIZ / "muestras").is_dir() else []
if muestras:
    print(f"{OK} {len(muestras)} muestra(s) de audio entregadas")
else:
    fallos += 1
    print(f"{FALLO} no hay audio en muestras/")

print("\n--- Codigo ---")
for s in ("tts_util.py", "adaptadores.py", "analisis.py", "evaluacion.py"):
    exige(f"scripts/{s}", f"scripts/{s}")

print("\n--- Lo que NO debe estar ---")
pesados = [p for p in RAIZ.rglob("*")
           if p.is_file() and ".git" not in p.parts and p.stat().st_size > 20_000_000]
if pesados:
    fallos += 1
    for p in pesados[:5]:
        print(f"{FALLO} fichero de {p.stat().st_size // 1_000_000} MB: "
              f"{p.relative_to(RAIZ)}")
    print("        Los modelos y el audio no se suben: revisad .gitignore")
else:
    print(f"{OK} ningun fichero por encima de 20 MB")

n_audio = len(list((RAIZ / "audio").rglob("*.wav"))) if (RAIZ / "audio").is_dir() else 0
sugiere(n_audio == 0, f"hay {n_audio} .wav en audio/, que no deberia subirse")

print("\n" + "=" * 62)
if fallos:
    print(f"RESULTADO: entrega INCOMPLETA - {fallos} elemento(s) faltan.")
    sys.exit(1)
if avisos:
    print(f"RESULTADO: entrega completa, con {avisos} aviso(s) que revisar.")
    sys.exit(0)
print("RESULTADO: entrega completa.")
sys.exit(0)

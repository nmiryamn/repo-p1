"""
Utilidades comunes de la Practica 1 — Sintesis de voz.

Procesamiento del Habla · ETSI · Universidad de Huelva

Este fichero lo proporciona el profesorado. No hace falta modificarlo.
"""

import contextlib
import os
import pathlib

# ---------------------------------------------------------------------------
# Rutas: la practica funciona en NIMROD y en un ordenador propio
# ---------------------------------------------------------------------------
# En NIMROD el trabajo vive en /data/equipo/p1-tts, que es un disco grande
# montado solo para vuestro equipo. En local, la raiz es la del repositorio.
# El resto del codigo usa BASE, AUDIO, RES... y no necesita saber donde esta.

_NIMROD = pathlib.Path("/data/equipo")

if _NIMROD.is_dir():
    ENTORNO = "nimrod"
    BASE = _NIMROD / "p1-tts"
    CACHE_HF = _NIMROD / "hf"
else:
    ENTORNO = "local"
    BASE = pathlib.Path(__file__).resolve().parent.parent
    CACHE_HF = None          # se usa la cache por defecto del sistema

AUDIO = BASE / "audio"
RES = BASE / "resultados"
MODELOS = BASE / "modelos"
MOS = BASE / "mos"
MUESTRAS = BASE / "muestras"

EQUIPO = int(os.environ.get("NIMROD_EQUIPO", 0))


def dispositivo():
    """Devuelve 'cuda' si hay GPU utilizable, 'cpu' en caso contrario.

    Permite que el mismo codigo corra en NIMROD y en un portatil sin GPU.
    En CPU todo funciona, solo que mas despacio.
    """
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def preparar_entorno(verbose=True):
    """Crea las carpetas de trabajo y, en NIMROD, fija HF_HOME.

    En NIMROD hay que llamarla ANTES de importar transformers o de
    descargar cualquier modelo: la cache por defecto del workspace es de
    solo lectura y las descargas fallarian.
    """
    if CACHE_HF is not None:
        os.environ["HF_HOME"] = str(CACHE_HF)
        CACHE_HF.mkdir(parents=True, exist_ok=True)

    for carpeta in (AUDIO, RES, MODELOS / "piper", MOS, MUESTRAS):
        carpeta.mkdir(parents=True, exist_ok=True)

    if ENTORNO == "nimrod":
        # Enlace para ver /data/equipo en el panel de ficheros de JupyterLab.
        # Sin el, la carpeta existe pero no aparece en el navegador lateral.
        enlace = pathlib.Path.home() / "equipo"
        if not enlace.exists():
            try:
                enlace.symlink_to("/data/equipo")
            except OSError:
                pass

    if verbose:
        dev = dispositivo()
        print(f"Entorno : {ENTORNO}" + (f" (equipo {EQUIPO})" if EQUIPO else ""))
        print(f"Trabajo : {BASE}")
        print(f"Computo : {dev}" + ("" if dev == "cuda" else "  (sin GPU: todo ira mas lento)"))
        if CACHE_HF is not None:
            print(f"HF_HOME : {os.environ['HF_HOME']}")
    return BASE


# ---------------------------------------------------------------------------
# Silenciado del ruido de ONNX Runtime
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def sin_ruido():
    """Silencia stderr dentro del bloque.

    ONNX Runtime (que usa Piper) detecta los 64 nucleos del servidor e
    intenta repartir hilos entre todos. Como el workspace solo tiene 8,
    falla en cada intento y escribe cientos de lineas
    'pthread_setaffinity_np failed'. Son inofensivas, pero tapan la
    salida util.

    Uso:
        with sin_ruido():
            voz = PiperVoice.load(ruta)
    """
    err = os.dup(2)
    nulo = os.open(os.devnull, os.O_WRONLY)
    os.dup2(nulo, 2)
    try:
        yield
    finally:
        os.dup2(err, 2)
        os.close(nulo)
        os.close(err)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def cargar_frases(ruta=None):
    """Lee frases.tsv y devuelve una lista de diccionarios.

    Columnas esperadas: id, categoria, texto, leida

    SOBRE LA COLUMNA 'leida': es como DEBERIA pronunciarse la frase, en
    letras y sin puntuacion. Sirve de referencia para medir inteligibilidad
    en la S3.

    Cuidado con las cifras y las siglas: si escribis "treinta de
    septiembre" y el reconocedor devuelve "30/09", el WER saldra altisimo
    aunque la voz se haya entendido perfectamente. Estareis midiendo
    vuestra convencion ortografica, no la calidad de la sintesis.
    Decidid la convencion del equipo, dejadla escrita en el informe y
    revisad a mano esas categorias (ver evaluacion.revisar).
    """
    import csv

    ruta = pathlib.Path(ruta) if ruta else BASE / "frases.tsv"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Copiad frases_ejemplo.tsv a frases.tsv "
            "y completadlo con vuestras 20 frases (tarea 1.1)."
        )
    with open(ruta, encoding="utf-8") as f:
        frases = list(csv.DictReader(f, delimiter="\t"))

    faltan = {"id", "categoria", "texto", "leida"} - set(frases[0])
    if faltan:
        raise ValueError(f"Faltan columnas en {ruta.name}: {faltan}")
    return frases


def resumen_corpus(frases):
    """Comprueba la composicion del corpus frente a lo que pide el guion."""
    from collections import Counter

    exigido = {"decl": 6, "int_abs": 4, "int_parc": 4, "cifras": 3, "siglas": 3}
    cuenta = Counter(f["categoria"] for f in frases)
    print(f"{len(frases)} frases en total\n")
    print(f"{'categoria':10s} {'hay':>4s} {'pide':>5s}")
    for cat, n in exigido.items():
        marca = "" if cuenta.get(cat, 0) >= n else "  <-- faltan"
        print(f"{cat:10s} {cuenta.get(cat, 0):4d} {n:5d}{marca}")
    extra = set(cuenta) - set(exigido)
    if extra:
        print(f"\nCategorias no previstas: {sorted(extra)}")

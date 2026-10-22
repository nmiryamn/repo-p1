"""
Evaluacion de calidad — Practica 1, sesion 3.

Tres vias independientes:
  - inteligibilidad: transcribir la voz sintetica y medir el WER
  - MOS: preparar las muestras ciegas y analizar las notas recibidas
  - consistencia: similitud entre embeddings de locutor (ECAPA-TDNN)
"""

import pathlib

import numpy as np

from tts_util import AUDIO, BASE, EQUIPO, MOS, dispositivo


# ===========================================================================
# 1. Inteligibilidad
# ===========================================================================
# Reconocedor de referencia de la asignatura. Es el MISMO para todos los
# equipos para que las cifras sean comparables entre grupos. NO cambiarlo.
ASR_REFERENCIA = "large-v3"

_asr = None


def cargar_asr():
    """Carga el reconocedor de referencia (una sola vez)."""
    global _asr
    if _asr is None:
        from faster_whisper import WhisperModel

        dev = dispositivo()
        # En GPU el compute_type DEBE ser float16: la cuantizacion int8 de
        # CTranslate2 falla en la arquitectura Blackwell de NIMROD.
        # En CPU int8 es la opcion sensata (mucho mas rapida, misma salida
        # a efectos practicos).
        tipo = "float16" if dev == "cuda" else "int8"
        if dev == "cpu":
            print("Reconocedor en CPU: la transcripcion de 20 frases puede")
            print("tardar varios minutos. Es normal.")
        _asr = WhisperModel(ASR_REFERENCIA, device=dev, compute_type=tipo)
    return _asr


def transcribe(ruta, idioma="es"):
    """Transcribe un WAV con el reconocedor de referencia.

    Se lee con librosa y se remuestrea a 16 kHz porque es lo que espera
    Whisper, y los sistemas TTS generan a otras frecuencias (Piper a
    22 050 Hz, MMS a 16 000 Hz).
    """
    import librosa

    y, _ = librosa.load(str(ruta), sr=16000)
    segmentos, _ = cargar_asr().transcribe(y, language=idioma, beam_size=5)
    return " ".join(s.text for s in segmentos).strip()


def normaliza_texto():
    """Transformacion comun antes de comparar referencia e hipotesis.

    Sin esto estariais midiendo diferencias de mayusculas y puntuacion,
    no errores de inteligibilidad.
    """
    import jiwer

    return jiwer.Compose([
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ])


def mide_inteligibilidad(candidatos, frases, carpeta_audio=None):
    """Transcribe el audio de cada sistema y calcula el WER por frase.

    AVISO sobre las cifras: el reconocedor puede devolver '30' donde la
    referencia dice 'treinta'. Eso cuenta como error sin serlo. Revisad a
    mano las frases de la categoria 'cifras' y explicad en el informe como
    lo tratasteis.
    """
    import jiwer
    import pandas as pd

    carpeta_audio = pathlib.Path(carpeta_audio) if carpeta_audio else AUDIO
    norma = normaliza_texto()
    filas = []
    for nombre in candidatos:
        for fr in frases:
            ruta = carpeta_audio / nombre / f"{fr['id']}.wav"
            if not ruta.exists():
                continue
            hip = transcribe(ruta)
            filas.append(dict(
                sistema=nombre,
                id=fr["id"],
                categoria=fr["categoria"],
                referencia=fr["leida"],
                hipotesis=hip,
                wer=jiwer.wer(norma(fr["leida"]), norma(hip)),
            ))
    return pd.DataFrame(filas)


def resumen_wer(intel):
    """Separa el WER de habla normal del de cifras y siglas.

    POR QUE HACE FALTA: en las categorias cifras y siglas, buena parte del
    WER NO es falta de inteligibilidad, sino desacuerdo de convencion entre
    la columna 'leida' y como escribe el reconocedor. Ejemplo real medido
    en NIMROD: la voz dice "dos mil veintiseis barra cuatro mil ochocientos
    quince", el reconocedor escribe "2026-4815" --perfecto-- y el WER sale
    0,615.

    El WER de habla normal (decl, int_abs, int_parc) esta libre de ese
    problema y es la cifra comparable entre sistemas. El de cifras y siglas
    hay que revisarlo A MANO, frase por frase, y clasificar cada error como
    artefacto de convencion o fallo real de sintesis.
    """
    import pandas as pd

    normal = ["decl", "int_abs", "int_parc"]
    filas = []
    for sistema, g in intel.groupby("sistema"):
        hn = g[g.categoria.isin(normal)]
        dif = g[~g.categoria.isin(normal)]
        filas.append({
            "sistema": sistema,
            "WER habla normal": round(hn.wer.mean(), 3),
            "WER cifras+siglas": round(dif.wer.mean(), 3),
            "n normal": len(hn),
            "n dificil": len(dif),
        })
    print("El WER de cifras+siglas NO es comparable entre sistemas sin")
    print("revisarlo a mano: mezcla errores reales y artefactos de convencion.\n")
    return pd.DataFrame(filas)


def revisar(intel, categorias=("cifras", "siglas")):
    """Muestra referencia e hipotesis lado a lado para revision manual."""
    import pandas as pd

    pd.set_option("display.max_colwidth", 90)
    return (intel[intel.categoria.isin(categorias)]
            [["sistema", "id", "categoria", "referencia", "hipotesis", "wer"]]
            .sort_values(["id", "sistema"]))


# ===========================================================================
# 2. MOS — panel cruzado y ciego
# ===========================================================================
def preparar_mos(candidatos, frases, ids_elegidos, semilla=1):
    """Copia las muestras a evaluar con nombres anonimos y guarda la clave.

    El oyente no debe saber que sistema escucha, asi que los ficheros se
    renombran y se barajan. La clave se guarda en mos_clave.json y NO se
    comparte con los demas equipos hasta que devuelvan sus notas.
    """
    import json
    import random
    import shutil

    random.seed(semilla)
    MOS.mkdir(parents=True, exist_ok=True)

    muestras = [(s, i) for s in candidatos for i in ids_elegidos]
    random.shuffle(muestras)

    clave = {}
    for n, (sistema, idf) in enumerate(muestras, start=1):
        origen = AUDIO / sistema / f"{idf}.wav"
        if not origen.exists():
            print(f"AVISO: falta {origen}")
            continue
        codigo = f"E{EQUIPO}_{n:02d}"
        shutil.copy(origen, MOS / f"{codigo}.wav")
        clave[codigo] = dict(sistema=sistema, frase=idf)

    with open(BASE / "mos_clave.json", "w", encoding="utf-8") as f:
        json.dump(clave, f, indent=2, ensure_ascii=False)

    print(f"{len(clave)} muestras en {MOS}")
    print(f"Clave guardada en {BASE / 'mos_clave.json'} — NO la comparta con otros equipos")
    return clave


def analiza_mos(csv_notas, ruta_clave=None):
    """Calcula el MOS por sistema con su intervalo de confianza.

    csv_notas: fichero con columnas evaluador, muestra, nota (1 a 5).
    """
    import json

    import pandas as pd

    from analisis import tabla_ic95

    ruta_clave = pathlib.Path(ruta_clave) if ruta_clave else BASE / "mos_clave.json"
    with open(ruta_clave, encoding="utf-8") as f:
        clave = json.load(f)

    notas = pd.read_csv(csv_notas)
    notas["sistema"] = notas.muestra.map(lambda m: clave.get(m, {}).get("sistema"))
    desconocidas = notas.sistema.isna().sum()
    if desconocidas:
        print(f"AVISO: {desconocidas} notas con codigo que no esta en la clave")
        notas = notas.dropna(subset=["sistema"])

    por_muestra = notas.groupby("muestra").size()
    pocas = por_muestra[por_muestra < 5]
    if len(pocas):
        print(f"AVISO: {len(pocas)} muestras con menos de 5 evaluaciones")

    return tabla_ic95(notas, "nota", "sistema")


# ===========================================================================
# 3. Consistencia de la voz
# ===========================================================================
_ecapa = None


def cargar_ecapa():
    """Carga el verificador de locutor ECAPA-TDNN (una sola vez)."""
    global _ecapa
    if _ecapa is None:
        from speechbrain.inference.speaker import EncoderClassifier

        _ecapa = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(BASE / "modelos" / "ecapa"),
            # "cuda:0", no "cuda": SpeechBrain no sabe interpretar la forma
            # corta y avisa de que cae a device 0 por su cuenta.
            run_opts={"device": "cuda:0" if dispositivo() == "cuda" else "cpu"},
        )
    return _ecapa


def embedding(ruta):
    """Vector que resume la identidad de la voz de un audio.

    Se lee con librosa a 16 kHz: ECAPA esta entrenado a esa frecuencia.
    NO useis torchaudio.load, que en este entorno requiere torchcodec.
    El vector se normaliza para que el producto escalar sea directamente
    la similitud coseno.
    """
    import librosa
    import torch

    y, _ = librosa.load(str(ruta), sr=16000)
    with torch.no_grad():
        e = cargar_ecapa().encode_batch(torch.tensor(y).unsqueeze(0)).squeeze()
    return torch.nn.functional.normalize(e, dim=0).cpu()


def consistencia(candidatos, frases, carpeta_audio=None):
    """Similitud media entre las frases de cada sistema y entre sistemas.

    Intra alta = el sistema mantiene la misma voz en todas las frases.
    Si la intra de un sistema es parecida a la similitud ENTRE sistemas,
    su identidad vocal no es distinguible.
    """
    import torch

    carpeta_audio = pathlib.Path(carpeta_audio) if carpeta_audio else AUDIO
    emb = {}
    for nombre in candidatos:
        vectores = []
        for fr in frases:
            ruta = carpeta_audio / nombre / f"{fr['id']}.wav"
            if ruta.exists():
                vectores.append(embedding(ruta))
        if vectores:
            emb[nombre] = torch.stack(vectores)

    print(f"{'sistema':24s} {'intra':>7s}")
    intra = {}
    for nombre, m in emb.items():
        sim = m @ m.T
        fuera = ~torch.eye(len(m), dtype=torch.bool)
        intra[nombre] = float(sim[fuera].mean())
        print(f"{nombre:24s} {intra[nombre]:7.3f}")

    nombres = list(emb)
    print(f"\n{'par de sistemas':40s} {'entre':>7s}")
    entre = {}
    for i, a in enumerate(nombres):
        for b in nombres[i + 1:]:
            v = float((emb[a] @ emb[b].T).mean())
            entre[(a, b)] = v
            print(f"{a + ' / ' + b:40s} {v:7.3f}")

    return emb, intra, entre

"""
Adaptadores de sistemas TTS — Practica 1.

Todos los sistemas se envuelven en una funcion con la MISMA firma:

    sintetiza(texto: str, ruta: Path | str, **parametros) -> None

de modo que el resto del codigo (medidas, prosodia, evaluacion) funcione
igual sea cual sea el sistema.

El profesorado proporciona dos adaptadores de ejemplo. El tercer candidato
(y los siguientes) los escribe cada equipo siguiendo el mismo patron:
mirad 'PLANTILLA PARA UN CANDIDATO NUEVO' al final del fichero.
"""

import pathlib

from tts_util import MODELOS, dispositivo, sin_ruido


# ===========================================================================
# Candidato A — Piper  (se ejecuta en CPU)
# ===========================================================================
def descargar_voz_piper(nombre="es_ES-sharvard-medium"):
    """Descarga una voz de Piper si no esta ya en el workspace.

    En piper-tts 1.7 las voces se gestionan con piper.download_voices y se
    identifican por NOMBRE (es_ES-voz-calidad), no por una ruta a fichero.
    El catalogo de voces en espanol esta en el repositorio rhasspy/piper-voices
    de Hugging Face.
    """
    import subprocess

    destino = MODELOS / "piper"
    destino.mkdir(parents=True, exist_ok=True)
    onnx = destino / f"{nombre}.onnx"
    if onnx.exists():
        print(f"{nombre} ya descargada ({onnx.stat().st_size / 1e6:.0f} MB)")
        return onnx

    print(f"Descargando {nombre} ...")
    subprocess.run(
        ["python3", "-m", "piper.download_voices", nombre, "--data-dir", str(destino)],
        check=True,
    )
    print(f"Listo: {onnx.stat().st_size / 1e6:.0f} MB")
    return onnx


def crear_piper(nombre="es_ES-sharvard-medium", usar_gpu=False):
    """Carga una voz de Piper y devuelve su funcion de sintesis.

    El modelo se carga UNA VEZ aqui. La funcion devuelta solo sintetiza,
    que es lo que debe cronometrarse al medir el RTF (tarea 1.4).

    Parametros admitidos al sintetizar (SynthesisConfig de piper):
        length_scale   duracion de los fonemas; >1 habla mas lento
        noise_scale    variabilidad general
        noise_w_scale  variabilidad de la duracion de los fonemas
        volume         multiplicador de volumen
    """
    import wave

    from piper import PiperVoice, SynthesisConfig

    onnx = descargar_voz_piper(nombre)
    with sin_ruido():
        voz = PiperVoice.load(str(onnx), use_cuda=usar_gpu)

    def sintetiza(texto, ruta, **parametros):
        cfg = SynthesisConfig(**parametros) if parametros else None
        ruta = pathlib.Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        with sin_ruido(), wave.open(str(ruta), "wb") as f:
            voz.synthesize_wav(texto, f, syn_config=cfg)

    sintetiza.nombre = f"piper_{nombre}"
    sintetiza.admite_parametros = True
    return sintetiza


# ===========================================================================
# Candidato B — MMS-TTS de Meta, via transformers  (se ejecuta en GPU)
# ===========================================================================
def crear_mms(modelo="facebook/mms-tts-spa", device=None):
    """Carga MMS-TTS y devuelve su funcion de sintesis.

    No admite parametros prosodicos: cualquier **parametros se ignora.
    Esa diferencia frente a Piper es en si misma un resultado de la
    sesion 2 (que anotar en el informe).
    """
    import soundfile as sf
    import torch
    from transformers import AutoTokenizer, VitsModel

    device = device or dispositivo()      # 'cuda' en NIMROD, 'cpu' en local
    red = VitsModel.from_pretrained(modelo).to(device)
    red.eval()
    tok = AutoTokenizer.from_pretrained(modelo)

    def sintetiza(texto, ruta, **parametros):
        ruta = pathlib.Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        x = tok(texto, return_tensors="pt").to(device)
        with torch.no_grad():
            y = red(**x).waveform.squeeze().cpu().numpy()
        sf.write(str(ruta), y, red.config.sampling_rate)

    sintetiza.nombre = modelo.split("/")[-1]
    sintetiza.admite_parametros = False
    return sintetiza


# ===========================================================================
# Registro de candidatos
# ===========================================================================
def registrar(candidatos, nombre, constructor, *args, **kwargs):
    """Intenta cargar un candidato y lo anade al diccionario si funciona.

    Si falla, imprime el error y sigue: un sistema que no arranca es un
    resultado valido de la tarea 1.2, siempre que se documente el motivo.
    """
    try:
        candidatos[nombre] = constructor(*args, **kwargs)
        print(f"[ OK ] {nombre}")
    except Exception as exc:
        print(f"[FALLO] {nombre}")
        print(f"        {type(exc).__name__}: {exc}")
        print("        Anotad este error en el informe y descartad el candidato.")
    return candidatos


# ===========================================================================
# PLANTILLA PARA UN CANDIDATO NUEVO
# ===========================================================================
#
# def crear_mi_sistema(...):
#     # 1. Cargar aqui el modelo, UNA SOLA VEZ.
#     ...
#
#     def sintetiza(texto, ruta, **parametros):
#         # 2. Generar el audio y escribirlo en 'ruta' como WAV.
#         #    Usad soundfile.write(...) o el modulo wave.
#         #    NO useis torchaudio.save: en este entorno requiere torchcodec,
#         #    que no esta disponible.
#         ...
#
#     sintetiza.nombre = "mi_sistema"
#     sintetiza.admite_parametros = False   # True si acepta prosodia
#     return sintetiza

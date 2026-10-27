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
    import sys

    destino = MODELOS / "piper"
    destino.mkdir(parents=True, exist_ok=True)
    onnx = destino / f"{nombre}.onnx"
    if onnx.exists():
        print(f"{nombre} ya descargada ({onnx.stat().st_size / 1e6:.0f} MB)")
        return onnx

    print(f"Descargando {nombre} ...")
    subprocess.run(
        [sys.executable, "-m", "piper.download_voices", nombre, "--data-dir", str(destino)],
        check=True,
    )
    print(f"Listo: {onnx.stat().st_size / 1e6:.0f} MB")
    return onnx


def datos_espeak(origen=None, nombre="espeak-ng-data"):
    """Devuelve una carpeta de datos de eSpeak-NG que se pueda abrir.

    En Windows, eSpeak-NG no abre rutas con caracteres no ASCII (tildes,
    enes...). Si la carpeta de datos esta en una ruta asi, falla y llama
    a exit(), lo que mata el proceso entero (y el kernel de Jupyter).
    En ese caso se copia una vez a ~/.cache/<nombre>.

    Por defecto usa la carpeta interna de piper. Kokoro trae su propia
    version de eSpeak-NG, asi que va a otra copia (ver crear_kokoro).
    """
    import shutil

    if origen is None:
        from piper.phonemize_espeak import ESPEAK_DATA_DIR as origen
    origen = pathlib.Path(origen)

    if str(origen).isascii():
        return origen
    copia = pathlib.Path.home() / ".cache" / nombre
    if not str(copia).isascii():
        print(f"AVISO: {copia} tiene caracteres no ASCII; eSpeak puede fallar")
    if not (copia / "phontab").exists():
        shutil.copytree(origen, copia, dirs_exist_ok=True)
    return copia


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
        voz = PiperVoice.load(str(onnx), use_cuda=usar_gpu,
                              espeak_data_dir=datos_espeak())

    # Si onnxruntime no tiene CUDA, use_cuda=True sigue en CPU SIN avisar
    # (el aviso de onnxruntime lo tapa sin_ruido). Una medida "con GPU"
    # hecha asi seria en realidad otra medida en CPU.
    if usar_gpu and "CUDAExecutionProvider" not in voz.session.get_providers():
        print("AVISO: usar_gpu=True pero onnxruntime no tiene CUDA; "
              "Piper se ejecuta en CPU")

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
# Candidato C — Kokoro-82M, via el paquete kokoro  (CPU o GPU)
# ===========================================================================
def crear_kokoro(voz="em_santa", device=None):
    """Carga Kokoro-82M y devuelve su funcion de sintesis.

    Otra familia distinta de Piper y MMS (que son VITS): Kokoro usa la
    arquitectura StyleTTS 2 con un vocoder iSTFTNet. Licencia Apache-2.0.
    Voces en espanol: ef_dora (mujer), em_alex y em_santa (hombres).
    Requiere: pip install kokoro

    Parametros admitidos al sintetizar:
        speed          velocidad; >1 habla mas rapido
        length_scale   como en Piper (>1 mas lento); se convierte a 1/speed
    """
    import sys
    import types

    import numpy as np
    import soundfile as sf

    # kokoro importa siempre misaki.en (solo para ingles), que carga spacy.
    # En Windows 11 con Control inteligente de aplicaciones activado, la DLL
    # de spacy esta bloqueada. Para espanol no hace falta: se sustituye por
    # un modulo vacio (kokoro solo lo usa en anotaciones de tipos).
    try:
        import misaki.en  # noqa: F401
    except ImportError:
        en = types.ModuleType("misaki.en")
        en.MToken = type("MToken", (), {})
        sys.modules["misaki.en"] = en

    import espeakng_loader
    from misaki import espeak  # noqa: F401  (fija las rutas de eSpeak)
    from phonemizer.backend.espeak.wrapper import EspeakWrapper
    EspeakWrapper.set_data_path(str(datos_espeak(
        espeakng_loader.get_data_path(), "espeak-ng-data-kokoro")))

    from kokoro import KPipeline

    device = device or dispositivo()
    tubo = KPipeline(lang_code="e", repo_id="hexgrad/Kokoro-82M", device=device)
    tubo.load_voice(voz)                    # descarga la voz ahora, no al medir

    def sintetiza(texto, ruta, **parametros):
        velocidad = parametros.get("speed", 1.0)
        if "length_scale" in parametros:
            velocidad = 1.0 / parametros["length_scale"]
        ruta = pathlib.Path(ruta)
        ruta.parent.mkdir(parents=True, exist_ok=True)
        trozos = [r.audio.cpu().numpy()
                  for r in tubo(texto, voice=voz, speed=velocidad)
                  if r.audio is not None]
        sf.write(str(ruta), np.concatenate(trozos), 24000)

    sintetiza.nombre = f"kokoro_{voz}"
    sintetiza.admite_parametros = True
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

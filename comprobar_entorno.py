#!/usr/bin/env python3
"""
Comprueba que el entorno esta listo para la Practica 1.

Uso:   python comprobar_entorno.py

Funciona tanto en NIMROD como en un ordenador propio. No descarga modelos:
solo verifica que las piezas estan instaladas y que el computo funciona.
"""

import importlib
import pathlib
import sys

OK, FALLO, AVISO = "[ OK ]", "[FALLO]", "[AVISO]"
fallos = 0


def comprueba(nombre, paquete=None, atributo="__version__"):
    global fallos
    try:
        mod = importlib.import_module(paquete or nombre)
        print(f"{OK} {nombre:22s} {getattr(mod, atributo, '?')}")
        return mod
    except Exception as exc:
        fallos += 1
        print(f"{FALLO} {nombre:22s} {type(exc).__name__}: {exc}")
        return None


def gpu_nvidia_presente():
    """Detecta una tarjeta NVIDIA SIN pasar por torch.

    Hace falta para distinguir dos situaciones que se confunden:
      - no hay GPU  ->  normal, todo va en CPU
      - hay GPU pero torch no la ve  ->  se instalo el paquete sin CUDA
    """
    import shutil
    import subprocess

    if shutil.which("nvidia-smi") is None:
        return None
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=15,
        )
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip().splitlines()[0]
    except Exception:
        pass
    return None


print("=" * 62)
print("COMPROBACION DEL ENTORNO - Practica 1")
print("=" * 62)

version = sys.version_info[:2]
print(f"\nPython {sys.version.split()[0]}")
if version < (3, 11):
    fallos += 1
    print(f"{FALLO} Python demasiado antiguo: necesitais 3.11 o superior.")
elif version >= (3, 14):
    print(f"{OK} version de Python reciente")
    print(f"{AVISO} Con Python {version[0]}.{version[1]} y GPU NVIDIA puede que")
    print("       no existan todavia los paquetes de PyTorch con CUDA.")
    print("       En CPU funciona. Si teneis GPU y falla, usad Python 3.12.")
else:
    print(f"{OK} version de Python soportada")

print("\n--- Nucleo ---")
torch = comprueba("torch")
comprueba("torchaudio")

if torch is not None:
    print("\n--- Computo ---")
    if torch.cuda.is_available():
        print(f"{OK} GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"{OK} VRAM: {vram:.1f} GiB")
        if vram < 10:
            print(f"{AVISO} Con menos de 10 GiB no cargueis los tres candidatos")
            print("       a la vez. Liberad con: del modelo; torch.cuda.empty_cache()")
        try:
            x = torch.randn(512, 512, device="cuda")
            _ = (x @ x).sum().item()
            print(f"{OK} operacion en GPU correcta")
        except Exception as exc:
            fallos += 1
            print(f"{FALLO} la GPU se detecta pero falla: {exc}")
    else:
        tarjeta = gpu_nvidia_presente()
        if tarjeta:
            fallos += 1
            print(f"{FALLO} Hay una GPU NVIDIA ({tarjeta}) pero PyTorch NO la usa.")
            print(f"       torch {torch.__version__}, CUDA {torch.version.cuda}")
            print()
            if torch.version.cuda is None:
                print("       CAUSA: teneis instalado el paquete de PyTorch SIN CUDA.")
                print("       En Windows es el que instala pip por defecto.")
                print("       SOLUCION:")
                print("         pip uninstall -y torch torchaudio")
                print("         pip install torch torchaudio \\")
                print("             --index-url https://download.pytorch.org/whl/cu128")
            else:
                print("       CAUSA: el paquete SI trae CUDA, asi que lo mas probable")
                print("       es que el driver de NVIDIA sea antiguo o no este cargado.")
                print("       Comprobadlo ejecutando  nvidia-smi  en el terminal.")
            print()
            print("       Mientras tanto todo funciona en CPU, solo que mas despacio.")
        else:
            print(f"{AVISO} Sin GPU: todo funcionara en CPU, mas despacio.")
            print("       Lo que mas se nota es el reconocedor de la sesion 3.")

print("\n--- Sintesis ---")
comprueba("piper", "piper")
comprueba("transformers")

print("\n--- Evaluacion ---")
comprueba("faster_whisper")
comprueba("jiwer")
comprueba("speechbrain")

print("\n--- Audio y fonetica ---")
comprueba("librosa")
comprueba("soundfile")
comprueba("parselmouth", "parselmouth", "VERSION")

print("\n--- Datos y figuras ---")
comprueba("numpy")
comprueba("pandas")
comprueba("matplotlib")

print("\n--- Material de la practica ---")
raiz = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(raiz / "scripts"))
try:
    import tts_util

    print(f"{OK} entorno detectado: {tts_util.ENTORNO}")
    print(f"{OK} carpeta de trabajo: {tts_util.BASE}")
except Exception as exc:
    fallos += 1
    print(f"{FALLO} no se pudo importar scripts/tts_util.py: {exc}")

for fichero in ("p1_plantilla.ipynb", "frases_ejemplo.tsv",
                "scripts/adaptadores.py", "scripts/analisis.py",
                "scripts/evaluacion.py"):
    if (raiz / fichero).exists():
        print(f"{OK} {fichero}")
    else:
        fallos += 1
        print(f"{FALLO} falta {fichero}")

print("\n" + "=" * 62)
if fallos:
    print(f"RESULTADO: {fallos} problema(s). Revisad lo marcado como FALLO.")
    print("Si falta algun paquete:  pip install -r requirements.txt")
    sys.exit(1)
print("RESULTADO: entorno listo.")
sys.exit(0)

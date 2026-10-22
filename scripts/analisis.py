"""
Analisis acustico y medidas de rendimiento — Practica 1.

Contiene lo que necesitan las sesiones 1 y 2:
  - medir_rtf()   factor de tiempo real, medido correctamente
  - analiza()     F0, duracion y movimiento final de un WAV
  - ic95()        intervalo de confianza por bootstrap
"""

import statistics
import time

import numpy as np


# ===========================================================================
# Velocidad de sintesis
# ===========================================================================
def medir_rtf(sintetiza, frases, carpeta, repeticiones=5, calentamiento=True):
    """Mide el factor de tiempo real de un sistema sobre un corpus.

    RTF = tiempo de sintesis / duracion del audio producido.
    Un RTF de 0,2 significa que generar 10 s de voz cuesta 2 s.

    Tres reglas que la funcion aplica y que debeis saber justificar:

    1. NO se cronometra la carga del modelo. Eso se mide aparte: un
       sistema real lo carga una vez y sintetiza miles de frases.
    2. Se descarta una primera sintesis de calentamiento (reserva de
       memoria, compilacion de nucleos, cache de la GPU).
    3. Cada frase se sintetiza varias veces y se toma la MEDIANA, no la
       media: una pausa del planificador del sistema dispararia la media.

    Devuelve una lista de diccionarios, uno por frase.
    """
    import pathlib

    import soundfile as sf

    carpeta = pathlib.Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)

    if calentamiento:
        sintetiza("Calentamiento.", carpeta / "_calentamiento.wav")

    filas = []
    for fr in frases:
        ruta = carpeta / f"{fr['id']}.wav"
        tiempos = []
        for _ in range(repeticiones):
            t0 = time.perf_counter()
            sintetiza(fr["texto"], ruta)
            tiempos.append(time.perf_counter() - t0)
        dur = sf.info(str(ruta)).duration
        t = statistics.median(tiempos)
        filas.append(
            dict(
                id=fr["id"],
                categoria=fr["categoria"],
                dur=round(dur, 3),
                t_sintesis=round(t, 4),
                rtf=round(t / dur, 4),
                sr=sf.info(str(ruta)).samplerate,
            )
        )
    (carpeta / "_calentamiento.wav").unlink(missing_ok=True)
    return filas


def medir_carga(constructor, *args, **kwargs):
    """Cronometra la carga de un modelo. Se reporta aparte del RTF."""
    t0 = time.perf_counter()
    sintetiza = constructor(*args, **kwargs)
    return sintetiza, round(time.perf_counter() - t0, 3)


# ===========================================================================
# Analisis acustico
# ===========================================================================
# NOTA sobre length_scale (medido en NIMROD con Piper es_ES-sharvard-medium):
# el control de velocidad NO es lineal. Pedir 0,6 da 0,69 de duracion
# relativa y pedir 1,6 da 1,51: el efecto esta comprimido en torno a un
# 25 % hacia 1,0, de forma consistente entre frases.
# Dos explicaciones plausibles quedaron DESCARTADAS al comprobarlas:
#   - no es el silencio inicial y final (recortarlo no cambia nada)
#   - no es el predictor de duracion estocastico (con noise_w_scale=0
#     la curva se aleja todavia mas de la diagonal)
# Conclusion: es una caracteristica del modelo. Para una velocidad objetivo
# hay que calibrar empiricamente, no fiarse del numero.


def analiza(ruta, f0_min=75, f0_max=500):
    """Extrae de un WAV las medidas prosodicas que usa la sesion 2.

    Devuelve un diccionario con:
      dur        duracion total en segundos
      f0_med     F0 mediana de los tramos sonoros, en Hz
      rango_st   recorrido de F0 en SEMITONOS sobre la mediana.
                 Se usan semitonos y no hercios para poder comparar una
                 voz grave con una aguda: 100->200 Hz y 200->400 Hz son
                 el mismo salto perceptivo (12 semitonos).
      final_st   movimiento de los ultimos 300 ms sonoros frente a los
                 300 ms anteriores, en semitonos.
                 POSITIVO = la frase termina subiendo (tipico de la
                 interrogativa absoluta en espanol).
                 NEGATIVO = termina bajando (tipico de la declarativa).
      t, f0      vectores para dibujar el contorno
    """
    import parselmouth

    son = parselmouth.Sound(str(ruta))
    pitch = son.to_pitch(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max)
    f0_todo = pitch.selected_array["frequency"]
    t_todo = pitch.xs()

    sonoro = f0_todo > 0
    f0 = f0_todo[sonoro]
    t = t_todo[sonoro]

    if len(f0) < 10:
        return dict(dur=son.get_total_duration(), f0_med=np.nan,
                    rango_st=np.nan, final_st=np.nan, t=t, f0=f0)

    mediana = float(np.median(f0))
    semitonos = 12 * np.log2(f0 / mediana)

    fin = t[-1]
    ultimo = f0[t > fin - 0.3]
    previo = f0[(t > fin - 0.6) & (t <= fin - 0.3)]
    final_st = (
        float(12 * np.log2(ultimo.mean() / previo.mean()))
        if len(previo) and len(ultimo)
        else np.nan
    )

    return dict(
        dur=round(son.get_total_duration(), 3),
        f0_med=round(mediana, 1),
        rango_st=round(float(np.ptp(semitonos)), 2),
        final_st=round(final_st, 2) if not np.isnan(final_st) else np.nan,
        t=t,
        f0=f0,
    )


def dibuja_contornos(rutas_etiquetadas, titulo="", salida=None):
    """Superpone los contornos de F0 de varios audios.

    rutas_etiquetadas: lista de pares (etiqueta, ruta).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for etiqueta, ruta in rutas_etiquetadas:
        a = analiza(ruta)
        ax.plot(a["t"], a["f0"], ".", ms=3,
                label=f"{etiqueta}   final {a['final_st']:+.1f} st")
    ax.set_xlabel("tiempo (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.set_title(titulo)
    ax.legend(fontsize=8)
    fig.tight_layout()
    if salida:
        fig.savefig(salida, dpi=150, bbox_inches="tight")
    return fig


# ===========================================================================
# Estadistica
# ===========================================================================
def ic95(valores, repeticiones=5000, semilla=0):
    """Intervalo de confianza del 95 % de la media, por bootstrap.

    Se remuestrea con reemplazo muchas veces y se toman los percentiles
    2,5 y 97,5 de las medias obtenidas. No supone que los datos sigan una
    distribucion normal, lo cual importa con notas de 1 a 5 y muestras
    pequenas.

    REGLA DE LECTURA: si los intervalos de dos sistemas se solapan de
    forma apreciable, NO podeis afirmar que uno sea mejor que el otro.
    Decidlo asi en el informe.
    """
    valores = np.asarray(valores, dtype=float)
    rng = np.random.default_rng(semilla)
    medias = [rng.choice(valores, len(valores), replace=True).mean()
              for _ in range(repeticiones)]
    lo, hi = np.percentile(medias, [2.5, 97.5])
    return float(valores.mean()), float(lo), float(hi)


def tabla_ic95(df, columna_valor, columna_grupo):
    """Aplica ic95 por grupo y devuelve un DataFrame ordenado."""
    import pandas as pd

    filas = []
    for grupo, g in df.groupby(columna_grupo):
        media, lo, hi = ic95(g[columna_valor].values)
        filas.append({columna_grupo: grupo, "n": len(g), "media": round(media, 3),
                      "ic_inf": round(lo, 3), "ic_sup": round(hi, 3)})
    return pd.DataFrame(filas).sort_values("media", ascending=False)

# Práctica 1 — Síntesis de voz · Equipo 6

**Procesamiento del Habla · Grado en Ingeniería Informática · ETSI · Universidad de Huelva**

Álvaro Zarzuela Moncada · Miryam Merchán León · Sergio Velázquez López · Tomás Estrada Gómez

Comparamos tres sistemas abiertos de síntesis de voz en español para decidir cuál usaríamos en un **asistente de voz de la secretaría universitaria**. Todos se miden sobre el mismo corpus de 20 frases del dominio: velocidad y recursos, control de la prosodia (velocidad, determinismo, preguntas, énfasis, cifras y siglas) y calidad (inteligibilidad, MOS y consistencia de la voz).

El informe completo está en [`informe/`](informe/). Todas las cifras salen de [`p1_plantilla.ipynb`](p1_plantilla.ipynb).

---

## Resultado en una tabla

| | Piper (`es_ES-sharvard-medium`) | MMS-TTS (`facebook/mms-tts-spa`) | **Kokoro-82M (`em_santa`)** |
|---|---|---|---|
| Familia | VITS (ONNX) | VITS (PyTorch) | StyleTTS 2 |
| Licencia de la voz | CC BY 3.0 | CC BY-NC 4.0 (no comercial) | Apache-2.0 |
| RTF mediano CPU / GPU | 0,032 / 0,034 | 0,162 / 0,044 | 0,242 / 0,042 |
| RAM al cargar | ~0,1 GB | ~0,3 GB | ~1,1 GB |
| Fallos reales de inteligibilidad (20 frases) | 3 | 3 (2 graves en siglas) | 0 |
| Consistencia de la voz (ECAPA, intra) | 0,80 | 0,58 | 0,79 |
| MOS (90 notas de otros equipos, IC 95 %) | 3,57 [3,20-3,90] | 2,77 [2,43-3,10] | 3,40 [3,07-3,73] |
| ¿Sube al final en las preguntas de sí/no? | No | No | No (casi plana) |

**Recomendación: Kokoro** con el texto normalizado previamente. Es el único sin fallos reales de inteligibilidad, mantiene la voz estable, empata con Piper en el MOS (los dos por encima de MMS) y su licencia permite el uso. Es más lento en CPU, pero sigue siendo unas cuatro veces más rápido que el tiempo real. Piper queda como alternativa para máquinas pequeñas. El razonamiento completo está en las secciones 6 y 7 del informe.

---

## Qué hay en cada carpeta

| Ruta | Qué contiene |
|---|---|
| `p1_plantilla.ipynb` | **El cuaderno de la práctica.** Ejecuta todas las tareas en orden (sesiones 1, 2 y 3) y guarda debajo de cada una sus notas con los resultados y lo que oímos |
| `frases.tsv` | Corpus de 20 frases del dominio: 6 declarativas, 4 interrogativas absolutas, 4 parciales, 3 con cifras y 3 con siglas. La columna `leida` es la pronunciación de referencia para el WER |
| `scripts/tts_util.py` | Rutas, detección del entorno (NIMROD o local), carga del corpus |
| `scripts/adaptadores.py` | Un adaptador por sistema (Piper, MMS, Kokoro) con la misma interfaz: `tts(texto, ruta, **parametros)` |
| `scripts/analisis.py` | Medida del RTF y del tiempo de carga, análisis acústico (duración, F0, `final_st`), contornos |
| `scripts/evaluacion.py` | Inteligibilidad con Whisper large-v3, preparación y análisis del MOS, consistencia con ECAPA-TDNN |
| `scripts/normalizacion.py` | `normaliza()`: pasa fechas, horas, importes y siglas a letras antes de sintetizar (tarea 2.5) |
| `resultados/` | Todas las tablas (`.csv`) y figuras (`.png`) que salen del cuaderno. Los ficheros `*_gpu.csv` son las medidas en GPU |
| `muestras/` | Los 15 audios del panel MOS (5 frases × 3 sistemas) con nombres claros, para escuchar los sistemas sin ejecutar nada. Ver abajo |
| `informe/` | Informe de la práctica (`informe.md` y su versión `informe.pdf`) |
| `mos_clave.json` | Clave del MOS: qué sistema y qué frase hay detrás de cada código `E6_xx`. Se subió cuando los otros equipos devolvieron sus notas |
| `docs/GUIA_MATERIAL.md` | La guía original del material de la asignatura (NIMROD, instalación, avisos conocidos) |
| `comprobar_entorno.py` | Comprueba que están instaladas todas las dependencias |
| `comprobar_entrega.py` | Comprueba que la entrega está completa. También se ejecuta en GitHub Actions en cada push |

**Carpetas que se crean al ejecutar y no están en el repositorio** (las excluye el `.gitignore` por tamaño):

| Ruta | Qué contiene |
|---|---|
| `audio/` | Todos los audios generados: `audio/<sistema>/` con las 20 frases originales, `audio/normalizado/<sistema>/` con el texto normalizado, y una subcarpeta por experimento (`velocidad`, `determinismo`, `modalidad`, `foco`...) |
| `modelos/` | Voces y modelos descargados (Piper, ECAPA) |
| `mos/` | Las 15 muestras anonimizadas que se envían a los otros equipos |

### Las muestras

Son las frases f01 (declarativa), f03 (pregunta de sí/no), f04 (pregunta con «qué»), f05 (fecha y hora) y f06 (siglas), sintetizadas a partir del texto normalizado. El nombre indica el sistema y la frase, por ejemplo `kokoro_santa_f05.wav`. Las frases completas están en `frases.tsv`.

Merece la pena comparar f03 en los tres sistemas (ninguno entona la pregunta) y f06 en MMS (no consigue decir las siglas).

---

## Cómo ejecutarlo

### En local (lo que usamos nosotros)

Lo hemos ejecutado en **Windows 11 con Python 3.12**, en CPU. La GPU es opcional: sin ella todo funciona, pero más despacio.

```bash
git clone https://github.com/nmiryamn/repo-p1.git
cd repo-p1

py -3.12 -m venv .venv            # Linux/macOS: python3.12 -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python comprobar_entorno.py

jupyter lab                       # o abrir el cuaderno en VS Code
```

Con GPU NVIDIA en Windows hay que instalar PyTorch con CUDA **antes** que el `requirements.txt`, porque el paquete por defecto no trae CUDA. Los detalles están en [`docs/GUIA_MATERIAL.md`](docs/GUIA_MATERIAL.md).

### En NIMROD

Se siguen los pasos de la «Opción A» de [`docs/GUIA_MATERIAL.md`](docs/GUIA_MATERIAL.md). El código detecta el entorno y ajusta solo las rutas y el dispositivo.

### Orden de ejecución

1. Abrir `p1_plantilla.ipynb` y ejecutar **siempre primero** la celda de puesta en marcha (`preparar_entorno()`), que crea las carpetas.
2. Ejecutar de arriba abajo. Lo ideal es **Kernel → Restart and Run All**.
3. La primera vez se descargan los modelos: Piper (~77 MB), MMS (~145 MB), Kokoro (~330 MB), Whisper large-v3 (~3 GB) y ECAPA (~80 MB).
4. Lo más lento en CPU es la transcripción con Whisper large-v3 de la sesión 3, que tarda varios minutos.

El cuaderno regenera todos los audios y todos los ficheros de `resultados/`. Piper y MMS no son deterministas con sus parámetros por defecto (tarea 2.2), así que al volver a ejecutarlo algunas cifras pueden cambiar en los decimales. Kokoro y Piper con el ruido a cero dan siempre lo mismo.

### Problemas que encontramos en Windows (ya resueltos en el código)

- **Rutas con tildes.** La ruta de nuestro proyecto contiene «Informática» y «PRÁCTICAS», y eSpeak-NG no abre rutas con caracteres no ASCII: Piper y Kokoro cerraban el kernel sin dar ningún mensaje. `adaptadores.py` copia los datos de eSpeak a una ruta sin tildes.
- **DLL de spacy bloqueada.** Kokoro importa spacy aunque para español no lo necesita, y el «Control inteligente de aplicaciones» de Windows 11 bloquea su DLL. `adaptadores.py` evita esa importación.
- **Enlaces simbólicos de SpeechBrain.** Al descargar ECAPA, SpeechBrain crea enlaces simbólicos, y en Windows eso exige permisos de administrador (`WinError 1314`). `evaluacion.py` le indica que copie los ficheros.
- **Piper con GPU.** Con `usar_gpu=True`, si onnxruntime no tiene CUDA, Piper sigue en CPU sin avisar. El cuaderno lo comprueba antes de medir, para no comparar CPU contra CPU.

### Medidas en GPU

No pudimos usar NIMROD. Con permiso del profesor, las medidas en GPU (tareas 1.3 y 1.4) se hicieron en un portátil con NVIDIA RTX 4060 Laptop (8 GB), ejecutando las mismas celdas. Se guardan aparte, en `resultados/*_gpu.csv`, para no pisar las de CPU.

---

## Cómo hemos trabajado con git

- Una rama por tarea con prefijo de sesión (`s1/1.4-medidas-gpu`, `s2/2.5-normalizacion`, `s3/3.3-consistencia`...), y merge a `main` con commit de merge. No se hace commit directamente en `main`.
- Los mensajes indican el tipo y la tarea: `data(1.1): ...` para datos, `feat(2.5): ...` para código y resultados, `docs(2.2): ...` para notas y textos, `fix(2.1): ...` para correcciones.

---

## Entrega

```bash
python comprobar_entrega.py
```

Comprueba que están el informe en PDF, el cuaderno ejecutado, `frases.tsv`, los scripts, los resultados, `muestras/` y `mos_clave.json`, y que ningún fichero pasa de 20 MB.

## Créditos y licencias

- Material de partida (cuaderno, `scripts/`, comprobadores): asignatura Procesamiento del Habla, DTI, ETSI, Universidad de Huelva. Lo que añadimos o cambiamos nosotros está en el historial de commits.
- [Piper](https://github.com/OHF-Voice/piper1-gpl) (GPL-3.0) con la voz `es_ES-sharvard-medium` (CC BY 3.0).
- [MMS-TTS](https://huggingface.co/facebook/mms-tts-spa) de Meta (CC BY-NC 4.0).
- [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (Apache-2.0).
- [Whisper large-v3](https://github.com/openai/whisper) vía faster-whisper, y [ECAPA-TDNN de SpeechBrain](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb).

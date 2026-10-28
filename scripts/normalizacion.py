"""
Normalizacion de texto para la Practica 1 (tarea 2.5).

Procesamiento del Habla · ETSI · Universidad de Huelva

normaliza(texto) convierte a letras lo que un sistema TTS puede leer mal:
fechas, horas, importes, numeros y siglas. Es la primera etapa de cualquier
sistema TTS (Tema 5). Aqui se hace a mano, ANTES de sintetizar.

Sigue la MISMA convencion que la columna 'leida' de frases.tsv, para que el
texto que se sintetiza coincida con la referencia del WER de la S3:

    30/09        -> treinta de septiembre
    14:00        -> las dos de la tarde      (reloj de 12 h con franja)
    10:45        -> las diez y cuarenta y cinco de la mañana
    26,30 euros  -> veintiséis con treinta euros
    TFG          -> te efe ge                (se deletrea)
    ETSI         -> etsi                     (se lee como palabra)

Se conserva la puntuacion: los sistemas la usan para la entonacion (2.3).
"""

import re

# ---------------------------------------------------------------------------
# Numeros a letras (0 a 9999, suficiente para fechas, horas e importes)
# ---------------------------------------------------------------------------
_UNIDADES = ["cero", "uno", "dos", "tres", "cuatro", "cinco", "seis", "siete",
             "ocho", "nueve", "diez", "once", "doce", "trece", "catorce",
             "quince", "dieciséis", "diecisiete", "dieciocho", "diecinueve",
             "veinte", "veintiuno", "veintidós", "veintitrés", "veinticuatro",
             "veinticinco", "veintiséis", "veintisiete", "veintiocho",
             "veintinueve"]
_DECENAS = {30: "treinta", 40: "cuarenta", 50: "cincuenta", 60: "sesenta",
            70: "setenta", 80: "ochenta", 90: "noventa"}
_CENTENAS = {100: "ciento", 200: "doscientos", 300: "trescientos",
             400: "cuatrocientos", 500: "quinientos", 600: "seiscientos",
             700: "setecientos", 800: "ochocientos", 900: "novecientos"}


def numero(n):
    """Escribe en letras un entero entre 0 y 9999 ('uno', no 'un')."""
    if n < 30:
        return _UNIDADES[n]
    if n < 100:
        d, u = divmod(n, 10)
        return _DECENAS[d * 10] + (f" y {_UNIDADES[u]}" if u else "")
    if n == 100:
        return "cien"
    if n < 1000:
        c, resto = divmod(n, 100)
        return _CENTENAS[c * 100] + (f" {numero(resto)}" if resto else "")
    if n < 10000:
        m, resto = divmod(n, 1000)
        miles = "mil" if m == 1 else f"{numero(m)} mil"
        return miles + (f" {numero(resto)}" if resto else "")
    raise ValueError(f"numero fuera de rango: {n}")


# ---------------------------------------------------------------------------
# Fechas, horas e importes
# ---------------------------------------------------------------------------
_MESES = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio",
          "agosto", "septiembre", "octubre", "noviembre", "diciembre"]


def _fecha(m):
    dia, mes, anio = int(m["dia"]), int(m["mes"]), m["anio"]
    texto = f"{numero(dia)} de {_MESES[mes - 1]}"
    if anio:
        anio = int(anio) + (2000 if len(anio) == 2 else 0)
        texto += f" de {numero(anio)}"
    return texto


def _franja(h):
    if h < 6:
        return "de la madrugada"
    if h < 12:
        return "de la mañana"
    if h < 21:
        return "de la tarde"
    return "de la noche"


def _hora(m):
    # El articulo ('a las 14:00') se reescribe: 'la una' frente a 'las dos'.
    # Si el texto no lo trae ('abre 9:00'), tampoco se añade.
    h, mins = int(m["h"]), int(m["min"])
    h12 = h % 12 or 12
    if m["art"]:
        articulo = "la" if h12 == 1 else "las"
        texto = f"{articulo} {'una' if h12 == 1 else numero(h12)}"
    else:
        texto = "una" if h12 == 1 else numero(h12)
    if mins:
        texto += f" y {numero(mins)}"
    return f"{texto} {_franja(h)}"


def _importe(m):
    euros, cent = int(m["euros"]), m["cent"]
    if cent and int(cent):
        return f"{numero(euros)} con {numero(int(cent.ljust(2, '0')))} euros"
    return "un euro" if euros == 1 else f"{numero(euros)} euros"


_REGLAS = [
    # 30/09, 15/11/2026
    (re.compile(r"\b(?P<dia>\d{1,2})/(?P<mes>\d{1,2})(?:/(?P<anio>\d{2}|\d{4}))?\b"), _fecha),
    # (a las) 14:00, 10:45
    (re.compile(r"\b(?:(?P<art>las?)\s+)?(?P<h>\d{1,2}):(?P<min>\d{2})\b"), _hora),
    # 26,30 euros, 26,30 €, 5 euros
    (re.compile(r"\b(?P<euros>\d+)(?:,(?P<cent>\d{1,2}))?\s*(?:euros?\b|€)"), _importe),
    # cualquier otro entero suelto
    (re.compile(r"\b\d{1,4}\b"), lambda m: numero(int(m[0]))),
]


# ---------------------------------------------------------------------------
# Siglas
# ---------------------------------------------------------------------------
# Por defecto una sigla (2 a 5 mayusculas) se deletrea. Las que en el
# dominio se pronuncian como palabra van en SIGLAS_PALABRA y se pasan a
# minusculas, que es lo que hace que el sistema las lea de corrido.
SIGLAS_PALABRA = {"ETSI", "UHU"}

_LETRAS = {"A": "a", "B": "be", "C": "ce", "D": "de", "E": "e", "F": "efe",
           "G": "ge", "H": "hache", "I": "i", "J": "jota", "K": "ka",
           "L": "ele", "M": "eme", "N": "ene", "Ñ": "eñe", "O": "o",
           "P": "pe", "Q": "cu", "R": "erre", "S": "ese", "T": "te",
           "U": "u", "V": "uve", "W": "uve doble", "X": "equis",
           "Y": "i griega", "Z": "zeta"}


def _sigla(m):
    sigla = m[0]
    if sigla in SIGLAS_PALABRA:
        return sigla.lower()
    return " ".join(_LETRAS[c] for c in sigla)


_SIGLA = re.compile(r"\b[A-ZÑ]{2,5}\b")


def normaliza(texto):
    """Convierte fechas, horas, importes, numeros y siglas a letras."""
    for patron, a_letras in _REGLAS:
        texto = patron.sub(a_letras, texto)
    return _SIGLA.sub(_sigla, texto)

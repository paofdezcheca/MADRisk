"""ETL de MADRisk: carga, unión de años, limpieza y variables derivadas.

Fuente: Accidentes de tráfico de la Ciudad de Madrid (Portal de Datos Abiertos).
Cada FILA es una persona/vehículo implicado, no un accidente: un mismo
`num_expediente` se repite. Por eso, para contar accidentes hay que contar
expedientes distintos y nunca filas.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Carpeta con los CSV anuales (ruta relativa a este fichero, no absoluta)
RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# Años completos. 2026 se trata aparte porque el año está incompleto.
ANIOS = range(2020, 2026)

# Solo cargamos las columnas que usaremos (Render free tiene 512 MB de RAM).
# Quedan fuera: localizacion, numero y coordenadas (no hacemos mapas) y
# cod_lesividad (contiene la misma información que el objetivo, nunca es predictor).
COLUMNAS = [
    "num_expediente", "fecha", "hora", "distrito", "tipo_accidente",
    "estado_meteorologico", "tipo_vehiculo", "tipo_persona", "rango_edad",
    "sexo", "lesividad", "positiva_alcohol", "positiva_droga",
]


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza los nombres de columna: minúsculas, sin espacios y sin tildes.

    Así 'estado_meteorológico' queda como 'estado_meteorologico' y no
    depende de la codificación del fichero ni del año.
    """
    out = df.copy()
    out.columns = [
        str(c).strip().lower().replace("ó", "o") for c in out.columns
    ]
    return out


def leer_anio(anio: int) -> pd.DataFrame:
    """Lee el CSV de un año: separador ';', UTF-8 con BOM y latin-1 como alternativa."""
    ruta = RAW_DIR / f"accidentes_{anio}.csv"
    try:
        df = pd.read_csv(ruta, sep=";", encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(ruta, sep=";", encoding="latin-1")
    # Los textos "NULL" (p. ej. en positiva_droga) ya los convierte pandas en
    # nulos al leer, porque están en su lista de valores nulos por defecto.
    return normalizar_columnas(df)


def franja_horaria(hora: float) -> str:
    """Devuelve la franja horaria a partir de la hora (0-23)."""
    if pd.isna(hora):
        return "Sin hora"
    if hora < 6:
        return "Madrugada (0-5)"
    if hora < 12:
        return "Mañana (6-11)"
    if hora < 16:
        return "Mediodía (12-15)"
    if hora < 21:
        return "Tarde (16-20)"
    return "Noche (21-23)"


def dia_semana(anio: float, mes: float, dia: float) -> float:
    """Día de la semana (0 = lunes ... 6 = domingo) o NaN si la fecha no es válida."""
    try:
        return date(int(anio), int(mes), int(dia)).weekday()
    except (ValueError, TypeError):
        # NaN o fecha imposible (p. ej. 31/02): la marcamos como no válida
        return np.nan


def anadir_variables_derivadas(df: pd.DataFrame) -> pd.DataFrame:
    """Crea año, mes, día, hora, día de la semana, franja horaria y fin de semana."""
    out = df.copy()

    # La fecha viene como texto 'dd/mm/aaaa': troceamos por posición y pasamos a número.
    # errors="coerce" convierte lo que no sea un número en NaN en lugar de fallar.
    out["dia"] = pd.to_numeric(out["fecha"].str[0:2], errors="coerce")
    out["mes"] = pd.to_numeric(out["fecha"].str[3:5], errors="coerce")
    out["anio"] = pd.to_numeric(out["fecha"].str[6:10], errors="coerce")

    # La hora viene como 'h:mm:ss' (a veces con un solo dígito en la hora):
    # nos quedamos con lo que hay antes de los dos puntos.
    out["hora_num"] = pd.to_numeric(out["hora"].str.split(":").str[0], errors="coerce")

    # Día de la semana con el `date` de Python (el curso importa `date`)
    out["dia_semana"] = [
        dia_semana(a, m, d) for a, m, d in zip(out["anio"], out["mes"], out["dia"])
    ]

    out["franja"] = out["hora_num"].map(franja_horaria)
    # Sábado (5) y domingo (6). Si no hay día de la semana, queda como False.
    out["fin_de_semana"] = out["dia_semana"] >= 5
    return out


def validar_anio(df: pd.DataFrame, anio: int) -> pd.DataFrame:
    """Descarta filas con fecha u hora no válidas o de otro año e imprime el recuento."""
    fecha_ok = df["dia_semana"].notna()
    hora_ok = df["hora_num"].between(0, 23)
    anio_ok = df["anio"] == anio
    validas = fecha_ok & hora_ok & anio_ok

    print(
        f"  {anio}: {len(df)} filas -> {int(validas.sum())} válidas, "
        f"{int((~validas).sum())} inválidas "
        f"(fecha: {int((~fecha_ok).sum())}, hora: {int((~hora_ok).sum())}, "
        f"otro año: {int((~anio_ok).sum())})"
    )
    return df[validas].copy()


# Columnas de texto en las que un nulo se muestra como "No informado"
COLUMNAS_TEXTO = [
    "distrito", "tipo_accidente", "estado_meteorologico",
    "tipo_vehiculo", "tipo_persona", "positiva_alcohol",
]


def marcar_no_informados(df: pd.DataFrame) -> pd.DataFrame:
    """Sustituye los nulos de las columnas de texto por una etiqueta legible.

    Importante: una lesividad vacía NO significa "sin lesión". Se muestra como
    "No informada" y más adelante se deja fuera del entrenamiento del modelo.
    """
    out = df.copy()
    for columna in COLUMNAS_TEXTO:
        out[columna] = out[columna].map(lambda v: "No informado" if pd.isna(v) else v)
    out["lesividad"] = out["lesividad"].map(lambda v: "No informada" if pd.isna(v) else v)
    return out


# Lesividades que consideramos GRAVES (definición confirmada con Pao)
LESIVIDADES_GRAVES = ["Ingreso superior a 24 horas", "Fallecido 24 horas"]
# Lesividades sin información útil: no sabemos si fue grave, así que no cuentan
LESIVIDADES_SIN_DATO = ["No informada", "Se desconoce"]


def valor_grave(lesividad: str) -> float:
    """1 si la lesión es grave, 0 si no lo es y NaN si no hay información."""
    if lesividad in LESIVIDADES_SIN_DATO:
        return np.nan
    if lesividad in LESIVIDADES_GRAVES:
        return 1.0
    return 0.0


def marcar_grave(df: pd.DataFrame) -> pd.DataFrame:
    """Añade la columna `grave` (1 / 0 / NaN) a partir de `lesividad`.

    Los NaN NO se tratan como "no grave": quedan fuera de los porcentajes
    y del entrenamiento del modelo. Cada fila es una persona, así que `grave`
    indica si LA PERSONA tuvo una lesión grave.
    """
    out = df.copy()
    out["grave"] = out["lesividad"].map(valor_grave)
    return out


def cargar_datos() -> pd.DataFrame:
    """Carga 2020-2025, valida cada año y devuelve un único DataFrame (una fila = una persona)."""
    print("Cargando años completos:")
    partes = []
    for anio in ANIOS:
        df = leer_anio(anio)
        df = df[COLUMNAS].copy()
        df = anadir_variables_derivadas(df)
        partes.append(validar_anio(df, anio))

    # OJO: pd.concat NO aparece en el material del curso (solo se concatenan
    # textos y listas). Es la forma estándar de unir los años en una tabla.
    datos = pd.concat(partes, ignore_index=True)
    datos = marcar_no_informados(datos)
    datos = marcar_grave(datos)
    informados = int(datos["grave"].notna().sum())
    graves = int(datos["grave"].sum())
    print(
        f"Lesividad informada en {informados} filas "
        f"({len(datos) - informados} sin dato); graves: {graves} "
        f"({100 * graves / informados:.1f} % de las informadas)"
    )
    print(f"Total: {len(datos)} filas (personas/vehículos implicados)")
    return datos


if __name__ == "__main__":
    # Prueba rápida: python src/etl.py
    datos = cargar_datos()
    print()
    print("Primeras filas:")
    print(datos.head())
    print()
    print("Nulos por columna:")
    print(datos.isna().sum())

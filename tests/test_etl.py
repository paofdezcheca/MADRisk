"""Tests de las funciones del ETL con datos pequeños inventados (no leen los CSV)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.etl import (
    anadir_variables_derivadas,
    dia_semana,
    franja_horaria,
    marcar_grave,
    marcar_no_informados,
    normalizar_columnas,
    valor_grave,
    validar_anio,
)


def test_normalizar_columnas_quita_tildes_y_espacios():
    df = pd.DataFrame(columns=[" Estado_Meteorológico ", "FECHA"])
    assert list(normalizar_columnas(df).columns) == ["estado_meteorologico", "fecha"]


def test_franja_horaria_en_los_limites():
    assert franja_horaria(0) == "Madrugada (0-5)"
    assert franja_horaria(5) == "Madrugada (0-5)"
    assert franja_horaria(6) == "Mañana (6-11)"
    assert franja_horaria(12) == "Mediodía (12-15)"
    assert franja_horaria(16) == "Tarde (16-20)"
    assert franja_horaria(21) == "Noche (21-23)"
    assert franja_horaria(np.nan) == "Sin hora"


def test_dia_semana_valido_e_invalido():
    # El 1 de enero de 2024 fue lunes
    assert dia_semana(2024, 1, 1) == 0
    # El 7 de enero de 2024 fue domingo
    assert dia_semana(2024, 1, 7) == 6
    # 31 de febrero no existe
    assert np.isnan(dia_semana(2024, 2, 31))
    assert np.isnan(dia_semana(np.nan, 1, 1))


def test_variables_derivadas_desde_fecha_y_hora_en_texto():
    df = pd.DataFrame({"fecha": ["06/01/2024", "07/01/2024"], "hora": ["7:30:00", "23:05:00"]})
    out = anadir_variables_derivadas(df)
    assert out["anio"].tolist() == [2024, 2024]
    assert out["mes"].tolist() == [1, 1]
    assert out["hora_num"].tolist() == [7, 23]
    # 6 de enero de 2024: sábado; 7: domingo
    assert out["dia_semana"].tolist() == [5, 6]
    assert out["fin_de_semana"].tolist() == [True, True]
    assert out["franja"].tolist() == ["Mañana (6-11)", "Noche (21-23)"]


def test_validar_anio_descarta_filas_malas():
    df = pd.DataFrame(
        {
            "dia_semana": [0, np.nan, 2, 3],
            "hora_num": [10, 10, 25, 10],
            "anio": [2024, 2024, 2024, 2023],
        }
    )
    validas = validar_anio(df, 2024)
    assert len(validas) == 1
    assert validas.iloc[0]["hora_num"] == 10


def test_valor_grave_tres_casos():
    assert valor_grave("Fallecido 24 horas") == 1.0
    assert valor_grave("Ingreso superior a 24 horas") == 1.0
    assert valor_grave("Asistencia sanitaria sólo en el lugar del accidente") == 0.0
    # Sin dato NO es "no grave": debe quedar como nulo
    assert np.isnan(valor_grave("No informada"))
    assert np.isnan(valor_grave("Se desconoce"))


def test_marcar_no_informados_y_grave():
    df = pd.DataFrame(
        {
            "distrito": ["CENTRO", None],
            "tipo_accidente": ["Colisión", None],
            "estado_meteorologico": ["Despejado", None],
            "tipo_vehiculo": ["Turismo", None],
            "tipo_persona": ["Conductor", None],
            "positiva_alcohol": ["N", None],
            "lesividad": ["Fallecido 24 horas", None],
        }
    )
    out = marcar_grave(marcar_no_informados(df))
    assert out.loc[1, "distrito"] == "No informado"
    assert out.loc[1, "lesividad"] == "No informada"
    assert out.loc[0, "grave"] == 1.0
    assert np.isnan(out.loc[1, "grave"])


if __name__ == "__main__":
    # Ejecutor sencillo (sin pytest): python tests/test_etl.py
    pruebas = [f for nombre, f in sorted(globals().items()) if nombre.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} pruebas correctas")

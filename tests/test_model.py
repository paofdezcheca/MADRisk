"""Tests de src/model.py con datos pequeños inventados y con el model.joblib guardado."""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

from src.model import (
    VARIABLES,
    VARIABLES_TEXTO,
    codificar,
    crear_codigos,
    dividir,
    evaluar_umbrales,
    nivel_prioridad,
    preparar_datos,
    probabilidad_persona,
)


def datos_de_prueba(n: int = 200) -> pd.DataFrame:
    """Tabla inventada con las variables del modelo y la columna `grave`."""
    generador = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "hora_num": generador.integers(0, 24, n),
            "dia_semana": generador.integers(0, 7, n),
            "distrito": generador.choice(["CENTRO", "USERA"], n),
            "tipo_accidente": generador.choice(["Colisión", "Atropello"], n),
            "estado_meteorologico": generador.choice(["Despejado", "Lluvia"], n),
            "tipo_vehiculo": generador.choice(["Turismo", "Moto"], n),
            "tipo_persona": generador.choice(["Conductor", "Peatón"], n),
            "rango_edad": generador.choice(["De 18 a 24 años", "Desconocido"], n),
        }
    )
    # Los 20 primeros son graves, las 20 siguientes no tienen dato y el resto no es grave
    df["grave"] = 0.0
    df.loc[:19, "grave"] = 1.0
    df.loc[20:39, "grave"] = np.nan
    return df


def test_crear_codigos_da_un_numero_a_cada_valor():
    df = datos_de_prueba()
    codigos = crear_codigos(df)
    assert set(codigos) == set(VARIABLES_TEXTO)
    assert codigos["distrito"] == {"CENTRO": 0, "USERA": 1}


def test_preparar_datos_deja_fuera_las_lesividades_sin_dato():
    df = datos_de_prueba()
    X, y = preparar_datos(df, crear_codigos(df))
    # 200 filas menos las 20 sin dato
    assert len(X) == len(y) == 180
    assert list(X.columns) == VARIABLES
    assert y.sum() == 20
    # Todo es numérico tras codificar
    assert X.notna().all().all()


def test_dividir_mantiene_la_proporcion_de_graves():
    df = datos_de_prueba()
    X, y = preparar_datos(df, crear_codigos(df))
    X_train, X_test, y_train, y_test = dividir(X, y)
    assert len(X_test) == round(0.3 * len(X))
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_nivel_prioridad_en_los_umbrales():
    assert nivel_prioridad(0.08) == "Alta"
    assert nivel_prioridad(0.079) == "Media"
    assert nivel_prioridad(0.03) == "Media"
    assert nivel_prioridad(0.029) == "Baja"


def test_evaluar_umbrales_sin_graves_no_falla():
    # Antes dividía entre cero cuando los datos de prueba no tenían ningún grave
    class ModeloFalso:
        def predict_proba(self, X):
            return np.array([[0.9, 0.1]] * len(X))

    X = pd.DataFrame({"a": [1, 2, 3]})
    y = pd.Series([0, 0, 0])
    evaluar_umbrales(ModeloFalso(), X, y, [0.05])


def test_modelo_guardado_da_una_probabilidad_valida():
    paquete = joblib.load(RAIZ / "model.joblib")
    modelo, codigos = paquete["modelo"], paquete["codigos"]
    # Persona inventada con el primer valor conocido de cada variable de texto
    persona = {v: sorted(codigos[v])[0] for v in VARIABLES_TEXTO}
    persona.update({"hora_num": 12, "dia_semana": 2})
    probabilidad = probabilidad_persona(modelo, codigos, persona)
    assert 0.0 <= probabilidad <= 1.0
    # Los diccionarios guardados codifican todas las variables de texto
    assert list(codificar(pd.DataFrame([persona]), codigos).columns) == VARIABLES


if __name__ == "__main__":
    # Ejecutor sencillo (sin pytest): python tests/test_model.py
    pruebas = [f for nombre, f in sorted(globals().items()) if nombre.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} pruebas correctas")

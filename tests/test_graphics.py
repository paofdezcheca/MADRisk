"""Tests de src/graphics.py con datos pequeños inventados."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.graphics import (
    accidentes_por_categoria,
    accidentes_por_mes,
    colores_por_valor,
    comparacion_probabilidades,
    grupo_mas_grave,
    mapa_calor_dia_hora,
    partir_titulo,
    tasa_gravedad,
)


def datos_de_prueba() -> pd.DataFrame:
    """Dos accidentes: el primero con dos personas implicadas (dos filas)."""
    return pd.DataFrame(
        {
            "num_expediente": ["A1", "A1", "B2"],
            "mes": [1, 1, 2],
            "dia_semana": [0, 0, 5],
            "hora_num": [8, 8, 23],
            "distrito": ["CENTRO", "CENTRO", "USERA"],
            "grave": [1.0, 0.0, 0.0],
        }
    )


def test_accidentes_cuentan_expedientes_y_no_filas():
    fig = accidentes_por_mes(datos_de_prueba(), "Accidentes por mes")
    # Enero tiene 2 filas pero 1 solo accidente; febrero, 1
    assert list(fig.data[0].y) == [1, 1]


def test_sin_datos_devuelve_una_figura_con_aviso():
    vacio = datos_de_prueba().iloc[0:0]
    fig = accidentes_por_categoria(vacio, "distrito", "Distritos", "Distrito")
    assert fig.layout.annotations[0].text == "Sin datos con estos filtros"
    assert len(fig.data) == 0


def test_mapa_de_calor_tiene_una_fila_por_dia_con_datos():
    fig = mapa_calor_dia_hora(datos_de_prueba(), "Calor")
    assert list(fig.data[0].y) == ["Lunes", "Sábado"]


def test_tasa_gravedad_oculta_grupos_pequenos():
    df = datos_de_prueba()
    # Con minimo=3 ningún distrito llega (CENTRO tiene 2 personas) y queda la figura vacía
    assert len(tasa_gravedad(df, "distrito", "t", "Distrito", minimo=3).data) == 0
    # Con minimo=1 salen los dos y CENTRO tiene un 50 % de graves
    fig = tasa_gravedad(df, "distrito", "t", "Distrito", minimo=1)
    tasas = dict(zip(fig.data[0].y, fig.data[0].x))
    assert tasas == {"CENTRO": 50.0, "USERA": 0.0}


def test_colores_por_valor_extremos_de_la_escala():
    colores = colores_por_valor([1, 5, 9], ["#000000", "#808080", "#ffffff"])
    assert colores[0] == "#000000"
    assert colores[1] == "#808080"
    assert colores[2] == "#ffffff"


def test_comparacion_probabilidades_pasa_a_porcentaje():
    fig = comparacion_probabilidades(0.146, 0.003, 2.2, "#c0392b", "#2e8b57")
    assert list(fig.data[0].y) == ["Accidente A", "Accidente B", "Media general"]
    assert [round(v, 1) for v in fig.data[0].x] == [14.6, 0.3, 2.2]


def test_grupo_mas_grave_elige_la_tasa_mayor_y_respeta_el_minimo():
    df = datos_de_prueba()
    mayor = grupo_mas_grave(df, "distrito", minimo=1)
    # CENTRO: 1 grave de 2 personas (50 %); la media general es 1/3
    assert mayor["valor"] == "CENTRO"
    assert mayor["tasa"] == 50.0
    assert mayor["personas"] == 2
    assert round(mayor["veces"], 1) == 1.5
    # Con un mínimo que ningún grupo alcanza no hay conclusión
    assert grupo_mas_grave(df, "distrito", minimo=5) is None


def test_titulos_largos_se_parten_en_lineas():
    corto, n_corto = partir_titulo("Accidentes por mes")
    assert corto == "Accidentes por mes" and n_corto == 1
    largo, n_largo = partir_titulo("Probabilidad de lesión grave de cada aviso frente a la media")
    assert n_largo == 2 and largo.count("<br>") == 1
    # Los gráficos reales usan el título partido y dejan más margen arriba
    fig = comparacion_probabilidades(0.1, 0.2, 2.2, "#c0392b", "#2e8b57")
    assert "<br>" in fig.layout.title.text
    assert fig.layout.margin.t > 54


if __name__ == "__main__":
    # Ejecutor sencillo (sin pytest): python tests/test_graphics.py
    pruebas = [f for nombre, f in sorted(globals().items()) if nombre.startswith("test_")]
    for prueba in pruebas:
        prueba()
        print(f"OK  {prueba.__name__}")
    print(f"\n{len(pruebas)} pruebas correctas")

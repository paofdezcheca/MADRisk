"""MADRisk: app en Dash para analizar la gravedad de los accidentes de tráfico en Madrid.

Bloque 1 (exploración): filtros y gráficos descriptivos.
Cada fila de los datos es una persona implicada; un accidente se repite en varias filas.
"""

from pathlib import Path

import joblib
from dash import Dash, Input, Output, State, dcc, html

from src.etl import cargar_datos
from src.model import VARIABLES, probabilidad_persona
from src.graphics import (
    DIAS,
    MESES,
    accidentes_por_categoria,
    accidentes_por_dia_semana,
    accidentes_por_hora,
    accidentes_por_mes,
    grafico_probabilidad,
    mapa_calor_dia_hora,
    personas_por_categoria,
    tasa_gravedad,
)

# Los datos se cargan UNA sola vez al arrancar la app, no en cada callback
datos = cargar_datos()

TODOS = "Todos"


def crear_opciones(columna, nombres=None):
    """Opciones de un desplegable: 'Todos' más cada valor distinto de la columna.

    `nombres` es un diccionario opcional valor -> texto (p. ej. 1 -> "Ene").
    """
    opciones = [{"label": "Todos", "value": TODOS}]
    for valor in sorted(datos[columna].unique().tolist()):
        texto = nombres[valor] if nombres else str(valor)
        opciones.append({"label": texto, "value": valor})
    return opciones


# Filtros: (id del desplegable, columna de los datos, texto de la etiqueta, nombres)
FILTROS = [
    ("f_anio", "anio", "Año", None),
    ("f_mes", "mes", "Mes", MESES),
    ("f_dia", "dia_semana", "Día de la semana", DIAS),
    ("f_hora", "hora_num", "Hora", {h: f"{h}:00" for h in range(24)}),
    ("f_distrito", "distrito", "Distrito", None),
    ("f_tipo_acc", "tipo_accidente", "Tipo de accidente", None),
    ("f_vehiculo", "tipo_vehiculo", "Tipo de vehículo", None),
    ("f_meteo", "estado_meteorologico", "Meteorología", None),
    ("f_lesividad", "lesividad", "Lesividad", None),
]


def filtrar(valores):
    """Aplica los filtros elegidos a los datos. `valores` sigue el orden de FILTROS."""
    filtrados = datos
    for (_, columna, _, _), valor in zip(FILTROS, valores):
        if valor is not None and valor != TODOS:
            filtrados = filtrados[filtrados[columna] == valor]
    return filtrados


def formato_miles(numero):
    """12345 -> '12.345' (separador de miles a la española)."""
    return f"{numero:,}".replace(",", ".")


def caja_grafico(id_grafico, ancho="48%"):
    """Contenedor con un gráfico; dos por fila si el ancho es 48%."""
    return html.Div(
        children=[dcc.Graph(id=id_grafico)],
        style={"width": ancho, "display": "inline-block", "vertical-align": "top"},
    )


# Un desplegable por filtro, con su etiqueta encima
controles = []
for id_filtro, columna, etiqueta, nombres in FILTROS:
    controles.append(
        html.Div(
            children=[
                html.P(etiqueta, style={"margin": "0", "font-weight": "bold"}),
                dcc.Dropdown(
                    options=crear_opciones(columna, nombres),
                    value=TODOS,
                    id=id_filtro,
                ),
            ],
            style={"width": "200px", "display": "inline-block", "margin": "5px 10px"},
        )
    )


# Factores para el análisis de gravedad:
# etiqueta -> (columna, nombres de los valores, ordenar por tasa)
# Si ordenar por tasa es False, se mantiene el orden natural (horas, días, meses...)
FACTORES = {
    "Hora del día": ("hora_num", {h: f"{h}:00" for h in range(24)}, False),
    "Franja horaria": ("franja", None, True),
    "Día de la semana": ("dia_semana", DIAS, False),
    "Mes": ("mes", MESES, False),
    "Distrito": ("distrito", None, True),
    "Tipo de accidente": ("tipo_accidente", None, True),
    "Tipo de vehículo": ("tipo_vehiculo", None, True),
    "Tipo de persona": ("tipo_persona", None, True),
    "Rango de edad": ("rango_edad", None, True),
    "Sexo": ("sexo", None, True),
    "Meteorología": ("estado_meteorologico", None, True),
    "Alcohol positivo": ("positiva_alcohol", None, True),
}
opciones_factor = []
for etiqueta in FACTORES:
    opciones_factor.append({"label": etiqueta, "value": etiqueta})

opciones_anio_gravedad = crear_opciones("anio")


# --- Predicción -------------------------------------------------------------
# El modelo se entrena en train.py; aquí SOLO se carga (nunca se hace fit en la app)
paquete = joblib.load(Path(__file__).resolve().parent / "model.joblib")
modelo = paquete["modelo"]
codigos = paquete["codigos"]

# % medio de personas graves entre las de lesividad informada (referencia del gráfico)
media_graves = datos["grave"].mean() * 100


def valor_mas_frecuente(columna):
    """Valor más habitual de una columna que no sea 'Desconocido' ni 'No informado'."""
    for valor in datos[columna].value_counts().index:
        if valor not in ("Desconocido", "No informado"):
            return valor


# Campos del formulario: variable -> (etiqueta, opciones)
# Las numéricas muestran un texto (p. ej. 0 -> "Lunes"); las de texto, sus valores.
CAMPOS = {
    "hora_num": ("Hora", [{"label": f"{h}:00", "value": h} for h in range(24)]),
    "dia_semana": ("Día de la semana", [{"label": DIAS[d], "value": d} for d in DIAS]),
    "mes": ("Mes", [{"label": MESES[m], "value": m} for m in MESES]),
    "distrito": ("Distrito", None),
    "tipo_accidente": ("Tipo de accidente", None),
    "estado_meteorologico": ("Meteorología", None),
    "tipo_vehiculo": ("Tipo de vehículo implicado", None),
    "tipo_persona": ("Tipo de persona", None),
    "rango_edad": ("Rango de edad", None),
    "sexo": ("Sexo", None),
    "positiva_alcohol": ("Positivo en alcohol (S/N)", None),
}

controles_prediccion = []
for variable in VARIABLES:
    etiqueta, opciones = CAMPOS[variable]
    if opciones is None:
        opciones = [{"label": v, "value": v} for v in sorted(codigos[variable])]
    controles_prediccion.append(
        html.Div(
            children=[
                html.P(etiqueta, style={"margin": "0", "font-weight": "bold"}),
                dcc.Dropdown(
                    options=opciones,
                    value=valor_mas_frecuente(variable),
                    id="p_" + variable,
                ),
            ],
            style={"width": "220px", "display": "inline-block", "margin": "5px 10px"},
        )
    )

app = Dash(__name__)
server = app.server  # obligatorio para gunicorn (Render)

app.layout = html.Div(
    children=[
        html.H1(
            "MADRisk",
            style={"text-align": "center", "backgroundColor": "lightblue"},
        ),
        html.P(
            "Análisis de los accidentes de tráfico en Madrid (2020-2025). "
            "Cada fila de los datos es una persona implicada, y un mismo accidente puede "
            "implicar a varias. Cada gráfico indica si cuenta accidentes o personas.",
            style={"margin": "10px 20px"},
        ),
        html.P(
            "Los filtros de vehículo y lesividad se aplican a las personas: un accidente "
            "se muestra si al menos una persona implicada cumple el filtro. "
            "Una lesividad vacía se muestra como \"No informada\" (no significa sin lesión).",
            style={"margin": "10px 20px", "color": "gray"},
        ),
        html.Div(children=controles, style={"margin": "0 10px"}),
        html.H3(id="resumen", style={"margin": "15px 20px"}),
        caja_grafico("g_mes"),
        caja_grafico("g_dia"),
        caja_grafico("g_hora"),
        caja_grafico("g_calor"),
        caja_grafico("g_distrito"),
        caja_grafico("g_tipo_acc"),
        caja_grafico("g_meteo"),
        caja_grafico("g_vehiculo"),
        caja_grafico("g_lesividad", ancho="98%"),
        html.H2("Análisis de la gravedad", style={"margin": "30px 20px 0"}),
        html.P(
            "Gravedad = persona con lesión grave: hospitalizada más de 24 horas o fallecida, "
            "criterio oficial de la DGT para herido grave. "
            "El porcentaje se calcula solo con las personas cuya lesividad está informada. "
            "Pasa el ratón por una barra para ver cuántas personas hay detrás. "
            "Se ocultan los grupos con menos de 100 personas.",
            style={"margin": "10px 20px"},
        ),
        html.Div(
            children=[
                html.P("Factor", style={"margin": "0", "font-weight": "bold"}),
                dcc.Dropdown(options=opciones_factor, value="Hora del día", id="g_factor"),
            ],
            style={"width": "260px", "display": "inline-block", "margin": "5px 10px"},
        ),
        html.Div(
            children=[
                html.P("Año", style={"margin": "0", "font-weight": "bold"}),
                dcc.Dropdown(options=opciones_anio_gravedad, value=TODOS, id="g_anio"),
            ],
            style={"width": "200px", "display": "inline-block", "margin": "5px 10px"},
        ),
        dcc.Graph(id="g_gravedad"),
        html.H2("Predicción de la gravedad", style={"margin": "30px 20px 0"}),
        html.P(
            "Describe a una persona implicada en un accidente que YA ha ocurrido y el modelo "
            "estima la probabilidad de que sufra una lesión grave. No predice si va a "
            "ocurrir un accidente. En un atropello, el tipo de vehículo es el que atropelló.",
            style={"margin": "10px 20px"},
        ),
        html.P(
            "Es una estimación orientativa: el modelo ordena bien el riesgo (la gravedad es "
            "poco frecuente, un 2,2 %) pero no es preciso para un caso individual.",
            style={"margin": "10px 20px", "color": "gray"},
        ),
        html.Div(children=controles_prediccion, style={"margin": "0 10px"}),
        html.Button(
            "Calcular probabilidad", id="p_boton", n_clicks=0,
            style={"margin": "10px 20px", "padding": "8px 16px"},
        ),
        html.H3(id="p_resultado", style={"margin": "10px 20px"}),
        dcc.Graph(id="p_grafico"),
    ]
)


@app.callback(
    Output("resumen", "children"),
    Output("g_mes", "figure"),
    Output("g_dia", "figure"),
    Output("g_hora", "figure"),
    Output("g_calor", "figure"),
    Output("g_distrito", "figure"),
    Output("g_tipo_acc", "figure"),
    Output("g_meteo", "figure"),
    Output("g_vehiculo", "figure"),
    Output("g_lesividad", "figure"),
    Input("f_anio", "value"),
    Input("f_mes", "value"),
    Input("f_dia", "value"),
    Input("f_hora", "value"),
    Input("f_distrito", "value"),
    Input("f_tipo_acc", "value"),
    Input("f_vehiculo", "value"),
    Input("f_meteo", "value"),
    Input("f_lesividad", "value"),
)
def actualizar(*valores):
    """Filtra los datos con los 9 desplegables y redibuja todos los gráficos."""
    filtrados = filtrar(valores)

    # Resumen: accidentes distintos frente a personas (filas)
    n_accidentes = len(filtrados.drop_duplicates(subset="num_expediente"))
    resumen = (
        f"{formato_miles(n_accidentes)} accidentes · "
        f"{formato_miles(len(filtrados))} personas implicadas"
    )

    return (
        resumen,
        accidentes_por_mes(filtrados, "Accidentes por mes"),
        accidentes_por_dia_semana(filtrados, "Accidentes por día de la semana"),
        accidentes_por_hora(filtrados, "Accidentes por hora del día"),
        mapa_calor_dia_hora(filtrados, "Accidentes por día de la semana y hora"),
        accidentes_por_categoria(
            filtrados, "distrito", "Accidentes por distrito", "Distrito"
        ),
        accidentes_por_categoria(
            filtrados, "tipo_accidente", "Accidentes por tipo de accidente", "Tipo"
        ),
        accidentes_por_categoria(
            filtrados, "estado_meteorologico", "Accidentes por meteorología", "Estado"
        ),
        personas_por_categoria(
            filtrados, "tipo_vehiculo", "Personas por tipo de vehículo (10 más frecuentes)",
            "Vehículo", maximo=10,
        ),
        personas_por_categoria(
            filtrados, "lesividad", "Personas por lesividad", "Lesividad"
        ),
    )


@app.callback(
    Output("g_gravedad", "figure"),
    Input("g_factor", "value"),
    Input("g_anio", "value"),
)
def actualizar_gravedad(factor, anio):
    """Dibuja el % de personas con lesión grave según el factor y el año elegidos."""
    columna, nombres, ordenar_por_tasa = FACTORES[factor]
    if anio is None or anio == TODOS:
        filtrados = datos
        sufijo = "2020-2025"
    else:
        filtrados = datos[datos["anio"] == anio]
        sufijo = str(anio)
    return tasa_gravedad(
        filtrados, columna, f"Gravedad según: {factor} ({sufijo})", factor,
        nombres=nombres, ordenar_por_tasa=ordenar_por_tasa,
    )


@app.callback(
    Output("p_resultado", "children"),
    Output("p_grafico", "figure"),
    Input("p_boton", "n_clicks"),
    *[State("p_" + variable, "value") for variable in VARIABLES],
)
def predecir(n_clicks, *valores):
    """Calcula la probabilidad de lesión grave de la persona descrita en el formulario."""
    caracteristicas = dict(zip(VARIABLES, valores))
    if None in valores:
        return "Completa todos los campos para calcular la probabilidad.", grafico_probabilidad(0, media_graves)

    probabilidad = probabilidad_persona(modelo, codigos, caracteristicas) * 100
    veces = probabilidad / media_graves
    texto = (
        f"Probabilidad estimada de lesión grave: {probabilidad:.1f} % "
        f"({veces:.1f} veces la media, que es {media_graves:.1f} %)"
    )
    return texto, grafico_probabilidad(probabilidad, media_graves)


if __name__ == "__main__":
    app.run(debug=True)

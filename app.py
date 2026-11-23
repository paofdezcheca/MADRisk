"""MADRisk: app en Dash para analizar la gravedad de los accidentes de tráfico en Madrid.

Versión mínima: un filtro por año y un gráfico de accidentes por mes.
"""

from dash import Dash, Input, Output, dcc, html

from src.etl import cargar_datos
from src.graphics import accidentes_por_mes

# Los datos se cargan UNA sola vez al arrancar la app, no en cada callback
datos = cargar_datos()

# Opciones del desplegable: "Todos" más cada año disponible (ordenados)
TODOS = "Todos"
anios = sorted(datos["anio"].unique().tolist())
opciones_anio = [{"label": "Todos los años", "value": TODOS}]
for anio in anios:
    opciones_anio.append({"label": str(anio), "value": anio})

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
            "Los gráficos cuentan accidentes, no personas: un mismo accidente "
            "puede implicar a varias personas.",
            style={"margin": "10px 20px"},
        ),
        dcc.Dropdown(
            options=opciones_anio,
            value=TODOS,
            id="dropdown_anio",
            style={"width": "300px", "margin-left": "20px"},
        ),
        dcc.Graph(id="grafico_mes"),
    ]
)


@app.callback(
    Output("grafico_mes", "figure"),
    Input("dropdown_anio", "value"),
)
def actualizar_grafico(anio):
    """Filtra los datos por año (si se elige uno) y dibuja los accidentes por mes."""
    if anio is None or anio == TODOS:
        filtrados = datos
        titulo = "Accidentes por mes (2020-2025)"
    else:
        filtrados = datos[datos["anio"] == anio]
        titulo = f"Accidentes por mes en {anio}"
    return accidentes_por_mes(filtrados, titulo)


if __name__ == "__main__":
    app.run(debug=True)

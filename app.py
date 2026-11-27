"""MADRisk: ayuda a decidir a qué accidente atender primero.

La página principal compara dos avisos de accidente y estima, para cada uno, la
probabilidad de que una persona implicada sufra una lesión grave. Es una estimación
sobre un accidente YA ocurrido; no predice si va a ocurrir un accidente.
Debajo, tras un botón, están los datos que respaldan el modelo (exploración y análisis).

Cada fila de los datos es una persona implicada; un accidente se repite en varias filas.
"""

from pathlib import Path

import joblib
from dash import Dash, Input, Output, State, dcc, html

from src.etl import cargar_datos
from src.graphics import (
    DIAS,
    MESES,
    accidentes_por_categoria,
    accidentes_por_dia_semana,
    accidentes_por_hora,
    accidentes_por_mes,
    comparacion_probabilidades,
    mapa_calor_dia_hora,
    personas_por_categoria,
    tasa_gravedad,
)
from src.model import nivel_prioridad, probabilidad_persona

# Los datos se cargan UNA sola vez al arrancar la app, no en cada callback
datos = cargar_datos()

# El modelo se entrena en train.py; aquí SOLO se carga (nunca se hace fit en la app)
paquete = joblib.load(Path(__file__).resolve().parent / "model.joblib")
modelo = paquete["modelo"]
codigos = paquete["codigos"]
niveles = paquete["niveles"]  # % real de graves dentro de cada nivel (datos de prueba)

# % medio de personas graves entre las de lesividad informada
media_graves = datos["grave"].mean() * 100
personas_modelo = int(datos["grave"].notna().sum())

# --- Aspecto de la web -------------------------------------------------------
AZUL = "#1f4e79"
GRIS_FONDO = "#f3f5f8"
COLOR_NIVEL = {"Alta": "#c0392b", "Media": "#d68910", "Baja": "#2e8b57"}

FUENTE = "Segoe UI, Arial, sans-serif"

ESTILO_PAGINA = {
    "backgroundColor": GRIS_FONDO,
    "fontFamily": FUENTE,
    "color": "#2b2b2b",
    "minHeight": "100vh",
}
ESTILO_CONTENIDO = {"maxWidth": "1100px", "margin": "0 auto", "padding": "0 16px 40px"}
ESTILO_TARJETA = {
    "backgroundColor": "white",
    "borderRadius": "12px",
    "padding": "20px 24px",
    "margin": "16px 0",
    "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
}
ESTILO_FILA = {"display": "flex", "flexWrap": "wrap", "gap": "16px"}
ESTILO_ETIQUETA = {"margin": "10px 0 4px", "fontWeight": "600", "fontSize": "14px"}
ESTILO_AYUDA = {"margin": "0 0 4px", "fontSize": "12px", "color": "#6b7280"}
ESTILO_BOTON = {
    "backgroundColor": AZUL,
    "color": "white",
    "border": "none",
    "borderRadius": "8px",
    "padding": "12px 28px",
    "fontSize": "16px",
    "fontWeight": "600",
    "cursor": "pointer",
}
ESTILO_BOTON_SECUNDARIO = {
    **ESTILO_BOTON,
    "backgroundColor": "white",
    "color": AZUL,
    "border": f"2px solid {AZUL}",
}


def formato_miles(numero):
    """12345 -> '12.345' (separador de miles a la española)."""
    return f"{numero:,}".replace(",", ".")


def un_decimal(numero):
    """12.34 -> '12,3' (coma decimal a la española)."""
    return f"{numero:.1f}".replace(".", ",")


def valor_mas_frecuente(columna):
    """Valor más habitual de una columna que no sea 'Desconocido' ni 'No informado'."""
    for valor in datos[columna].value_counts().index:
        if valor not in ("Desconocido", "No informado"):
            return valor


# --- Formulario de cada aviso ------------------------------------------------
# Solo se piden datos que se conocen al recibir el aviso. Orden en la pantalla:
# variable -> (etiqueta, ayuda, opciones; None = los valores que conoce el modelo)
CAMPOS = {
    "tipo_persona": ("Persona implicada", "Conductor, pasajero o peatón", None),
    "tipo_vehiculo": (
        "Vehículo implicado",
        "En un atropello, el vehículo que atropelló", None,
    ),
    "tipo_accidente": ("Tipo de accidente", "Cómo se produjo", None),
    "rango_edad": ("Edad aproximada", "Si no se sabe, 'Desconocido'", None),
    "distrito": ("Distrito", "Dónde ha ocurrido", None),
    "hora_num": ("Hora del aviso", "", [{"label": f"{h}:00", "value": h} for h in range(24)]),
    "dia_semana": ("Día de la semana", "", [{"label": DIAS[d], "value": d} for d in DIAS]),
    "estado_meteorologico": ("Meteorología", "", None),
}
ORDEN_FORMULARIO = list(CAMPOS)

# Valores con los que arranca cada desplegable (los más habituales en los datos)
VALORES_POR_DEFECTO = {variable: valor_mas_frecuente(variable) for variable in ORDEN_FORMULARIO
                       if variable not in ("hora_num", "dia_semana")}
VALORES_POR_DEFECTO.update({"hora_num": 12, "dia_semana": 0})

# Ejemplos para probar la herramienta con un clic: cada uno es una pareja (aviso A, aviso B)
# con orden de gravedad muy distinto. Cada pulsación del botón pasa al siguiente.
EJEMPLOS = [
    (  # Una persona mayor atropellada de noche frente a un choque lateral de día
        {"tipo_persona": "Peatón", "tipo_vehiculo": "Turismo",
         "tipo_accidente": "Atropello a persona", "rango_edad": "Más de 74 años",
         "distrito": "CENTRO", "hora_num": 22, "dia_semana": 4,
         "estado_meteorologico": "Lluvia débil"},
        {"tipo_persona": "Conductor", "tipo_vehiculo": "Turismo",
         "tipo_accidente": "Colisión lateral", "rango_edad": "De 30 a 34 años",
         "distrito": "SALAMANCA", "hora_num": 9, "dia_semana": 1,
         "estado_meteorologico": "Despejado"},
    ),
    (  # Una caída en moto grande de madrugada frente a un alcance con pasajero
        {"tipo_persona": "Conductor", "tipo_vehiculo": "Motocicleta > 125cc",
         "tipo_accidente": "Caída", "rango_edad": "De 25 a 29 años",
         "distrito": "FUENCARRAL-EL PARDO", "hora_num": 3, "dia_semana": 5,
         "estado_meteorologico": "Despejado"},
        {"tipo_persona": "Pasajero", "tipo_vehiculo": "Turismo",
         "tipo_accidente": "Alcance", "rango_edad": "De 40 a 44 años",
         "distrito": "CHAMBERÍ", "hora_num": 8, "dia_semana": 2,
         "estado_meteorologico": "Nublado"},
    ),
    (  # Dos casos parecidos: la herramienta debe decir que decide el operador
        {"tipo_persona": "Conductor", "tipo_vehiculo": "Turismo",
         "tipo_accidente": "Alcance", "rango_edad": "De 35 a 39 años",
         "distrito": "LATINA", "hora_num": 18, "dia_semana": 3,
         "estado_meteorologico": "Despejado"},
        {"tipo_persona": "Conductor", "tipo_vehiculo": "Turismo",
         "tipo_accidente": "Alcance", "rango_edad": "De 40 a 44 años",
         "distrito": "CARABANCHEL", "hora_num": 17, "dia_semana": 3,
         "estado_meteorologico": "Despejado"},
    ),
]


def tarjeta_aviso(letra):
    """Tarjeta con los desplegables de un aviso (ids: 'A_tipo_persona', 'B_distrito'...)."""
    controles = []
    for variable in ORDEN_FORMULARIO:
        etiqueta, ayuda, opciones = CAMPOS[variable]
        if opciones is None:
            opciones = [{"label": v, "value": v} for v in sorted(codigos[variable])]
        bloque = [html.P(etiqueta, style=ESTILO_ETIQUETA)]
        if ayuda:
            bloque.append(html.P(ayuda, style=ESTILO_AYUDA))
        bloque.append(
            dcc.Dropdown(
                options=opciones,
                value=valor_mas_frecuente(variable),
                id=f"{letra}_{variable}",
            )
        )
        controles.append(html.Div(children=bloque))

    return html.Div(
        children=[
            html.H3(
                f"Accidente {letra}",
                style={"margin": "0 0 4px", "color": AZUL, "fontSize": "20px"},
            ),
            *controles,
        ],
        style={**ESTILO_TARJETA, "flex": "1 1 380px", "margin": "0"},
    )


def tarjeta_resultado(letra, probabilidad, borde):
    """Resultado de un aviso: probabilidad, nivel de prioridad y qué significa."""
    nivel = nivel_prioridad(probabilidad)
    color = COLOR_NIVEL[nivel]
    tasa = niveles[nivel]["tasa_grave"]
    return html.Div(
        children=[
            html.H3(f"Accidente {letra}", style={"margin": "0", "color": AZUL}),
            html.P(
                f"{un_decimal(probabilidad * 100)} %",
                style={"fontSize": "48px", "fontWeight": "700", "margin": "4px 0",
                       "color": color},
            ),
            html.P("probabilidad de lesión grave", style={**ESTILO_AYUDA, "fontSize": "14px"}),
            html.Span(
                f"Prioridad {nivel.upper()}",
                style={"backgroundColor": color, "color": "white", "padding": "4px 12px",
                       "borderRadius": "999px", "fontWeight": "600", "fontSize": "14px"},
            ),
            html.P(
                f"En los datos de prueba, el {un_decimal(tasa)} % de las personas con "
                f"prioridad {nivel.lower()} sufrió una lesión grave (la media general es "
                f"{un_decimal(media_graves)} %).",
                style={"fontSize": "14px", "marginTop": "14px"},
            ),
        ],
        style={**ESTILO_TARJETA, "flex": "1 1 300px", "margin": "0",
               "borderTop": f"6px solid {borde}", "textAlign": "center"},
    )


# --- Filtros y factores del análisis de datos ---------------------------------
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
opciones_factor = [{"label": etiqueta, "value": etiqueta} for etiqueta in FACTORES]


def caja_desplegable(etiqueta, id_desplegable, opciones, ancho="200px", valor=TODOS):
    """Desplegable con su etiqueta encima, para los filtros del análisis."""
    return html.Div(
        children=[
            html.P(etiqueta, style={**ESTILO_ETIQUETA, "margin": "0 0 4px"}),
            dcc.Dropdown(options=opciones, value=valor, id=id_desplegable),
        ],
        style={"width": ancho},
    )


def caja_grafico(id_grafico, ancho="48%"):
    """Contenedor con un gráfico; dos por fila si el ancho es 48%."""
    return html.Div(
        children=[dcc.Graph(id=id_grafico)],
        style={"flex": f"1 1 {ancho}", "minWidth": "320px"},
    )


filtros_exploracion = [
    caja_desplegable(etiqueta, id_filtro, crear_opciones(columna, nombres))
    for id_filtro, columna, etiqueta, nombres in FILTROS
]

# Tabla de niveles: qué significa cada nivel con datos reales
filas_niveles = []
for nivel in ["Alta", "Media", "Baja"]:
    filas_niveles.append(
        html.Div(
            children=[
                html.Span(
                    f"Prioridad {nivel}",
                    style={"backgroundColor": COLOR_NIVEL[nivel], "color": "white",
                           "padding": "3px 12px", "borderRadius": "999px",
                           "fontWeight": "600", "fontSize": "13px", "minWidth": "120px",
                           "textAlign": "center"},
                ),
                html.Span(
                    f"{un_decimal(niveles[nivel]['tasa_grave'])} % de lesiones graves "
                    f"({formato_miles(niveles[nivel]['personas'])} personas de prueba)"
                ),
            ],
            style={"display": "flex", "gap": "14px", "alignItems": "center",
                   "margin": "6px 0"},
        )
    )

app = Dash(__name__)
server = app.server  # obligatorio para gunicorn (Render)

# --- Carrusel de fotos -------------------------------------------------------
# Fotos de intervenciones del SAMUR de Madrid en accidentes de tráfico (carpeta assets/,
# que Dash sirve solo). Autor: Diario de Madrid, licencia CC BY 4.0, vía Wikimedia Commons.
FOTOS = [f"carrusel/foto{i}.jpg" for i in range(1, 6)]
SEGUNDOS_POR_FOTO = 5
ALTO_CARRUSEL = "300px"

# Para que el carrusel avance siempre hacia la derecha y pueda "dar la vuelta" sin que
# se note, la tira repite la secuencia de fotos varias veces. Solo al final de la tira
# (cada ~3 minutos) hay un salto sin animación.
REPETICIONES = 8
N_FOTOS = len(FOTOS)
HUECOS = N_FOTOS * REPETICIONES  # número de fotos en la tira
CICLO = HUECOS - N_FOTOS  # avances antes de volver al principio de la tira


def estilo_tira(avance):
    """Estilo de la tira de fotos tras `avance` avances: la desplaza hacia la derecha."""
    posicion = avance % CICLO
    # La foto 0 está en el hueco de más a la derecha; cada avance baja un hueco, es
    # decir, mueve la tira hacia la derecha.
    hueco = HUECOS - 1 - posicion
    # Sin animación solo cuando se vuelve al principio de la tira
    animacion = "none" if (posicion == 0 and avance > 0) else "transform 1s ease-in-out"
    return {
        "display": "flex",
        "width": f"{HUECOS * 100}%",
        "height": "100%",
        "transform": f"translateX(-{hueco * 100 / HUECOS}%)",
        "transition": animacion,
    }


# En el hueco `h` va la foto (HUECOS - 1 - h) % N_FOTOS: así, al bajar de hueco, la
# foto que entra por la izquierda es la siguiente de la lista
fotos_tira = []
for hueco in range(HUECOS):
    foto = FOTOS[(HUECOS - 1 - hueco) % N_FOTOS]
    fotos_tira.append(
        html.Img(
            src=app.get_asset_url(foto),
            alt="Intervención del SAMUR en un accidente de tráfico en Madrid",
            style={"width": f"{100 / HUECOS}%", "height": "100%", "objectFit": "cover",
                   "display": "block"},
        )
    )

carrusel = html.Div(
    children=[
        html.Div(children=fotos_tira, id="carrusel_tira", style=estilo_tira(0)),
        dcc.Interval(id="carrusel_reloj", interval=SEGUNDOS_POR_FOTO * 1000),
    ],
    style={"width": "100%", "height": ALTO_CARRUSEL, "overflow": "hidden",
           "backgroundColor": "#10253a"},
)

app.layout = html.Div(
    style=ESTILO_PAGINA,
    children=[
        carrusel,
        # Cabecera
        html.Div(
            children=[
                html.H1("MADRisk", style={"margin": "0", "fontSize": "34px"}),
                html.P(
                    "¿A qué accidente atender primero?",
                    style={"margin": "4px 0 0", "fontSize": "20px", "opacity": "0.9"},
                ),
            ],
            style={"backgroundColor": AZUL, "color": "white", "padding": "28px 16px",
                   "textAlign": "center"},
        ),
        html.Div(
            style=ESTILO_CONTENIDO,
            children=[
                # Qué predice, exactamente
                html.Div(
                    children=[
                        html.H2("Qué predice, exactamente",
                                style={"marginTop": "0", "color": AZUL}),
                        html.P([
                            "Para un accidente que ",
                            html.B("ya ha ocurrido"),
                            ", estima la probabilidad de que una persona implicada sufra una ",
                            html.B("lesión grave"),
                            ": ingreso hospitalario de más de 24 horas o fallecimiento "
                            "(criterio oficial de la DGT para herido grave).",
                        ]),
                        html.P("Compara dos avisos y propone atender primero el de mayor "
                               "probabilidad."),
                        html.P(
                            "No predice si va a ocurrir un accidente, ni cuántos heridos habrá, "
                            "ni sustituye al criterio de los servicios de emergencia.",
                            style={"color": "#6b7280"},
                        ),
                    ],
                    style=ESTILO_TARJETA,
                ),
                # Formulario de los dos avisos
                html.H2("Describe los dos avisos", style={"color": AZUL, "marginBottom": "4px"}),
                html.P(
                    "Cambia lo que sepas de cada aviso. Los campos vienen con el valor "
                    "más habitual.",
                    style={"marginTop": "0", "color": "#6b7280"},
                ),
                html.Div(children=[tarjeta_aviso("A"), tarjeta_aviso("B")], style=ESTILO_FILA),
                html.Div(
                    children=[
                        html.Button("Cargar un ejemplo", id="boton_ejemplo",
                                    n_clicks=0, style=ESTILO_BOTON_SECUNDARIO),
                        html.Button("Comparar avisos", id="boton_comparar",
                                    n_clicks=0, style=ESTILO_BOTON),
                    ],
                    style={"textAlign": "center", "margin": "24px 0", "display": "flex",
                           "gap": "12px", "justifyContent": "center", "flexWrap": "wrap"},
                ),
                # Resultado (se rellena en el callback)
                html.Div(id="resultado"),
                # Cómo interpretarlo
                html.Div(
                    children=[
                        html.H2("Cómo interpretar el resultado",
                                style={"marginTop": "0", "color": AZUL}),
                        html.P(
                            f"El modelo se ha entrenado con {formato_miles(personas_modelo)} "
                            "personas implicadas en accidentes de Madrid (2020-2025). "
                            f"Solo el {un_decimal(media_graves)} % sufre una lesión grave. Esto es lo "
                            "que pasó en la realidad con las personas que el modelo puso en "
                            "cada nivel (datos que no vio al entrenar):"
                        ),
                        *filas_niveles,
                        html.P(
                            "Es una ayuda para ordenar el riesgo, no un diagnóstico: aun con "
                            "prioridad alta, la mayoría de las personas no sufre una lesión "
                            "grave. El modelo solo conoce las características del aviso, no el "
                            "estado real de los heridos.",
                            style={"color": "#6b7280", "marginTop": "14px"},
                        ),
                        html.P(
                            "Proyecto académico (DAVD, ICAI) con datos abiertos del "
                            "Ayuntamiento de Madrid. No es una herramienta oficial.",
                            style={"color": "#6b7280", "marginBottom": "0"},
                        ),
                    ],
                    style=ESTILO_TARJETA,
                ),
                # Datos detrás del modelo (se muestran con el botón)
                html.Div(
                    children=[html.Button(
                        "Ver los datos detrás del modelo", id="boton_analisis",
                        n_clicks=0, style=ESTILO_BOTON,
                    )],
                    style={"textAlign": "center", "margin": "24px 0"},
                ),
                html.Div(
                    id="analisis",
                    style={"display": "none"},
                    children=[
                        html.Div(
                            children=[
                                html.H2("Exploración de los accidentes",
                                        style={"marginTop": "0", "color": AZUL}),
                                html.P(
                                    "Cada gráfico indica si cuenta accidentes o personas: cada "
                                    "fila de los datos es una persona implicada y un mismo "
                                    "accidente puede implicar a varias. Los filtros de vehículo "
                                    "y lesividad se aplican a las personas. Una lesividad vacía "
                                    "se muestra como \"No informada\" (no significa sin lesión)."
                                ),
                                html.Div(children=filtros_exploracion,
                                         style={**ESTILO_FILA, "gap": "12px"}),
                                html.H3(id="resumen", style={"color": AZUL}),
                                html.Div(
                                    children=[
                                        caja_grafico("g_mes"), caja_grafico("g_dia"),
                                        caja_grafico("g_hora"), caja_grafico("g_calor"),
                                        caja_grafico("g_distrito"), caja_grafico("g_tipo_acc"),
                                        caja_grafico("g_meteo"), caja_grafico("g_vehiculo"),
                                        caja_grafico("g_lesividad", ancho="98%"),
                                    ],
                                    style=ESTILO_FILA,
                                ),
                            ],
                            style=ESTILO_TARJETA,
                        ),
                        html.Div(
                            children=[
                                html.H2("Qué factores se asocian a la gravedad",
                                        style={"marginTop": "0", "color": AZUL}),
                                html.P(
                                    "Gravedad = persona con lesión grave: hospitalizada más de "
                                    "24 horas o fallecida, criterio oficial de la DGT para "
                                    "herido grave. El porcentaje se calcula solo con las "
                                    "personas cuya lesividad está informada. Pasa el ratón por "
                                    "una barra para ver cuántas personas hay detrás. Se ocultan "
                                    "los grupos con menos de 100 personas."
                                ),
                                html.Div(
                                    children=[
                                        caja_desplegable("Factor", "g_factor", opciones_factor,
                                                         ancho="260px", valor="Tipo de persona"),
                                        caja_desplegable("Año", "g_anio", crear_opciones("anio")),
                                    ],
                                    style=ESTILO_FILA,
                                ),
                                dcc.Graph(id="g_gravedad"),
                            ],
                            style=ESTILO_TARJETA,
                        ),
                    ],
                ),
                html.P(
                    "Datos: Portal de Datos Abiertos del Ayuntamiento de Madrid, "
                    "accidentes de tráfico 2020-2025. Fotos: Diario de Madrid "
                    "(CC BY 4.0), vía Wikimedia Commons.",
                    style={"textAlign": "center", "color": "#6b7280", "fontSize": "13px"},
                ),
            ],
        ),
    ],
)


@app.callback(
    *[Output(f"{letra}_{variable}", "value")
      for letra in ("A", "B") for variable in ORDEN_FORMULARIO],
    Input("boton_ejemplo", "n_clicks"),
)
def cargar_ejemplo(n_clicks):
    """Rellena los dos avisos con el siguiente ejemplo (sin pulsar nada, los valores habituales)."""
    if n_clicks == 0:
        aviso_a = aviso_b = VALORES_POR_DEFECTO
    else:
        aviso_a, aviso_b = EJEMPLOS[(n_clicks - 1) % len(EJEMPLOS)]
    return [aviso_a[v] for v in ORDEN_FORMULARIO] + [aviso_b[v] for v in ORDEN_FORMULARIO]


@app.callback(
    Output("resultado", "children"),
    Input("boton_comparar", "n_clicks"),
    *[State(f"A_{variable}", "value") for variable in ORDEN_FORMULARIO],
    *[State(f"B_{variable}", "value") for variable in ORDEN_FORMULARIO],
)
def comparar(n_clicks, *valores):
    """Estima la probabilidad de lesión grave de cada aviso y dice cuál atender primero."""
    n = len(ORDEN_FORMULARIO)
    if n_clicks == 0:
        # Al cargar la página todavía no hay nada que comparar
        return html.P("Pulsa «Comparar avisos» para ver qué accidente atender primero.",
                      style={"textAlign": "center", "color": "#6b7280"})
    if None in valores:
        return html.P("Completa todos los campos de los dos avisos para compararlos.",
                      style={"textAlign": "center", "color": "#6b7280"})

    # Los 8 primeros valores son del aviso A y los 8 siguientes del B
    aviso_a = dict(zip(ORDEN_FORMULARIO, valores[:n]))
    aviso_b = dict(zip(ORDEN_FORMULARIO, valores[n:]))
    prob_a = probabilidad_persona(modelo, codigos, aviso_a)
    prob_b = probabilidad_persona(modelo, codigos, aviso_b)

    # Decisión: el de mayor probabilidad. Si son del mismo nivel y casi iguales, empate.
    mismo_nivel = nivel_prioridad(prob_a) == nivel_prioridad(prob_b)
    parecidas = max(prob_a, prob_b) < 1.25 * min(prob_a, prob_b)
    if mismo_nivel and parecidas:
        titulo, color = "Prioridad similar: decide el criterio del operador", AZUL
        borde_a = borde_b = AZUL
    elif prob_a > prob_b:
        titulo, color = "Atender primero: Accidente A", COLOR_NIVEL[nivel_prioridad(prob_a)]
        borde_a, borde_b = color, "#d0d5db"
    else:
        titulo, color = "Atender primero: Accidente B", COLOR_NIVEL[nivel_prioridad(prob_b)]
        borde_a, borde_b = "#d0d5db", color

    return [
        html.Div(
            titulo,
            style={"backgroundColor": color, "color": "white", "borderRadius": "12px",
                   "padding": "18px", "textAlign": "center", "fontSize": "26px",
                   "fontWeight": "700", "marginBottom": "16px"},
        ),
        html.Div(
            children=[tarjeta_resultado("A", prob_a, borde_a),
                      tarjeta_resultado("B", prob_b, borde_b)],
            style=ESTILO_FILA,
        ),
        html.Div(
            children=[dcc.Graph(
                figure=comparacion_probabilidades(
                    prob_a, prob_b, media_graves,
                    COLOR_NIVEL[nivel_prioridad(prob_a)], COLOR_NIVEL[nivel_prioridad(prob_b)],
                ),
                config={"displayModeBar": False},
            )],
            style={**ESTILO_TARJETA, "marginTop": "16px"},
        ),
    ]


@app.callback(
    Output("carrusel_tira", "style"),
    Input("carrusel_reloj", "n_intervals"),
)
def avanzar_carrusel(n_intervals):
    """Cada 5 segundos mueve la tira una foto hacia la derecha."""
    return estilo_tira(n_intervals or 0)


@app.callback(
    Output("analisis", "style"),
    Output("boton_analisis", "children"),
    Input("boton_analisis", "n_clicks"),
)
def mostrar_analisis(n_clicks):
    """Muestra u oculta los datos detrás del modelo cada vez que se pulsa el botón."""
    if n_clicks % 2 == 1:
        return {"display": "block"}, "Ocultar los datos"
    return {"display": "none"}, "Ver los datos detrás del modelo"


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


if __name__ == "__main__":
    app.run(debug=True)

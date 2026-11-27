"""Gráficos de MADRisk.

Cada función recibe un DataFrame y devuelve una figura de Plotly (go.Figure).
Este fichero no sabe nada de Dash: así se puede probar y reutilizar por separado.

Regla importante: cada fila del DataFrame es una PERSONA implicada, y un accidente
(num_expediente) se repite en varias filas. Por eso:
  - los gráficos "de accidentes" se quedan antes con una fila por accidente;
  - los gráficos "de personas" cuentan filas.
Cada título indica qué se está contando.
"""

from __future__ import annotations

import textwrap

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Nombre de cada mes y de cada día de la semana (0 = lunes) para los ejes
MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}
DIAS = {
    0: "Lunes", 1: "Martes", 2: "Miércoles", 3: "Jueves",
    4: "Viernes", 5: "Sábado", 6: "Domingo",
}


# Paleta: azules (la cabecera de la web) y los amarillos, naranjas y rojos de los
# chalecos y vehículos del SAMUR de las fotos del carrusel.
COLOR = "#1f4e79"
AZUL_CLARO = "#9dc3e6"
AZUL_OSCURO = "#12355b"
AMARILLO = "#f2b705"
NARANJA = "#e8710a"
ROJO = "#b03a2e"
ESCALA_AZULES = [AZUL_CLARO, COLOR, AZUL_OSCURO]  # de menos a más
ESCALA_CALIDA = [AMARILLO, NARANJA, ROJO]  # de menos grave a más grave


def mezclar(color_a: str, color_b: str, t: float) -> str:
    """Color intermedio entre dos colores hexadecimales (t = 0 -> a, t = 1 -> b)."""
    a = [int(color_a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(color_b[i:i + 2], 16) for i in (1, 3, 5)]
    mezcla = [round(x + (y - x) * t) for x, y in zip(a, b)]
    return "#" + "".join(f"{c:02x}" for c in mezcla)


def colores_por_valor(valores: list, escala: list) -> list:
    """Un color por valor: el más bajo usa el primer color de la escala y el más alto el último."""
    minimo, maximo = min(valores), max(valores)
    colores = []
    for valor in valores:
        t = 0.5 if maximo == minimo else (valor - minimo) / (maximo - minimo)
        # Con 3 colores, el primer tramo va del 1.º al 2.º y el segundo del 2.º al 3.º
        tramo = min(int(t * (len(escala) - 1)), len(escala) - 2)
        t_tramo = t * (len(escala) - 1) - tramo
        colores.append(mezclar(escala[tramo], escala[tramo + 1], t_tramo))
    return colores


# Caracteres por línea del título: así cabe en una pantalla de móvil sin cortarse
ANCHO_TITULO = 38


def partir_titulo(texto: str, ancho: int = ANCHO_TITULO) -> tuple[str, int]:
    """Parte un título largo en líneas con <br> y devuelve (texto, número de líneas)."""
    lineas = textwrap.wrap(texto, ancho) or [texto]
    return "<br>".join(lineas), len(lineas)


def estilo(fig: go.Figure) -> go.Figure:
    """Aplica el mismo aspecto a todos los gráficos: fondo blanco, letra y márgenes.

    Los títulos largos se parten en varias líneas y los márgenes de los ejes se ajustan
    solos (automargin), para que los gráficos se lean bien también en pantallas pequeñas.
    """
    titulo = fig.layout.title.text
    lineas = 1
    if titulo:
        texto, lineas = partir_titulo(titulo)
        fig.update_layout(title_text=texto)
    fig.update_layout(
        template="plotly_white",
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#2b2b2b"},
        margin={"l": 20, "r": 30, "t": 30 + 24 * lineas, "b": 40},
        title_font_size=15,
    )
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    return fig


def figura_vacia(titulo: str) -> go.Figure:
    """Figura con un mensaje, para cuando los filtros no dejan ningún dato."""
    fig = go.Figure()
    fig.update_layout(
        title=titulo,
        annotations=[{
            "text": "Sin datos con estos filtros",
            "showarrow": False,
            "font": {"size": 16},
        }],
    )
    return estilo(fig)


def un_registro_por_accidente(df: pd.DataFrame) -> pd.DataFrame:
    """Deja una fila por accidente (num_expediente) para poder contar accidentes."""
    return df.drop_duplicates(subset="num_expediente")


def accidentes_por_mes(df: pd.DataFrame, titulo: str) -> go.Figure:
    """Barras con el número de ACCIDENTES que hay en cada mes."""
    if len(df) == 0:
        return figura_vacia(titulo)
    accidentes = un_registro_por_accidente(df)

    conteo = accidentes.groupby("mes").size().reset_index(name="accidentes")
    conteo["mes_nombre"] = conteo["mes"].map(MESES)

    fig = px.bar(
        conteo, x="mes_nombre", y="accidentes", title=titulo,
        labels={"mes_nombre": "Mes", "accidentes": "Número de accidentes"},
    )
    fig.update_traces(marker_color=colores_por_valor(conteo["accidentes"].tolist(), ESCALA_AZULES))
    return estilo(fig)


def accidentes_por_dia_semana(df: pd.DataFrame, titulo: str) -> go.Figure:
    """Barras con el número de ACCIDENTES por día de la semana."""
    if len(df) == 0:
        return figura_vacia(titulo)
    accidentes = un_registro_por_accidente(df)

    conteo = accidentes.groupby("dia_semana").size().reset_index(name="accidentes")
    conteo["dia_nombre"] = conteo["dia_semana"].map(DIAS)

    fig = px.bar(
        conteo, x="dia_nombre", y="accidentes", title=titulo,
        labels={"dia_nombre": "Día de la semana", "accidentes": "Número de accidentes"},
    )
    fig.update_traces(marker_color=colores_por_valor(conteo["accidentes"].tolist(), ESCALA_AZULES))
    return estilo(fig)


def accidentes_por_hora(df: pd.DataFrame, titulo: str) -> go.Figure:
    """Línea con el número de ACCIDENTES por hora del día (0-23)."""
    if len(df) == 0:
        return figura_vacia(titulo)
    accidentes = un_registro_por_accidente(df)

    conteo = accidentes.groupby("hora_num").size().reset_index(name="accidentes")

    fig = px.line(
        conteo, x="hora_num", y="accidentes", markers=True, title=titulo,
        labels={"hora_num": "Hora del día", "accidentes": "Número de accidentes"},
    )
    fig.update_traces(
        line_color=COLOR,
        marker={"color": AMARILLO, "size": 10, "line": {"color": COLOR, "width": 1.5}},
    )
    return estilo(fig)


def accidentes_por_categoria(
    df: pd.DataFrame, columna: str, titulo: str, etiqueta: str, maximo: int = 0
) -> go.Figure:
    """Barras horizontales con el número de ACCIDENTES de cada valor de `columna`.

    `maximo`: si es mayor que 0, solo se dibujan las `maximo` categorías más frecuentes.
    """
    if len(df) == 0:
        return figura_vacia(titulo)
    accidentes = un_registro_por_accidente(df)
    return _barras_horizontales(accidentes, columna, titulo, etiqueta,
                                "Número de accidentes", maximo)


def personas_por_categoria(
    df: pd.DataFrame, columna: str, titulo: str, etiqueta: str, maximo: int = 0
) -> go.Figure:
    """Barras horizontales con el número de PERSONAS (filas) de cada valor de `columna`."""
    if len(df) == 0:
        return figura_vacia(titulo)
    return _barras_horizontales(df, columna, titulo, etiqueta,
                                "Número de personas", maximo)


def _barras_horizontales(
    df: pd.DataFrame, columna: str, titulo: str, etiqueta: str, eje_x: str, maximo: int
) -> go.Figure:
    """Cuenta filas por categoría y dibuja las barras de mayor a menor (arriba la mayor)."""
    conteo = df.groupby(columna).size().reset_index(name="total")
    conteo = conteo.sort_values("total", ascending=False)
    if maximo > 0:
        conteo = conteo.head(maximo)
    # En un gráfico horizontal la primera barra se dibuja abajo: invertimos el orden
    conteo = conteo.sort_values("total", ascending=True)

    fig = px.bar(
        conteo, x="total", y=columna, orientation="h", title=titulo,
        labels={"total": eje_x, columna: etiqueta},
    )
    fig.update_traces(marker_color=colores_por_valor(conteo["total"].tolist(), ESCALA_AZULES))
    # Sin título en el eje vertical: ya lo dice el título del gráfico y así las barras
    # ganan ancho en pantallas pequeñas
    fig.update_yaxes(title=None)
    return estilo(fig)


def mapa_calor_dia_hora(df: pd.DataFrame, titulo: str) -> go.Figure:
    """Heatmap de ACCIDENTES: día de la semana (filas) por hora del día (columnas)."""
    if len(df) == 0:
        return figura_vacia(titulo)
    accidentes = un_registro_por_accidente(df)

    # Tabla con una fila por día y una columna por hora; cada celda cuenta accidentes
    tabla = accidentes.pivot_table(
        index="dia_semana", columns="hora_num", values="num_expediente",
        aggfunc="count", fill_value=0,
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=tabla.values,
            x=tabla.columns.tolist(),
            y=[DIAS[d] for d in tabla.index],
            colorscale=[[0, "#fff6d9"], [0.4, AMARILLO], [0.75, NARANJA], [1, "#8c2f23"]],
            colorbar={"title": "Accidentes"},
        )
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="Hora del día", yaxis_title="Día de la semana",
        yaxis={"autorange": "reversed"},  # lunes arriba
    )
    return estilo(fig)


def comparacion_probabilidades(
    prob_a: float, prob_b: float, media: float, color_a: str, color_b: str
) -> go.Figure:
    """Barras horizontales con la probabilidad (en %) de los dos avisos y la media general.

    `prob_a` y `prob_b` van de 0 a 1; `media` ya está en %. Los colores de A y B son los
    de su nivel de prioridad, para que el gráfico se lea igual que las tarjetas.
    """
    valores = [prob_a * 100, prob_b * 100, media]
    nombres = ["Accidente A", "Accidente B", "Media general"]
    fig = go.Figure(
        data=go.Bar(
            x=valores, y=nombres, orientation="h",
            marker_color=[color_a, color_b, "#9aa5b1"],
            text=[f"{v:.1f} %".replace(".", ",") for v in valores],
            textposition="outside", cliponaxis=False,
        )
    )
    fig.update_layout(
        title="Probabilidad de lesión grave de cada aviso frente a la media",
        xaxis_title="Probabilidad de lesión grave (%)",
        # Margen a la derecha para que quepa el texto de la barra más larga
        xaxis_range=[0, max(valores) * 1.3],
        yaxis={"autorange": "reversed"},  # A arriba, la media abajo
        height=260,
    )
    return estilo(fig)


def grupo_mas_grave(df: pd.DataFrame, columna: str, minimo: int = 100) -> dict | None:
    """Valor de `columna` con mayor % de personas graves, o None si no hay datos.

    Igual que en `tasa_gravedad`, solo cuentan las personas con lesividad informada y
    se ignoran los grupos con menos de `minimo` personas. Devuelve el valor, su tasa
    (en %), cuántas personas hay detrás y cuántas veces supera la media.
    """
    informados = df[df["grave"].notna()]
    if len(informados) == 0:
        return None
    tabla = informados.groupby(columna)["grave"].agg(["count", "mean"]).reset_index()
    tabla = tabla[tabla["count"] >= minimo]
    if len(tabla) == 0:
        return None
    mejor = tabla.sort_values("mean", ascending=False).iloc[0]
    media = informados["grave"].mean()
    return {
        "valor": mejor[columna],
        "tasa": float(mejor["mean"] * 100),
        "personas": int(mejor["count"]),
        "veces": float(mejor["mean"] / media) if media > 0 else 0.0,
    }


def tasa_gravedad(
    df: pd.DataFrame,
    columna: str,
    titulo: str,
    etiqueta: str,
    nombres: dict | None = None,
    ordenar_por_tasa: bool = True,
    minimo: int = 100,
) -> go.Figure:
    """Porcentaje de PERSONAS con lesión grave según los valores de `columna`.

    Solo cuentan las personas con lesividad informada (grave = 0 o 1); las que no
    tienen dato quedan fuera porque no sabemos si fue grave.
    `nombres`: texto a mostrar para cada valor (p. ej. 0 -> "Lunes").
    `ordenar_por_tasa`: True dibuja barras horizontales de mayor a menor tasa;
    False mantiene el orden natural de la columna (horas, días, meses...).
    `minimo`: se ocultan las categorías con menos personas que este número,
    porque un porcentaje calculado con muy pocas personas no es fiable.
    """
    informados = df[df["grave"].notna()]
    if len(informados) == 0:
        return figura_vacia(titulo)

    # Por cada valor: personas con lesividad informada y cuántas fueron graves
    tabla = informados.groupby(columna)["grave"].agg(["count", "sum"]).reset_index()
    tabla.columns = [columna, "personas", "graves"]
    tabla = tabla[tabla["personas"] >= minimo].copy()
    if len(tabla) == 0:
        return figura_vacia(titulo)
    tabla["tasa"] = tabla["graves"] / tabla["personas"] * 100

    # Texto del eje: el nombre si lo tenemos, y si no el propio valor
    if nombres:
        tabla["categoria"] = tabla[columna].map(nombres)
    else:
        tabla["categoria"] = tabla[columna].astype(str)

    # Tasa media de los datos mostrados, como referencia en el título
    media = informados["grave"].mean() * 100
    titulo_completo = f"{titulo} · media {media:.1f}".replace(".", ",") + " %"
    etiquetas = {"tasa": "% de personas con lesión grave", "categoria": etiqueta,
                 "personas": "Personas (lesividad informada)", "graves": "Personas graves"}

    if ordenar_por_tasa:
        tabla = tabla.sort_values("tasa", ascending=True)  # la mayor queda arriba
        fig = px.bar(
            tabla, x="tasa", y="categoria", orientation="h", title=titulo_completo,
            labels=etiquetas, hover_data=["personas", "graves"],
        )
        fig.update_yaxes(title=None)  # mismo motivo que en _barras_horizontales
    else:
        fig = px.bar(
            tabla, x="categoria", y="tasa", title=titulo_completo,
            labels=etiquetas, hover_data=["personas", "graves"],
        )
    fig.update_traces(marker_color=colores_por_valor(tabla["tasa"].tolist(), ESCALA_CALIDA))
    return estilo(fig)


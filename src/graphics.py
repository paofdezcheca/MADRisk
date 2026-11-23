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


# Color principal de los gráficos (el mismo azul que la cabecera de la web)
COLOR = "#1f4e79"


def estilo(fig: go.Figure) -> go.Figure:
    """Aplica el mismo aspecto a todos los gráficos: fondo blanco, letra y márgenes."""
    fig.update_layout(
        template="plotly_white",
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#2b2b2b"},
        margin={"l": 50, "r": 20, "t": 60, "b": 50},
        title_font_size=16,
    )
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
    fig.update_traces(marker_color=COLOR)
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
    fig.update_traces(marker_color=COLOR)
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
    fig.update_traces(marker_color=COLOR, line_color=COLOR)
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
    fig.update_traces(marker_color=COLOR)
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
            colorscale="Blues",
            colorbar={"title": "Accidentes"},
        )
    )
    fig.update_layout(
        title=titulo,
        xaxis_title="Hora del día", yaxis_title="Día de la semana",
        yaxis={"autorange": "reversed"},  # lunes arriba
    )
    return estilo(fig)


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
    titulo_completo = f"{titulo} · media {media:.1f} %"
    etiquetas = {"tasa": "% de personas con lesión grave", "categoria": etiqueta,
                 "personas": "Personas (lesividad informada)", "graves": "Personas graves"}

    if ordenar_por_tasa:
        tabla = tabla.sort_values("tasa", ascending=True)  # la mayor queda arriba
        fig = px.bar(
            tabla, x="tasa", y="categoria", orientation="h", title=titulo_completo,
            labels=etiquetas, hover_data=["personas", "graves"],
        )
    else:
        fig = px.bar(
            tabla, x="categoria", y="tasa", title=titulo_completo,
            labels=etiquetas, hover_data=["personas", "graves"],
        )
    fig.update_traces(marker_color=COLOR)
    return estilo(fig)


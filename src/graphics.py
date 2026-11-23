"""Gráficos de MADRisk.

Cada función recibe un DataFrame y devuelve una figura de Plotly (go.Figure).
Este fichero no sabe nada de Dash: así se puede probar y reutilizar por separado.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Nombre de cada mes para el eje X
MESES = {
    1: "Ene", 2: "Feb", 3: "Mar", 4: "Abr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Ago", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dic",
}


def accidentes_por_mes(df: pd.DataFrame, titulo: str) -> go.Figure:
    """Barras con el número de ACCIDENTES (no de personas) que hay en cada mes."""
    # Cada fila es una persona implicada y un accidente (num_expediente) se repite
    # en varias filas. Nos quedamos con una fila por accidente antes de contar.
    accidentes = df.drop_duplicates(subset="num_expediente")

    # Contamos accidentes por mes
    conteo = accidentes.groupby("mes").size().reset_index(name="accidentes")
    conteo["mes_nombre"] = conteo["mes"].map(MESES)

    fig = px.bar(
        conteo,
        x="mes_nombre",
        y="accidentes",
        title=titulo,
        labels={"mes_nombre": "Mes", "accidentes": "Número de accidentes"},
    )
    fig.update_layout(template="plotly_white")
    return fig

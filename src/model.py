"""Modelo de MADRisk: entrenamiento, evaluación y predicción.

Qué predice: DADO UN ACCIDENTE YA OCURRIDO y las características de una persona
implicada, la probabilidad de que esa persona sufra una lesión grave.
NO predice si va a ocurrir un accidente (no conocemos los desplazamientos sin accidente).

Cada fila de los datos es una persona, y `grave` indica si esa persona tuvo una
lesión grave (ingreso de más de 24 horas o fallecimiento).
"""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Variables que conocemos cuando el accidente ya ha ocurrido.
# NO se usa `lesividad` ni `cod_lesividad`: contienen la misma información que el objetivo.
VARIABLES_NUMERICAS = ["hora_num", "dia_semana", "mes"]
VARIABLES_TEXTO = [
    "distrito", "tipo_accidente", "estado_meteorologico", "tipo_vehiculo",
    "tipo_persona", "rango_edad", "sexo", "positiva_alcohol",
]
VARIABLES = VARIABLES_NUMERICAS + VARIABLES_TEXTO


def crear_codigos(df: pd.DataFrame) -> dict:
    """Para cada variable de texto, un diccionario valor -> número (0, 1, 2...).

    El modelo solo entiende números. Se hace con un diccionario y `.map()`, como el
    curso hace con el sexo. Hay que guardar estos diccionarios junto al modelo para
    codificar igual los datos nuevos en la app.
    """
    codigos = {}
    for columna in VARIABLES_TEXTO:
        valores = sorted(df[columna].unique().tolist())
        codigos[columna] = {valor: i for i, valor in enumerate(valores)}
    return codigos


def codificar(df: pd.DataFrame, codigos: dict) -> pd.DataFrame:
    """Devuelve solo las variables del modelo, con las de texto pasadas a número."""
    X = df[VARIABLES].copy()
    for columna in VARIABLES_TEXTO:
        X[columna] = X[columna].map(codigos[columna])
    return X


def preparar_datos(df: pd.DataFrame, codigos: dict):
    """Deja solo las personas con lesividad informada y separa X (variables) e y (grave)."""
    # Las filas sin dato de lesividad NO son "no grave": se dejan fuera del entrenamiento
    informados = df[df["grave"].notna()]
    X = codificar(informados, codigos)
    y = informados["grave"].astype(int)
    return X, y


def dividir(X: pd.DataFrame, y: pd.Series):
    """70 % entrenamiento / 30 % prueba, manteniendo la proporción de graves en ambos."""
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def entrenar_regresion_logistica(X_train, y_train) -> LogisticRegression:
    """Modelo base: regresión logística."""
    modelo = LogisticRegression(max_iter=1000)
    modelo.fit(X_train, y_train)
    return modelo


def entrenar_random_forest(X_train, y_train) -> RandomForestClassifier:
    """Random Forest con tamaño limitado para que `model.joblib` pese poco en Render.

    Con 50 árboles de profundidad 8 el fichero pesa ~1,4 MB y rinde igual que con
    100 árboles de profundidad 12 (13,8 MB): lo comprobamos antes de elegirlo.
    """
    modelo = RandomForestClassifier(n_estimators=50, max_depth=8, random_state=42)
    modelo.fit(X_train, y_train)
    return modelo


def evaluar(modelo, X_test, y_test, nombre: str) -> None:
    """Imprime accuracy, informe por clase y matriz de confusión.

    La accuracy sola no sirve: con un 2 % de graves, decir siempre "no grave" ya
    acierta el 98 %. Hay que mirar el recall y el F1 de la clase grave.
    """
    predicciones = modelo.predict(X_test)
    print(f"=== {nombre} ===")
    print("Accuracy:", round(accuracy_score(y_test, predicciones), 4))
    print(classification_report(
        y_test, predicciones, target_names=["no grave", "grave"], zero_division=0
    ))
    print("Matriz de confusión (filas = real, columnas = predicho; [no grave, grave]):")
    print(confusion_matrix(y_test, predicciones))
    print()


def probabilidad_grave(modelo, X: pd.DataFrame) -> float:
    """Probabilidad (0-1) de que la persona tenga una lesión grave.

    Usa `predict_proba`, método de los modelos de scikit-learn que NO aparece en el
    curso: devuelve la probabilidad de cada clase [no grave, grave]. Nos quedamos con
    la columna 1 (grave).
    """
    return float(modelo.predict_proba(X)[0][1])


def evaluar_umbrales(modelo, X_test, y_test, umbrales) -> None:
    """Recall, precisión y F1 de la clase grave según el umbral de probabilidad.

    Con un 2 % de graves, el modelo casi nunca da más de 0,5 de probabilidad, así que
    el umbral por defecto (0,5) no detecta ningún grave. Bajando el umbral se detectan
    más graves (sube el recall) a costa de más falsas alarmas (baja la precisión).
    Usa `predict_proba`, que no aparece en el curso (ver `probabilidad_grave`).
    """
    probabilidades = modelo.predict_proba(X_test)[:, 1]
    graves = int((y_test == 1).sum())
    print("Clase grave según el umbral de probabilidad:")
    for umbral in umbrales:
        marcados = probabilidades >= umbral
        aciertos = int((marcados & (y_test == 1)).sum())
        recall = aciertos / graves
        precision = aciertos / max(int(marcados.sum()), 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        print(
            f"  umbral {umbral:.2f}: recall {recall:.2f} | precisión {precision:.2f} "
            f"| F1 {f1:.2f} | personas marcadas como graves: {int(marcados.sum())}"
        )
    print()

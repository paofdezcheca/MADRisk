"""Entrena los modelos, los evalúa y guarda el Random Forest en model.joblib.

Se ejecuta a mano: python train.py. La app NUNCA entrena, solo carga el fichero.
"""

import joblib
import pandas as pd

from src.etl import cargar_datos
from src.model import (
    VARIABLES,
    crear_codigos,
    dividir,
    entrenar_random_forest,
    entrenar_regresion_logistica,
    evaluar,
    evaluar_umbrales,
    preparar_datos,
)

datos = cargar_datos()

codigos = crear_codigos(datos)
X, y = preparar_datos(datos, codigos)
print(f"\nPersonas para el modelo: {len(X)} (graves: {int(y.sum())}, {100 * y.mean():.1f} %)\n")

X_train, X_test, y_train, y_test = dividir(X, y)
print(f"Entrenamiento: {len(X_train)} filas | Prueba: {len(X_test)} filas\n")

regresion = entrenar_regresion_logistica(X_train, y_train)
evaluar(regresion, X_test, y_test, "Regresión logística")

bosque = entrenar_random_forest(X_train, y_train)
evaluar(bosque, X_test, y_test, "Random Forest")
evaluar_umbrales(bosque, X_test, y_test, [0.03, 0.05, 0.08, 0.10, 0.12])

# Qué variables pesan más en el Random Forest
importancias = pd.Series(bosque.feature_importances_, index=VARIABLES)
print("Importancia de las variables (Random Forest):")
print(importancias.sort_values(ascending=False).round(3))

# Guardamos el modelo junto a los diccionarios de codificación
joblib.dump({"modelo": bosque, "codigos": codigos}, "model.joblib")
print("\nModelo guardado en model.joblib")

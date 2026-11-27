# MADRisk

### Análisis interactivo y predicción de la gravedad de los accidentes de tráfico en Madrid

**App desplegada:** https://madrisk.onrender.com/

## Descripción

**MADRisk** es una aplicación interactiva para el análisis de los accidentes de tráfico registrados en la ciudad de Madrid.

A partir de los datos abiertos publicados por el Ayuntamiento de Madrid, la aplicación permite explorar la accidentalidad desde una perspectiva temporal, geográfica y contextual, identificando patrones relacionados con factores como la localización, la hora, el tipo de vehículo, el tipo de accidente o las condiciones meteorológicas.

El proyecto busca ir más allá de la visualización de estadísticas históricas. Además de permitir al usuario explorar los accidentes registrados, MADRisk analiza qué factores están asociados a una mayor lesividad y incluye un modelo predictivo que estima la gravedad de un accidente a partir de sus características.

## Objetivos

Los principales objetivos del proyecto son:

- Integrar y procesar los diferentes conjuntos de datos anuales de accidentes de tráfico de Madrid.
- Analizar la evolución temporal y la distribución geográfica de la accidentalidad.
- Identificar patrones y circunstancias asociados a una mayor gravedad de los accidentes.
- Crear visualizaciones interactivas que permitan explorar y comparar diferentes escenarios.
- Desarrollar y evaluar un modelo predictivo de gravedad de los accidentes.
- Construir una aplicación web interactiva utilizando Dash y Plotly.
- Desplegar la aplicación en una URL pública.

## Funcionalidades

MADRisk se estructura en tres áreas principales:

### 1. Exploración de la accidentalidad

El usuario puede analizar los accidentes mediante diferentes filtros, como año, mes, día de la semana, hora, distrito, tipo de accidente, tipo de vehículo, condiciones meteorológicas o nivel de lesividad.

Los resultados se muestran mediante indicadores y gráficos interactivos (barras, líneas y mapas de calor) que permiten identificar patrones temporales y por distrito. Cada gráfico indica si cuenta **accidentes** o **personas**: cada fila de los datos es una persona implicada y un mismo accidente puede implicar a varias.

### 2. Análisis de los factores asociados a la gravedad

La aplicación permite estudiar qué características y circunstancias están relacionadas con una mayor lesividad, mostrando el porcentaje de personas con lesión grave según el factor elegido (hora, distrito, tipo de accidente, tipo de vehículo, edad, etc.).

Se considera **lesión grave** la de una persona hospitalizada más de 24 horas o fallecida a las 24 horas (`Ingreso superior a 24 horas` y `Fallecido 24 horas`), que es el criterio oficial de la **DGT** (Dirección General de Tráfico) para definir a un herido grave. Las personas con lesividad vacía o "Se desconoce" (alrededor del 46 % de las filas) no se cuentan como "sin lesión": quedan fuera de los porcentajes y del entrenamiento del modelo.

Se analizan factores como el momento y la localización del accidente, el tipo de vehículo, el tipo de accidente, las condiciones meteorológicas y otras variables disponibles en los datos.

### 3. Predicción de la gravedad

La web está pensada como ayuda para los servicios de emergencia: **compara dos avisos de accidente y propone atender primero el de mayor probabilidad de lesión grave**. Un modelo de clasificación supervisada (Random Forest) estima esa probabilidad para una persona implicada en un accidente que ya ha ocurrido, usando solo datos que se conocen al recibir el aviso: tipo de persona, vehículo, tipo de accidente, edad, distrito, hora, día y meteorología. Según la probabilidad se asigna una prioridad alta (≥ 8 %), media (3-8 %) o baja (< 3 %). En los datos de prueba, el 16,1 % de las personas con prioridad alta sufrió una lesión grave, frente al 5,6 % en la media y el 0,5 % en la baja.

El objetivo no es predecir la probabilidad de que ocurra un accidente, sino estimar su gravedad **una vez ocurrido**.

Resultados (sobre el 30 % de datos de prueba):

- Solo el 2,2 % de las personas con lesividad informada tiene una lesión grave, así que la *accuracy* no es una buena medida (decir siempre "no grave" acierta el 97,8 %).
- Con el umbral habitual de 0,5 el modelo no detecta ningún grave. Por eso la app muestra la **probabilidad** y la compara con la media (2,2 %), en lugar de dar un sí o un no.
- El modelo ordena bien el riesgo: el 10 % de personas con mayor probabilidad contiene un 12,3 % de graves frente al 2,2 % de base. Con un umbral de 0,08 el F1 de la clase grave es 0,22 (recall 0,33, precisión 0,16). Es una estimación orientativa, no precisa para un caso individual.

## Fuente de datos

Los datos utilizados proceden del **Portal de Datos Abiertos del Ayuntamiento de Madrid**.

Se utilizan los conjuntos de datos históricos de accidentes de tráfico publicados anualmente, que contienen información sobre:

- Fecha y hora.
- Localización y distrito.
- Tipo de accidente.
- Tipo de vehículo.
- Tipo de persona implicada.
- Edad y sexo.
- Condiciones meteorológicas.
- Nivel de lesividad.
- Alcohol y drogas.
- Coordenadas geográficas (disponibles en los datos, pero la app no las usa).

Los diferentes ficheros anuales se integran y procesan mediante Python para construir una única base de datos destinada al análisis y al desarrollo del modelo predictivo.

## Fases del proyecto

### Fase 1 — Preparación de los datos
Recopilación e integración de los conjuntos de datos anuales. Limpieza, transformación de variables y tratamiento de valores ausentes e inconsistencias.

### Fase 2 — Análisis exploratorio
Análisis de la evolución y distribución de la accidentalidad e identificación de patrones temporales, geográficos y relacionados con las características de los accidentes.

### Fase 3 — Modelo predictivo
Definición de la variable objetivo, selección y creación de variables explicativas, entrenamiento de modelos de clasificación y evaluación de su capacidad predictiva.

### Fase 4 — Desarrollo de la aplicación
Creación de visualizaciones interactivas mediante Plotly y desarrollo del dashboard utilizando Dash.

### Fase 5 — Despliegue y mejora
Despliegue de la aplicación en una URL pública, pruebas de funcionamiento y mejora progresiva de la interfaz y las funcionalidades.

## Cómo ejecutarlo en local

```bash
python -m venv .venv
.venv\Scripts\activate        # en Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Después, abre http://127.0.0.1:8050 en el navegador.

### Tests

Comprueban el ETL y el modelo con datos inventados (no hace falta instalar nada más):

```bash
python tests/test_etl.py
python tests/test_model.py
```

## Estructura del proyecto

- `data/raw/`: CSV anuales 2020-2025 del Ayuntamiento de Madrid.
- `src/etl.py`: carga, limpieza y variables derivadas.
- `src/graphics.py`: funciones que reciben datos y devuelven figuras de Plotly.
- `src/model.py`: codificación, entrenamiento, evaluación y predicción.
- `train.py`: entrena el modelo y guarda `model.joblib` (`python train.py`).
- `tests/`: pruebas del ETL y del modelo.
- `app.py`: layout y callbacks de Dash; solo carga el modelo, nunca lo entrena.
- `model.joblib`: Random Forest entrenado (50 árboles, profundidad 8, ~1,4 MB).
- `assets/carrusel/`: fotos del carrusel de la cabecera.

## Créditos de las fotos

Las fotos del carrusel son del **Diario de Madrid**, publicadas en Wikimedia Commons con licencia [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/):

- [Accidente de moto en la calle Antonio Machado](https://commons.wikimedia.org/wiki/File:Accidente_de_Moto_en_la_calle_Antonio-Machado-29.03.08.jpg)
- [Intervención del SAMUR en un accidente de tráfico](https://commons.wikimedia.org/wiki/File:Intervenci%C3%B3n_del_SAMUR_en_un_accidente_de_Tr%C3%A1fico.jpg)
- [Intervención del SAMUR en un accidente de moto en la M30](https://commons.wikimedia.org/wiki/File:Intervenci%C3%B3n_del_SAMUR_en_un_accidente_de_moto_en_la_M30.jpg)
- [Intervención del SAMUR en un accidente de moto](https://commons.wikimedia.org/wiki/File:Intervenci%C3%B3n_del_SAMUR_en_un_accidente_de_moto.jpg)
- [Intervención del SAMUR en un atrapado en un accidente de tráfico](https://commons.wikimedia.org/wiki/File:Intervenci%C3%B3n_del_SAMUR_en_un_atrapado_en_una_accidente_de_tr%C3%A1fico.jpg)

## Tecnologías

- Python
- Pandas
- Scikit-learn
- Plotly
- Dash
- Git y GitHub
- Render

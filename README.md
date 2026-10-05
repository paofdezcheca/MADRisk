# MADRisk

### Análisis interactivo y predicción de la gravedad de los accidentes de tráfico en Madrid

## Descripción

**MADRisk** es una aplicación interactiva para el análisis de los accidentes de tráfico registrados en la ciudad de Madrid.

A partir de los datos abiertos publicados por el Ayuntamiento de Madrid, la aplicación permitirá explorar la accidentalidad desde una perspectiva temporal, geográfica y contextual, identificando patrones relacionados con factores como la localización, la hora, el tipo de vehículo, el tipo de accidente o las condiciones meteorológicas.

El proyecto busca ir más allá de la visualización de estadísticas históricas. Además de permitir al usuario explorar los accidentes registrados, MADRisk analizará qué factores están asociados a una mayor lesividad y desarrollará un modelo predictivo capaz de estimar la gravedad de un accidente a partir de sus características.

## Objetivos

Los principales objetivos del proyecto son:

- Integrar y procesar los diferentes conjuntos de datos anuales de accidentes de tráfico de Madrid.
- Analizar la evolución temporal y la distribución geográfica de la accidentalidad.
- Identificar patrones y circunstancias asociados a una mayor gravedad de los accidentes.
- Crear visualizaciones interactivas que permitan explorar y comparar diferentes escenarios.
- Desarrollar y evaluar un modelo predictivo de gravedad de los accidentes.
- Construir una aplicación web interactiva utilizando Dash y Plotly.
- Desplegar la aplicación en una URL pública.

## Funcionalidades previstas

MADRisk se estructurará en tres áreas principales:

### 1. Exploración de la accidentalidad

El usuario podrá analizar los accidentes mediante diferentes filtros, como año, mes, día de la semana, hora, distrito, tipo de accidente, tipo de vehículo, condiciones meteorológicas o nivel de lesividad.

Los resultados se mostrarán mediante mapas, indicadores y gráficos interactivos que permitirán identificar patrones temporales y geográficos.

### 2. Análisis de los factores asociados a la gravedad

La aplicación permitirá estudiar qué características y circunstancias están relacionadas con una mayor lesividad.

Se analizarán factores como el momento y la localización del accidente, el tipo de vehículo, el tipo de accidente, las condiciones meteorológicas y otras variables disponibles en los datos.

### 3. Predicción de la gravedad

Se desarrollará un modelo de clasificación supervisada que permita estimar la gravedad de un accidente a partir de sus características.

El objetivo no será predecir la probabilidad de que ocurra un accidente, sino estimar su posible nivel de gravedad una vez conocidas determinadas circunstancias del mismo.

## Fuente de datos

Los datos utilizados proceden del **Portal de Datos Abiertos del Ayuntamiento de Madrid**.

Se utilizarán los conjuntos de datos históricos de accidentes de tráfico publicados anualmente, que contienen información sobre:

- Fecha y hora.
- Localización y distrito.
- Tipo de accidente.
- Tipo de vehículo.
- Tipo de persona implicada.
- Edad y sexo.
- Condiciones meteorológicas.
- Nivel de lesividad.
- Alcohol y drogas.
- Coordenadas geográficas.

Los diferentes ficheros anuales serán integrados y procesados mediante Python para construir una única base de datos destinada al análisis y al desarrollo del modelo predictivo.

## Plan de trabajo inicial

### Fase 1 — Preparación de los datos
Recopilación e integración de los conjuntos de datos anuales. Limpieza, transformación de variables y tratamiento de valores ausentes e inconsistencias.

### Fase 2 — Análisis exploratorio
Análisis de la evolución y distribución de la accidentalidad e identificación de patrones temporales, geográficos y relacionados con las características de los accidentes.

### Fase 3 — Modelo predictivo
Definición de la variable objetivo, selección y creación de variables explicativas, entrenamiento de modelos de clasificación y evaluación de su capacidad predictiva.

### Fase 4 — Desarrollo de la aplicación
Creación de mapas y visualizaciones interactivas mediante Plotly y desarrollo del dashboard utilizando Dash.

### Fase 5 — Despliegue y mejora
Despliegue de la aplicación en una URL pública, pruebas de funcionamiento y mejora progresiva de la interfaz y las funcionalidades.

## Tecnologías

- Python
- Pandas
- Scikit-learn
- Plotly
- Dash
- Git y GitHub
- Render

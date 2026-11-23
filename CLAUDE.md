# CLAUDE.md — MADRisk

Contexto e instrucciones para trabajar en este repositorio. Léelo entero antes de tocar código.

## Proyecto

**MADRisk**: app en Dash para analizar y predecir la **gravedad** de los accidentes de tráfico en Madrid.
Proyecto de la asignatura *Desarrollo de Aplicaciones para la Visualización de Datos* (DAVD, ICAI, curso 2026-27, prof. David Martín-Corral).
Autora: Paola Fernández-Checa (Pao).

- Repo del proyecto: https://github.com/paofdezcheca/MADRisk
- Material del curso (referencia de estilo y herramientas): https://github.com/dmartincc/davd-26-27

### Tres bloques de la app
1. **Exploración**: filtros por año, mes, día de la semana, hora, distrito, tipo de accidente, tipo de vehículo, meteorología y lesividad.
2. **Análisis de gravedad**: qué factores se asocian a lesiones graves y cómo varían por distrito, franja horaria, tipo de accidente o perfil de persona.
3. **Predicción**: *dado un accidente ya ocurrido* y sus características, probabilidad de que sea grave.
   **No** es predecir si va a ocurrir un accidente (no conocemos los desplazamientos sin accidente). Explícalo siempre así en la app y en el código.

## Regla principal: solo lo visto en el curso

Usa únicamente herramientas y patrones que aparecen en el repo del curso. **Si algo no aparece, avisa antes de usarlo y propón la alternativa del curso.**

| Área | Permitido (visto en el curso) |
| --- | --- |
| Datos | `pandas`, `numpy`, `pathlib.Path`, `read_csv`, `to_numeric(errors="coerce")`, `.map()`, `groupby/agg`, `pivot_table`, resumen de nulos por columna |
| Modelo | `train_test_split(..., random_state=42, stratify=y)`, `LogisticRegression`, `RandomForestClassifier`, `classification_report`, `confusion_matrix`, `accuracy_score`, `joblib` |
| Gráficos | `plotly.express` y `plotly.graph_objects`: barras (también apiladas y en relativo), líneas, pie/donut, histogramas, box, `go.Heatmap`, `make_subplots` |
| App | Dash: `html.Div/H1/P/Button`, `dcc.Dropdown`, `dcc.Checklist`, `dcc.Graph`, callbacks con `Input`/`Output`/`State`, estilos CSS en `style={}` |
| Despliegue | Render + gunicorn (`server = app.server`), `Procfile`, `render.yaml`, `runtime.txt`, `requirements.txt` |

**No aparecen en el curso, así que hay que avisar antes de usarlos:**
- Mapas (`scatter_map`, `density_map`). `choropleth` solo sale en un notebook de ejemplo. Alternativa: barras o heatmap por distrito.
- `ROC-AUC` (aunque la propuesta lo menciona). Alternativa: `classification_report` (precision, recall, F1) y `confusion_matrix`.
- `predict_proba` (hace falta para dar una probabilidad). Es un método de los modelos del curso: úsalo, pero explícalo en el comentario.
- `class_weight`, `OneHotEncoder`, `get_dummies`, `Pipeline`, `GridSearchCV`, `dcc.Tabs`, `dcc.Slider`, `dcc.Store`, `RadioItems`, `dash_bootstrap_components`, `PreventUpdate`.
- Cualquier librería que no esté en la tabla (seaborn y matplotlib solo para explorar, nunca dentro de la app).

## Estructura del repo (la que recomienda el curso)

```
MADRisk/
├── data/            # CSV anuales (decidir si se suben o se descargan; ver "Datos")
├── src/
│   ├── __init__.py
│   ├── etl.py       # carga, unión de años, limpieza, variables derivadas
│   ├── model.py     # entrenamiento, evaluación y predicción
│   └── graphics.py  # funciones que reciben un DataFrame y devuelven figuras Plotly
├── train.py         # entrena y guarda model.joblib
├── app.py           # layout y callbacks; carga model.joblib
├── requirements.txt
├── Procfile         # web: gunicorn --timeout 600 app:server
├── render.yaml
├── runtime.txt      # python-3.12.0
├── .gitignore       # plantilla del curso (gitignore_davd.txt)
└── README.md
```

Reglas:
- `app.py` define `app = Dash(__name__)` y `server = app.server` (obligatorio para gunicorn).
- **Nunca se hace `fit` dentro de un callback.** El modelo se entrena en `train.py` y la app solo lo carga con `joblib.load`.
- Los datos se cargan una sola vez al arrancar la app, no en cada callback.
- `graphics.py` no sabe nada de Dash: recibe datos y devuelve `go.Figure`.
- Las dependencias de Render van en `requirements.txt`: dash, plotly, pandas, numpy, scikit-learn, joblib y gunicorn.
- Plan gratuito de Render (512 MB de RAM): cargar solo las columnas necesarias y vigilar el tamaño de `model.joblib` (limitar `n_estimators` y `max_depth` del RandomForest).

## Datos

Fuente: Portal de Datos Abiertos del Ayuntamiento de Madrid, "Accidentes de tráfico de la Ciudad de Madrid". Ficheros anuales 2020-2025 (unos 270.000 registros). **2026 se trata aparte** porque el año está incompleto; no se mezcla con los años completos.

Lectura:
- Separador `;`. Algunos años vienen en UTF-8 con BOM y otros pueden venir en latin-1: probar `encoding="utf-8-sig"` y, si falla, `"latin-1"`.
- Hay un fichero en `.xlsx`; usar su versión CSV si existe.
- Las columnas pueden cambiar un poco entre años (por ejemplo, `estado_meteorológico` lleva tilde). Normalizar los nombres antes de concatenar.
- Hay textos `"NULL"` (por ejemplo en `positiva_droga`) que hay que tratar como nulos.

Columnas: `num_expediente, fecha, hora, localizacion, numero, cod_distrito, distrito, tipo_accidente, estado_meteorológico, tipo_vehiculo, tipo_persona, rango_edad, sexo, cod_lesividad, lesividad, coordenada_x_utm, coordenada_y_utm, positiva_alcohol, positiva_droga`.

**Personas frente a accidentes:**
- Cada fila es una persona o vehículo implicado, y un mismo `num_expediente` se repite. Por ejemplo, en 2021 hay 41.783 filas para 17.952 expedientes.
- Los conteos de accidentes se hacen con `num_expediente.nunique()`, nunca con el número de filas. Indicar en cada gráfico si cuenta personas o accidentes.

**Variable objetivo (`grave`) — DECIDIDO:**
- `grave = 1` si `lesividad` es `Ingreso superior a 24 horas` o `Fallecido 24 horas`. Es el criterio oficial de la DGT (herido grave = hospitalizado más de 24 h, más fallecidos). Citarlo así en la app y en la presentación.
- `grave = 0` para el resto de categorías informadas (sin asistencia, asistencia en el lugar, ambulatoria, urgencias sin ingreso, ingreso de 24 h o menos).
- `lesividad` vacía (en torno al 45 % de las filas) o `Se desconoce`: **no se trata como "sin lesión"**. Se deja fuera del entrenamiento y del cálculo de tasas, y se muestra como "No informada" en la exploración.
- **`cod_lesividad` no se usa nunca como predictor** (contiene la misma información que el objetivo).
- Clases muy desbalanceadas (los graves son pocos), así que la accuracy por sí sola no sirve. Hay que reportar recall y F1 de la clase grave.

Variables derivadas: año, mes, día de la semana, hora, franja horaria y fin de semana.

## Estilo de código

- Igual que en el material del curso: funciones cortas con docstring, comentarios en español y nombres claros.
- Rutas con `pathlib.Path`. Nada de rutas absolutas.
- Validación explícita: contar filas válidas e inválidas tras cada limpieza e imprimirlo.
- No dejar código muerto ni notebooks como fuente de la app.

## Cómo trabajar con Pao

- **Siempre en español y de forma concisa.**
- **Pasos pequeños.** Cada cambio tiene que poder subirse como un commit independiente. Al terminar cada paso, da:
  1. qué archivos cambiaron;
  2. cómo probarlo (`python app.py` y abrir http://127.0.0.1:8050, o `python train.py`);
  3. un **mensaje de commit sugerido** (`etl: ...`, `app: ...`, `model: ...`, `graphics: ...`, `deploy: ...`, `docs: ...`).
- **No hagas commits ni push** salvo que Pao lo pida explícitamente (la actividad en GitHub cuenta un 20 %).
- **Mensajes de commit** escritos como los escribiría Pao: en español, cortos y naturales (`etl: ...`, `app: ...`). Sin `Co-Authored-By`, sin menciones a Claude ni a herramientas de IA.
- **Código entendible y comentado**: Pao tiene que poder revisarlo y explicarlo. Comentarios en español que expliquen el porqué de cada paso, sin código rebuscado.
- **Comprueba que el código se ejecuta antes de dar el paso por terminado.**
- Explica brevemente las decisiones para que Pao pueda defenderlas en la presentación.
- Pao prefiere iterar viendo resultados: primero una versión que funcione y luego la mejoramos.

## Evaluación (para priorizar)

- App desplegada en una URL: **40 %** (diseño, interacción e innovación).
- Calidad y estructura del código: 20 %.
- Presentación: 20 %. Son 5 minutos más 2 de preguntas, el **26 y el 30 de noviembre de 2026**.
- Actividad en GitHub: 20 %. Se valora la mejora continua.

Prioridad: tener **cuanto antes una versión mínima desplegada en Render** y después iterar sobre ella.

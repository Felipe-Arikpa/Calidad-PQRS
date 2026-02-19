# CALIDAD PQRS

Sistema de clasificación automática y validación de calidad para quejas de la capacidad de salud.

## Descripción

Este proyecto implementa modelos de machine learning para:
- Clasificar automáticamente PQRS en **Procesos** y **Causas**
- Validar la calidad de tipificaciones existentes
- Generar alertas sobre clasificaciones con alta probabilidad de error

El sistema utiliza dos modelos de Regresión Logística con TF-IDF que trabajan de manera secuencial:
1. **Modelo de Procesos**: Clasifica la queja en uno de los procesos principales
2. **Modelo de Causas**: Clasifica la causa específica considerando el proceso predicho

## 🗂️ Estructura del Proyecto

```
CALIDAD PQRS/
├── Evaluation/
│   └── validate_model.py
├── Homework/
│   ├── Procesos.ipynb
│   ├── Causas.ipynb
│   └── predict.ipynb
├── Input/
│   ├── Filter4/
│   │   └── dict.json
│   ├── Predict/
│   │   └── *.xlsx                                   # Archivo con las quejas para las que se quiere validar su tipificación
│   ├── Thresholds/
│   │   ├── causes_thresholds.parquet
│   │   └── process_thresholds.parquet
│   └── Train/
│       └── *.xlsx                                   # Archivo(s) con los datos para enternar los modelos
├── Models/
│   ├── Causes/
│   │   └── salud_causes_classifier.pkl
│   └── Process/
│       └── salud_process_classifier.pkl
├── Output/
│   ├── alerts/
│   │   └── Alertas calidad.xlsx                     # Archivo con las quejas revisadas (salida)
│   └── monitoring/
│       └── transactions_log.parquet
├── src/
│   └── calidad_pqrs/
│       ├── __init__.py
│       ├── config.py
│       ├── utils.py
│       └── models/
│           ├── preprocessing_causes.py
│           ├── preprocessing_process.py
│           ├── tuning_causes.py
│           ├── tuning_process.py
│           ├── train_causes.py
│           ├── train_process.py
│           └── predict.py
├── Test/
├── .gitignore
├── pyproject.toml
└── README.md
```

## Instalación

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd CALIDAD\ PQRS
```

### 2. Crear ambiente virtual

**En Windows:**
```bash
python -m venv venv                         # Este comando se ejecuta una única vez
venv\Scripts\activate                       # Este se ejecuta siempre.
```

**En macOS/Linux:**
```bash
python3 -m venv venv                         # Este comando se ejecuta una única vez
source venv/bin/activate                     # Este se ejecuta siempre.
```

### 3. Instalar dependencias

```bash
pip install -e .                            # Este comando se ejecuta una única vez
```

### 4. Descargar modelo de spaCy

El proyecto utiliza el modelo de lenguaje español de spaCy:

```bash
py -m spacy download es_core_news_lg        # Este comando se ejecuta una única vez
```

## 📚 Uso

### Entrenamiento de Modelos

El entrenamiento se realiza en dos etapas secuenciales:

#### 1. Preparar datos de entrenamiento

Coloca los archivos Excel con datos históricos (año actual y año anterior) en la carpeta `Input/Train/`. Los archivos deben contener las siguientes columnas:
- `Número del caso`
- `Prestación`
- `Filtro 3`
- `Filtro 4`
- `Proceso`
- `Causa`
- `Descripción`
- `Fecha de apertura`

#### 2. Entrenar modelo de Procesos

```bash
python -m calidad_pqrs.models.train_process
```

Este comando:
- Entrena el modelo de clasificación de procesos
- Guarda el modelo en `Models/Process/salud_process_classifier.pkl`
- Calcula y guarda umbrales óptimos en `Input/Thresholds/process_thresholds.parquet`

#### 3. Entrenar modelo de Causas

```bash
python -m calidad_pqrs.models.train_causes
```

Este comando:
- Entrena el modelo de clasificación de causas
- Guarda el modelo en `Models/Causes/salud_causes_classifier.pkl`
- Calcula y guarda umbrales óptimos en `Input/Thresholds/causes_thresholds.parquet`

**Nota:** Los modelos solo se actualizan si el nuevo modelo supera el F1-score macro del modelo anterior.

### Realizar Predicciones

```bash
python Evaluation/validate_model.py
```

Este comando:
- Realiza predicciones sobre los datos en `Input/Predict/`
- Genera alertas en `Output/alerts/Alertas calidad.xlsx`
- Registra métricas en `Output/monitoring/transactions_log.parquet`
- Evalúa si el modelo requiere reentrenamiento

Si el modelo está degradado, mostrará una alerta indicando la necesidad de reentrenamiento.

### Interpretar resultados

El archivo `Output/alerts/Alertas calidad.xlsx` contiene:

| Columna | Descripción |
|---------|-------------|
| Número del caso | ID del caso |
| Descripción | Texto de la queja |
| Proceso | Proceso asignado originalmente |
| Causa | Causa asignada originalmente |
| Final Validation | "Revisar", "No se identifican alertas", "Proceso o Causa desconocidos" |
| Proceso_Sugerido | Proceso sugerido por el modelo |
| Causa_Sugerida | Causa sugerido por el modelo |

**Estados de validación:**
- **"Revisar"**: Se identifican elementos en el texto de la queja que pueden pertenecer a otro proceso o causa.
- **"No se identifican alertas"**: Clasificación correcta o incertidumbre alta (no se genera alertas si el modelo noestá muy seguro)
- **"Proceso o Causa desconocidos"**: Categorías no vistas durante el entrenamiento

## 📧 Contacto

Para preguntas o soporte, contactar a Felipe Aricapa.

**Última actualización**: Febrero 2026
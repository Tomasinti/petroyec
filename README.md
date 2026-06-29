# 🛢️ Predicción de Calidad de Hidrocarburos mediante Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![Status](https://img.shields.io/badge/Status-Completado-green)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://petroyec-l6q3se9cqg6wtgggcr5qe4.streamlit.app/)

## Descripción

Sistema de Machine Learning para predecir si un hidrocarburo es **dulce (sweet)** o **agrio (sour)** según su contenido de azufre, utilizando exclusivamente propiedades físico-químicas medidas en laboratorio — sin necesidad de medir el azufre directamente.

El proyecto utiliza datos reales del **U.S. Department of Energy — Bureau of Mines Crude Oil Analysis Database**, con 9.069 muestras de crudos de todo el mundo recolectadas entre 1920 y 1983, bajo métodos estandarizados ASTM.

> Desarrollado como proyecto de portfolio orientado a la industria de inspección y certificación de hidrocarburos (AmSpec, SGS, Bureau Veritas).

---

## Problema de negocio

Empresas de inspección como por ejemplo **AmSpec** realizan análisis físico-químicos de muestras de crudo para certificar su calidad antes de la comercialización. Este proyecto demuestra cómo un modelo de ML puede:

- **Clasificar** el tipo de crudo (Liviano / Mediano / Pesado) a partir de sus propiedades físicas.
- **Predecir** si un crudo es dulce o agrio **sin medir el azufre directamente**, reduciendo tiempos de análisis.
- **Explicar** cada predicción individualmente mediante análisis SHAP.
- **Cuantificar** las relaciones entre propiedades físico-químicas con respaldo estadístico sobre 9.000+ muestras reales.

---

## Dataset

| Característica | Detalle |
|---|---|
| Fuente | U.S. DOE — Bureau of Mines, disponible en data.gov |
| Muestras | 9.069 análisis de crudo (tras limpieza) |
| Período | 1920 – 1983 |
| Cobertura | EE.UU., América del Sur, Medio Oriente, Europa |
| Variables clave | Gravedad API, azufre, nitrógeno, viscosidad, punto de fluidez, residuo de carbono, volúmenes de destilación |

---

## Hallazgos del análisis exploratorio

| Relación | Correlación | Interpretación |
|---|---|---|
| API ↔ Vol. gasolina/nafta | +0.88 | Crudos livianos producen más nafta |
| API ↔ Vol. residuo | -0.77 | Crudos pesados generan más residuo |
| API ↔ Azufre | -0.56 | Los crudos livianos tienden a ser más dulces |
| Residuo carbono ↔ Azufre | +0.64 | Crudos con más residuo son más sucios |

- **70.9%** de las muestras son crudos dulces (< 0.5% azufre)
- Texas (2.492) y Oklahoma (1.299) concentran la mayor parte de las muestras analizadas

---

## Modelos desarrollados

### Tarea 1 — Clasificación de tipo de crudo

| Modelo | Accuracy | F1 (macro) |
|---|---|---|
| Random Forest | 100% | 1.00 |

Las clases están naturalmente bien separadas por los rangos de gravedad API (Liviano >31.1°, Mediano 22.3–31.1°, Pesado <22.3°), lo que explica el resultado.

### Tarea 2 — Predicción Dulce vs Agrio (sin medir azufre)

| Modelo | Accuracy | AUC-ROC |
|---|---|---|
| Gradient Boosting | 88% | **0.935** |
| Red Neuronal (PyTorch) | 88% | 0.932 |

El Gradient Boosting supera a la red neuronal, resultado consistente con la literatura de ML aplicado a datos tabulares de tamaño medio. La red neuronal agrega valor como punto de comparación y demuestra implementación en PyTorch.

### Interpretabilidad — SHAP

El análisis SHAP revela que **Gravedad API** y **Residuo de carbono** son los predictores dominantes, con importancia media de 1.26 y 1.14 respectivamente. Todas las relaciones aprendidas son físicamente coherentes con la química del petróleo, validando que el modelo capturó patrones reales y no correlaciones espurias.

---

## Estructura del proyecto

```
petroyec/
├── data/
│   ├── crude_oil_clean.csv           # Dataset limpio (9.069 muestras)
│   ├── hidrocarburos.db              # Base de datos SQLite
│   ├── modelo_rf_tipo.pkl            # Random Forest — clasificación de tipo
│   ├── scaler_tipo.pkl               # Scaler correspondiente
│   ├── modelo_gb_dulce_agrio.pkl     # Gradient Boosting — dulce/agrio
│   ├── scaler_dulce_agrio.pkl        # Scaler correspondiente
│   ├── modelo_nn.pth                 # Red neuronal PyTorch
│   ├── scaler_nn.pkl                 # Scaler correspondiente
│   └── *.png                         # Visualizaciones generadas
├── notebooks/
│   ├── 01_exploracion_datos.ipynb    # EDA completo con conclusiones
│   ├── 02_modelos_ml.ipynb           # Random Forest y Gradient Boosting
│   ├── 03_red_neuronal.ipynb         # Red neuronal PyTorch
│   ├── 04_base_de_datos.ipynb        # SQLite — esquema y consultas SQL
│   └── 05_interpretabilidad_shap.ipynb # Análisis SHAP global e individual
├── app/
│   └── app.py                        # Aplicación Streamlit
├── requirements.txt
└── README.md
```

---

## Tecnologías

| Categoría | Herramientas |
|---|---|
| Lenguaje | Python 3.13 |
| Manipulación de datos | Pandas, NumPy |
| Visualización | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, XGBoost |
| Deep Learning | PyTorch |
| Interpretabilidad | SHAP |
| Base de datos | SQLite |
| Serialización | Joblib |
| App web | Streamlit |
| Control de versiones | Git, GitHub |

---

## Autor

**Tomás Inti Malafiej** — Estudiante de Licenciatura en Ciencia de Datos, UCASAL Argentina

[![GitHub](https://img.shields.io/badge/GitHub-Tomasinti-black)](https://github.com/Tomasinti)

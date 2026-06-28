# 🛢️ Predicción de Calidad de Hidrocarburos mediante Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![Status](https://img.shields.io/badge/Status-En%20desarrollo-yellow)

## Descripción

Sistema de Machine Learning para predecir si un hidrocarburo es **dulce (sweet)** o **agrio (sour)** según su contenido de azufre, utilizando exclusivamente propiedades físico-químicas medidas en laboratorio — sin necesidad de medir el azufre directamente.

El proyecto utiliza datos reales del **U.S. Department of Energy — Bureau of Mines Crude Oil Analysis Database**, con 9.069 muestras de crudos de todo el mundo recolectadas entre 1920 y 1983, bajo métodos estandarizados ASTM.

> Desarrollado como proyecto de portfolio orientado a la industria de inspección y certificación de hidrocarburos (AmSpec, SGS, Bureau Veritas).

---

## Problema de negocio

Empresas de inspección como **AmSpec Argentina** realizan análisis físico-químicos de muestras de crudo para certificar su calidad antes de la comercialización. Este proyecto demuestra cómo un modelo de ML puede:

- **Clasificar** el tipo de crudo (Liviano / Mediano / Pesado) a partir de sus propiedades físicas
- **Predecir** si un crudo es dulce o agrio **sin medir el azufre directamente**, reduciendo tiempos de análisis
- **Cuantificar** las relaciones entre propiedades físico-químicas con respaldo estadístico sobre 9.000+ muestras reales

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

---

# 🗄️ Base de Datos SQLite — Hidrocarburos

**Proyecto:** Predicción de calidad de crudos mediante Machine Learning  
**Autor:** Tomás Malafiej — Licenciatura en Ciencia de Datos, UCASAL

---

## Objetivo

Estructurar los datos del dataset DOE en una base de datos relacional normalizada, 
demostrando capacidad de modelado de datos y consultas SQL aplicadas a problemas reales 
de la industria petrolera.

---

## Conclusiones del análisis SHAP

### ¿Qué aprendimos sobre el modelo?

**Gravedad API y Residuo de carbono** son los dos predictores dominantes,
con importancia SHAP media de 1.26 y 1.14 respectivamente. Juntos explican
la mayor parte de las decisiones del modelo.

**Interpretación física:**
- API alto (crudo liviano) → empuja hacia dulce ✅
- Residuo carbono alto (crudo sucio) → empuja hacia agrio ✅
- Vol. gasolina/nafta alto → empuja hacia dulce ✅

Estas relaciones son **físicamente coherentes** con la química del petróleo,
lo que valida que el modelo aprendió patrones reales y no correlaciones espurias.

**Punto de fluidez y Viscosidad** tienen impacto mínimo en la mayoría de
las muestras, consistente con su baja correlación con el azufre observada
en el EDA (notebook 01).

### Valor para AmSpec

El análisis SHAP permite **auditar cada predicción individualmente**:
en lugar de un resultado de caja negra, el sistema puede explicar exactamente
qué propiedad de la muestra determinó la clasificación. Esto es crítico
en un contexto de certificación industrial donde la trazabilidad de las
decisiones es un requisito.

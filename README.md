# 🛢️ Predicción de Calidad de Hidrocarburos mediante Machine Learning

![Python](https://img.shields.io/badge/Python-3.13-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.x-orange)
![Status](https://img.shields.io/badge/Status-Completado-green)

## 📋 Descripción

Sistema de Machine Learning para clasificar tipos de crudo y predecir si un hidrocarburo es **dulce (sweet)** o **agrio (sour)** según su contenido de azufre, utilizando propiedades físico-químicas medidas en laboratorio bajo normas ASTM/API.

El proyecto utiliza datos reales del **U.S. Department of Energy — Bureau of Mines Crude Oil Analysis Database**, con 9.069 muestras de crudos de todo el mundo recolectadas entre 1920 y 1983.

> Desarrollado como proyecto de portfolio orientado a la industria de inspección y certificación de hidrocarburos.

---

## 🎯 Problema de negocio

Empresas de inspección como **AmSpec** realizan análisis físico-químicos de muestras de crudo para certificar su calidad antes de la comercialización. Este proyecto demuestra cómo un modelo de ML puede:

- **Clasificar** el tipo de crudo (Liviano / Mediano / Pesado) a partir de sus propiedades
- **Predecir** si un crudo es dulce o agrio **sin necesidad de medir el azufre directamente**
- **Reducir tiempo y costo** de laboratorio mediante predicciones preliminares

---

## 📊 Dataset

| Característica | Detalle |
|---|---|
| Fuente | U.S. DOE — Bureau of Mines (data.gov) |
| Muestras | 9.069 análisis de crudo |
| Período | 1920 – 1983 |
| Origen | Estados Unidos, América del Sur, Medio Oriente, Europa |
| Variables | Gravedad API, azufre, nitrógeno, viscosidad, punto de fluidez, residuo de carbono, volúmenes de destilación |

---

## 🔍 Hallazgos del Análisis Exploratorio

- **Correlación API vs Azufre: -0.56** — Los crudos más livianos tienden a tener menos azufre (crudos dulces)
- **Residuo de carbono y azufre: +0.64** — Los crudos con más residuo son generalmente más sucios
- **70.9%** de las muestras son crudos dulces (<0.5% azufre)
- Texas (2.492) y Oklahoma (1.299) concentran la mayor parte de las muestras

---

## 🤖 Modelos desarrollados

### Tarea 1 — Clasificación de tipo de crudo
| Modelo | Accuracy | F1 (macro) |
|---|---|---|
| Random Forest | 100% | 1.00 |

> Resultado esperado: la clasificación por tipo está directamente relacionada con la gravedad API y la gravedad específica.

### Tarea 2 — Predicción Dulce vs Agrio (sin medir azufre)
| Modelo | Accuracy | AUC-ROC |
|---|---|---|
| Gradient Boosting | 89% | **0.9416** |
| Red Neuronal (PyTorch) | 88% | 0.9324 |

> El Gradient Boosting supera a la red neuronal en este dataset tabular, lo cual es consistente con la literatura de ML aplicado a datos estructurados.

---

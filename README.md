# 🧠 Quiz 3 – Inteligencia Artificial: Predicción con IA de Mascotas

Este proyecto implementa y compara tres técnicas de aprendizaje supervisado para predecir la edad de una mascota y clasificar si es apta para adopción, basándose en su nivel de energía.

## 📂 Estructura del proyecto

- `app_ia_mascotas.py`: Aplicación principal con Streamlit
- `entrenar_modelos.py`: Script que entrena y guarda los modelos
- `dataset_mascotas.csv`: Dataset base con edad, raza, tamaño, nivel de energía y aptitud
- `modelo_regresion_lineal.pkl`: Modelo entrenado de regresión lineal
- `modelo_regresion_logistica.pkl`: Modelo entrenado de regresión logística
- `modelo_red_neuronal.h5`: Red neuronal entrenada con Keras
- `Quiz3_Defensa_Completa.docx`: Documento guía para sustentación

## 🚀 Cómo ejecutar

1. Instala las dependencias necesarias (preferiblemente en un entorno virtual):
   ```bash
   pip install -r requirements.txt
   ```

2. Ejecuta la aplicación:
   ```bash
   streamlit run app_ia_mascotas.py
   ```

## 📈 Modelos utilizados

| Técnica               | Objetivo                             | Métricas calculadas              |
|-----------------------|--------------------------------------|----------------------------------|
| Regresión Lineal      | Estimar la edad de la mascota        | MSE, R² Score                    |
| Regresión Logística   | Clasificar si es apta para adopción | Accuracy, Precisión, Recall, F1  |
| Red Neuronal (Keras)  | Clasificación binaria (es_apta)     | Accuracy, Precisión, Recall, F1  |

## 🧪 Requisitos

- Python 3.10+
- scikit-learn
- pandas
- numpy
- matplotlib
- tensorflow
- streamlit
- joblib

## 📄 Licencia

Uso académico bajo fines educativos - Politécnico Colombiano Jaime Isaza Cadavid.

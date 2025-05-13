import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model

st.set_page_config(page_title="IA Mascotas", layout="centered")

@st.cache_resource
def cargar_modelos():
    reg_lineal = joblib.load("modelo_regresion_lineal.pkl")
    reg_log = joblib.load("modelo_regresion_logistica.pkl")
    red_nn = load_model("modelo_red_neuronal.h5")
    return reg_lineal, reg_log, red_nn

reg_lineal, reg_log, red_nn = cargar_modelos()

@st.cache_data
def cargar_datos():
    return pd.read_csv("dataset_mascotas.csv")

df = cargar_datos()

st.title("🧠 Predicción con IA de Mascotas")
st.write("Basado en nivel de energía, este sistema estima la edad de la mascota y si es apta para adopción usando 3 modelos.")

energia = st.slider("🔋 Nivel de energía (1-5):", 1, 5, 3)

# Predicciones
edad_pred = reg_lineal.predict([[energia]])[0]
apta_log = reg_log.predict([[energia]])[0]
apta_nn = red_nn.predict(np.array([[energia]])).round()[0][0]

st.subheader("📊 Comparación entre modelos")
st.table(pd.DataFrame({
    "Modelo": ["Regresión Lineal", "Regresión Logística", "Red Neuronal"],
    "Resultado": [f"{edad_pred:.1f} años", "Apta" if apta_log else "No Apta", "Apta" if apta_nn else "No Apta"],
    "Tipo": ["Valor continuo", "Clasificación binaria", "Clasificación binaria"]
}))

# Métricas
st.markdown("### 📈 Métricas del rendimiento de cada modelo")
X = df[["nivel_energia"]]
y_edad = df["edad"]
y_apta = df["es_apta"]
y_pred_lin = reg_lineal.predict(X)
y_pred_log = reg_log.predict(X)
y_pred_nn = (red_nn.predict(X) > 0.5).astype(int)

st.markdown(f"""**Regresión Lineal**
- MSE: {mean_squared_error(y_edad, y_pred_lin):.2f}
- R² Score: {r2_score(y_edad, y_pred_lin):.2f}
""")

st.markdown(f"""**Regresión Logística**
- Precisión (positiva): {precision_score(y_apta, y_pred_log):.2f}
- Exactitud (Accuracy): {accuracy_score(y_apta, y_pred_log):.2f}
- Recall: {recall_score(y_apta, y_pred_log):.2f}
- F1-Score: {f1_score(y_apta, y_pred_log):.2f}
""")

st.markdown(f"""**Red Neuronal**
- Precisión (positiva): {precision_score(y_apta, y_pred_nn):.2f}
- Exactitud (Accuracy): {accuracy_score(y_apta, y_pred_nn):.2f}
- Recall: {recall_score(y_apta, y_pred_nn):.2f}
- F1-Score: {f1_score(y_apta, y_pred_nn):.2f}
""")

# Exportar predicción
if st.button("💾 Guardar predicción como CSV"):
    pd.DataFrame({
        "modelo": ["Regresión Lineal", "Regresión Logística", "Red Neuronal"],
        "prediccion": [edad_pred, apta_log, apta_nn]
    }).to_csv("prediccion_resultado.csv", index=False)
    st.success("Archivo prediccion_resultado.csv guardado correctamente.")

# Gráfico
if st.checkbox("📉 Ver gráfica de dispersión"):
    fig, ax = plt.subplots()
    ax.scatter(df["nivel_energia"], df["edad"], label="Datos reales", alpha=0.6)
    ax.scatter([energia], [edad_pred], color="red", label="Predicción")
    ax.set_xlabel("Nivel de energía")
    ax.set_ylabel("Edad")
    ax.set_title("Dispersión: Nivel de energía vs Edad")
    ax.legend()
    st.pyplot(fig)

# Dataset
if st.checkbox("📄 Mostrar datos del dataset"):
    st.dataframe(df)

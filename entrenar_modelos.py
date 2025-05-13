import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix, precision_score

# ------------------ Cargar datos ------------------
df = pd.read_csv('dataset_mascotas.csv')
X = df[['nivel_energia']]
y_edad = df['edad']
y_apta = df['es_apta']

# ------------------ Punto 1: Regresión Lineal ------------------
print("\n🔹 Punto 1: Regresión Lineal (edad)")
modelo_reg_lineal = LinearRegression()
modelo_reg_lineal.fit(X, y_edad)
pred_lineal = modelo_reg_lineal.predict(X)
mse = mean_squared_error(y_edad, pred_lineal)
r2 = r2_score(y_edad, pred_lineal)
print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")
joblib.dump(modelo_reg_lineal, 'modelo_regresion_lineal.pkl')

# ------------------ Punto 2: Regresión Logística ------------------
print("\n🔹 Punto 2: Regresión Logística (es_apta)")
modelo_reg_log = LogisticRegression()
modelo_reg_log.fit(X, y_apta)
pred_log = modelo_reg_log.predict(X)
matriz = confusion_matrix(y_apta, pred_log)
accuracy = accuracy_score(y_apta, pred_log)
precision = precision_score(y_apta, pred_log)
print("Matriz de Confusión:")
print(matriz)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precisión: {precision:.2f}")
joblib.dump(modelo_reg_log, 'modelo_regresion_logistica.pkl')

# ------------------ Punto 3: Red Neuronal ------------------
print("\n🔹 Punto 3: Red Neuronal (clasificación es_apta)")
model_nn = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_shape=[1])
])
model_nn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nn.fit(X, y_apta, epochs=50, verbose=0)

# Evaluación
loss, accuracy = model_nn.evaluate(X, y_apta, verbose=0)
pred_nn = (model_nn.predict(X) > 0.5).astype(int)
precision_nn = precision_score(y_apta, pred_nn)
print(f"Loss (error): {loss:.4f}")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precisión: {precision_nn:.2f}")
model_nn.save('modelo_red_neuronal.h5')
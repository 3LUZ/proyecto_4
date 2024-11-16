import streamlit as st
import joblib
import numpy as np
# Título de la aplicación
st.title("Predictor")
st.header("Predecir el valor de venta del vehiculo")
# Instrucciones para el usuario
st.write("Ingrese las características del coche para predecir el precio.")
# Front
year = float(st.text_input("Anio de Fabricacion", value="2000"))
km = int(st.text_input("km", value="20000"))
fuel = int(st.slider('Número de habitaciones :',0 ,1 ,0))
engine = int(st.text_input("Tamano del motor", value="2000"))
 # Backend
# Creamos el array de entrada
X_list = [ year, km, fuel, engine ]
X = np.array(X_list, dtype=np.float64)
X = X.reshape(1,-1)
# Botón para ejecutar el modelo
if st.button("Predecir"):
 if len(X) > 0:
 # Cargar el modelo y los parámetros de normalización guardados
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('model.joblib')
 # Mostrar las primeras filas del DataFrame cargado
    X_scaled = scaler.transform(X)
 # Realizar predicciones con el modelo
    predicciones = model.predict(X_scaled)
 # Mostrar las predicciones
    st.write("Predicciones del precio de venta:")
    st.write(predicciones)

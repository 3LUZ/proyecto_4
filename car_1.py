#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Cargamos los parámetros de entrenamiento y parámetros del escalamiento
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

def predict(data):
    input_features = np.array(data).reshape(1, -1)
    
    # Normalizar las características de entrada utilizando el scaler cargado
    input_features = scaler.transform(input_features)
    # Hacer predicción
    prediction = model.predict(input_features)
    
    return prediction

age    = float(input('Year: '))
km     = float(input('Km: '))
fuel   = float(input('Fuel_Type: '))
engine = float(input('EngineSize: '))

y = predict([[age, km, fuel, engine]])

y = np.round(y[0], 2)
print(f"Resultado : {y:,.2f}")


# In[ ]:





# In[ ]:





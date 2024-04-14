import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# cargamos los datos
data = pd.read_csv('Dataset_multivariable.csv')

# hacemos shuffle de los datos
data = data.sample(frac=1, random_state=0)

# dividimos los datos en entrenamiento 0.7 y test 0.2
train_data = data.iloc[:int(len(data)*0.7)]
test_data = data.iloc[int(len(data)*0.7):]

print('tr:', train_data)
print('ts:', test_data)

""" def y(x, w):
  resultados = 0
  print(w)
  print(x)
  for (i,xi) in enumerate(x):
    print(i)
    resultados += np.dot(w[i], xi)
  return resultados """

numero_variables = train_data.shape[1] - 1
w = np.zeros(numero_variables)
alpha = 0.00001
y = lambda x: np.dot(w, x)
y_pred = [] # predicciones
error = [] # errores
ws = [] # pesos

# entrenamiento
for i in range(5):
  # calculamos los pesos
  for column in range(numero_variables):
    w[column] = w[column] - 2*alpha * np.dot(w[column]*train_data.iloc[:, column] - train_data.iloc[:, numero_variables], train_data.iloc[:, column])
  ws.append(w.copy())
  # calculamos una predicción
  y_pred.append(y(train_data.iloc[:, :numero_variables].T.values))
  # calculamos el error
  error.append(np.mean(abs(y_pred[-1] - train_data.iloc[:, numero_variables])))

# pintamos los pesos
print(*ws, end='\n\n', sep='\n')

# pintamos las predicciones
print(*y_pred, end='\n\n', sep='\n')

# pintamos los errores
print(*error, end='\n\n', sep='\n')
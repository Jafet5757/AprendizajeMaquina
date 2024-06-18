import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import prettytable as pt

class SVM:
  def __init__(self, filename = 'iris.csv'):
    self.iris = pd.read_csv(filename)
    self.train = None
    self.test = None
    self.total_intances = 0
    self.ci = [] # Vector de pesos de la clase i
    self.Ni = [] # Total de instancias positivas de la clase i
    self.clases = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']

  def prepare_data(self):
    # Load the iris dataset
    self.train, self.test = train_test_split(self.iris, test_size=0.2, random_state=50)

    self.total_intances = len(self.train)

    # Aplicamos la estrategia One vs Rest al conjunto de entrenamiento

    self.train['Iris-setosa'] = self.train['species'].apply(lambda x: 1 if x == 'Iris-setosa' else -1)

    self.train['Iris-versicolor'] = self.train['species'].apply(lambda x: 1 if x == 'Iris-versicolor' else -1)

    self.train['Iris-virginica'] = self.train['species'].apply(lambda x: 1 if x == 'Iris-virginica' else -1)

    print(self.train)
  
  def svm(self, data, target):
    clases = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    # Calculamos el promedio de la columna donde su target es 1
    vector1 = data[clases].loc[data[target] == 1].mean()
    vector2 = data[clases].loc[data[target] == -1].mean()
    # Obtenemos el vector de pesos c = (v1 + v2)/2
    c = (vector1 + vector2) / 2
    # calculamos el total de instancias positivas
    N = len(data.loc[data[target] == 1])

    return c, N

  def predict(self, data_to_predict):
    # Calculamos la norma de los vectores de pesos
    norm1 = np.linalg.norm(self.ci[0])
    norm2 = np.linalg.norm(self.ci[1])
    norm3 = np.linalg.norm(self.ci[2])

    # Calculamos la proyección de los datos con el vector de pesos de cada clase
    projection1 = np.dot(data_to_predict, self.ci[0]) / norm1
    projection2 = np.dot(data_to_predict, self.ci[1]) / norm2
    projection3 = np.dot(data_to_predict, self.ci[2]) / norm3

    # calculamos la probabilidad de pertenencia a cada clase
    prob1 = projection1 * self.Ni[0] / self.total_intances
    prob2 = projection2 * self.Ni[1] / self.total_intances
    prob3 = projection3 * self.Ni[2] / self.total_intances

    results = []

    # Buscamos cual es la clase con mayor probabilidad
    for i in range(len(data_to_predict)):
      if prob1[i] > prob2[i] and prob1[i] > prob3[i]:
        results.append('Iris-setosa')
      elif prob2[i] > prob1[i] and prob2[i] > prob3[i]:
        results.append('Iris-versicolor')
      else:
        results.append('Iris-virginica')

    return results

  def fit(self):
    self.prepare_data()
    c1, N1 = self.svm(self.train, 'Iris-setosa')
    c2, N2 = self.svm(self.train, 'Iris-versicolor')
    c3, N3 = self.svm(self.train, 'Iris-virginica')
    self.ci = [c1, c2, c3]
    self.Ni = [N1, N2, N3]


if __name__ == '__main__':
  svm = SVM()
  svm.fit()
  species_pred = svm.predict(svm.test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']])
  species_true = (svm.test['species']).to_list()

  print('Accuracy: ', accuracy_score(species_true, species_pred))
  
  # Creamos la tabla de resultados
  table = pt.PrettyTable()
  table.field_names = ['Clase Verdadera', 'Clase Predicha']
  for i in range(len(species_true)):
    table.add_row([species_true[i], species_pred[i]])
  print(table)

  print(classification_report(species_true, species_pred))

  # Matriz de confusión
  cm = confusion_matrix(species_true, species_pred)
  disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Setosa', 'Versicolor', 'Virginica'])
  disp.plot()
  plt.show()
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 21:57:32 2023

@author: DOUGLAS
"""

# Importar bibliotecas necesarias
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# Leer archivo CSV
data = pd.read_csv('gender_submission.csv')

# Dividir los datos en conjuntos de entrenamiento y prueba
X = data.drop(['Survived'], axis=1)
y = data['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar algoritmo CART para seleccionar características importantes
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
model = SelectFromModel(clf, prefit=True)
X_train_new = model.transform(X_train)
X_test_new = model.transform(X_test)

# Clasificar los datos de prueba utilizando el modelo seleccionado
clf.fit(X_train_new, y_train)
y_pred = clf.predict(X_test_new)

# Calcular la precisión de la clasificación
accuracy = accuracy_score(y_test, y_pred)
print("La precisión de la clasificación es: {:.2f}%".format(accuracy * 100))

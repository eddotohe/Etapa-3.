# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:00:08 2023

@author: DOUGLAS
"""

# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Leer archivo CSV
data = pd.read_csv('test.csv')

# Eliminar columnas innecesarias
data.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Codificar variable categórica 'Sex'
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])

# Tratar valores faltantes
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
data['Age'] = imputer.fit_transform(data[['Age']])

# Crear características adicionales
data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
data['IsAlone'] = 1
data.loc[data['FamilySize'] > 1, 'IsAlone'] = 0

# Codificar variables categóricas 'Embarked'
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(), [6])], remainder='passthrough')
data = ct.fit_transform(data)






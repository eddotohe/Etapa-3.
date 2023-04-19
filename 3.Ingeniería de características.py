# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 22:00:13 2023

@author: DOUGLAS
"""

import pandas as pd

# Cargar el archivo CSV en un DataFrame de Pandas
df_train = pd.read_csv('train.csv')

# Crear una nueva columna "Title" que extrae el título de la columna "Name"
df_train['Title'] = df_train['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)

# Reemplazar títulos poco comunes con 'Rare'
df_train['Title'] = df_train['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

# Reemplazar los títulos en francés por sus equivalentes en inglés
df_train['Title'] = df_train['Title'].replace(['Mlle', 'Ms'], 'Miss')
df_train['Title'] = df_train['Title'].replace('Mme', 'Mrs')

# Convertir la columna "Sex" a valores numéricos
df_train['Sex'] = df_train['Sex'].map({'female': 0, 'male': 1}).astype(int)

# Convertir la columna "Embarked" a valores numéricos
df_train['Embarked'] = df_train['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Completar los valores faltantes en la columna "Age" con la media de edad
mean_age = df_train['Age'].mean()
df_train['Age'] = df_train['Age'].fillna(mean_age)

# Crear una nueva columna "FamilySize" que es la suma de las columnas "SibSp" y "Parch"
df_train['FamilySize'] = df_train['SibSp'] + df_train['Parch'] + 1

# Crear una nueva columna "IsAlone" que es 1 si "FamilySize" es 1, 0 de lo contrario
df_train['IsAlone'] = 0
df_train.loc[df_train['FamilySize'] == 1, 'IsAlone'] = 1

# Eliminar las columnas que no se utilizarán en el modelo
df_train = df_train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp', 'Parch'], axis=1)

# Mostrar las primeras filas del DataFrame
print(df_train.head())





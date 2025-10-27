# Proyecto con Arboles de Decision
#  CRISP-DM CRISP-DM significa Cross Industry Standard Process for Data Mining,
# y es el estándar más usado en la industria para desarrollar proyectos de análisis

###############   0. Business Understanding
# Este conjunto de datos proviene originalmente del Instituto Nacional de Diabetes y Enfermedades Digestivas y Renales.
# El objetivo es predecir en base a medidas diagnósticas si un paciente tiene o no diabetes.

###############   1. Data Understanding

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Cargar los datos
total_data = pd.read_csv("https://breathecode.herokuapp.com/asset/internal-link?id=930&path=diabetes.csv", sep=',')
total_data.to_csv("/workspaces/arboles-decision-danny/data/raw/total_data.csv", sep=',', index=False)

# Exploración inicial
print("Exploración inicial")
print(total_data.head())
print("Filas y columnas:", total_data.shape)
print("Info:")
print(total_data.info())
print("Estadísticas Descriptivas (originales):")
print(total_data.describe())

# 2.1 Verificar valores nulos o ceros sospechosos
print("\nValores nulos por columna:")
print(total_data.isnull().sum())

# 2.2 Contar valores cero en variables que no deberían tenerlos
cols_with_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
for col in cols_with_zeros:
    zeros = (total_data[col] == 0).sum()
    print(f"{col}: {zeros} valores cero")

# 2.3 Visualización rápida de distribuciones
total_data.hist(figsize=(10, 8))
plt.suptitle("Distribución de variables antes de limpieza")
plt.show()

###############   3. Data Preparation
interim_data = total_data.copy()

# Reemplazar ceros por NaN (ya hecho antes)
interim_data[cols_with_zeros] = interim_data[cols_with_zeros].replace(0, np.nan)

# Imputar valores faltantes con la mediana sin usar inplace en el slice
for col in cols_with_zeros:
    median_val = interim_data[col].median()
    interim_data[col] = interim_data[col].fillna(median_val)

# Verificar que ya no haya nulos
print("\nValores nulos después de la imputación:")
print(interim_data.isnull().sum())

# Estadísticas descriptivas después de la limpieza
print("\nEstadísticas Descriptivas (dataset limpio):")
print(interim_data.describe())

#No se va a escalar o normalizar ya que se utilizará el modelo de arboles de decision 

# Identificación de Outliers

# se utilizará IQR IQR significa Interquartile Range, o rango intercuartílico. 
# Es una medida de dispersión que se centra en la parte “central” de tus datos, ignorando valores extremos.

# Columnas numéricas a revisar
num_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

# Función para detectar outliers
def detect_outliers(df, cols):
    outlier_dict = {}
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)][col]
        outlier_dict[col] = outliers
        print(f"{col}: {len(outliers)} outliers")
    return outlier_dict

outliers = detect_outliers(interim_data, num_cols)

#Capear (winsorize): reemplazar valores extremos por percentiles.
#Eliminar: solo si hay muy pocos outliers.
#quí usamos capping al 1er y 99º percentil, robusto para árboles:

for col in num_cols:
    lower = interim_data[col].quantile(0.01)
    upper = interim_data[col].quantile(0.99)
    interim_data[col] = np.clip(interim_data[col], lower, upper)

# ------------Importancia de las variables

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# Separar variables
X = interim_data.drop("Outcome", axis=1)
y = interim_data["Outcome"]

# Separar la data en train y test

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar árbol de decisión
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)

# Predicción y evaluación
y_pred = tree.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Feature importance
importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature Importance:")
print(importances)

# Visualización
importances.plot(kind="bar", figsize=(10,5), title="Feature Importance")
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.tree import export_text
import joblib  # Para guardar el modelo

############### 4. Probar distintos criterios de pureza #################

criterios = ['gini', 'entropy', 'log_loss']
for crit in criterios:
    tree = DecisionTreeClassifier(random_state=42, criterion=crit)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print(f"\nCriterio: {crit}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    
    # Visualizar feature importance
    importances = pd.Series(tree.feature_importances_, index=X.columns).sort_values(ascending=False)
    print("Feature Importance:")
    print(importances)
    importances.plot(kind="bar", figsize=(10,5), title=f"Feature Importance ({crit})")
    plt.show()

############### 5. Optimización de hiperparámetros #################

# Definir el grid de búsqueda
param_grid = {
    'max_depth': [3, 5, 7, 9, None],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10],
    'criterion': ['gini', 'entropy', 'log_loss']
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\nMejores hiperparámetros encontrados:")
print(grid_search.best_params_)

# Evaluar modelo optimizado
best_tree = grid_search.best_estimator_
y_pred_best = best_tree.predict(X_test)
print("\nEvaluación del árbol optimizado:")
print("Accuracy:", accuracy_score(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# Guardar el modelo optimizado
joblib.dump(best_tree, "decision_tree_optimized.pkl")
print("\nModelo guardado como 'decision_tree_optimized.pkl'")

# visualizar árbol en texto
tree_rules = export_text(best_tree, feature_names=list(X.columns))
print("\nReglas del árbol optimizado:")
print(tree_rules)
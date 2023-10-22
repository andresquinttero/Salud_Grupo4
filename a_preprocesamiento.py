import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


################# 1. Cargar el archivo Excel #################

# Se carga el archivo Excel
df_cronicos = pd.read_excel('databases/RETO_df_cronicos.xlsx', engine='openpyxl')

# Obtener las primeras cinco filas del DataFrame para revisión
preview_cronicos = df_cronicos.head(5)
print(preview_cronicos)

# Obtener los nombres de las columnas y sus tipos de datos del DataFrame
columns_info = df_cronicos.dtypes
print(columns_info)


################# Función para corregir estos errores en las tildes #################

def correct_encoding(column):
    column = column.replace('Ã³', 'ó')
    column = column.replace('Ã\xad', 'í')
    column = column.replace('Ã©', 'é')
    column = column.replace('Ã¡', 'á')
    column = column.replace('Ãº', 'ú')
    column = column.replace('Ã±', 'ñ')
    column = column.replace('Ã', 'í')
    return column

# Aplica la función a cada columna del DataFrame
df_cronicos.columns = [correct_encoding(col) for col in df_cronicos.columns]

print(df_cronicos.columns) # Vemos que ya no hay errores de codificación

# Estadísticas descriptivas de la base de datos
print(df_cronicos.describe(include='all').T)

# Ver el número de columnas
print(len(df_cronicos.columns)) # 290 columnas

################# Análisis de valores nulos #################

missing_values = df_cronicos.isnull().sum()
print(missing_values)
total_nulls = df_cronicos.isnull().sum().sum()
print(total_nulls)

# Vemos el porcentaje de valores nulos por columna
percentage_missing = (df_cronicos.isnull().sum() / len(df_cronicos)) * 100
print(percentage_missing.sort_values(ascending=False))

# Por lo que crearemos un mapa de calor para ver los valores nulos por columna
plt.figure(figsize=(15, 10))
sns.heatmap(df_cronicos.isnull(), cbar=False, cmap='viridis')
plt.show()

# Identificar las columnas de diagnóstico
diagnostic_columns = [col for col in df_cronicos.columns if "Diagnostico" in col or "NombreDiagnostico" in col]

# Crear una copia del DataFrame original para guardar los cambios
df_cronicos_cleaned = df_cronicos.copy()

# Reemplazar valores nulos por '0' en las columnas de diagnóstico y dejar los otros valores nulos como están
for col in diagnostic_columns:
    df_cronicos_cleaned[col] = df_cronicos_cleaned[col].apply(lambda x: 0 if pd.isnull(x) else x)

# Verificar los cambios realizados en el mapa de calor
plt.figure(figsize=(15, 10))
sns.heatmap(df_cronicos_cleaned.isnull(), cbar=False, cmap='viridis')
plt.show()
# Aqui vemos que donde había valores nulos ahora hay ceros y donde si habian diagnositcos se dejaron como estaban





# Handle missing values
df_cronicos = df_cronicos.dropna()

# Step 3: Check for duplicates and handle them appropriately
# Check for duplicates
duplicates = df_cronicos.duplicated()
print(duplicates)

# Handle duplicates
df_cronicos = df_cronicos.drop_duplicates()

# Step 4: Check for outliers and handle them appropriately
# Check for outliers
outliers = df_cronicos[df_cronicos.apply(lambda x: np.abs(x - x.mean()) / x.std() < 3).all(axis=1)]

# Handle outliers
df_cronicos = outliers

# Step 5: Check for inconsistent data types and handle them appropriately
# Check data types
data_types = df_cronicos.dtypes
print(data_types)

# Handle inconsistent data types
# Step 7: Save the pre-processed data to a new file
df_cronicos.to_csv('preprocessed_data.csv', index=False)

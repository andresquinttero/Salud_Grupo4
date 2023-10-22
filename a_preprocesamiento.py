import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
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


# Adicionalemnte hay 15 columans que tienen valores nulos, por lo que se procede a llenarlos con valores por defecto segun cada una 

# Definir las columnas y los valores correspondientes para llenar
columns_to_fill = [
    "Espirometria", "VEF1/CVF", "VEF1/VFC Posbroncodilatador", "Gravedad", "Diagnóstico EPOC",
    "Disnea MMRC", "Clasificación", "CAT", "Número de exacerbaciones último año (Que hayan necesitado hospitalizado)",
    "Clasificación GOLD", "Clasificación1", "Clasificación BODEX", "Oxígeno dependiente", "Tiene gases arteriales", "Resultado"
]

fill_values = ["no", 0, 0, "no", "no", 0, "no", 0, 0, 0, "no", 0, "no", "no", 0]

# Llenar las columnas vacías según las instrucciones
for col, value in zip(columns_to_fill, fill_values):
    df_cronicos_cleaned[col].fillna(value, inplace=True)

# Verificar los cambios en las primeras filas y en el mapa de calor
df_cronicos_cleaned[columns_to_fill].head()
plt.figure(figsize=(15, 10))
sns.heatmap(df_cronicos_cleaned.isnull(), cbar=False, cmap='viridis')
plt.show()


################# Manejar atipicos #################
# Seleccionar las primeras 5 columnas numéricas
numeric_cols = df_cronicos_cleaned.select_dtypes(include=[np.number]).columns[:5]

# Crear boxplots para cada una de las columnas numéricas seleccionadas
for col in numeric_cols:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=df_cronicos_cleaned[col])
    plt.title(f"Boxplot para {col}")
    plt.show()

# Vemos que en la Box de peso hay 2 valores atipicos, por lo que se procede a correjirlos
# Ordenamos los valores de la columna "Peso" en orden descendente y visualizar los primeros registros
largest_weights = df_cronicos_cleaned["Peso"].nlargest(5)
print(largest_weights)

# Hay 4 valores sin sentido por lo que los aproximamos pensando que tienen error en la coma
# Definir un diccionario con los valores actuales como claves y los nuevos valores como valores
replacement_values = {
    62153.0: 62,
    6151.0: 61,
    693.0: 69,
    666.0: 67
}

# Reemplazar los valores en la columna "Peso" usando el diccionario
df_cronicos_cleaned["Peso"].replace(replacement_values, inplace=True)

largest_weights = df_cronicos_cleaned["Peso"].nlargest(5)
print(largest_weights)
# Vemos que ya no hay valores atipicos
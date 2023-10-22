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


# Adicionalemnte hay 15 columnas que tienen valores nulos, por lo que se procede a llenarlos con valores por defecto segun cada una 

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

# Vemos que en la Box de peso hay 2 valores atipicos, por lo que se procede a corregirlos
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


# Exploración inicial con el dataset limpio

# Para ver todas las columnas
pd.set_option('display.max_columns', None)
print(df_cronicos_cleaned.columns.tolist())

# Para ver todas las columnas con 2 valores únicos
binary_cols = [col for col in df_cronicos_cleaned.columns if df_cronicos_cleaned[col].nunique() == 2]

# Para ver los valores únicos de cada columna con 2 valores únicos
for col in binary_cols:
    print(f"Unique values in {col}: {df_cronicos_cleaned[col].unique()}")

# # Cambiamos "no" con  "No" en algunas columnas
df_cronicos_cleaned["Oxígeno dependiente"].replace("no", "No", inplace=True)
df_cronicos_cleaned['Oxígeno dependiente']
df_cronicos_cleaned["Tiene gases arteriales"].replace("no", "No", inplace=True)
df_cronicos_cleaned['Tiene gases arteriales']

# Cambiamos "no" con  "No" en "Diagnóstico EPOC" 
df_cronicos_cleaned["Diagnóstico EPOC"].replace("no", "No", inplace=True)
df_cronicos_cleaned['Diagnóstico EPOC']
print(df_cronicos_cleaned['Diagnóstico EPOC'].value_counts())

# Hacemos la variable 'Diagnóstico EPOC' numérica para luego hacer una matriz de correlación
df_cronicos_cleaned["Diagnóstico EPOC"].replace("No", 0, inplace=True)
df_cronicos_cleaned["Diagnóstico EPOC"].replace("Si", 1, inplace=True)
df_cronicos_cleaned['Diagnóstico EPOC']
df_cronicos_cleaned['Diagnóstico EPOC'] = df_cronicos_cleaned['Diagnóstico EPOC'].astype(int)


# Vemos que hay algunos valores atípicos en la columna talla, vamos a borrarlos
df_cronicos_cleaned = df_cronicos_cleaned[df_cronicos_cleaned['Talla'] < 200]

# Vemos gráficamente cómo está el dataset

# Gráfica para ver en qué mes es más frecuente el diagnóstico de EPOC
plt.subplots(figsize=(12, 8))
sns.countplot(x='MES', data=df_cronicos_cleaned)
plt.title('Frecuencia de diagnóstico de EPOC por mes')
plt.show()

# Gráfica para ver si el peso tiene relación con la frecuencia de diagnóstico de EPOC
sns.boxplot(x='Diagnóstico EPOC', y='Peso', data=df_cronicos_cleaned)
plt.title('Relación entre peso y diagnóstico de EPOC')
plt.show()
# No hay ninguna relación aparente entre el diagnóstico de EPOC y el peso



# Gráfica para ver si la talla tiene relación con la frecuencia de diagnóstico de EPOC
sns.boxplot(x='Diagnóstico EPOC', y='Talla', data=df_cronicos_cleaned)
plt.title('Relación entre talla y diagnóstico de EPOC')
plt.show()
# No hay ninguna relación aparente entre el diagnóstico de EPOC y la talla


# Hacemos una matriz de correlación para ver las variables que más se relacionan con el diagnóstico de EPOC
numeric_cols = df_cronicos_cleaned.select_dtypes(include=[np.number]).columns
corr_matrix = df_cronicos_cleaned[numeric_cols].corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, cmap='viridis')
plt.show()
print(corr_matrix["Diagnóstico EPOC"].sort_values(ascending=False))

# Aunque las correlaciones no son muy altas, vemos que las variables que más 
# se relacionan con el diagnóstico de EPOC son:
# CAT, Clasificación GOLD, Disnea MMRC, Clasificación BODEX, VEF1/VFC posbroncodilatador

# Lo cual tiene sentido, 

# La variable que tiene la mayor correlación con el diagnóstico de EPOC es
# el CAT, que es un cuestionario que evalúa el impacto de la EPOC en la calidad de vida del paciente.
# Esto significa que a mayor puntuación en el CAT, mayor es la probabilidad de tener EPOC. 

# La segunda variable con mayor correlación es la clasificación GOLD, que estratifica a los pacientes
# con EPOC según el riesgo de exacerbaciones y el impacto en la calidad de vida. Esto significa 
# que a mayor categoría en la clasificación GOLD, mayor es la probabilidad de tener EPOC

# La tercera variable con mayor correlación es la disnea MMRC, que es una escala que mide el grado de
# dificultad para respirar que experimenta el paciente. Esto significa que a mayor nivel de disnea,
# mayor es la probabilidad de tener EPOC. Esto se explica por el hecho de que la disnea es uno de 
# los síntomas más frecuentes y limitantes de la EPOC.

# La cuarta variable con mayor correlación es la clasificación BODEX, que predice la mortalidad 
# en los pacientes con EPOC. Esto significa que a mayor puntuación en el BODEX, mayor es la 
# probabilidad de tener EPOC. Esto se debe a que el BODEX se basa en cuatro variables que reflejan
# el estado nutricional, la función pulmonar, la capacidad de ejercicio y la severidad de los 
# síntomas del paciente con EPOC.

# La quinta variable con mayor correlación es el VEF1/VFC posbroncodilatador, que es el cociente 
# entre el volumen espiratorio forzado en el primer segundo y la capacidad vital forzada 
# después de administrar un broncodilatador.

# Las variables que tienen una correlación muy baja o nula con el diagnóstico de EPOC son 
# la velocidad (m/s) y el tiempo en segundos (apoyo monopodal). 
# Estas variables miden aspectos relacionados con el equilibrio y la movilidad del paciente,
# que no parecen estar directamente asociados con la presencia o ausencia de EPOC.

import pandas as pd
import numpy as np


# --- FUNCIONES DE ANÁLISIS ---
def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe un DataFrame y devuelve otro DataFrame con una fila 
    por cada columna del DataFrame original. El índice del resultado debe ser 
    el nombre de cada columna. Las columnas del resultado deben ser"""

    if not isinstance(df, pd.DataFrame):
        print(f"El argumento es de tipo {type(df)} cuando debería ser un DataFrame")
        return None
    
    tipo = df.dtypes
    porcentaje_nulos = round((df.isnull().sum() / len(df)) * 100, 2)
    valores_unicos = df.nunique()
    porcentaje_cardinalidad = round((valores_unicos / len(df)) * 100, 2)

    columnas = pd.DataFrame({
        "tipo" : tipo,
        "porcentaje_nulos": porcentaje_nulos,
        "valores_unicos": valores_unicos,
        "porcentaje_cardinalidad": porcentaje_cardinalidad
    })

    return columnas
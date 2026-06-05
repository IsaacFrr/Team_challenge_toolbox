import pandas as pd
import numpy as np

# --- FUNCIONES DE ANÁLISIS ---
# Default de umbral de categoría entre 15 y 80
def tipifica_variables(df: pd.DataFrame, umbral_categoria: int = 15, umbral_continua: float = 80.0) -> pd.DataFrame:
    """Devuelve un DataFrame con dos columnas, `nombre_variable` y `tipo_sugerido`,
      con tantas filas como columnas tiene el DataFrame de entrada."""
    
    # Comprobaciones: Que sea un dataframe, que el umbral_categoría sea un entero positivo y que umbral continua sea un float entre 0 y 100
    if not isinstance(df, pd.DataFrame):
        print(f"El argumento es de tipo {type(df)} cuando debería ser un DataFrame")
        return None
    if not isinstance(umbral_categoria, int) or umbral_categoria < 0:
        print("El umbral_categoria debe ser un entero positivo")
        return None
    if not isinstance(umbral_continua, (int, float)) or umbral_continua < 0 or umbral_continua > 100:
        print("El umbral_continua debe ser un float entre 0 y 100")
        return None
    
    # Normalizar el umbral_continua por si han pasado un entero
    umbral_continua = float(umbral_continua)
    describe = describe_df(df)

    cardinalidad = describe["valores_unicos"]
    porcentaje_cardinalidad = describe["porcentaje_cardinalidad"]

    lista = []
    for clave, valor in cardinalidad.items():
        # Cardinalidad = 2 entonces Binaria
        if valor == 2:
            lista.append("Binaria")
        # Cardinalidad < umbral entonces Categórica
        elif valor < umbral_categoria:
            lista.append("Categórica")
        # Cardinalidad >= umbral Y porcentaje cardinalidad >= umbral_continua entonces Numérica Continua
        elif valor >= umbral_categoria and porcentaje_cardinalidad[clave] >= umbral_continua:
            lista.append("Numérica Continua")
        # Cardinalidad >= umbral Y porcentaje cardinalidad  <= umbral_continua entonces Numérica Discreta
        elif valor >= umbral_categoria and porcentaje_cardinalidad[clave] <= umbral_continua:
            lista.append("Numérica Discreta")



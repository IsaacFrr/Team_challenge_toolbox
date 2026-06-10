import pandas as pd
import numpy as np

# --- FUNCIONES DE ANÁLISIS ---

# TIPIFICAR VARIABLES
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

    columnas = pd.DataFrame({
        "nombre_variable": cardinalidad.index,
        "tipo_sugerido": lista
        })
    
    return resultado

# DESCRIBIR EL DATAFRAME
def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe un DataFrame y devuelve otro DataFrame con una fila 
    por cada columna del DataFrame original. El índice del resultado debe ser 
    el nombre de cada columna."""

    if not isinstance(df, pd.DataFrame):
        print(f"El argumento es de tipo {type(df)} cuando debería ser un DataFrame")
        return None
    
    columnas = pd.DataFrame({
        "tipo" : df.dtypes,
        "porcentaje_nulos": round((df.isnull().sum() / len(df)) * 100, 2),
        "valores_unicos": df.nunique(),
        "porcentaje_cardinalidad": round((df.nunique() / len(df)) * 100, 2)
    })

    return columnas

# VALIDAR EL DATAFRAME
def verificar_dataframe(df: pd.DataFrame, exigir_numericas: bool = False) -> None:
    """Verifica que el DataFrame esté en condiciones para empezar a modelar o si debemos tratar algo.
    Comprueba: que no esté vacío, que no tenga duplicados, nulos, infinitos ni columnas constantes.
    Si exigir_numericas=True, comprueba además que todas las columnas son
    numéricas (depende del flujo de trabajo que tengas igual interesa o no)."""
    # Comprueba si tiene filas y columnas (comprueba si está vacío por ambas partes)
    assert df.shape[0] > 0, "El DataFrame no tiene filas"
    assert df.shape[1] > 0, "El DataFrame no tiene columnas"

    # Comprueba si hay fuilas duplicadas
    num_duplicados = df.duplicated().sum()
    assert num_duplicados == 0, f"Hay {num_duplicados} filas duplicadas"

    # Comprueba si hay nulos
    num_nulos = df.isnull().sum().sum()
    assert num_nulos == 0, f"Hay {num_nulos} valores nulos sin tratar"

    # Solo si estamos justo antes de modelar exigimos que todo sea numérico o no
    # Dependerá de cómo se desee trabajar
    if exigir_numericas:
        columnas_no_numericas = []
        for columna in df.columns:
            # Más estricto que dtype. Esto cubre todos los tipos numéricos
            if not pd.api.types.is_numeric_dtype(df[columna]):
                columnas_no_numericas.append(columna)
        assert len(columnas_no_numericas) == 0, f"Hay columnas no numéricas sin encodear: {columnas_no_numericas}"

    # Comprueba si hay valores infinitos en las columnas numéricas
    num_infinitos = np.isinf(df.select_dtypes(include="number")).sum().sum()
    assert num_infinitos == 0, f"Hay {num_infinitos} valores infinitos"

    # No puede haber columnas constantes (columnas que tienen un valor único)
    # (No aporta nada. Varianza cero, no da información y seguramente ralentice todo)
    columnas_constantes = []
    for columna in df.columns:
        if df[columna].nunique() == 1:
            columnas_constantes.append(columna)
    assert len(columnas_constantes) == 0, f"Hay columnas constantes sin información: {columnas_constantes}"

    print("El DataFrame está listo para modelar")
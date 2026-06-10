import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

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
    
    return columnas

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

    return

# --- FUNCIONES DE REGRESIÓN CATEGÓRICA---

def get_features_cat_regression(
    df: pd.DataFrame,
    target_col: str,
    pvalue: float = 0.05
) -> list:
    if not isinstance(df, pd.DataFrame):
        print("Provided dataframe is not a pd.Dataframe.")
        return
    if target_col not in df.columns:
        print("Target column is not in dataframe.")
        return
    if not (isinstance(pvalue, float) and (0 < pvalue and pvalue < 1)):
        print("P-value must be a floating point number between 0 and 1.")
        return

    features = []
    df_tipos = tipifica_variables(df, umbral_categoria = 15, umbral_continua=85)

    if df_tipos.loc[df_tipos["nombre_variable"]==target_col, "tipo_sugerido"].isin(["Binaria", "Categórica"]).iloc[0]:
        print("Target variable must be numerical.")
        return

    variables = df_tipos.loc[df_tipos.tipo_sugerido.isin(["Binaria", "Categórica"])]
    for indep_var, tipo in zip(variables["nombre_variable"].tolist(), variables["tipo_sugerido"].tolist()):
        if tipo == "Binaria":
            # Estadístico U de Mann-Whitney
            grupos = [df.loc[df[indep_var]==clase][target_col] for clase in df[indep_var].unique()]
            pval_stat = scipy.stats.mannwhitneyu(grupos[0], grupos[1])[1]

        elif tipo == "Categórica":
            # Estadístico F de ANOVA
            grupos = [df.loc[df[indep_var] == clase, target_col] for clase in df[indep_var].dropna(inplace=False).unique()]
            if 1 in [len(grupo) for grupo in grupos]:
                pval_stat = scipy.stats.kruskal(*grupos)[1] # para poblaciones con tamaño muestral pequeño
            else:
                pval_stat = scipy.stats.f_oneway(*grupos)[1]

        if pval_stat in [np.nan, np.inf]: # Con algunos sets de datos puede devolver np.nan o np.inf
            print(f"Data size for target variable groups by {indep_var} is not sufficient.")
        elif pval_stat <= pvalue:
            features.append(indep_var)

    return features


def plot_features_cat_regression(
    df: pd.DataFrame,
    target_col: str = "",
    columns: list = [],
    pvalue: float = 0.05,
    with_individual_plot: bool = False
) -> list:
    # Comprobaciones
    if not isinstance(df, pd.DataFrame):
        print("Provided dataframe is not a pd.Dataframe.")
        return
    if target_col not in df.columns:
        print("Target column is not in dataframe.")
        return
    if not (isinstance(pvalue, float) and (0 < pvalue and pvalue < 1)):
        print("P-value must be a floating point number between 0 and 1.")
        return

    df_tipos = tipifica_variables(df)
    if df_tipos.loc[df_tipos["nombre_variable"]==target_col, "tipo_sugerido"].isin(["Binaria", "Categórica"]).iloc[0]:
        print("target variable must be numerical.")
        return

    # Lista de features según las columnas dadas
    if len(columns) == 0:
        features = df_tipos.loc[df_tipos.tipo_sugerido.isin(["Binaria", "Categórica"]), "nombre_variable"].tolist()
    else:
        cats_sign = get_features_cat_regression(df, target_col, pvalue)
        if (type(cats_sign) == None):
            return
        elif len(cats_sign) == 0:
            print("Selected variables are not significantly correlated to target.")
            return
        features = list(set(cats_sign).intersection(set(columns)))
    
    # Plot de target según las features
    if with_individual_plot:
        for var in features:
            categorias = df[var].unique().tolist()
            plt.figure(figsize=(8, 6))

            sns.histplot(data=df, x=target_col, hue=var, hue_order=categorias, bins=20, kde=True, stat="count", alpha=0.4,)

            plt.title(f"{target_col} según {var}: "f"{', '.join(map(str, categorias))}")    # map convierte bools y numeros en str
            plt.xlabel(target_col)
            plt.show()
    else:
        n = len(features)
        ncols = min(3, n)
        nrows = int(np.ceil(n / ncols))

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(7*min(2, ncols), 5*nrows))
        axes = np.array(axes).reshape(-1)

        for i, var in enumerate(features):
            ax = axes[i]
            tabla = pd.crosstab(df[var], df[target_col]) # Tabla de contingencia
            ylabel = "Frecuencia absoluta"

            sns.histplot(data=df, x=target_col, hue=var, hue_order=df[var].unique().tolist(), bins=20, kde=True, stat="count", alpha=0.4, ax=ax)

            ax.set_title(f"{target_col} según {var}")
            ax.set_xlabel(var)
            ax.set_ylabel(ylabel)
            ax.tick_params(axis="x", rotation=45)

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        plt.show()

    return

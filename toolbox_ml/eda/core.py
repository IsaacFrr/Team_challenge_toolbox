import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

# --- FUNCIONES DE ANÁLISIS ---

# TIPIFICAR VARIABLES
# Default de umbral de categoría entre 15 y 80
def tipifica_variables(df: pd.DataFrame, umbral_categoria: int = 15, umbral_continua: float = 80.0) -> pd.DataFrame:
    """Returns a DataFrame with two columns, "nombre_variable" and "tipo_sugerido,
    with as many rows as columns in the input DataFrame.

    Args:
    df (pd.DataFrame): Input DataFrame.
    umbral_categoria (int): Cardinality threshold below which a column 
        is considered categorical. Must be a positive integer. Defaults to 15.
    umbral_continua (float): Cardinality percentage threshold above which 
        a numeric column is considered continuous. Must be between 0 and 100. 
        Defaults to 80.0.

    Returns:
        pd.DataFrame: DataFrame with columns 'nombre_variable' and 'tipo_sugerido', 
        where each row corresponds to a column of the input. Returns None if any 
        input argument is invalid.
    """
    
    # Checks: that it is a DataFrame, that umbral_categoria is a positive integer
    # and that umbral_continua is a float between 0 and 100
    if not isinstance(df, pd.DataFrame):
        print(f"The argument is of type {type(df)} when it should be a DataFrame")
        return None
    if not isinstance(umbral_categoria, int) or umbral_categoria < 0:
        print("umbral_categoria must be a positive integer")
        return None
    if not isinstance(umbral_continua, (int, float)) or umbral_continua < 0 or umbral_continua > 100:
        print("umbral_continua must be a float between 0 and 100")
        return None
    
    # Normalize umbral_continua in case an integer was passed as an argument
    umbral_continua = float(umbral_continua)
    describe = describe_df(df)

    cardinalidad = describe["valores_unicos"]
    porcentaje_cardinalidad = describe["porcentaje_cardinalidad"]

    lista = []
    for clave, valor in cardinalidad.items():
        # Cardinalidad = 2 then Binary
        if valor == 2:
            lista.append("Binaria")
        # Cardinalidad < umbral_categoria then Categórica
        elif valor < umbral_categoria:
            lista.append("Categórica")
        # Cardinalidad >= umbral_categoria AND porcentaje_cardinalidad >= umbral_continua then Numérica Continua
        elif valor >= umbral_categoria and porcentaje_cardinalidad[clave] >= umbral_continua:
            lista.append("Numérica Continua")
        # Cardinalidad >= umbral_categoria AND porcentaje_cardinalidad <= umbral_continua then Numérica Discreta
        elif valor >= umbral_categoria and porcentaje_cardinalidad[clave] <= umbral_continua:
            lista.append("Numérica Discreta")

    columnas = pd.DataFrame({
        "nombre_variable": cardinalidad.index,
        "tipo_sugerido": lista
        })
    
    return columnas

# DESCRIBIR EL DATAFRAME
def describe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Receives a DataFrame and returns another DataFrame with one row
    per column of the original DataFrame. The index of the result is
    the name of each column.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze.

    Returns:
        pd.DataFrame: DataFrame with one row per column of the input, containing 
        'tipo', 'porcentaje_nulos', 'valores_unicos' and 'porcentaje_cardinalidad'. 
        Returns None if the input is not a DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        print(f"The argument is of type {type(df)} when it should be a DataFrame")
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
    """Verifies that the DataFrame is in condition to start modeling or if something needs to be handled.

    Checks:
    That it is not empty, has no duplicates, nulls, infinite values or constant columns.
    If exigir_numericas=True, it also checks that all columns are numeric (depending on your workflow
    this may or may not be relevant).

    Args:
    df (pd.DataFrame): DataFrame to verify.
    exigir_numericas (bool): If True, also checks that all columns 
    are numeric. Defaults to False.

    Raises:
        AssertionError: If any of the data quality checks fails.
    """
    # Check if it has rows and columns (checks if it is empty from both sides)
    assert df.shape[0] > 0, "DataFrame has no rows"
    assert df.shape[1] > 0, "DataFrame has no columns"

    # Check if there are duplicated rows
    num_duplicados = df.duplicated().sum()
    assert num_duplicados == 0, f"There are {num_duplicados} duplicated rows"

    # Check if there are null values
    num_nulos = df.isnull().sum().sum()
    assert num_nulos == 0, f"There are {num_nulos} null values to handle"

    # This depends on how you want to work
    # False by default
    if exigir_numericas:
        columnas_no_numericas = []
        for columna in df.columns:
            # Stricter than dtype. This covers all numeric types
            if not pd.api.types.is_numeric_dtype(df[columna]):
                columnas_no_numericas.append(columna)
        assert len(columnas_no_numericas) == 0, f"There are non-numeric columns not yet encoded: {columnas_no_numericas}"

    # Checks if there are infinite values in numeric columns
    num_infinitos = np.isinf(df.select_dtypes(include="number")).sum().sum()
    assert num_infinitos == 0, f"There are {num_infinitos} infinite values"

    # There cannot be constant columns (columns with a single unique value)
    # Constant columns provide nothing. Zero variance, no information and likely slow things down
    columnas_constantes = []
    for columna in df.columns:
        if df[columna].nunique() == 1:
            columnas_constantes.append(columna)
    assert len(columnas_constantes) == 0, f"There are constant columns with no information: {columnas_constantes}"

    print("The DataFrame is ready for modeling")

    return

# --- FUNCIONES DE REGRESIÓN CATEGÓRICA---

def get_features_cat_regression( df: pd.DataFrame, target_col: str, pvalue: float = 0.05) -> list:
    """
    Returns all categorical variables significantly correlated to target in dataset.

    By default, level of significance is set to 0.05.

        Parameters:
            df (pd.DataFrame): dataframe where each column is a variable and each row is an observation
            target_col (str):  target variable to compare.
            pvalue (float): level of significance to use for test statistic.

        Returns:
            features (list): list of cualitative features significantly correlated to target.
    """

    if not isinstance(df, pd.DataFrame):
        print("Provided dataframe is not a pd.Dataframe.")
        return
    if target_col not in df.columns:
        print("Target column is not in dataframe.")
        return
    if not (isinstance(pvalue, float) and (0 < pvalue and pvalue < 1)):
        print("Set p-value must be a floating point number between 0 and 1.")
        return

    features = []
    df_tipos = tipifica_variables(df)

    if df_tipos.loc[df_tipos["nombre_variable"]==target_col, "tipo_sugerido"].isin(["Binaria", "Categórica"]).iloc[0]:
        print("Target variable must contain numerical data.")
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


def plot_features_cat_regression(df: pd.DataFrame, target_col: str = "", columns: list = [], pvalue: float = 0.05, with_individual_plot: bool = False) -> list:
    """
    Plot of target variable classified by given categorical column variables.

    If no columns are given, uses all cualitative variables in dataset.
    Default level of significance is set to 0.05.

        Parameters:
            df (pd.DataFrame): dataframe where each column is a variable and each row is an observation
            target_col (str):  target variable to compare.
            columns (list of strings): variables within df to use.
            with_individual_plot (bool): if True, draws one plot per categorical variable.
            pvalue (float): level of significance to use for test statistic.

        Returns:
            features (list): list of cualitative features used in plot.
    """
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
        print("Target variable must contain numerical data.")
        return

    # Lista de features según las columnas dadas
    if len(columns) == 0:
        features = df_tipos.loc[df_tipos.tipo_sugerido.isin(["Binaria", "Categórica"]), "nombre_variable"].tolist()
    else:
        cats_sign = get_features_cat_regression(df, target_col, pvalue)
        if cats_sign == None:
            return
        elif len(cats_sign) == 0:
            print("No categorical variables are significantly correlated to target for the given thresholds.")
            return
        features = list(set(cats_sign)&set(columns))
    
    if len(features)==0:    # Si la intersección es vacía
        print("None of the given variables are significantily related to target.")
        return
    
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

    return features

# --- FUNCIONES DE REGRESIÓN NUMÉRICA---
def get_features_num_regression(df: pd.DataFrame, target_col: str, umbral_corr: float, pvalue: float = 1.) -> list:
    """
    Returns all numerical variables significantly correlated to target in dataset.

    By default, level of significance is set to 1.

        Parameters:
            df (pd.DataFrame): dataframe where each column is a variable and each row is an observation
            target_col (str):  target variable to compare.
            umbral_corr (float): correlation threshold to consider a variable statistically significant.
            pvalue (float): level of significance to use for test statistic.

        Returns:
            features (list): list of cuantitative features significantly correlated to target.
    """
    # Comprobaciones
    if not isinstance(df, pd.DataFrame):
        print("Provided dataframe is not a pd.Dataframe.")
        return
    if target_col not in df.columns:
        print("Target column is not in dataframe.")
        return
    if not (isinstance(pvalue, float) and (0 < pvalue and pvalue <= 1)):    # sin permitir none
        print("Set p-value must be a float between 0 and 1.")
        return
    if not (isinstance(umbral_corr, float) and (0 <= umbral_corr and umbral_corr < 1)):
        print("Set correlation threshold must be a float between 0 and 1.")
        return

    features = []
    df_tipos = tipifica_variables(df)
    
    if df_tipos.loc[df_tipos["nombre_variable"]==target_col, "tipo_sugerido"].isin(["Binaria", "Categórica"]).iloc[0]:
        print("Target variable must contain numerical data.")
        return

    df_tipos.drop(df_tipos.loc[df_tipos["nombre_variable"]==target_col].index, inplace=True)  # Quitar el target

    for indep_var in df_tipos.loc[df_tipos.tipo_sugerido.isin(["Numérica Discreta", "Numérica Continua"]), "nombre_variable"]:
        stat, pval_stat = scipy.stats.pearsonr(df[indep_var], df[target_col])

        if pval_stat in [np.nan, np.inf]:
            print(f"Data size for target variable groups by {indep_var} is not sufficient.")
        elif (abs(stat) > umbral_corr) and (pval_stat <= pvalue):
            features.append(indep_var)
    
    return features

def plot_features_num_regression(df: pd.DataFrame, target_col: str = "", columns: list = [], umbral_corr: float = 0., pvalue: float = 1.) -> list:
    """
    Plot of target variable classified by given numerical column variables.

    If no columns are given, uses all cuantitative variables in dataset.
    Default level of significance is set to 1 and correlation threshold is 0 (all variables are considered).

        Parameters:
            df (pd.DataFrame): dataframe where each column is a variable and each row is an observation
            target_col (str):  target variable to compare.
            columns (list of strings): variables within df to use.
            umbral_corr (float): correlation threshold to consider a variable statistically significant.
            pvalue (float): level of significance to use for test statistic.

        Returns:
            features (list): list of cuantitative features used in plot.
    """
    # Comprobaciones
    if not isinstance(df, pd.DataFrame):
        print("Provided dataframe is not a pd.Dataframe.")
        return
    if target_col not in df.columns:
        print("Target column is not in dataframe.")
        return
    if not (isinstance(pvalue, float) and (0 < pvalue and pvalue <= 1)):    # sin permitir none
        print("Set p-value must be a float between 0 and 1.")
        return
    if not (isinstance(umbral_corr, float) and (0 <= umbral_corr and umbral_corr < 1)):
        print("Set correlation threshold must be a float between 0 and 1.")
        return

    features = []
    df_tipos = tipifica_variables(df)
    
    if df_tipos.loc[df_tipos["nombre_variable"]==target_col, "tipo_sugerido"].isin(["Binaria", "Categórica"]).iloc[0]:
        print("Target variable must contain numerical data.")

    df_tipos.drop(df_tipos.loc[df_tipos["nombre_variable"]==target_col].index, inplace=True)  # Quitar la target
    
    # Lista de features según las columnas dadas
    if len(columns) == 0:
        features = df_tipos.loc[df_tipos.tipo_sugerido.isin(["Numérica Discreta", "Numérica Continua"]), "nombre_variable"].tolist()
    else:
        vars_sign = get_features_num_regression(df, target_col, umbral_corr, pvalue)
        if vars_sign == None:
            return
        elif len(vars_sign) == 0:
            print("No numerical variables are significantly correlated to target for the given thresholds.")
            return
        features = list(set(vars_sign)&set(columns))
    
    if len(features)==0:
        print("None of the given variables are significantily related to target.")
        return

    # Plot de target según las features
    grupos = [features[i:i + 5] for i in range(0, len(features), 5)]

    for var_set in grupos:
        var_set.insert(0, target_col)
        sns.pairplot(df, vars=var_set)
        plt.show()

    return features
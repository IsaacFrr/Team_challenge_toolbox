import pytest
import pandas as pd
import numpy as np
from toolbox_ml.eda.core import describe_df, tipifica_variables, verificar_dataframe

# DESCRIBE_DF()
def test_describe_df_devuelve_dataframe():
    """Caso correcto: input válido -> retorna DataFrame."""
    df = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "z"]})
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame)

def test_describe_df_columnas_correctas():
    """El DataFrame resultado tiene exactamente las columnas esperadas."""
    df = pd.DataFrame({'a': [1, 2, 3]})
    resultado = describe_df(df)
    assert set(resultado.columns) == {
        'tipo', 'porcentaje_nulos', 'valores_unicos', 'porcentaje_cardinalidad'
    }

def test_describe_df_dataframe_vacio():
    """Caso límite: DataFrame sin filas → debe seguir funcionando."""
    df = pd.DataFrame({'a': [], 'b': []})
    resultado = describe_df(df)
    assert isinstance(resultado, pd.DataFrame)
    assert len(resultado) == 2

def test_describe_df_porcentaje_nulos_correcto():
    """Calcula correctamente el porcentaje de nulos."""
    df = pd.DataFrame({'a': [1, None, None, None]})
    resultado = describe_df(df)
    assert resultado.loc['a', 'porcentaje_nulos'] == pytest.approx(75.0, abs=0.01)

def test_describe_df_retorna_none_con_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert describe_df("esto no es un dataframe") is None
    assert describe_df([1, 2, 3]) is None




# TIPIFICA_VARIABLES()
def test_tipifica_variables_devuelve_dataframe():
    """Caso correcto: input válido -> retorna DataFrame clasificando cada columna"""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5],
        "b": ["x", "y", "z", "x", "y"]
        })
    resultado = tipifica_variables(df)
    # Se afirma que lo que le he pasado es un dataframe. En caso contrario, falla.
    assert isinstance(resultado, pd.DataFrame)
    # Se afirma que las columnas que ha devuelto son estas dos
    assert set(resultado.columns) == {"nombre_variable", "tipo_sugerido"}

def test_tipifica_variables_clasificacion_correcta():
    """La función asigna correctamente cada tipo"""
    df = pd.DataFrame({
        "bin": [0, 1, 1, 0, 1, 0, 0],
        "cat": [1, 2, 3, 1, 2, 3, 1],
        "cont": [1, 2, 3, 4, 5, 6, 7]
    })

    resultado = tipifica_variables(df, umbral_categoria=5, umbral_continua=80)

    # Dict para comprobar más rápido más abajo en vez de usar .loc o cualquier cosa.
    # Zip "empaqueta" las dos columnas y los "vectoriza" juntos. Los une por parejas 
    tipos_variables = dict(zip(resultado["nombre_variable"], resultado["tipo_sugerido"]))

    assert tipos_variables["bin"] == "Binaria"
    assert tipos_variables["cat"] == "Categórica"
    assert tipos_variables["cont"] == "Numérica Continua"

def test_tipifica_variables_retorna_none_con_input_invalido():
    """Caso de error: input no es DataFrame → retorna None."""
    assert tipifica_variables("esto no es un df") is None
    assert tipifica_variables([1, 2, 3]) is None

def test_tipifica_variables_retorna_none_con_umbrales_invalidos():
    """Caso de error: umbrales inválidos → retorna None."""
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert tipifica_variables(df, umbral_categoria=-1) is None
    assert tipifica_variables(df, umbral_continua=101) is None


# VERIFICAR DATAFRAME

def test_verificar_dataframe_caso_correcto():
    """Un DataFrame limpio no debe lanzar ningún error."""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [5, 6, 7, 8]
    })
    verificar_dataframe(df)

def test_verificar_dataframe_vacio():
    """Un DataFrame vacío debería lanzar error"""
    df = pd.DataFrame({})
    with pytest.raises(AssertionError):
        verificar_dataframe(df)
    df_no_columns = pd.DataFrame({"a": []})
    with pytest.raises(AssertionError):
        verificar_dataframe(df_no_columns)

def test_verificar_dataframe_duplicados():
    """Un dataframe con nulos debería lanzar error"""
    df = pd.DataFrame({
        "a": [1, 2, 1],
        "b": [1, 2, 1],
        "c": [1, 2, 1]
    })
    with pytest.raises(AssertionError):
        verificar_dataframe(df)

def test_verificar_dataframe_nulos():
    """Un dataframe con nulos debería lanzar error"""
    df = pd.DataFrame({
        "a": [1, None, np.nan],
        "b": [None, np.nan, 3],
    })
    with pytest.raises(AssertionError):
        verificar_dataframe(df)

def test_verificar_dataframe_infinitos():
    """Un dataframe con infinitos debería lanzar error"""
    df = pd.DataFrame({
        "a": [1, 2, 3],
        "b": [0, 0, 1],
    })
    # Esto va a guardar un infinito porque divide entre cero y guarda eso en una fila de la
    # nueva columna "c"
    df["c"] = df["a"] / df["b"]
    with pytest.raises(AssertionError):
        verificar_dataframe(df)

def test_verificar_dataframe_constante():
    """Un dataframe con una columna constante debería lanzar error"""
    df = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [1, 1, 1, 1, 1, 1]
    })
    with pytest.raises(AssertionError):
        verificar_dataframe(df)
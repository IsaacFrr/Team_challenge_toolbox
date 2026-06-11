# Team Challenge Toolbox

Librería Python de utilidades para análisis exploratorio de datos (EDA) y preprocesamiento orientada a proyectos de Machine Learning.

## Instalación

```bash
git clone https://github.com/IsaacFrr/Team_challenge_toolbox.git
cd Team_challenge_toolbox
pip install -e .
```

O instalando las dependencias directamente:

```bash
pip install -r requirements.txt
```

## Uso rápido

```python
from toolbox_ml.eda.core import (
    describe_df,
    tipifica_variables,
    verificar_dataframe,
    get_features_cat_regression,
    plot_features_cat_regression,
)
```

## Funciones disponibles

### `describe_df(df)`

Devuelve un DataFrame con información resumida de cada columna del DataFrame de entrada.

**Parámetros:**
- `df` *(pd.DataFrame)*: DataFrame a analizar.

- **Retorna:** `pd.DataFrame` con las columnas `tipo`, `porcentaje_nulos`, `valores_unicos`, `porcentaje_cardinalidad`. Retorna `None` si el input no es un DataFrame.

- ---

### `tipifica_variables(df, umbral_categoria=15, umbral_continua=80.0)`

Clasifica cada columna del DataFrame en un tipo sugerido según su cardinalidad.

**Parámetros:**
- `df` *(pd.DataFrame)*: DataFrame a analizar.
- - `umbral_categoria` *(int)*: Umbral de valores únicos por debajo del cual una variable se considera categórica. Por defecto `15`.
  - - `umbral_continua` *(float)*: Porcentaje de cardinalidad a partir del cual una variable numérica se considera continua. Por defecto `80.0`.
   
    - **Retorna:** `pd.DataFrame` con columnas `nombre_variable` y `tipo_sugerido`. Los tipos posibles son: `Binaria`, `Categórica`, `Numérica Continua`, `Numérica Discreta`. Retorna `None` si los argumentos no son válidos.
   
    - ---

    ### `verificar_dataframe(df, exigir_numericas=False)`

    Verifica que un DataFrame esté listo para modelar. Comprueba que no esté vacío, no tenga filas duplicadas, no tenga nulos, no tenga valores infinitos y no tenga columnas constantes.

    **Parámetros:**
    - `df` *(pd.DataFrame)*: DataFrame a verificar.
    - - `exigir_numericas` *(bool)*: Si es `True`, comprueba además que todas las columnas son numéricas. Por defecto `False`.
     
      - **Retorna:** `None`. Lanza `AssertionError` con mensaje descriptivo si alguna comprobación falla. Imprime un mensaje de confirmación si el DataFrame está correcto.
     
      - ---

      ### `get_features_cat_regression(df, target_col, pvalue=0.05)`

      Devuelve las columnas categóricas (Binarias o Categóricas) del DataFrame que tienen una relación estadísticamente significativa con la variable target numérica continua.

      Utiliza el test de Mann-Whitney U para variables binarias y ANOVA (F de Snedecor o Kruskal-Wallis para muestras pequeñas) para variables categóricas.

      **Parámetros:**
      - `df` *(pd.DataFrame)*: DataFrame de entrada.
      - - `target_col` *(str)*: Nombre de la columna target. Debe ser de tipo numérico continuo.
        - - `pvalue` *(float)*: Nivel de significación estadística. Por defecto `0.05`.
         
          - **Retorna:** Lista de nombres de columnas con relación significativa con el target. Retorna `None` si los argumentos no son válidos.
         
          - ---

          ### `plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False)`

          Genera gráficos de distribución del target para cada feature categórica significativa. Llama internamente a `get_features_cat_regression` para filtrar las variables relevantes.

          **Parámetros:**
          - `df` *(pd.DataFrame)*: DataFrame de entrada.
          - - `target_col` *(str)*: Nombre de la columna target numérica continua.
            - - `columns` *(list)*: Lista de columnas a considerar. Si está vacía, se usan todas las categóricas del DataFrame.
              - - `pvalue` *(float)*: Nivel de significación estadística. Por defecto `0.05`.
                - - `with_individual_plot` *(bool)*: Si es `True`, genera un gráfico independiente por cada feature. Si es `False`, genera un grid de subplots. Por defecto `False`.
                 
                  - **Retorna:** Lista de columnas graficadas.
                 
                  - ---

                  ## Tests

                  ```bash
                  pytest tests/
                  ```

                  Los tests cubren las funciones `describe_df`, `tipifica_variables` y `verificar_dataframe`.

                  ## Estado del proyecto

                  | Función | Implementada | Tests | Demo notebook |
                  |---|---|---|---|
                  | `describe_df` | ✅ | ✅ | ✅ |
                  | `tipifica_variables` | ✅ | ✅ | ✅ |
                  | `verificar_dataframe` | ✅ | ✅ | ❌ |
                  | `get_features_cat_regression` | ✅ | ❌ | ✅ |
                  | `plot_features_cat_regression` | ✅ | ❌ | ✅ |
                  | `get_features_num_regression` | 🚧 En desarrollo | ❌ | ❌ |

                  ## Dependencias

                  - `pandas`
                  - - `numpy`
                    - - `scipy`
                      - - `matplotlib`
                        - - `seaborn`

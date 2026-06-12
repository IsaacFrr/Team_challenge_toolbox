# Team Challenge Toolbox (`toolbox_ml`)

Librería de Python con herramientas de análisis exploratorio (EDA) y selección de
features para Machine Learning, desarrollada como Team Challenge del Bootcamp de
Data Science de The Bridge.

El paquete agrupa funciones reutilizables para describir un DataFrame, tipificar
sus variables y seleccionar/visualizar features relevantes frente a un target
numérico en problemas de regresión.

## Instalación

```bash
git clone https://github.com/IsaacFrr/Team_challenge_toolbox.git
cd Team_challenge_toolbox

# (recomendado) crear y activar un entorno virtual
python -m venv venv
source venv/bin/activate      # En Windows: venv\Scripts\activate

# instalar dependencias e instalar el paquete en modo editable
pip install -r requirements.txt
pip install -e .
```

Una vez instalado, las funciones se importan desde cualquier notebook o script:

```python
from toolbox_ml.eda.core import (
    describe_df,
    tipifica_variables,
    verificar_dataframe,
    get_features_num_regression,
    plot_features_num_regression,
    get_features_cat_regression,
    plot_features_cat_regression,
)
```

## Funciones disponibles

Todas las funciones validan la entrada: si reciben algo que no es un DataFrame
(o argumentos fuera de rango) muestran el motivo por pantalla y devuelven `None`
en lugar de lanzar una excepción.

### `describe_df(df)`

Devuelve un resumen por columna del DataFrame: tipo de dato, porcentaje de nulos,
número de valores únicos y porcentaje de cardinalidad.

```python
import seaborn as sns
from toolbox_ml.eda.core import describe_df

df = sns.load_dataset("diamonds")
describe_df(df)
```

### `tipifica_variables(df, umbral_categoria=15, umbral_continua=80.0)`

Clasifica automáticamente cada variable como `"Binaria"` (cardinalidad = 2),
`"Categórica"`, `"Numérica Discreta"` o `"Numérica Continua"`, según su
cardinalidad y su porcentaje de cardinalidad.

```python
from toolbox_ml.eda.core import tipifica_variables

tipifica_variables(df, umbral_categoria=15, umbral_continua=80.0)
```

### `verificar_dataframe(df, exigir_numericas=False)`

Chequeo de calidad previo al modelado. Comprueba que el DataFrame no esté vacío
y que no tenga filas duplicadas, valores nulos, valores infinitos ni columnas
constantes. Con `exigir_numericas=True` comprueba además que todas las columnas
sean numéricas. Lanza un `AssertionError` con un mensaje claro si algo falla.

```python
from toolbox_ml.eda.core import verificar_dataframe

verificar_dataframe(df)                         # chequeo general
verificar_dataframe(df, exigir_numericas=True)  # además exige columnas numéricas
```

### `get_features_num_regression(df, target_col, umbral_corr, pvalue=1.0)`

Devuelve la lista de columnas numéricas cuya correlación de Pearson con
`target_col` supere `umbral_corr` en valor absoluto. Si se pasa `pvalue`, filtra
además por significación estadística (p-valor del test de Pearson menor o igual
que el valor dado).

```python
from toolbox_ml.eda.core import get_features_num_regression

get_features_num_regression(df, target_col="price", umbral_corr=0.8)
# -> ['carat', 'x', 'y', 'z']
```

### `plot_features_num_regression(df, target_col="", columns=[], umbral_corr=0.0, pvalue=1.0)`

Pinta un `pairplot` de `target_col` frente a las columnas numéricas que cumplen
los criterios de correlación (misma lógica que `get_features_num_regression`).
Si hay más de 5 columnas las reparte en grupos de 5, incluyendo siempre el target
en cada grupo. Devuelve la lista de columnas representadas.

```python
from toolbox_ml.eda.core import plot_features_num_regression

plot_features_num_regression(df, target_col="price", umbral_corr=0.8)
```

### `get_features_cat_regression(df, target_col, pvalue=0.05)`

Devuelve la lista de columnas categóricas cuya relación con `target_col`
(numérico) es estadísticamente significativa. Elige el test automáticamente:
**Mann-Whitney U** si la categórica tiene 2 grupos, **ANOVA de un factor** si
tiene más de 2 (y **Kruskal-Wallis** como alternativa robusta cuando algún grupo
es muy pequeño).

```python
from toolbox_ml.eda.core import get_features_cat_regression

get_features_cat_regression(df, target_col="price", pvalue=0.05)
```

### `plot_features_cat_regression(df, target_col="", columns=[], pvalue=0.05, with_individual_plot=False)`

Pinta histogramas de `target_col` agrupados por cada variable categórica que
supere el test estadístico. Con `with_individual_plot=True` genera una figura por
variable; con `False` (por defecto) las agrupa en subplots. Devuelve la lista de
columnas representadas.

```python
from toolbox_ml.eda.core import plot_features_cat_regression

plot_features_cat_regression(df, target_col="price", pvalue=0.05)
```

## Tests

Los tests están escritos con `pytest`. Desde la raíz del repositorio, con el
entorno virtual activado:

```bash
pytest tests/ -v
```

## Equipo y reparto de tareas

Proyecto desarrollado por el equipo del TC 5 del Bootcamp de Data Science
(Online Madrid, Sept 2025 – Abr 2026) de The Bridge.

| Integrante | Rol | Responsabilidad |
| --- | --- | --- |
| Isaac | Scrum Master | Setup del repositorio, `setup.py`, `__init__.py`, integración final y notebook de demostración |
| Gabriel | Desarrollador | `describe_df`, `tipifica_variables` (+ `verificar_dataframe`) y sus tests |
| Ana | Desarrolladora | `get/plot_features_num_regression`, `get/plot_features_cat_regression` y sus tests |

## Licencia

MIT

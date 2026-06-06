# Team Challenge Toolbox

Libreria Python de herramientas de EDA y Machine Learning, desarrollada como Team Challenge del Bootcamp de Data Science de The Bridge.

## Instalacion

Clona el repositorio e instala el paquete en modo desarrollo:

```bash
git clone https://github.com/IsaacFrr/Team_challenge_toolbox.git
cd Team_challenge_toolbox
pip install -e .
```

## Funciones disponibles

### describe_df(df)

Recibe un DataFrame y devuelve un resumen por columna: tipo de dato, porcentaje de nulos, valores unicos y porcentaje de cardinalidad.

### tipifica_variables(df, umbral_categoria=15, umbral_continua=80.0)

Clasifica automaticamente las variables: Binaria (cardinalidad=2), Categorica, Numerica Discreta o Numerica Continua.

### get_features_num_regression(df, target_col, umbral_corr, pvalue=None)

En desarrollo. Devuelve columnas numericas con correlacion superior a umbral_corr respecto al target.

### get_features_cat_regression(df, target_col, pvalue=0.05)

En desarrollo. Devuelve columnas categoricas cuya relacion con el target supera el test estadistico.

## Tests

```bash
pytest tests/
```

## Equipo

Desarrollado por el equipo del Bootcamp DS Online Madrid (Sept 2025 - Abr 2026) de The Bridge.

## Licencia

MIT

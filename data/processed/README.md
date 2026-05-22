# `data/processed/`

Este directorio se **genera automáticamente** por el pipeline. Para facilitar la revisión sin configurar ODBC/Supabase, la subcarpeta `powerbi/` sí se versiona como mini data lake local en Parquet.

## Artefactos esperados tras ejecutar el pipeline

| Archivo | Generado por | Descripción |
|---|---|---|
| `abt.parquet` | [0_Master_Pipeline.ipynb](../../0_Master_Pipeline.ipynb), [scripts/load_powerbi_predictions.py](../../scripts/load_powerbi_predictions.py) y [src/consolidacion.py](../../src/consolidacion.py) | Analytical Base Table consolidada con las 42 features del modelo y la variable objetivo `es_moroso`. Insumo directo del entrenamiento. |

## Mini data lake Power BI

La carpeta `data/processed/powerbi/` contiene tablas Parquet listas para conectarse desde Power BI con el conector **Folder** o **Parquet**, sin credenciales de base de datos:

| Archivo | Equivalente lógico | Uso recomendado |
|---|---|---|
| `abt_analitica_riesgo.parquet` | `abt_analitica_riesgo` | Tabla analítica completa para auditoría y exploración. |
| `predicciones_riesgo.parquet` | `predicciones_riesgo` | Salida del modelo con probabilidad y predicción de mora. |
| `fact_creditos.parquet` | `fact_creditos` | Hechos de crédito para métricas y medidas DAX. |
| `dim_cliente.parquet` | `dim_cliente` | Dimensión demográfica y socioeconómica de clientes. |
| `dim_producto_credito.parquet` | `dim_producto_credito` | Catálogo de productos de crédito. |
| `dim_tiempo.parquet` | `dim_tiempo` | Calendario local para relaciones por fecha de desembolso. |
| `perfil_descriptivo_cliente.parquet` | `perfil_descriptivo_cliente` | Perfil agregado por cliente para vistas descriptivas. |
| `manifest.json` | Diccionario de carga | Conteo de filas generado en la última ejecución. |

## Cómo regenerarlo

```bash
jupyter nbconvert --to notebook --execute --inplace 0_Master_Pipeline.ipynb
```

> El directorio se crea con `mkdir(parents=True, exist_ok=True)` desde `src/config.py` (`settings.DATA_PROCESSED_DIR`). No requiere creación manual.

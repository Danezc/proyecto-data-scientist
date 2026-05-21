# Evaluación Técnica: Procesamiento, Arquitectura y Modelado de Riesgo

Este documento presenta una auditoría detallada de la arquitectura del proyecto, la calidad del procesamiento de datos y la idoneidad del modelo LightGBM para el caso de negocio **TumiPay**.

---

## 1. Análisis y Procesamiento de Datos (ETL)

El procesamiento de datos cumple con los estándares más estrictos de un proyecto de producción de nivel empresarial.

### Puntos Fuertes:
* **Estandarización Semántica**: La normalización de ciudades mediante diccionario robusto (`src/procesamiento.py`) resuelve inconsistencias típicas de formularios web (ej: *'bogot'*, *'medellin '* con espacios, *'barranquila'*).
* **Tratamiento de Nulos Metódico**: Imputar ingresos mensuales y score externo usando medianas previene la distorsión por valores atípicos (outliers) y evita el sesgo en el modelo.
* **Cálculo del Target Libre de Sesgo**: La redefinición del target usando `fecha_vencimiento <= cutoff_date` en lugar de `fecha_pago` fue crítica. El enfoque anterior eliminaba a los clientes en mora activa que nunca pagaron (pues su `fecha_pago` es nula), lo cual es un error metodológico grave en riesgo de crédito.
* **Ingeniería de Características sin Fuga Temporal**: Al agregar los eventos de la App (`eventos_app.csv`), la restricción `fecha_evento < fecha_desembolso` garantiza que el modelo use únicamente el historial digital del usuario *antes* de recibir el crédito. Esto previene el **Data Leakage** y emula fielmente el proceso de originación real.

---

## 2. Arquitectura del Software y Producción

El diseño del proyecto demuestra madurez en la separación de conceptos y la preparación para MLOps.

```
Clean Architecture:
[Datos Raw] ──> [procesamiento.py] ──> [consolidacion.py (ABT)]
                                             │
   ┌─────────────────────────────────────────┴────────────────────────┐
   ▼                                                                  ▼
[train_mora.py (ML)]                                       [sql/transformaciones.sql]
   │                                                                  │
   ▼                                                                  ▼
[Pickle / API FastAPI]                                     [Tablas PostgreSQL]
                                                                      │
                                                                      ▼
                                                           [Power BI Dashboard]
```

### Puntos Fuertes:
* **Principio de Responsabilidad Única (SRP)**: Cada módulo en `src/` realiza una tarea específica (ingesta, limpieza, ABT, entrenamiento, API).
* **Modularidad del Pipeline**: El pipeline completo se orquesta limpiamente desde `0_Master_Pipeline.ipynb` y cuenta con un DAG de Airflow funcional (`dags/tumipay_pipeline_dag.py`) que reutiliza la lógica del código de `src/`, asegurando reproducibilidad y consistencia entre desarrollo y producción.
* **Robustez en la Inferencia (FastAPI)**: La alineación dinámica de categorías de LightGBM en el endpoint `/predict` mediante `pandas_categorical` evita caídas del sistema cuando la petición web carece de ciertas variables categóricas o introduce etiquetas no vistas.

---

## 3. Modelo LightGBM: Evaluación y Alternativas

El uso de **LightGBM** es una excelente elección técnica, pero requiere ciertas precauciones debido al volumen del dataset (1,527 registros).

### Ventajas de LightGBM en este Caso:
1. **Manejo Nativo de Categóricas**: Acepta variables como `ciudad` y `ocupacion` directamente como tipo `category` de pandas. Esto evita la necesidad de aplicar One-Hot Encoding, lo cual incrementaría drásticamente la dimensionalidad del dataset (generando columnas dispersas/sparse).
2. **Tratamiento de Nulos**: LightGBM maneja valores faltantes de manera automática e inteligente durante los splits de árboles.
3. **Métrica Ponderada**: El uso de `class_weight='balanced'` es fundamental dada la proporción del target (~27% de morosidad).

### Limitaciones de LightGBM en este Contexto:
* **Tamaño del Dataset**: LightGBM está diseñado para conjuntos de datos masivos. Con solo 1,527 filas, los árboles profundos pueden sobreajustar (overfitting) con facilidad si no se regulariza de forma agresiva. El ROC-AUC del set de pruebas (0.7476) refleja un modelo robusto y real tras eliminar la fuga de datos de `estado_credito_operativo`, pero tiene margen de mejora.

---

## 4. Comparativa de Modelos Alternativos

Para la sustentación de la prueba técnica, es ideal comparar LightGBM con alternativas:

| Algoritmo | Idoneidad para TumiPay | Ventajas | Desventajas / Retos |
|---|---|---|---|
| **CatBoost** | ⭐⭐⭐⭐⭐ *(Altamente recomendado)* | Diseñado específicamente para optimizar variables categóricas (usa target encoding ordenado). Evita el sobreajuste en datasets pequeños. | Entrenamiento ligeramente más lento (no es factor crítico para 1.5k filas). |
| **XGBoost** | ⭐⭐⭐⭐ | Excelente rendimiento predictivo, regularización L1 y L2 integrada para evitar sobreajuste. | Históricamente requiere pre-procesamiento manual para categóricas. |
| **Regresión Logística + WoE** | ⭐⭐⭐⭐ *(Estándar Bancario)* | Explainability total (Scorecard). Cumple con regulaciones bancarias tradicionales de Basilea. | Requiere discretizar (binnear) todas las variables manualmente. Pierde interacciones complejas no lineales. |
| **Random Forest** | ⭐⭐⭐ | Estable y difícil de sobreajustar. | No maneja nulos ni categóricas de forma nativa. Mayor costo computacional en inferencia. |

---

## 5. Próximos Pasos Recomendados para la Sustentación

Para elevar la calificación técnica del proyecto al máximo nivel, se sugieren los siguientes ajustes prácticos:

1. **Validación Cruzada (K-Fold CV)**:
   * Reemplazar la partición simple `train_test_split` de 80/20 por un esquema de **Stratified 5-Fold Cross-Validation**. Esto proporcionará una estimación del ROC-AUC mucho más estable (ej. `0.748 ± 0.015`) y evitará la variabilidad en los reportes de métricas.
2. **Regularización en LightGBM**:
   * Ajustar hiperparámetros limitando la complejidad de los árboles para prevenir overfitting en este tamaño de datos:
     ```python
     model = LGBMClassifier(
         objective='binary',
         class_weight='balanced',
         n_estimators=80,          # Reducido para evitar sobreajuste
         learning_rate=0.03,        # Ritmo de aprendizaje más suave
         max_depth=4,               # Árboles poco profundos
         num_leaves=15,             # Hojas limitadas
         min_child_samples=30,      # Evita hojas con pocos clientes
         random_state=42
     )
     ```
3. **Implementación de una Torta de Modelos (CatBoost vs LightGBM)**:
   * Entrenar en paralelo un clasificador `CatBoostClassifier` y comparar sus métricas de test mediante validación cruzada. Presentar esta comparación en la sustentación demostrará un conocimiento sobresaliente de ingeniería de Machine Learning.

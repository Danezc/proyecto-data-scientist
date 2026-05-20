from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Suponiendo que Airflow monta el repositorio en /opt/airflow/dags/proyecto-data-scientist
# Ajustamos el sys.path para poder embeber nuestras clases de 'src'
dags_folder = Path(__file__).resolve().parent
project_root = dags_folder.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def _run_ingesta_limpieza(**kwargs):
    from src.ingesta import DataIngestor
    from src.procesamiento import DataCleaner
    from src.config import settings
    
    ingestor = DataIngestor(data_dir=str(settings.DATA_RAW_DIR))
    dfs = ingestor.load_all()
    
    cleaner = DataCleaner()
    df_clientes = cleaner.clean_clientes(dfs['clientes'])
    df_creditos = cleaner.clean_creditos(dfs['creditos'])
    
    # Podríamos pasar las rutas/archivos por XCom, pero 
    # las guardaremos como un checkpoint temporal.
    df_clientes.to_parquet(settings.DATA_PROCESSED_DIR / 'clientes_clean.parquet', index=False)
    df_creditos.to_parquet(settings.DATA_PROCESSED_DIR / 'creditos_clean.parquet', index=False)
    dfs['pagos'].to_parquet(settings.DATA_PROCESSED_DIR / 'pagos_raw.parquet', index=False)
    
def _run_consolidacion(**kwargs):
    import pandas as pd
    from src.consolidacion import DataConsolidator
    from src.config import settings
    
    df_clientes = pd.read_parquet(settings.DATA_PROCESSED_DIR / 'clientes_clean.parquet')
    df_creditos = pd.read_parquet(settings.DATA_PROCESSED_DIR / 'creditos_clean.parquet')
    df_pagos = pd.read_parquet(settings.DATA_PROCESSED_DIR / 'pagos_raw.parquet')
    
    consolidator = DataConsolidator()
    abt = consolidator.build_analytical_base_table(df_clientes, df_creditos, df_pagos)
    
    abt.to_parquet(settings.DATA_PROCESSED_DIR / 'abt.parquet', index=False)

def _run_almacenamiento_y_entrenamiento(**kwargs):
    import pandas as pd
    from sqlalchemy import create_engine
    from src.train_mora import ModeloRiesgo
    from src.config import settings
    
    abt = pd.read_parquet(settings.DATA_PROCESSED_DIR / 'abt.parquet')
    
    modelo = ModeloRiesgo(abt)
    modelo.preparar_datos()
    metrics = modelo.entrenar()
    
    # Persistir para PBI
    engine = create_engine(settings.DATABASE_URL)
    X_full = modelo.preprocesador.transform(modelo.X)
    abt['probabilidad_mora'] = modelo.modelo.predict_proba(X_full)[:, 1]
    abt['prediccion_mora'] = modelo.modelo.predict(X_full)
    
    abt.to_sql('predicciones_riesgo', engine, if_exists='replace', index=False)
    print(f"Métricas alcanzadas: {metrics}")


default_args = {
    'owner': 'data_science_team',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'tumipay_riesgo_pipeline_diario',
    default_args=default_args,
    description='Pipeline orquestado de Machine Learning para Tumipay',
    schedule_interval='@daily',
    start_date=datetime(2026, 5, 20),
    catchup=False,
    tags=['mlops', 'riesgo', 'tumipay'],
) as dag:

    t1 = PythonOperator(
        task_id='ingesta_y_limpieza',
        python_callable=_run_ingesta_limpieza,
    )

    t2 = PythonOperator(
        task_id='consolidacion_abt',
        python_callable=_run_consolidacion,
    )

    t3 = PythonOperator(
        task_id='entrenamiento_lightgbm_y_escritura_bd',
        python_callable=_run_almacenamiento_y_entrenamiento,
    )

    # Definir dependencias
    t1 >> t2 >> t3

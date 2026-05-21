from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Ajustamos el sys.path para poder embeber nuestras clases de 'src'
dags_folder = Path(__file__).resolve().parent
project_root = dags_folder.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def _run_ingesta_limpieza(**kwargs):
    from src.ingesta import DataIngestor
    from src.procesamiento import DataCleaner
    from src.config import settings
    from src.db_utils import create_supabase_engine, upsert_dataframe
    
    # Crear carpeta processed si no existe
    settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    ingestor = DataIngestor(raw_dir=settings.DATA_RAW_DIR)
    dfs = ingestor.load_all()
    
    cleaner = DataCleaner()
    df_clientes = cleaner.clean_clientes(dfs['clientes'])
    df_creditos = cleaner.clean_creditos(dfs['creditos'])
    df_pagos    = cleaner.clean_pagos(dfs['pagos'])
    df_eventos  = dfs['eventos']  # no requiere limpieza especial previa
    
    # Guardar checkpoints en parquet
    df_clientes.to_parquet(settings.DATA_PROCESSED_DIR / 'clientes_clean.parquet', index=False)
    df_creditos.to_parquet(settings.DATA_PROCESSED_DIR / 'creditos_clean.parquet', index=False)
    df_pagos.to_parquet(settings.DATA_PROCESSED_DIR / 'pagos_clean.parquet', index=False)
    df_eventos.to_parquet(settings.DATA_PROCESSED_DIR / 'eventos_clean.parquet', index=False)
    
    engine = create_supabase_engine(settings.DATABASE_URL)
    upsert_dataframe(df_clientes, engine, 'raw_clientes', ['cliente_id'])
    upsert_dataframe(df_creditos, engine, 'raw_creditos', ['credito_id'])
    upsert_dataframe(df_pagos, engine, 'raw_pagos', ['pago_id'])
    upsert_dataframe(df_eventos, engine, 'raw_eventos_app', ['evento_id'])
    print("ETL Ingesta y Limpieza completada con éxito.")

def _run_consolidacion(**kwargs):
    import pandas as pd
    from src.consolidacion import DataConsolidator
    from src.config import settings
    
    df_clientes = pd.read_parquet(settings.DATA_PROCESSED_DIR / 'clientes_clean.parquet')
    df_creditos = pd.read_parquet(settings.DATA_PROCESSED_DIR / 'creditos_clean.parquet')
    df_pagos = pd.read_parquet(settings.DATA_PROCESSED_DIR / 'pagos_clean.parquet')
    df_eventos = pd.read_parquet(settings.DATA_PROCESSED_DIR / 'eventos_clean.parquet')
    
    consolidator = DataConsolidator(cutoff_date=settings.CUTOFF_DATE)
    abt = consolidator.build_analytical_base_table(
        df_clientes=df_clientes, 
        df_creditos=df_creditos, 
        df_pagos=df_pagos,
        df_eventos=df_eventos
    )
    
    abt.to_parquet(settings.DATA_PROCESSED_DIR / 'abt.parquet', index=False)
    print("Consolidación de la ABT finalizada.")

def _run_almacenamiento_y_entrenamiento(**kwargs):
    import pandas as pd
    import joblib
    from src.train_mora import MoraModelTrainer
    from src.config import settings
    from src.db_utils import create_supabase_engine, upsert_dataframe
    
    abt_path = settings.DATA_PROCESSED_DIR / 'abt.parquet'
    model_path = settings.MODELS_DIR / 'modelo_mora.pkl'
    
    # 1. Entrenar el modelo LightGBM usando el pipeline centralizado
    trainer = MoraModelTrainer(data_path=abt_path, model_output_path=model_path)
    trainer.run_pipeline()
    
    # 2. Cargar modelo para predecir probabilidades sobre todo el set
    model_artifact = joblib.load(model_path)
    model = model_artifact['model']
    features = model_artifact['features']
    
    abt = pd.read_parquet(abt_path)
    X_full = abt[features].copy()
    
    # Asegurar codificación de categorías
    for col in X_full.select_dtypes(include=['object', 'str']).columns:
        X_full[col] = X_full[col].astype('category')
        
    abt['probabilidad_mora'] = model.predict_proba(X_full)[:, 1]
    abt['prediccion_mora'] = model.predict(X_full)
    
    # 3. Guardar en Supabase sin recrear la tabla consumida por Power BI
    engine = create_supabase_engine(settings.DATABASE_URL)
    upsert_dataframe(abt, engine, 'predicciones_riesgo', ['credito_id'])
    print("Entrenamiento y persistencia en base de datos completado.")


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

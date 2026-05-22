import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import settings  # noqa: E402
from src.consolidacion import DataConsolidator  # noqa: E402
from src.db_utils import create_supabase_engine, get_database_host, table_counts, upsert_dataframe  # noqa: E402
from src.ingesta import DataIngestor  # noqa: E402
from src.powerbi_datalake import export_powerbi_datalake  # noqa: E402
from src.procesamiento import DataCleaner  # noqa: E402
from src.train_mora import MoraModelTrainer  # noqa: E402


def main() -> None:
    engine = create_supabase_engine(settings.DATABASE_URL)
    print(f"Base de datos activa: {get_database_host(settings.DATABASE_URL)}")

    ingestor = DataIngestor(raw_dir=settings.DATA_RAW_DIR)
    dfs = ingestor.load_all()

    cleaner = DataCleaner()
    df_clientes = cleaner.clean_clientes(dfs["clientes"])
    df_creditos = cleaner.clean_creditos(dfs["creditos"])
    df_pagos = cleaner.clean_pagos(dfs["pagos"])
    df_eventos = dfs["eventos"].copy()

    settings.DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    consolidator = DataConsolidator(cutoff_date=settings.CUTOFF_DATE)
    abt = consolidator.build_analytical_base_table(df_clientes, df_creditos, df_pagos, df_eventos)
    abt.to_csv(settings.DATA_PROCESSED_DIR / "abt.csv", index=False)

    trainer = MoraModelTrainer(
        data_path=settings.DATA_PROCESSED_DIR / "abt.csv",
        model_output_path=settings.MODELS_DIR / "modelo_mora.pkl",
    )
    X_train, X_test, y_train, y_test = trainer.train_test_split_custom(abt)
    model = trainer.train_model(X_train, y_train)
    trainer.evaluate_model(model, X_test, y_test)
    trainer.export_model(model, list(X_train.columns))

    X_full = abt[X_train.columns].copy()
    for col in X_full.select_dtypes(include=["object", "str"]).columns:
        X_full[col] = X_full[col].astype("category")

    abt["probabilidad_mora"] = model.predict_proba(X_full)[:, 1]
    abt["prediccion_mora"] = model.predict(X_full)

    lake_counts = export_powerbi_datalake(
        abt=abt.drop(columns=["probabilidad_mora", "prediccion_mora"]),
        predicciones=abt,
        output_dir=settings.DATA_PROCESSED_DIR / "powerbi",
    )
    print("Mini data lake Power BI generado en data/processed/powerbi:")
    for table_name, row_count in lake_counts.items():
        print(f"  {table_name}: {row_count} filas")

    affected = upsert_dataframe(abt, engine, "predicciones_riesgo", ["credito_id"])
    counts = table_counts(engine, ["predicciones_riesgo"])
    print(f"predicciones_riesgo: {len(abt)} filas fuente, {affected} filas insertadas/actualizadas")
    print(f"Filas actuales en Supabase: {counts['predicciones_riesgo']}")


if __name__ == "__main__":
    main()

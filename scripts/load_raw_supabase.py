import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.config import settings  # noqa: E402
from src.db_utils import create_supabase_engine, get_database_host, table_counts, upsert_dataframe  # noqa: E402
from src.ingesta import DataIngestor  # noqa: E402
from src.procesamiento import DataCleaner  # noqa: E402


def main() -> None:
    engine = create_supabase_engine(settings.DATABASE_URL)
    print(f"Base de datos activa: {get_database_host(settings.DATABASE_URL)}")

    ingestor = DataIngestor(raw_dir=settings.DATA_RAW_DIR)
    dfs = ingestor.load_all()

    cleaner = DataCleaner()
    raw_loads = [
        ("raw_clientes", cleaner.clean_clientes(dfs["clientes"]), ["cliente_id"]),
        ("raw_creditos", cleaner.clean_creditos(dfs["creditos"]), ["credito_id"]),
        ("raw_pagos", cleaner.clean_pagos(dfs["pagos"]), ["pago_id"]),
        ("raw_eventos_app", dfs["eventos"].copy(), ["evento_id"]),
    ]

    for table_name, dataframe, keys in raw_loads:
        affected = upsert_dataframe(dataframe, engine, table_name, keys)
        print(f"{table_name}: {len(dataframe)} filas fuente, {affected} filas insertadas/actualizadas")

    counts = table_counts(engine, [table for table, _, _ in raw_loads])
    print(pd.DataFrame(counts.items(), columns=["tabla", "filas_en_supabase"]).to_string(index=False))


if __name__ == "__main__":
    main()
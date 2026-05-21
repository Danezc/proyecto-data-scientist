from __future__ import annotations

from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import MetaData, Table, create_engine, inspect
from sqlalchemy.dialects.postgresql import insert


LOCAL_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0"}


def create_supabase_engine(database_url: str):
    parsed = urlparse(database_url)
    if not database_url or not parsed.hostname:
        raise ValueError("DATABASE_URL no está configurado.")
    if parsed.hostname in LOCAL_HOSTS:
        raise ValueError(
            f"DATABASE_URL apunta a '{parsed.hostname}'. La carga a Docker/local está bloqueada."
        )
    return create_engine(database_url, pool_pre_ping=True)


def get_database_host(database_url: str) -> str:
    parsed = urlparse(database_url)
    return parsed.hostname or ""


def align_dataframe_to_table(df: pd.DataFrame, engine, table_name: str) -> pd.DataFrame:
    table_columns = [column["name"] for column in inspect(engine).get_columns(table_name)]
    available_columns = [column for column in table_columns if column in df.columns]
    missing_columns = [column for column in table_columns if column not in df.columns]

    aligned = df.loc[:, available_columns].copy()
    
    # Convertir columnas datetime a object para permitir None en vez de NaT
    for col in aligned.columns:
        if pd.api.types.is_datetime64_any_dtype(aligned[col]):
            aligned[col] = aligned[col].astype(object)
            
    aligned = aligned.where(pd.notna(aligned), None)

    if not available_columns:
        raise ValueError(f"No hay columnas compatibles para cargar en {table_name}.")

    return aligned, missing_columns


def upsert_dataframe(
    df: pd.DataFrame,
    engine,
    table_name: str,
    conflict_columns: list[str],
    chunk_size: int = 1000,
) -> int:
    if df.empty:
        return 0

    aligned, _ = align_dataframe_to_table(df, engine, table_name)
    missing_keys = [column for column in conflict_columns if column not in aligned.columns]
    if missing_keys:
        raise ValueError(f"Faltan llaves de conflicto en {table_name}: {missing_keys}")

    aligned = aligned.drop_duplicates(subset=conflict_columns, keep="last")
    metadata = MetaData()
    table = Table(table_name, metadata, autoload_with=engine)
    updated_columns = [column for column in aligned.columns if column not in conflict_columns]

    affected_rows = 0
    with engine.begin() as connection:
        for start in range(0, len(aligned), chunk_size):
            chunk = aligned.iloc[start : start + chunk_size]
            records = chunk.to_dict(orient="records")
            statement = insert(table).values(records)
            if updated_columns:
                statement = statement.on_conflict_do_update(
                    index_elements=conflict_columns,
                    set_={column: getattr(statement.excluded, column) for column in updated_columns},
                )
            else:
                statement = statement.on_conflict_do_nothing(index_elements=conflict_columns)
            result = connection.execute(statement)
            affected_rows += result.rowcount or 0

    return affected_rows


def table_counts(engine, table_names: list[str]) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    with engine.connect() as connection:
        for table_name in table_names:
            if not inspect(engine).has_table(table_name):
                counts[table_name] = None
                continue
            counts[table_name] = pd.read_sql_query(f"SELECT COUNT(*) AS total FROM {table_name}", connection)[
                "total"
            ].iloc[0]
    return counts
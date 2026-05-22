from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


CLIENTE_COLUMNS = [
    "cliente_id",
    "fecha_registro",
    "departamento",
    "ciudad",
    "edad",
    "genero",
    "estrato",
    "nivel_educativo",
    "ocupacion",
    "ingreso_mensual_estimado",
    "ingreso_mensual_bi",
    "flag_ingreso",
    "canal_adquisicion",
    "score_externo",
    "tiene_producto_ahorro",
    "numero_dependientes",
    "dispositivo_principal",
]


FACT_COLUMNS = [
    "credito_id",
    "cliente_id",
    "fecha_desembolso",
    "producto_credito",
    "monto_credito",
    "plazo_meses",
    "tasa_interes_mensual",
    "valor_cuota_pactada",
    "canal_originacion",
    "score_interno_originacion",
    "relacion_cuota_ingreso",
    "politica_aprobacion",
    "estado_credito_operativo",
    "es_moroso",
    "probabilidad_mora",
    "prediccion_mora",
]


def _add_bi_income_fields(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    p99 = result["ingreso_mensual_estimado"].quantile(0.99)
    result["ingreso_mensual_bi"] = result["ingreso_mensual_estimado"].clip(upper=p99)
    result["flag_ingreso"] = "normal"
    outlier_mask = result["ingreso_mensual_estimado"] > p99
    result.loc[outlier_mask, "flag_ingreso"] = "outlier_real"
    result.loc[outlier_mask & (result["estrato"].fillna(4) <= 3), "flag_ingreso"] = "error_captura"
    return result


def _build_dim_tiempo(df: pd.DataFrame) -> pd.DataFrame:
    fechas = pd.to_datetime(df["fecha_desembolso"]).dropna()
    dim = pd.DataFrame({"fecha": pd.date_range(fechas.min(), fechas.max(), freq="D")})
    dim["fecha_id"] = dim["fecha"].dt.strftime("%Y%m%d").astype(int)
    dim["anio"] = dim["fecha"].dt.year
    dim["mes"] = dim["fecha"].dt.month
    dim["nombre_mes"] = dim["fecha"].dt.month_name(locale=None)
    dim["trimestre"] = dim["fecha"].dt.quarter
    dim["dia_semana"] = dim["fecha"].dt.dayofweek + 1
    return dim


def _build_dim_producto(df: pd.DataFrame) -> pd.DataFrame:
    dim = df[["producto_credito"]].drop_duplicates().sort_values("producto_credito").reset_index(drop=True)
    dim.insert(0, "producto_credito_id", range(1, len(dim) + 1))
    dim["categoria_producto"] = dim["producto_credito"].fillna("Desconocido")
    return dim


def _build_perfil_cliente(predicciones: pd.DataFrame) -> pd.DataFrame:
    aggregations = predicciones.groupby("cliente_id", as_index=False).agg(
        creditos_total=("credito_id", "count"),
        creditos_morosos=("es_moroso", "sum"),
        monto_total_credito=("monto_credito", "sum"),
        monto_promedio_credito=("monto_credito", "mean"),
        probabilidad_mora_promedio=("probabilidad_mora", "mean"),
        probabilidad_mora_max=("probabilidad_mora", "max"),
        ultima_fecha_desembolso=("fecha_desembolso", "max"),
    )
    aggregations["tasa_mora_cliente"] = aggregations["creditos_morosos"] / aggregations["creditos_total"]

    cliente_attrs = predicciones[[col for col in CLIENTE_COLUMNS if col in predicciones.columns]].drop_duplicates("cliente_id")
    return cliente_attrs.merge(aggregations, on="cliente_id", how="left")


def export_powerbi_datalake(abt: pd.DataFrame, predicciones: pd.DataFrame, output_dir: Path) -> dict[str, int]:
    """Exporta tablas locales en CSV para Power BI sin depender de ODBC/Supabase."""
    output_dir.mkdir(parents=True, exist_ok=True)

    abt_export = _add_bi_income_fields(abt)
    pred_export = _add_bi_income_fields(predicciones)

    dim_cliente = pred_export[[col for col in CLIENTE_COLUMNS if col in pred_export.columns]].drop_duplicates("cliente_id")
    dim_producto = _build_dim_producto(pred_export)
    dim_tiempo = _build_dim_tiempo(pred_export)
    fact_creditos = pred_export[[col for col in FACT_COLUMNS if col in pred_export.columns]].copy()
    fact_creditos["fecha_id"] = pd.to_datetime(fact_creditos["fecha_desembolso"]).dt.strftime("%Y%m%d").astype(int)
    perfil_cliente = _build_perfil_cliente(pred_export)

    tables = {
        "abt_analitica_riesgo": abt_export,
        "predicciones_riesgo": pred_export,
        "dim_cliente": dim_cliente,
        "dim_producto_credito": dim_producto,
        "dim_tiempo": dim_tiempo,
        "fact_creditos": fact_creditos,
        "perfil_descriptivo_cliente": perfil_cliente,
    }

    counts: dict[str, int] = {}
    for table_name, dataframe in tables.items():
        dataframe.to_csv(output_dir / f"{table_name}.csv", index=False)
        counts[table_name] = len(dataframe)

    manifest = {
        "descripcion": "Mini data lake local para Power BI. Regenerado por 0_Master_Pipeline.ipynb.",
        "formato": "csv",
        "tablas": counts,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return counts
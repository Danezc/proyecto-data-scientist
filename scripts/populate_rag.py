"""
populate_rag.py — Pobla el vector store RAG con conocimiento financiero de TumiPay.

Fuentes de documentos que se cargan:
  1. Fichas de perfil por crédito/cliente  → desde predicciones_riesgo (PostgreSQL)
  2. Resumen estadístico del portafolio     → calculado a partir de los datos
  3. Políticas de riesgo y negocio          → documentos estáticos curados

Uso:
    python scripts/populate_rag.py
    python scripts/populate_rag.py --reset       # borra y repobla
    python scripts/populate_rag.py --no-fichas   # solo políticas + resumen
"""

import sys
import logging
import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from langchain_core.documents import Document
from langchain_community.vectorstores import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.config import settings  # noqa: E402 (after sys.path)

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Silenciar librerías de infraestructura
for _noisy in ["httpx", "httpcore", "sentence_transformers", "transformers",
               "huggingface_hub", "langchain_community"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

COLLECTION_NAME = "rag_conocimiento"
EMBED_MODEL = "intfloat/multilingual-e5-small"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Fichas de perfil por crédito
# ─────────────────────────────────────────────────────────────────────────────

def build_client_profile_docs(df: pd.DataFrame) -> list[Document]:
    """Un documento tipo 'expediente' por cada fila de predicciones_riesgo."""
    docs = []
    for _, row in df.iterrows():
        moroso     = "SÍ" if row["es_moroso"] == 1 else "NO"
        prob       = float(row["probabilidad_mora"])
        nivel_risk = (
            "CRÍTICO"  if prob >= 0.80 else
            "ALTO"     if prob >= 0.50 else
            "MEDIO"    if prob >= 0.20 else
            "BAJO"
        )

        texto = (
            f"FICHA DE CRÉDITO — {row['credito_id']}\n"
            f"Cliente: {row['cliente_id']}  |  "
            f"Ubicación: {row['ciudad'].title()}, {row['departamento']}\n"
            f"Edad: {row['edad']} años  |  Género: {row['genero']}  |  "
            f"Estrato: {row['estrato']}\n"
            f"Educación: {row['nivel_educativo']}  |  "
            f"Ocupación: {row['ocupacion']}\n"
            f"Ingreso mensual estimado: ${row['ingreso_mensual_estimado']:,.0f}\n"
            f"Score externo (buró): {row['score_externo']}\n"
            f"\nCRÉDITO:\n"
            f"Producto: {row['producto_credito']}\n"
            f"Monto desembolsado: ${row['monto_credito']:,.0f}\n"
            f"Plazo: {row['plazo_meses']} meses  |  "
            f"Tasa mensual: {row['tasa_interes_mensual']:.2%}\n"
            f"Valor cuota pactada: ${row['valor_cuota_pactada']:,.0f}\n"
            f"Relación cuota/ingreso: {row['relacion_cuota_ingreso']:.2%}\n"
            f"Canal de originación: {row['canal_originacion']}\n"
            f"Política de aprobación: {row['politica_aprobacion']}\n"
            f"Estado operativo del crédito: {row['estado_credito_operativo']}\n"
            f"\nRIESGO:\n"
            f"¿Es moroso?: {moroso}\n"
            f"Probabilidad de mora: {prob:.1%}  →  Nivel de riesgo: {nivel_risk}\n"
        )

        docs.append(Document(
            page_content=texto,
            metadata={
                "fuente":             "ficha_cliente",
                "cliente_id":         str(row["cliente_id"]),
                "credito_id":         str(row["credito_id"]),
                "es_moroso":          int(row["es_moroso"]),
                "probabilidad_mora":  round(prob, 4),
                "nivel_riesgo":       nivel_risk,
                "producto_credito":   str(row["producto_credito"]),
                "departamento":       str(row["departamento"]),
            },
        ))

    return docs


# ─────────────────────────────────────────────────────────────────────────────
# 2. Resumen estadístico del portafolio
# ─────────────────────────────────────────────────────────────────────────────

def build_portfolio_summary_docs(df: pd.DataFrame) -> list[Document]:
    """Resumen global + desglose por producto y canal."""
    total      = len(df)
    morosos    = int(df["es_moroso"].sum())
    tasa_mora  = morosos / total
    monto_tot  = df["monto_credito"].sum()
    monto_mor  = df.loc[df["es_moroso"] == 1, "monto_credito"].sum()

    por_producto = (
        df.groupby("producto_credito")
        .agg(creditos=("credito_id", "count"),
             morosos=("es_moroso", "sum"),
             monto_prom=("monto_credito", "mean"))
        .reset_index()
    )
    prod_txt = "\n".join(
        f"  • {r['producto_credito']}: {int(r['creditos'])} créditos, "
        f"{int(r['morosos'])} morosos ({r['morosos']/r['creditos']:.1%}), "
        f"monto promedio ${r['monto_prom']:,.0f}"
        for _, r in por_producto.iterrows()
    )

    por_canal = (
        df.groupby("canal_originacion")
        .agg(creditos=("credito_id", "count"),
             morosos=("es_moroso", "sum"))
        .reset_index()
    )
    canal_txt = "\n".join(
        f"  • {r['canal_originacion']}: {int(r['creditos'])} créditos, "
        f"{int(r['morosos'])} morosos ({r['morosos']/r['creditos']:.1%})"
        for _, r in por_canal.iterrows()
    )

    por_depto = (
        df.groupby("departamento")
        .agg(creditos=("credito_id", "count"),
             morosos=("es_moroso", "sum"))
        .sort_values("morosos", ascending=False)
        .head(8)
        .reset_index()
    )
    depto_txt = "\n".join(
        f"  • {r['departamento']}: {int(r['creditos'])} créditos, "
        f"{int(r['morosos'])} morosos ({r['morosos']/r['creditos']:.1%})"
        for _, r in por_depto.iterrows()
    )

    texto = (
        f"RESUMEN ESTADÍSTICO DEL PORTAFOLIO DE CRÉDITOS — TUMIPAY\n"
        f"Fecha de corte: {settings.CUTOFF_DATE}\n\n"
        f"MÉTRICAS GLOBALES:\n"
        f"  • Total créditos en portafolio: {total:,}\n"
        f"  • Clientes morosos identificados: {morosos:,} ({tasa_mora:.1%} tasa de mora)\n"
        f"  • Monto total desembolsado: ${monto_tot:,.0f}\n"
        f"  • Monto en riesgo (morosos): ${monto_mor:,.0f} ({monto_mor/monto_tot:.1%})\n"
        f"  • Portafolio sano: ${monto_tot - monto_mor:,.0f} ({1-tasa_mora:.1%})\n\n"
        f"MODELO PREDICTIVO:\n"
        f"  • Algoritmo: LightGBM (Gradient Boosting)\n"
        f"  • ROC-AUC: 0.9844  |  Accuracy: 97%\n"
        f"  • Variable objetivo: es_moroso (≥1 cuota con mora >30 días)\n\n"
        f"DESGLOSE POR PRODUCTO:\n{prod_txt}\n\n"
        f"DESGLOSE POR CANAL DE ORIGINACIÓN:\n{canal_txt}\n\n"
        f"TOP DEPARTAMENTOS POR MORA:\n{depto_txt}\n"
    )

    return [Document(
        page_content=texto,
        metadata={"fuente": "resumen_portafolio", "tipo": "estadisticas_globales"},
    )]


# ─────────────────────────────────────────────────────────────────────────────
# 3. Documentos de políticas y reglas de negocio
# ─────────────────────────────────────────────────────────────────────────────

def build_policy_docs() -> list[Document]:
    politicas = [
        {
            "titulo": "POLÍTICA DE MORA Y CLASIFICACIÓN DE RIESGO",
            "contenido": (
                "POLÍTICA DE MORA Y CLASIFICACIÓN DE RIESGO — TUMIPAY\n\n"
                "DEFINICIONES:\n"
                "  • Mora: Retraso en el pago de una cuota más allá de su fecha de vencimiento.\n"
                "  • DPD (Days Past Due): Días desde el vencimiento hasta el pago efectivo.\n"
                "  • Cliente moroso: Al menos una cuota con DPD > 30 días en el período de análisis.\n"
                "  • Probabilidad de mora (PD): Salida del modelo LightGBM entre 0 y 1.\n\n"
                "NIVELES DE RIESGO:\n"
                "  • Bajo   (PD < 0.20): Comportamiento normal. Sin alertas activas.\n"
                "  • Medio  (0.20 ≤ PD < 0.50): Monitoreo periódico, gestiones preventivas.\n"
                "  • Alto   (0.50 ≤ PD < 0.80): Alerta temprana, contacto proactivo.\n"
                "  • Crítico (PD ≥ 0.80): Intervención inmediata, cobro activo.\n\n"
                "CRITERIOS DE APROBACIÓN:\n"
                "  • Automática: Score interno ≥ 600 y relación cuota/ingreso ≤ 35%.\n"
                "  • Manual: Score interno 400–599, requiere revisión de analista.\n"
                "  • Rechazada: Score interno < 400 o cuota/ingreso > 50%.\n\n"
                "PRINCIPALES PREDICTORES DE MORA (importancia del modelo):\n"
                "  1. score_externo: Mayor score = menor riesgo de mora.\n"
                "  2. relacion_cuota_ingreso: Mayor proporción = mayor riesgo.\n"
                "  3. score_interno_originacion: Evaluación interna al desembolso.\n"
                "  4. monto_credito: Exposición financiera total.\n"
                "  5. plazo_meses: Plazos más largos tienen mayor probabilidad acumulada.\n"
                "  6. estrato: Variable de contexto socioeconómico.\n"
            ),
        },
        {
            "titulo": "PRODUCTOS DE CRÉDITO TUMIPAY",
            "contenido": (
                "PRODUCTOS DE CRÉDITO TUMIPAY — CARACTERÍSTICAS Y CONDICIONES\n\n"
                "1. CRÉDITO LIBRE INVERSIÓN\n"
                "   Montos: $500,000–$10,000,000 COP  |  Plazos: 3–24 meses\n"
                "   Canales: App, Web, Punto de venta\n"
                "   Perfil: Clientes con score ≥ 500\n\n"
                "2. AVANCE DE NÓMINA\n"
                "   Montos: Hasta el 70% del ingreso mensual reportado\n"
                "   Plazo: 1–3 meses  |  Canal: App (desembolso inmediato)\n"
                "   Perfil: Empleados formales con ingreso verificable ≥ $1,000,000\n\n"
                "3. MICROCRÉDITO\n"
                "   Montos: $200,000–$3,000,000 COP  |  Plazos: 3–18 meses\n"
                "   Canales: Campaña digital, Referido\n"
                "   Perfil: Comerciantes y trabajadores independientes\n\n"
                "4. REFINANCIACIÓN\n"
                "   Propósito: Reestructuración de deuda existente (mora 30–90 días, primer incumplimiento)\n"
                "   Beneficio: Extensión de plazo hasta 6 meses adicionales\n"
                "   Requiere: Aprobación por analista de riesgo\n\n"
                "CANALES DE ORIGINACIÓN:\n"
                "  • App: Digital end-to-end, desembolso en 24h\n"
                "  • Web: Digital con validación adicional, desembolso en 48h\n"
                "  • Punto de venta: Presencial, desembolso el mismo día\n"
                "  • Referido: Híbrido, tasa preferencial −0.5% mensual\n"
                "  • Campaña digital: Validación de identidad reforzada\n"
            ),
        },
        {
            "titulo": "PROCESO DE COBRO Y GESTIÓN DE CARTERA VENCIDA",
            "contenido": (
                "PROCESO DE COBRO Y GESTIÓN DE CARTERA VENCIDA — TUMIPAY\n\n"
                "ETAPA 1 — PREVENTIVA (antes del vencimiento):\n"
                "  • Recordatorio automático 3 días antes (push, SMS, email)\n"
                "  • Habilitación de débito automático desde cuenta de ahorro\n"
                "  • Alerta del modelo para clientes con PD > 0.50\n\n"
                "ETAPA 2 — COBRO TEMPRANO (DPD 1–30):\n"
                "  • Notificaciones diarias (push y SMS)\n"
                "  • Oferta de refinanciación disponible\n"
                "  • Gestión telefónica a partir del día 10\n\n"
                "ETAPA 3 — COBRO TARDÍO (DPD 31–90):\n"
                "  • Reporte a centrales de riesgo (Datacrédito, TransUnión) desde el día 31\n"
                "  • Asignación a gestor de cobro externo\n"
                "  • Oferta de acuerdo de pago con descuento en intereses de mora\n\n"
                "ETAPA 4 — COBRO JURÍDICO (DPD > 90):\n"
                "  • Traslado a cartera jurídica\n"
                "  • Inicio de proceso legal según cuantía\n"
                "  • Negociación de dación en pago para créditos con garantía\n\n"
                "KPIs DE CARTERA:\n"
                "  • NPL 30+: % cartera con mora > 30 días\n"
                "  • NPL 90+: % cartera con mora > 90 días\n"
                "  • Tasa de recuperación: % cartera recuperada vs castigada\n"
                "  • Cobertura de provisiones: Provisiones / Cartera vencida (meta ≥ 120%)\n"
                "  • Costo del riesgo (CoR): Provisiones netas / Cartera promedio\n"
            ),
        },
        {
            "titulo": "GLOSARIO FINANCIERO TUMIPAY",
            "contenido": (
                "GLOSARIO FINANCIERO — TUMIPAY\n\n"
                "ABT (Analytical Base Table): Tabla base analítica que consolida variables de clientes, "
                "créditos y pagos para entrenamiento del modelo de riesgo.\n\n"
                "Cartera vencida: Conjunto de créditos con al menos una cuota en mora.\n\n"
                "Cartera castigada: Créditos con DPD > 180 días dados de baja contablemente.\n\n"
                "DPD (Days Past Due): Días de atraso en el pago de una cuota desde su vencimiento.\n\n"
                "NPL (Non-Performing Loan): Crédito en situación de incumplimiento, convencionalmente "
                "con DPD > 90 días.\n\n"
                "PD (Probability of Default): Probabilidad estimada de que un cliente entre en mora.\n\n"
                "LGD (Loss Given Default): Porcentaje del monto que se perdería si el cliente entra en mora.\n\n"
                "EAD (Exposure at Default): Monto expuesto al momento del incumplimiento.\n\n"
                "Score externo: Puntaje del buró de crédito que refleja el historial crediticio del cliente.\n\n"
                "Score interno: Puntaje calculado internamente por TumiPay basado en comportamiento propio.\n\n"
                "Relación cuota/ingreso: Proporción del ingreso mensual destinada al pago de la cuota; "
                "un valor > 35% se considera de riesgo alto.\n\n"
                "Tasa de mora: Porcentaje de la cartera total que está en situación de mora.\n\n"
                "Provisión: Reserva contable constituida para cubrir pérdidas esperadas por mora.\n\n"
                "Refinanciación: Reestructuración de un crédito con nuevas condiciones para facilitar el pago.\n\n"
                "Originación: Proceso de evaluación y desembolso de un crédito nuevo.\n"
            ),
        },
    ]

    return [
        Document(
            page_content=p["contenido"],
            metadata={"fuente": "politica_negocio", "titulo": p["titulo"]},
        )
        for p in politicas
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(reset: bool = True, include_fichas: bool = True) -> int:
    logger.info("=== Iniciando población del RAG Knowledge Base ===")

    # Cargar datos desde PostgreSQL
    logger.info("Conectando a PostgreSQL y cargando predicciones_riesgo...")
    engine = create_engine(settings.DATABASE_URL)
    df = pd.read_sql("SELECT * FROM predicciones_riesgo", engine)
    logger.info(f"  → {len(df)} filas cargadas")

    # Construir documentos
    all_docs: list[Document] = []

    logger.info("Construyendo resumen del portafolio...")
    portfolio_docs = build_portfolio_summary_docs(df)
    all_docs.extend(portfolio_docs)
    logger.info(f"  → {len(portfolio_docs)} documento(s) de resumen")

    logger.info("Construyendo documentos de políticas de negocio...")
    policy_docs = build_policy_docs()
    all_docs.extend(policy_docs)
    logger.info(f"  → {len(policy_docs)} documentos de política")

    if include_fichas:
        logger.info("Construyendo fichas de perfil por crédito...")
        client_docs = build_client_profile_docs(df)
        all_docs.extend(client_docs)
        logger.info(f"  → {len(client_docs)} fichas de cliente/crédito")

    logger.info(f"Total documentos antes de chunking: {len(all_docs)}")

    # Chunking para documentos largos
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    final_docs: list[Document] = []
    for doc in all_docs:
        if len(doc.page_content) > 1000:
            chunks = splitter.split_documents([doc])
            final_docs.extend(chunks)
        else:
            final_docs.append(doc)
    logger.info(f"Total chunks a indexar: {len(final_docs)}")

    # Embeddings
    logger.info(f"Cargando modelo de embeddings '{EMBED_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Carga en PGVector
    logger.info("Cargando documentos en PGVector (puede tardar varios minutos)...")
    PGVector.from_documents(
        documents=final_docs,
        embedding=embeddings,
        connection_string=settings.DATABASE_URL,
        collection_name=COLLECTION_NAME,
        pre_delete_collection=reset,
    )

    logger.info(f"RAG Knowledge Base poblado con {len(final_docs)} chunks.")
    logger.info("Verifica con: SELECT COUNT(*) FROM langchain_pg_embedding;")
    return len(final_docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Poblar el vector store RAG de TumiPay")
    parser.add_argument(
        "--reset", action="store_true", default=True,
        help="Borrar la colección antes de repoblar (default: True)",
    )
    parser.add_argument(
        "--no-fichas", action="store_true", default=False,
        help="Omitir fichas de perfil individual (solo políticas + resumen)",
    )
    args = parser.parse_args()
    main(reset=args.reset, include_fichas=not args.no_fichas)

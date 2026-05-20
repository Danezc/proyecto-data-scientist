-- =========================================================================
-- MIGRACIÓN 001: Schema Base - TumiPay Data Pipeline
-- =========================================================================
-- Descripción: Habilita la extensión vectorial y crea las tablas de datos
--              crudos (raw) que alimentan el pipeline de ciencia de datos.
--
-- Compatibilidad: PostgreSQL 15+ / Supabase (pgvector nativo)
-- =========================================================================

-- 1. EXTENSIÓN VECTORIAL
-- En Supabase se habilita desde: Dashboard → Database → Extensions → vector
-- O con el siguiente comando (requiere permisos de superusuario):
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;


-- =========================================================================
-- 2. TABLAS RAW (Datos de origen del pipeline ETL)
-- =========================================================================

CREATE TABLE IF NOT EXISTS raw_clientes (
    cliente_id                  VARCHAR(50) PRIMARY KEY,
    fecha_registro              DATE,
    departamento                VARCHAR(100),
    ciudad                      VARCHAR(100),
    edad                        INTEGER,
    genero                      VARCHAR(10),
    estrato                     INTEGER,
    nivel_educativo             VARCHAR(100),
    ocupacion                   VARCHAR(100),
    ingreso_mensual_estimado    NUMERIC(15, 2),
    canal_adquisicion           VARCHAR(50),
    score_externo               NUMERIC(10, 2),
    tiene_producto_ahorro       BOOLEAN,
    numero_dependientes         INTEGER,
    dispositivo_principal       VARCHAR(50),
    email_hash                  TEXT
);

CREATE TABLE IF NOT EXISTS raw_creditos (
    credito_id                  VARCHAR(50) PRIMARY KEY,
    cliente_id                  VARCHAR(50) REFERENCES raw_clientes(cliente_id),
    fecha_desembolso            DATE,
    producto_credito            VARCHAR(100),
    monto_credito               NUMERIC(15, 2),
    plazo_meses                 INTEGER,
    tasa_interes_mensual        NUMERIC(6, 4),
    valor_cuota_pactada         NUMERIC(15, 2),
    canal_originacion           VARCHAR(50),
    score_interno_originacion   INTEGER,
    relacion_cuota_ingreso      NUMERIC(8, 4),
    politica_aprobacion         VARCHAR(50),
    estado_credito_operativo    VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS raw_pagos (
    pago_id         VARCHAR(50) PRIMARY KEY,
    credito_id      VARCHAR(50) REFERENCES raw_creditos(credito_id),
    cliente_id      VARCHAR(50) REFERENCES raw_clientes(cliente_id),
    numero_cuota    INTEGER,
    fecha_pago      DATE,
    valor_pagado    NUMERIC(15, 2),
    estado_pago     VARCHAR(50),
    dias_mora       INTEGER
);

CREATE TABLE IF NOT EXISTS raw_eventos_app (
    evento_id           VARCHAR(50) PRIMARY KEY,
    cliente_id          VARCHAR(50) REFERENCES raw_clientes(cliente_id),
    fecha_evento        DATE,
    tipo_evento         VARCHAR(100),
    resultado_evento    VARCHAR(100),
    duracion_sesion_seg INTEGER
);


-- =========================================================================
-- NOTA: Las tablas del vector store RAG (langchain_pg_collection,
-- langchain_pg_embedding) son creadas automáticamente por LangChain
-- al instanciar RAGAgentPipeline. No requieren migración manual.
-- =========================================================================

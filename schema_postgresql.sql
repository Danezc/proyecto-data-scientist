-- =========================================================================
-- SCHEMA POSTGRESQL - TUMIPAY DATA PIPELINE & RAG VECTOR STORE
-- =========================================================================

-- 1. EXTENSIÓN VECTORIAL (Obligatorio para RAG con pgvector)
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. TABLA DE CONOCIMIENTO (Embeddings / RAG)
-- Almacena los insights, resúmenes del EDA y políticas de crédito
CREATE TABLE IF NOT EXISTS rag_conocimiento (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    page_content TEXT NOT NULL,
    metadata JSONB,
    -- NOTA: La dimensión (ej. 1536) depende del modelo. 
    -- Para modelos Open Source como Qwen, puede ser 896, 1024 o 1536. 
    -- Se define genéricamente como VECTOR si la dimensión cambia dinámicamente,
    -- o estrictamente VECTOR(1536) para optimizar índices de búsqueda.
    embedding VECTOR(1536) 
);

-- Crear un índice HNSW para búsqueda rápida de similitud de coseno
CREATE INDEX ON rag_conocimiento USING hnsw (embedding vector_cosine_ops);


-- =========================================================================
-- TABLAS RAW INICIALES (Mapeo de datos tabulares)
-- =========================================================================

CREATE TABLE IF NOT EXISTS raw_clientes (
    cliente_id VARCHAR(50) PRIMARY KEY,
    fecha_registro DATE,
    departamento VARCHAR(100),
    ciudad VARCHAR(100),
    edad INT,
    genero VARCHAR(10),
    estrato INT,
    nivel_educativo VARCHAR(100),
    ocupacion VARCHAR(100),
    ingreso_mensual_estimado NUMERIC(15, 2),
    canal_adquisicion VARCHAR(50),
    score_externo NUMERIC(10, 2),
    tiene_producto_ahorro BOOLEAN
);

CREATE TABLE IF NOT EXISTS raw_creditos (
    credito_id VARCHAR(50) PRIMARY KEY,
    cliente_id VARCHAR(50) REFERENCES raw_clientes(cliente_id),
    fecha_desembolso DATE,
    monto_credito NUMERIC(15, 2),
    plazo_meses INT,
    tasa_interes_mensual NUMERIC(5, 4),
    valor_cuota_pactada NUMERIC(15, 2),
    canal_originacion VARCHAR(50)
);

CREATE TABLE IF NOT EXISTS raw_pagos (
    pago_id VARCHAR(50) PRIMARY KEY,
    credito_id VARCHAR(50) REFERENCES raw_creditos(credito_id),
    cliente_id VARCHAR(50) REFERENCES raw_clientes(cliente_id),
    numero_cuota INT,
    fecha_pago DATE,
    valor_pagado NUMERIC(15, 2),
    estado_pago VARCHAR(50),
    dias_mora INT
);

CREATE TABLE IF NOT EXISTS raw_eventos_app (
    evento_id VARCHAR(50) PRIMARY KEY,
    cliente_id VARCHAR(50) REFERENCES raw_clientes(cliente_id),
    fecha_evento DATE,
    tipo_evento VARCHAR(100),
    resultado_evento VARCHAR(100),
    duracion_sesion_seg INT
);

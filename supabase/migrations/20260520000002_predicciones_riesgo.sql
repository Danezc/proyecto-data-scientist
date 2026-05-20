-- =========================================================================
-- MIGRACIÓN 002: Tabla de Predicciones LightGBM
-- =========================================================================
-- Descripción: Crea la tabla de salida del modelo LightGBM de riesgo de mora.
--              Contiene features del ABT + probabilidades + predicciones.
--              Es la fuente principal de datos para Power BI y el agente RAG.
--
-- Generada a partir del pipeline: src/train_mora.py → abt.to_sql(...)
-- Schema verificado: 2026-05-20 (1527 registros)
-- =========================================================================

CREATE TABLE IF NOT EXISTS predicciones_riesgo (
    -- Identificadores
    credito_id                  TEXT,
    cliente_id                  TEXT,

    -- Datos del crédito
    fecha_desembolso            TIMESTAMP,
    producto_credito            TEXT,
    monto_credito               BIGINT,
    plazo_meses                 BIGINT,
    tasa_interes_mensual        DOUBLE PRECISION,
    valor_cuota_pactada         BIGINT,
    canal_originacion           TEXT,
    score_interno_originacion   BIGINT,
    relacion_cuota_ingreso      DOUBLE PRECISION,
    politica_aprobacion         TEXT,
    estado_credito_operativo    TEXT,

    -- Variable objetivo (target)
    es_moroso                   BIGINT,

    -- Datos del cliente (features demográficas)
    fecha_registro              TIMESTAMP,
    departamento                TEXT,
    ciudad                      TEXT,
    edad                        BIGINT,
    genero                      TEXT,
    estrato                     BIGINT,
    nivel_educativo             TEXT,
    ocupacion                   TEXT,
    ingreso_mensual_estimado    DOUBLE PRECISION,
    canal_adquisicion           TEXT,
    score_externo               DOUBLE PRECISION,
    tiene_producto_ahorro       BOOLEAN,
    numero_dependientes         BIGINT,
    dispositivo_principal       TEXT,
    email_hash                  TEXT,

    -- Salida del modelo LightGBM
    probabilidad_mora           DOUBLE PRECISION,   -- Score 0.0–1.0
    prediccion_mora             BIGINT              -- 0 = sano, 1 = moroso
);

-- Índice para consultas frecuentes por Power BI y el agente RAG
CREATE INDEX IF NOT EXISTS idx_predicciones_credito_id
    ON predicciones_riesgo (credito_id);

CREATE INDEX IF NOT EXISTS idx_predicciones_probabilidad
    ON predicciones_riesgo (probabilidad_mora DESC);

CREATE INDEX IF NOT EXISTS idx_predicciones_es_moroso
    ON predicciones_riesgo (es_moroso);

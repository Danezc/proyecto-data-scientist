-- =========================================================================
-- CONSULTAS SQL DE TRANSFORMACIÓN, CONSOLIDACIÓN E INSIGHTS DE NEGOCIO
-- =========================================================================
-- Este archivo contiene las consultas SQL estructuradas para el motor PostgreSQL.
-- Aplica mejores prácticas: CTEs, JOINS, Funciones Ventana y Agregaciones.
-- =========================================================================

-- ─────────────────────────────────────────────────────────────────────────
-- 1. CREACIÓN DE LA TABLA ANALÍTICA BASE (ABT) EN SQL
-- ─────────────────────────────────────────────────────────────────────────
-- Esta consulta consolida los datos de originación (créditos) con demografía
-- (clientes) y calcula la variable objetivo (es_moroso) libre de data leakage
-- evaluando únicamente las cuotas vencidas antes de la fecha de corte (2026-04-30).
-- Adicionalmente, incluye la agregación de eventos de la App previos a la originación.

CREATE TABLE IF NOT EXISTS abt_analitica_riesgo AS
WITH 
-- A. Estandarización de clientes e imputación de nulos
clientes_clean AS (
    SELECT 
        cliente_id,
        fecha_registro,
        departamento,
        -- Estandarización de nombres de ciudades
        CASE 
            WHEN LOWER(TRIM(ciudad)) IN ('bogot', 'bogota', 'bogota d.c.', 'bogotá d.c.', 'bogota dc', 'bogotá dc') THEN 'Bogotá'
            WHEN LOWER(TRIM(ciudad)) IN ('medellin ', 'medellin', 'medellín') THEN 'Medellín'
            WHEN LOWER(TRIM(ciudad)) IN ('barranquila', 'barranquilla') THEN 'Barranquilla'
            ELSE INITCAP(TRIM(ciudad))
        END AS ciudad,
        -- Imputación de edad (outliers > 110 o < 18 a la mediana de 35 años)
        CASE 
            WHEN edad < 18 OR edad > 110 THEN 35
            ELSE edad
        END AS edad,
        genero,
        estrato,
        nivel_educativo,
        ocupacion,
        -- Imputación de ingreso mensual por la mediana general ($2,200,000 COP)
        COALESCE(ingreso_mensual_estimado, 2200000.00) AS ingreso_mensual_estimado,
        canal_adquisicion,
        -- Imputación de score externo por la mediana general (580)
        COALESCE(score_externo, 580) AS score_externo,
        tiene_producto_ahorro,
        numero_dependientes,
        dispositivo_principal
    FROM raw_clientes
),

-- B. Limpieza de créditos e imputación de tipo de producto nulo
creditos_clean AS (
    SELECT 
        credito_id,
        cliente_id,
        fecha_desembolso,
        COALESCE(producto_credito, 'Desconocido') AS producto_credito,
        monto_credito,
        plazo_meses,
        tasa_interes_mensual,
        valor_cuota_pactada,
        canal_originacion,
        score_interno_originacion,
        relacion_cuota_ingreso,
        politica_aprobacion
    FROM raw_creditos
    WHERE credito_id IS NOT NULL AND cliente_id IS NOT NULL AND monto_credito IS NOT NULL
),

-- C. Cálculo de target de mora (cuotas vencidas hasta el corte: 2026-04-30)
pagos_target AS (
    SELECT 
        credito_id,
        MAX(dias_mora) AS max_dias_mora,
        CASE 
            WHEN MAX(dias_mora) > 30 THEN 1 
            ELSE 0 
        END AS es_moroso
    FROM raw_pagos
    -- Filtro anti-leakage: solo evaluar cuotas con vencimiento hasta la fecha de corte
    WHERE fecha_vencimiento <= '2026-04-30'
    GROUP BY credito_id
),

-- D. Agregación de eventos digitales del cliente previos al desembolso del crédito
eventos_previos AS (
    SELECT 
        cr.credito_id,
        COUNT(CASE WHEN ev.tipo_evento = 'login' THEN 1 END) AS prev_evento_login,
        COUNT(CASE WHEN ev.tipo_evento = 'simulacion_credito' THEN 1 END) AS prev_evento_simulacion,
        COUNT(CASE WHEN ev.tipo_evento = 'pago_fallido' THEN 1 END) AS prev_evento_pago_fallido,
        COUNT(CASE WHEN ev.tipo_evento = 'pago_exitoso' THEN 1 END) AS prev_evento_pago_exitoso,
        COUNT(CASE WHEN ev.tipo_evento = 'solicitud_soporte' THEN 1 END) AS prev_evento_soporte,
        COALESCE(SUM(ev.duracion_sesion_seg), 0) AS prev_evento_sesion_seg_tot,
        COALESCE(ROUND(AVG(ev.duracion_sesion_seg), 1), 0) AS prev_evento_sesion_seg_avg
    FROM raw_creditos cr
    INNER JOIN raw_eventos_app ev ON cr.cliente_id = ev.cliente_id
    -- Filtro de tiempo para evitar fuga de información
    WHERE ev.fecha_evento < cr.fecha_desembolso
    GROUP BY cr.credito_id
)

-- E. Consolidación de la ABT final
SELECT 
    c.credito_id,
    c.cliente_id,
    c.fecha_desembolso,
    c.producto_credito,
    c.monto_credito,
    c.plazo_meses,
    c.tasa_interes_mensual,
    c.valor_cuota_pactada,
    c.canal_originacion,
    c.score_interno_originacion,
    c.relacion_cuota_ingreso,
    c.politica_aprobacion,
    cl.fecha_registro,
    cl.departamento,
    cl.ciudad,
    cl.edad,
    cl.genero,
    cl.estrato,
    cl.nivel_educativo,
    cl.ocupacion,
    cl.ingreso_mensual_estimado,
    cl.canal_adquisicion,
    cl.score_externo,
    cl.tiene_producto_ahorro,
    cl.numero_dependientes,
    cl.dispositivo_principal,
    -- Características calculadas de originación
    (c.fecha_desembolso - cl.fecha_registro) AS antiguedad_cliente_dias,
    EXTRACT(MONTH FROM c.fecha_desembolso) AS mes_desembolso,
    EXTRACT(ISODOW FROM c.fecha_desembolso) AS dia_semana_desembolso,
    -- Características de comportamiento digital previo
    COALESCE(ev.prev_evento_login, 0) AS prev_evento_login,
    COALESCE(ev.prev_evento_simulacion, 0) AS prev_evento_simulacion,
    COALESCE(ev.prev_evento_pago_fallido, 0) AS prev_evento_pago_fallido,
    COALESCE(ev.prev_evento_pago_exitoso, 0) AS prev_evento_pago_exitoso,
    COALESCE(ev.prev_evento_soporte, 0) AS prev_evento_soporte,
    COALESCE(ev.prev_evento_sesion_seg_tot, 0) AS prev_evento_sesion_seg_tot,
    COALESCE(ev.prev_evento_sesion_seg_avg, 0) AS prev_evento_sesion_seg_avg,
    -- Target
    COALESCE(t.es_moroso, 0) AS es_moroso
FROM creditos_clean c
LEFT JOIN clientes_clean cl ON c.cliente_id = cl.cliente_id
LEFT JOIN pagos_target t ON c.credito_id = t.credito_id
LEFT JOIN eventos_previos ev ON c.credito_id = ev.credito_id;


-- ─────────────────────────────────────────────────────────────────────────
-- 2. CONSULTAS DE NEGOCIO (INSIGHTS)
-- ─────────────────────────────────────────────────────────────────────────

-- Insight A: ¿Qué factores demográficos/socioeconómicos (Ocupación, Estrato) están asociados a mayor mora?
SELECT 
    ocupacion,
    estrato,
    COUNT(*) AS total_creditos,
    SUM(es_moroso) AS creditos_morosos,
    ROUND(100.0 * SUM(es_moroso) / COUNT(*), 2) AS tasa_mora_pct,
    ROUND(AVG(monto_credito), 0) AS monto_promedio
FROM abt_analitica_riesgo
GROUP BY ocupacion, estrato
HAVING COUNT(*) > 10
ORDER BY tasa_mora_pct DESC;

-- Insight B: Comportamiento por Score Externo (Buró) y Score Interno de Originación
SELECT 
    CASE 
        WHEN score_externo < 400 THEN '1. Muy Bajo (<400)'
        WHEN score_externo BETWEEN 400 AND 550 THEN '2. Bajo (400-550)'
        WHEN score_externo BETWEEN 550 AND 700 THEN '3. Medio (550-700)'
        ELSE '4. Alto (>700)'
    END AS rango_score_externo,
    COUNT(*) AS total_clientes,
    SUM(es_moroso) AS creditos_morosos,
    ROUND(100.0 * SUM(es_moroso) / COUNT(*), 2) AS tasa_mora_pct,
    ROUND(AVG(score_interno_originacion), 0) AS promedio_score_interno
FROM abt_analitica_riesgo
GROUP BY rango_score_externo
ORDER BY rango_score_externo;

-- Insight C: Análisis de canales y productos con mayor nivel de riesgo
SELECT 
    producto_credito,
    canal_originacion,
    COUNT(*) AS total_creditos,
    SUM(es_moroso) AS creditos_morosos,
    ROUND(100.0 * SUM(es_moroso) / COUNT(*), 2) AS tasa_mora_pct,
    ROUND(AVG(relacion_cuota_ingreso) * 100, 2) AS relacion_cuota_ingreso_avg_pct
FROM abt_analitica_riesgo
GROUP BY producto_credito, canal_originacion
ORDER BY tasa_mora_pct DESC;

-- Insight D: Comportamiento digital previo y su relación con la mora (Uso de Funciones Ventana)
-- Clasifica la duración de sesión acumulada en quintiles para analizar la correlación con el default
WITH quintiles_sesiones AS (
    SELECT 
        es_moroso,
        prev_evento_pago_fallido,
        prev_evento_simulacion,
        NTILE(5) OVER (ORDER BY prev_evento_sesion_seg_tot) AS quintil_sesion
    FROM abt_analitica_riesgo
)
SELECT 
    quintil_sesion,
    COUNT(*) AS total_clientes,
    SUM(es_moroso) AS morosos,
    ROUND(100.0 * SUM(es_moroso) / COUNT(*), 2) AS tasa_mora_pct,
    ROUND(AVG(prev_evento_pago_fallido), 2) AS promedio_pagos_fallidos_previos,
    ROUND(AVG(prev_evento_simulacion), 2) AS promedio_simulaciones_previas
FROM quintiles_sesiones
GROUP BY quintil_sesion
ORDER BY quintil_sesion;

-- Insight E: Análisis temporal de comportamiento de pago (Ejemplo de función ventana sobre pagos)
-- Determina el número de días transcurridos entre la fecha de vencimiento y la fecha de pago
-- para identificar patrones de pago tardío por número de cuota
SELECT 
    numero_cuota,
    COUNT(*) AS total_cuotas_evaluadas,
    SUM(CASE WHEN estado_pago = 'Pagado tardío' THEN 1 ELSE 0 END) AS total_pagos_tardios,
    ROUND(AVG(dias_mora), 1) AS dias_mora_promedio,
    ROUND(100.0 * SUM(CASE WHEN estado_pago = 'Pagado tardío' THEN 1 ELSE 0 END) / COUNT(*), 2) AS tasa_pago_tardio_pct
FROM raw_pagos
WHERE fecha_vencimiento <= '2026-04-30'
GROUP BY numero_cuota
ORDER BY numero_cuota;

# Diccionario de Datos — TumiPay

Diccionario reconstruido a partir de [data/raw/schema_postgresql.sql](../data/raw/schema_postgresql.sql) y de la exploración realizada en [notebooks/1_eda_y_calidad.ipynb](../notebooks/1_eda_y_calidad.ipynb). Sustituye al `diccionario_datos.xlsx` original (no provisto con el dataset).

- **Fecha de corte del dataset:** `2026-04-30`
- **Granularidad:** un cliente puede tener múltiples créditos; cada crédito tiene múltiples cuotas (pagos); cada cliente puede generar múltiples eventos de app.
- **Convención de IDs:** `CL#####` (clientes), `CR#####` (créditos), `PG######` (pagos), `EV######` (eventos).

---

## 1. `clientes` — Maestro de clientes

Una fila por cliente. Variables conocidas al momento del análisis (información de adquisición y enriquecimiento).

| Campo | Tipo | Descripción | Notas de calidad / uso |
|---|---|---|---|
| `cliente_id` | VARCHAR(20) PK | Identificador único del cliente. | Llave para joins con `creditos`, `pagos`, `eventos_app`. |
| `fecha_registro` | DATE | Fecha de alta del cliente en TumiPay. | Útil para antigüedad del cliente al momento de un crédito. |
| `departamento` | VARCHAR(80) | Departamento de residencia. | Sin variantes detectadas. |
| `ciudad` | VARCHAR(80) | Ciudad de residencia. | **Calidad:** variantes ortográficas (`bogot`, `medellin `, `barranquila`). Normalizado en `DataCleaner.clean_clientes`. |
| `edad` | INTEGER | Edad declarada en años. | **Calidad:** outliers fuera de `[18, 110]` se imputan a la mediana (35). |
| `genero` | VARCHAR(20) | Género autodeclarado. | — |
| `estrato` | INTEGER | Estrato socioeconómico (1–6). | — |
| `nivel_educativo` | VARCHAR(80) | Nivel educativo alcanzado. | — |
| `ocupacion` | VARCHAR(80) | Ocupación declarada. | — |
| `ingreso_mensual_estimado` | NUMERIC(14,2) | Ingreso mensual reportado (COP). | **Calidad:** nulos imputados con mediana ($2.200.000). Outlier extremo `CL00493 = 120M COP` con estrato 2 conservado en raw; winsorizado al P99 en `v_clientes_bi` con `flag_ingreso='error_captura'`. |
| `canal_adquisicion` | VARCHAR(80) | Canal por el que se captó al cliente. | — |
| `score_externo` | NUMERIC(8,2) | Score de buró externo. | **Calidad:** nulos imputados con mediana (580). Predictor dominante del modelo. |
| `tiene_producto_ahorro` | BOOLEAN | Indicador de tenencia de producto de ahorro. | — |
| `numero_dependientes` | INTEGER | Número de dependientes económicos. | — |
| `dispositivo_principal` | VARCHAR(40) | Dispositivo principal del cliente. | — |
| `email_hash` | VARCHAR(80) | Hash del correo (anonimizado). | **No usar como feature**: alta cardinalidad, sin valor predictivo. Excluido del modelo. |

---

## 2. `creditos` — Créditos desembolsados

Una fila por crédito desembolsado. Contiene las condiciones pactadas al momento de la originación.

| Campo | Tipo | Descripción | Notas de calidad / uso |
|---|---|---|---|
| `credito_id` | VARCHAR(20) PK | Identificador único del crédito. | Llave para `pagos` y `predicciones_riesgo`. |
| `cliente_id` | VARCHAR(20) FK | Cliente titular del crédito. | Join con `clientes`. |
| `fecha_desembolso` | DATE | Fecha de desembolso del crédito. | **Crítico**: define la línea temporal anti-leakage. Cualquier feature de comportamiento debe filtrar `<= fecha_desembolso`. |
| `producto_credito` | VARCHAR(80) | Tipo de producto (Libranza, Libre Inversión, etc.). | **Calidad:** nulos imputados a categoría `'Desconocido'`. |
| `monto_credito` | NUMERIC(14,2) | Valor desembolsado (COP). | Segmento `500K–1M` con mayor tasa de mora (34.2%). |
| `plazo_meses` | INTEGER | Plazo pactado en meses. | — |
| `tasa_interes_mensual` | NUMERIC(8,4) | Tasa de interés mensual. | Predictor relevante (selección adversa). |
| `valor_cuota_pactada` | NUMERIC(14,2) | Valor de la cuota fija (COP). | — |
| `canal_originacion` | VARCHAR(80) | Canal de originación (App, Sucursal, etc.). | — |
| `score_interno_originacion` | NUMERIC(8,2) | Score interno calculado al originar. | Disponible en originación → válido como feature. |
| `relacion_cuota_ingreso` | NUMERIC(10,4) | Cuota / ingreso mensual estimado. | Predictor importante (top-2 en feature importance). |
| `politica_aprobacion` | VARCHAR(40) | Política aplicada en la aprobación. | — |
| `estado_credito_operativo` | VARCHAR(40) | Estado actual: `Activo`, `Finalizado`, `Mora moderada`, `Mora severa`. | **⚠️ DATA LEAKAGE:** se conoce **después** de observar comportamiento de pago. **Excluido** del feature set del modelo. Su inclusión inflaba ROC-AUC artificialmente a 0.98. |

---

## 3. `pagos` — Historial de cuotas

Una fila por cuota generada. Fuente para construir la variable objetivo `es_moroso`.

| Campo | Tipo | Descripción | Notas de calidad / uso |
|---|---|---|---|
| `pago_id` | VARCHAR(20) PK | Identificador único de la cuota. | — |
| `credito_id` | VARCHAR(20) FK | Crédito al que pertenece la cuota. | Join con `creditos`. |
| `cliente_id` | VARCHAR(20) FK | Cliente titular. | Redundante con `creditos.cliente_id`. |
| `numero_cuota` | INTEGER | Número secuencial de la cuota en el crédito. | — |
| `fecha_vencimiento` | DATE | Fecha en que la cuota debe pagarse. | **Crítico:** se usa como filtro temporal del target (`fecha_vencimiento <= CUTOFF_DATE`) para incluir cuotas en mora activa sin `fecha_pago`. |
| `fecha_pago` | DATE | Fecha real del pago. NULL si está en mora activa. | **No filtrar por aquí**: excluiría a los peores morosos. |
| `valor_cuota` | NUMERIC(14,2) | Valor esperado de la cuota. | — |
| `valor_pagado` | NUMERIC(14,2) | Valor efectivamente pagado. | Permite detectar pagos parciales. |
| `medio_pago` | VARCHAR(80) | Medio usado para pagar. | — |
| `estado_pago` | VARCHAR(40) | Estado de la cuota. | Post-hoc: **no usar como feature**, sí como auditoría. |
| `dias_mora` | INTEGER | Días transcurridos entre `fecha_vencimiento` y `fecha_pago` (o corte). | **Base del target**: `es_moroso = 1` si `MAX(dias_mora) > 30` para cuotas con `fecha_vencimiento <= CUTOFF_DATE`. |

### Variable objetivo derivada — `es_moroso`

| Campo | Definición | Justificación |
|---|---|---|
| `es_moroso` | `1` si el crédito tiene al menos una cuota con `dias_mora > 30` y `fecha_vencimiento <= 2026-04-30`; `0` en caso contrario. | Umbral estándar de "default" en cartera de consumo. El filtro por `fecha_vencimiento` (no por `fecha_pago`) preserva morosos activos sin pago registrado, elevando la tasa real de mora del 25.4% al 27.2%. |

---

## 4. `eventos_app` — Interacción digital (opcional)

Una fila por evento de interacción del cliente con la app, web o canales de contacto.

| Campo | Tipo | Descripción | Notas de calidad / uso |
|---|---|---|---|
| `evento_id` | VARCHAR(20) PK | Identificador del evento. | — |
| `cliente_id` | VARCHAR(20) FK | Cliente que generó el evento. | Join con `clientes`. |
| `fecha_evento` | DATE | Fecha del evento. | **Crítico anti-leakage:** los agregados deben filtrar `fecha_evento < creditos.fecha_desembolso`. |
| `tipo_evento` | VARCHAR(80) | Tipo de interacción (login, pago_fallido, simulacion, etc.). | Base de las features `prev_evento_*`. |
| `canal` | VARCHAR(40) | Canal usado (app, web, call center). | — |
| `resultado_evento` | VARCHAR(80) | Resultado del evento (éxito, fallo, abandonado). | — |
| `dispositivo` | VARCHAR(40) | Dispositivo usado. | — |
| `duracion_sesion_seg` | INTEGER | Duración de la sesión en segundos. | Base de `prev_evento_sesion_seg_tot`. |

### Features derivadas (calculadas pre-desembolso)

| Feature | Cálculo | Señal de negocio |
|---|---|---|
| `prev_evento_pago_fallido` | `COUNT(*)` de eventos con `tipo_evento='pago_fallido'` y `fecha_evento < fecha_desembolso`. | Tensión financiera previa. |
| `prev_evento_sesion_seg_tot` | `SUM(duracion_sesion_seg)` con `fecha_evento < fecha_desembolso`. | Engagement / madurez digital. |

---

## 5. Convenciones de uso para modelado

| Regla | Aplicación |
|---|---|
| **Anti-leakage temporal** | Toda feature derivada de `pagos` o `eventos_app` debe filtrar por la fecha de desembolso del crédito evaluado. |
| **Exclusión de variables post-hoc** | `estado_credito_operativo`, `fecha_pago`, `valor_pagado`, `dias_mora` y `estado_pago` **no** son features del modelo. |
| **Exclusión de IDs y hashes** | `cliente_id`, `credito_id`, `pago_id`, `evento_id`, `email_hash` se descartan por alta cardinalidad y nulo poder predictivo. |
| **Tratamiento de outliers** | Conservados en raw; corregidos en capa de features (`v_clientes_bi`, `DataCleaner`) con flags de trazabilidad. |

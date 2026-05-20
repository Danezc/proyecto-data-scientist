# Dataset base - Prueba Data Science Engineer TUMIPAY

## Descripción general

Dataset sintético diseñado para evaluar habilidades de análisis, SQL, modelado, visualización y documentación en un contexto fintech.
La información representa clientes, créditos, pagos y eventos de uso de canales digitales.

Fecha de corte del dataset: **2026-04-30**

## Archivos

1. `clientes.csv`
   - Maestro de clientes.
   - Contiene características demográficas, de adquisición, comportamiento general y variables disponibles al momento de análisis.

2. `creditos.csv`
   - Créditos desembolsados.
   - Contiene condiciones del crédito, canal de originación, score interno de originación y estado operativo.

3. `pagos.csv`
   - Historial de cuotas y pagos.
   - Permite construir indicadores de comportamiento de pago y una variable objetivo de mora.

4. `eventos_app.csv`
   - Eventos opcionales de interacción digital.
   - Puede usarse para enriquecer el análisis, pero no es obligatorio.

5. `schema_postgresql.sql`
   - Script base para crear tablas en PostgreSQL.

6. `diccionario_datos.xlsx`
   - Diccionario de campos y reglas de uso.

## Recomendación para la prueba

La variable objetivo no viene construida de forma directa. Se espera que el candidato defina una etiqueta de riesgo a partir de `pagos.csv`, por ejemplo:
- crédito con mora mayor a 30 días,
- probabilidad de mora en las primeras cuotas,
- cliente con máximo atraso superior a cierto umbral.

Se espera que el candidato evite fuga de información: variables calculadas después del comportamiento de pago no deberían usarse como predictores si el objetivo es anticipar riesgo.

## Consideraciones

El dataset contiene intencionalmente algunos problemas de calidad:
- valores nulos,
- valores atípicos,
- inconsistencias menores de texto,
- posibles duplicidades de negocio,
- pagos parciales o vencidos,
- datos que requieren validación antes de ser usados.

El objetivo no es limpiar todo de forma perfecta, sino demostrar criterio analítico, documentación de supuestos y decisiones justificadas.

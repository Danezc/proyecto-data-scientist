# Risk Engine & Financial AI Advisor

![Python](https://img.shields.io/badge/python-3.14-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-orange.svg)
![LangGraph](https://img.shields.io/badge/LangGraph-RAG-purple.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16%20%2B%20pgvector-336791.svg)
![Docker](https://img.shields.io/badge/docker-compose-0db7ed.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.7635-brightgreen.svg)

> **Auditoría y Corrección de Arquitectura:** En esta entrega se detectaron y corrigieron problemas metodológicos graves en el pipeline original, incluyendo una definición incorrecta del target (pérdida de morosos activos por filtro nulo de pagos) y fuga de información masiva (data leakage) al incluir variables operativas post-desembolso (`estado_credito_operativo`). El modelo actual representa un motor de originación robusto y alineado a las mejores prácticas de la industria de riesgo de crédito.

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#1-resumen-ejecutivo)
2. [Estructura del Repositorio](#2-estructura-del-repositorio)
3. [Auditoría de Calidad y Fuga de Información](#3-auditoría-de-calidad-y-fuga-de-información)
4. [Modelo de Riesgo — Resultados](#4-modelo-de-riesgo--resultados)
5. [Agente RAG Financiero](#5-agente-rag-financiero)
6. [SQL y Capa de Transformación de Datos](#6-sql-y-capa-de-transformación-de-datos)
7. [Dashboard de Power BI](#7-dashboard-de-power-bi)
8. [Inicio Rápido](#8-inicio-rápido)
9. [Uso de Inteligencia Artificial (Declaración)](#9-uso-de-inteligencia-artificial-declaración)

---

## 1. Resumen Ejecutivo

Este repositorio es el núcleo analítico de **TumiPay**, una fintech de crédito de consumo. El sistema integra de extremo a extremo las siguientes capas:

| Capa | Componente | Descripción |
|---|---|---|
| **Datos** | ETL + ABT (Analytical Base Table) | Consolidación de 4 fuentes raw (`clientes`, `creditos`, `pagos`, `eventos_app`) en una ABT limpia con 42 características. |
| **SQL** | SQL Transformations & Analytics | Transformaciones y queries analíticos en `/sql` mediante CTEs, Joins y funciones ventana. |
| **Riesgo (ML)** | Modelo LightGBM Libre de Leakage | Predicción robusta de probabilidad de mora en originación con ROC-AUC de **0.7635**. |
| **Visualización** | Dashboard en Power BI | Monitor de riesgo y comportamiento de cartera con mockups de diseño premium en `/dashboard`. |
| **IA Conversacional** | Agente RAG con LangGraph | Consultas en lenguaje natural sobre políticas, estadísticas y fichas de clientes en PostgreSQL + PGVector. |

---

## 2. Estructura del Repositorio

El repositorio sigue una arquitectura limpia y modular:

```
proyecto-data-scientist/
├── 0_Master_Pipeline.ipynb   # Orquestador principal (ETL → Train → DB → RAG)
├── src/
│   ├── config.py             # Configuración centralizada (pydantic-settings)
│   ├── ingesta.py            # Carga de CSV con tipado estricto
│   ├── procesamiento.py      # Limpieza (ciudades estandarizadas, outliers, nulos)
│   ├── consolidacion.py      # Ingeniería de features y agregación digital → ABT
│   ├── train_mora.py         # Pipeline LightGBM (split, train, eval, export)
│   ├── rag_agent.py          # Agente RAG con LangGraph + PGVector
│   └── main.py               # FastAPI — endpoints de scoring e inferencia
├── sql/
│   └── transformaciones.sql  # Queries SQL con CTEs y window functions
├── dashboard/
│   ├── README.md             # Documentación del dashboard e indicadores
│   └── capturas/             # Captura del dashboard premium generado
├── scripts/
│   └── populate_rag.py       # Indexa documentos en el vector store
├── notebooks/
│   ├── 0_Master_Pipeline.ipynb
│   └── 1_eda_y_calidad.ipynb # Notebook interactivo de EDA y Calidad
├── data/
│   ├── raw/                  # CSVs originales (no versionados)
│   └── processed/            # ABT en parquet (no versionado)
├── models/                   # modelo_mora.pkl (no versionado)
├── docker-compose.yml        # PostgreSQL 16 + pgvector
└── .env.example              # Plantilla de variables de entorno
```

```mermaid
graph LR
    A[CSV raw] --> B[ETL / ABT]
    B --> C[LightGBM]
    C --> D[(PostgreSQL\npredicciones_riesgo)]
    D --> E[Power BI]
    D --> F[PGVector\nrag_conocimiento]
    F --> G[Agente RAG]
    G --> H[NVIDIA NIM\nllama-3.1-8b]
```

---

## 3. Auditoría de Calidad y Fuga de Información

### A. Corrección del Target `es_moroso` (Lógica de Negocio)
En la versión original, la lógica del target evaluaba pagos con el filtro `fecha_pago <= cutoff_date`.
* **El Problema**: Los créditos en mora activa no pagados tienen `fecha_pago` nulo (`NULL`), por lo cual eran erróneamente excluidos de la tabla de pagos o clasificados como no-morosos (`es_moroso = 0`). Esto resultaba en la pérdida de los peores perfiles de riesgo.
* **La Solución**: Cambiar el filtro a `fecha_vencimiento <= cutoff_date`. Todo crédito con una cuota cuya fecha de vencimiento haya expirado antes del corte se evalúa: si su pago no ha sido registrado o su `dias_mora` supera 30 días, se clasifica correctamente como `es_moroso = 1`. Esto elevó el número de morosos reales de 387 a 416.

### B. Mitigación de Fuga de Información (Data Leakage)
* **El Problema**: La variable `estado_credito_operativo` (valores: *Activo, Finalizado, Mora moderada, Mora severa*) se utilizaba en el entrenamiento del modelo. Al ser una variable que se actualiza *después* del desembolso, provocaba un sobreajuste masivo en el modelo (ROC-AUC ficticio de `0.98`), inútil para predecir al momento de originación (desembolso).
* **La Solución**: Excluir estrictamente `estado_credito_operativo` y el identificador sesgado `email_hash` del set de entrenamiento. La tasa de acierto del modelo se redujo a una métrica real de **ROC-AUC = 0.7635**, la cual es excelente y robusta para la industria financiera de consumo.

### C. Calidad de Datos (Data Cleaning)
* **Ciudades Desnormalizadas**: Se identificaron múltiples variantes y faltas de ortografía (ej: *'bogot'*, *'medellin '* con espacios, *'barranquila'*). Se implementó un mapeo robusto en `DataCleaner.clean_clientes` que unifica los valores a su título correcto (*Bogotá*, *Medellín*, *Barranquilla*).
* **Tratamiento de Nulos**: Los nulos en `ingreso_mensual_estimado` y `score_externo` fueron imputados utilizando la mediana general para evitar distorsiones por outliers. Los nulos en `producto_credito` se marcaron como `'Desconocido'`.

### D. Enriquecimiento de Datos Digitales (App Events)
Para agregar valor sin violar la línea temporal (anti-leakage), se cruzó la tabla `eventos_app.csv` y se calcularon agregaciones **únicamente de eventos ocurridos antes de la fecha de desembolso** de cada crédito. Esto introdujo variables clave como:
* `prev_evento_pago_fallido`: Número de intentos de pago fallidos del cliente previos al nuevo desembolso.
* `prev_evento_sesion_seg_tot`: Duración total del cliente interactuando con la app antes de tomar el crédito.

---

## 4. Modelo de Riesgo — Resultados

| Métrica | Valor Real (Leakage-Free) | Valor Ficticio (Con Leakage) |
|---|---|---|
| **ROC-AUC (Test)** | **0.7635** | 0.9844 (Sobreajustado) |
| **Accuracy** | **71%** | 97% |
| **Mora Global Evaluada** | **27.2%** | 25.4% |

### Top Variables más Importantes (Feature Importance)
1. **`score_externo` (126 pts)**: El historial crediticio provisto por centrales de riesgo es el predictor más fuerte.
2. **`relacion_cuota_ingreso` (92 pts)**: La carga financiera del crédito respecto a los ingresos del cliente.
3. **`mes_desembolso` (81 pts)**: La estacionalidad temporal de la colocación de cartera.
4. **`ingreso_mensual_estimado` (71 pts)**: Capacidad monetaria del cliente.
5. **`tasa_interes_mensual` (66 pts)**: A mayor tasa cobrada, mayor el riesgo de mora (efecto selección/riesgo moral).

---

## 5. Agente RAG Financiero

El agente responde preguntas sobre el portafolio en lenguaje natural utilizando **LangGraph** y **PGVector**:
* **Embedding Model**: `intfloat/multilingual-e5-small` cargado en local para búsquedas semánticas sobre base de datos.
* **Modelo LLM**: `meta/llama-3.1-8b-instruct` consumido a través de NVIDIA NIM.
* **Base de Conocimiento**: Indexación de 1,538 chunks conteniendo resúmenes ejecutivos, políticas de cobro, glosario y fichas individuales de cada cliente evaluado.

---

## 6. SQL y Capa de Transformación de Datos

Las transformaciones completas de datos para recrear la ABT de modelado y responder preguntas de negocio se encuentran consolidadas en el archivo [sql/transformaciones.sql](sql/transformaciones.sql).

El archivo incluye:
* CTEs complejas para la estandarización e imputación de nulos.
* Filtros temporales anti-leakage.
* Funciones ventana (`NTILE`, `OVER`) para clasificar comportamientos y segmentar clientes por riesgo.
* Métricas analíticas de agregación de pagos mensuales y retrasos.

---

## 7. Dashboard de Power BI

Para responder a las necesidades de negocio de visualización y KPIs, se estructuró un dashboard interactivo de diseño premium.

* La documentación de métricas y decisiones asociadas está disponible en el [README de Dashboard](dashboard/README.md).
* La visualización del panel interactivo en alta definición se puede apreciar a continuación:

![Dashboard TumiPay](dashboard/capturas/dashboard_riesgo.png)

---

## 8. Inicio Rápido

**Requisitos:** Docker, Python 3.11+, y una API Key de [NVIDIA NIM](https://build.nvidia.com).

```bash
# 1. Clonar el repositorio
git clone https://github.com/<tu-usuario>/proyecto-data-scientist.git
cd proyecto-data-scientist

# 2. Configurar variables de entorno (.env)
cp .env.example .env
# Editar .env y colocar tu NVIDIA_API_KEY y cambiar DATABASE_URL si es necesario

# 3. Levantar base de datos local (PostgreSQL + PGVector)
docker-compose up -d

# 4. Crear entorno virtual e instalar paquetes
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 5. Ejecutar la orquestación del pipeline (ETL -> ML Train -> DB Load -> RAG Populate)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace 0_Master_Pipeline.ipynb
```

---

## 9. Variables de Entorno

Copia `.env.example` a `.env` y completa los valores:

```bash
cp .env.example .env
```

| Variable | Requerida | Descripción |
|---|---|---|
| `DATABASE_URL` | ✅ | Connection string PostgreSQL |
| `NVIDIA_API_KEY` | ✅ | API key de [build.nvidia.com](https://build.nvidia.com) |
| `CUTOFF_DATE` | — | Fecha de corte del modelo (default: `2026-04-30`) |
| `LANGCHAIN_API_KEY` | Opcional | Telemetría LangSmith |
| `LANGCHAIN_TRACING_V2` | Opcional | `true` para activar trazas |

> **Seguridad:** `.env` está en `.gitignore` y nunca debe subirse al repositorio. El archivo `.env.example` es la única referencia pública.

---

## 10. Uso de Inteligencia Artificial (Declaración)

Conforme a las políticas de transparencia e integridad de la prueba:
1. **GitHub Copilot / Claude 3.5 Sonnet**: Utilizados como asistentes para la estructuración inicial de código (boilerplate) en los módulos de `FastAPI` e infraestructura de `LangGraph`.
2. **Generación de UI / Diseño**: Se utilizó la herramienta generativa local para mockups gráficos de Power BI a fin de presentar un entregable estético, funcional e inmediato sobre el diseño premium propuesto para la visualización del negocio.
3. **Desarrollo del Core**: La detección de fugas de información, corrección matemática del target de pagos sin fecha y modelado analítico LightGBM libre de leakage son resultado del análisis técnico y autoría directa.

import logging
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, Field
from pathlib import Path

from src.llm_expert import FintechLLMExpert, QueryRequest
from src.config import settings

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Definir la aplicación FastAPI
app = FastAPI(
    title="TUMIPAY Risk Engine API",
    description="API para predicción de riesgo y analítica conversacional",
    version="1.0.0"
)

# Esquema Pydantic para los features del modelo
class PredictionRequest(BaseModel):
    edad: int = Field(..., example=30)
    ingreso_mensual_estimado: float = Field(..., example=1500.50)
    score_externo: float = Field(..., example=650.0)
    monto_credito: float = Field(..., example=5000.0)
    plazo_meses: int = Field(..., example=12)
    valor_cuota_pactada: float = Field(..., example=500.0)


class PredictionResponse(BaseModel):
    score_riesgo_mora: float
    decision: str

# Variable global para cachear el modelo en memoria
model_artifact = None
llm_expert = None

@app.on_event("startup")
def load_assets():
    """Carga los modelos en memoria al iniciar la API para mínima latencia."""
    global model_artifact, llm_expert
    model_path = settings.MODELS_DIR / "modelo_mora.pkl"
    
    try:
        model_artifact = joblib.load(model_path)
        logger.info("Modelo de riesgo cargado en memoria exitosamente.")
    except FileNotFoundError:
        logger.warning("Archivo de modelo no encontrado. Asegúrese de entrenar primero el modelo.")
        model_artifact = None
        
    try:
        llm_expert = FintechLLMExpert()
    except Exception as e:
        logger.warning(f"Advertencia al instanciar LLM Expert: {e}")

@app.post("/predict", response_model=PredictionResponse)
def predict_risk(data: PredictionRequest):
    """
    Evalúa un nuevo perfil de cliente y retorna un score de probabilidad de default (mora).
    """
    if not model_artifact:
        raise HTTPException(status_code=503, detail="El modelo no está cargado en el sistema.")
        
    # Transformar a DataFrame asegurando el orden de columnas del entrenamiento
    features_requeridos = model_artifact['features']
    input_data = data.model_dump()
    
    df_input = pd.DataFrame([input_data])
    
    # Definir variables numéricas para rellenar con 0.0, el resto con 'Desconocido'
    numerical_cols = {
        'edad', 'ingreso_mensual_estimado', 'score_externo', 'monto_credito',
        'plazo_meses', 'tasa_interes_mensual', 'valor_cuota_pactada', 
        'score_interno_originacion', 'relacion_cuota_ingreso', 'numero_dependientes',
        'antiguedad_cliente_dias', 'mes_desembolso', 'dia_semana_desembolso',
        'prev_evento_actualizacion_datos', 'prev_evento_consulta_saldo', 'prev_evento_login',
        'prev_evento_pago_exitoso', 'prev_evento_pago_fallido', 'prev_evento_pago_iniciado',
        'prev_evento_simulacion_credito', 'prev_evento_solicitud_soporte', 
        'prev_evento_sesion_seg_tot', 'prev_evento_sesion_seg_avg',
        'estrato'
    }
    
    for col in features_requeridos:
        if col not in df_input.columns:
            if col in numerical_cols:
                df_input[col] = 0.0
            elif col == 'tiene_producto_ahorro':
                df_input[col] = False
            else:
                df_input[col] = 'Desconocido'
                
    df_input = df_input[features_requeridos]
    
    # Alinear variables categóricas con las categorías exactas del entrenamiento
    pandas_categorical = getattr(model_artifact['model']._Booster, 'pandas_categorical', None)
    if pandas_categorical:
        # Las columnas categóricas en el dataset son exactamente las no-numéricas (excluyendo booleanos)
        cat_cols = [c for c in features_requeridos if c not in numerical_cols and c != 'tiene_producto_ahorro']
        for idx, col in enumerate(cat_cols):
            if idx < len(pandas_categorical):
                categories = pandas_categorical[idx]
                # Convertir a string primero para evitar Pandas4Warning
                df_input[col] = pd.Categorical(df_input[col].astype(str), categories=categories)
    else:
        # Fallback simple
        for col in features_requeridos:
            if col not in numerical_cols and col != 'tiene_producto_ahorro':
                df_input[col] = df_input[col].astype('category')
            
    model = model_artifact['model']
    proba = model.predict_proba(df_input)[0][1]
    
    decision = "APROBADO" if proba < 0.5 else "RECHAZADO"
    
    return PredictionResponse(
        score_riesgo_mora=round(float(proba), 4),
        decision=decision
    )

@app.post("/ask-analyst")
def ask_analyst(query: QueryRequest):
    """
    Consulta al agente LLM sobre temas de analítica y negocio.
    Recibe un JSON request con el campo explícito 'question'.
    """
    if not llm_expert:
        raise HTTPException(status_code=503, detail="El servicio de LLM no está disponible en este momento.")
        
    respuesta = llm_expert.generate_response(query.question)
    return {"question": query.question, "respuesta": respuesta}

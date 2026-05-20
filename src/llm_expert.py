import os
import json
import logging
from openai import OpenAI
from pydantic import BaseModel
from pathlib import Path
from src.config import settings

logger = logging.getLogger(__name__)

class QueryRequest(BaseModel):
    question: str

class FintechLLMExpert:
    """
    Agente de IA especializado en analítica de riesgos para TUMIPAY.
    Emplea inyección de contexto estático para responder consultas de negocio.
    """
    
    def __init__(self, context_file: str = "contexto_negocio.json"):
        """
        Inicializa cargando variables de entorno y el contexto serializado.
        """
        api_key = os.getenv("NVIDIA_API_KEY", "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")
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        
        # Carga el archivo de contexto
        self.context_path = settings.DATA_PROCESSED_DIR / context_file
        self.context_data = self._load_context_file()
        self.system_prompt = self._build_system_prompt()
        logger.info("FintechLLMExpert inicializado con éxito.")

    def _load_context_file(self) -> dict:
        """Lee el contexto de negocio desde un JSON manejando excepciones."""
        if not self.context_path.exists():
            logger.warning(f"Archivo de contexto {self.context_path} no hallado. Se usará contexto vacío.")
            return {}
        try:
            with open(self.context_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Fallo al leer contexto: {e}")
            return {}

    def _build_system_prompt(self) -> str:
        """
        Crea un System Prompt determinista empoderado con los datos de negocio extraídos de JSON.
        """
        metrics = "\n".join([f"- {k}: {v}" for k, v in self.context_data.items()])
        
        return f"""Eres un Consultor de Riesgo Senior de TUMIPAY. 
Responde la pregunta del usuario con base exclusiva y estricta en el siguiente contexto extraído del motor ML:

CONTEXTO TÉCNICO Y DE NEGOCIO:
{metrics}

Instrucciones adicionales:
1. Responde de manera concisa, ejecutiva y orientada a la toma de decisiones.
2. Si la pregunta del usuario excede el contexto proporcionado, responde que no tienes información suficiente en el último corte.
3. Puedes formatear métricas importantes en Bullet Points.
"""

    def generate_response(self, user_question: str) -> str:
        """
        Evalúa una pregunta de negocio vía el modelo LLM.
        """
        try:
            logger.info(f"Procesando evaluación LLM para la pregunta: {user_question[:30]}...")
            completion = self.client.chat.completions.create(
                model="minimaxai/minimax-m2.7",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_question}
                ],
                temperature=0.3, # Formato analítico: Baja alucinación
                top_p=0.95,
                max_tokens=8192,
                stream=True
            )
            
            respuesta_completa = ""
            for chunk in completion:
                if not getattr(chunk, "choices", None):
                    continue
                if chunk.choices[0].delta.content is not None:
                    respuesta_completa += chunk.choices[0].delta.content
                    
            return respuesta_completa.strip()
        except Exception as e:
            logger.error(f"Error generativo en LLM: {str(e)}")
            return "Lo siento, ha ocurrido un error al procesar tu solicitud analítica."
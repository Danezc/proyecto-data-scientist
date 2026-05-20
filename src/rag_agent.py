import os
import re
import logging
from sqlalchemy import create_engine, text
from typing import TypedDict, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import PGVector
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

# Silenciar librerías de infraestructura (HTTP, embeddings, vector store)
for _noisy in ["httpx", "httpcore", "sentence_transformers", "transformers",
               "huggingface_hub", "langchain_community", "openai"]:
    logging.getLogger(_noisy).setLevel(logging.WARNING)

# 1. Definición del Estado de LangGraph
class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    context: str

class RAGAgentPipeline:
    """
    Pipeline controlado por LangGraph para inyectar contexto desde PGVector y generar respuestas.
    Asegura que el LLM responda estricamente sobre la base de datos documental.
    """
    
    def __init__(self, db_connection: str, nvidia_api_key: str = ""):
        if not nvidia_api_key:
            raise EnvironmentError(
                "NVIDIA_API_KEY no configurada. "
                "Agrega 'NVIDIA_API_KEY=<tu-clave>' en el archivo .env y pásala con settings.NVIDIA_API_KEY."
            )
        
        # 1. Embedding Model (Local Open Source - Qwen / Sentence Transformers)
        # Nota: Qwen3 u otro modelo equivalente puede cargarse usando huggingface. 
        # Tamaño 0.6B es espectacular para RAG asíncrono con buena retención de semántica financiera.
        self.embeddings = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small", # Cambiar por el repo exacto de Qwen si está disponible en local
            model_kwargs={'device': 'cpu'}, # 'cuda' si tienes GPU libre
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # 2. Vector Store (PostgreSQL + PGVector)
        self.vector_store = PGVector(
            connection_string=db_connection,
            embedding_function=self.embeddings,
            collection_name="rag_conocimiento"
        )
        
        # 3. LLM Model (Nvidia NIM — parámetros según documentación oficial de NVIDIA)
        self.llm = ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=nvidia_api_key,
            model="minimaxai/minimax-m2.7",
            temperature=1,
            top_p=0.95,
            max_tokens=8192,
            streaming=True,   # streaming reduce la latencia del primer token
            timeout=60.0,     # minimax-m2.7 puede tardar más que modelos pequeños
        )
        
        # 4. Construcción del Grafo (LangGraph Workflow)
        self.graph = self._build_graph()

    def _retrieve_node(self, state: AgentState) -> dict:
        """Nodo 1: Búsqueda Semántica con Timeout preventivo"""
        last_message = state["messages"][-1].content
        logger.debug(f"Buscando contexto para: {last_message}")
        
        # Retrieval top k=3
        docs = self.vector_store.similarity_search(last_message, k=3)
        context_str = "\n".join([doc.page_content for doc in docs])
        
        if not context_str:
            context_str = "No se encontró información relevante en los datos de la corporación TUMIPAY."
            
        return {"context": context_str}

    def _generate_node(self, state: AgentState) -> dict:
        """Nodo 2: Generación restringida al contexto"""
        context = state["context"]
        question = state["messages"][-1].content
        
        # Instrucción Anti-Hallucination
        system_prompt = f"""Eres el Analista Principal de Riesgos de TUMIPAY.
Debes responder la pregunta del usuario BASÁNDOTE ÚNICA Y EXCLUSIVAMENTE en el siguiente contexto extraído de nuestra base de datos vectorial PostgreSQL. 
Si el contexto no contiene la respuesta, di claramente "No dispongo de esa información en los expedientes actuales". 
No inventes datos ni asumas políticas genéricas. Guarda un tono ejecutivo.

CONTEXTO:
{context}
"""
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=question)
        ]
        
        # Invocación al LLM con Timeout
        response = self.llm.invoke(messages)
        return {"messages": [response]}

    def _build_graph(self):
        """Orquesta la máquina de estados"""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        
        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
        
    def stream_query(self, query: str):
        """Ejecuta una consulta en streaming usando LangGraph"""
        state_input = {"messages": [HumanMessage(content=query)], "context": ""}
        
        # Para exponer el streaming (yield) en FastAPI
        for output in self.graph.stream(state_input):
            # LangGraph emite en el iterador un dict con el nodo procesado
            for key, value in output.items():
                logger.debug(f"LangGraph -> Nodo completado: {key}")
                if key == "generate":
                    yield value["messages"][-1].content

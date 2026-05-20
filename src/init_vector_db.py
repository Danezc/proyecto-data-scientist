import logging
from src.rag_agent import TextToSQLRAGAgent
from langchain_core.documents import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def embed_database_schema_to_vector():
    """
    Lee el archivo de esquema ddl e inserta su definición como contexto en PGVector
    para que nuestro LLM Text-To-SQL sepa qué tablas existen al ser preguntado.
    """
    db_url = "postgresql://root:root@localhost:5432/tumipay_db" # Ajusta con tus credenciales
    
    agent = TextToSQLRAGAgent(db_connection=db_url)
    
    schema_path = "schema_postgresql.sql"
    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            esquema_completo = f.read()
            
        # Podríamos partirlo text-splitter si fuera inmenso, 
        # pero es útil entregarle a todo el DDL un bloque o bloques por tablas
        
        doc = Document(
            page_content=f"Este es el esquema de tablas del Core Bancario de TUMIPAY. Úsalo para crear queries SQL:\n{esquema_completo}",
            metadata={"source": "schema_postgresql.sql"}
        )
        
        logger.info("Insertando esquema de DB hacia PGVector, calculando Embeddings...")
        agent.vector_store.add_documents([doc])
        logger.info("Esquema DDL introducido exitosamente a la base vectorial.")
        
    except Exception as e:
        logger.error(f"Error introduciendo esquema a vector db: {e}")

if __name__ == "__main__":
    embed_database_schema_to_vector()

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Cargar las variables de entorno desde el archivo .env
load_dotenv()

# Obtener la URL de conexión
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise ValueError("No se encontró DATABASE_URL en el archivo .env")

# Crear el motor de base de datos
engine = create_engine(DATABASE_URL)

# Script SQL para truncar las tablas requeridas
truncate_query = text("""
TRUNCATE TABLE 
    raw_eventos_app, 
    raw_pagos, 
    raw_creditos, 
    raw_clientes, 
    predicciones_riesgo,
    dim_cliente,
    dim_producto_credito,
    dim_tiempo,
    fact_creditos,
    abt_analitica_riesgo,
    perfil_descriptivo_cliente,
    langchain_pg_collection,
    langchain_pg_embedding
CASCADE;
""")

def limpiar_base_de_datos():
    print("Conectando a la base de datos...")
    try:
        with engine.connect() as conn:
            print("Ejecutando TRUNCATE en las tablas...")
            conn.execute(truncate_query)
            conn.commit()
            print("Tablas truncadas exitosamente.")
    except Exception as e:
        print(f"Error al intentar limpiar la base de datos: {e}")

if __name__ == "__main__":
    limpiar_base_de_datos()

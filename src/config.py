import os
from pathlib import Path
from pydantic_settings import BaseSettings

# Determinar el directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """
    Configuración centralizada del proyecto TUMIPAY.
    Permite cargar variables desde entorno o utilizar valores por defecto.
    """
    
    # Rutas de directorios
    DATA_RAW_DIR: Path = BASE_DIR / "data" / "raw"
    DATA_PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Nombres de archivos fuente
    CLIENTES_FILE: str = "clientes.csv"
    CREDITOS_FILE: str = "creditos.csv"
    PAGOS_FILE: str = "pagos.csv"
    
    # Lógica de Negocio
    CUTOFF_DATE: str = "2026-04-30"
    
    # Base de Datos (PGVector / PostgreSQL)
    DATABASE_URL: str = "postgresql://root:root@localhost:5432/tumipay_db"
    
    # Credenciales de API (leer siempre desde .env, nunca hardcodear)
    NVIDIA_API_KEY: str = ""
    
    class Config:
        env_file = str(BASE_DIR / ".env")   # ruta absoluta: no depende del cwd del kernel
        env_file_encoding = 'utf-8'
        extra = 'ignore'
        env_ignore_empty = True             # ignora variables OS vacías, usa el valor del .env

# Instancia global de las configuraciones
settings = Settings()

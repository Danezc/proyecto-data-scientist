import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List

from src.config import settings

# Configuración de logging nativo
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

class DataIngestor:
    """
    Se encarga de la extracción e ingesta segura de fuentes de datos (CSV).
    Aplica tipado estricto y parseo de fechas de manera robusta.
    """
    
    def __init__(self, raw_dir: Path = settings.DATA_RAW_DIR):
        self.raw_dir = raw_dir

    def read_csv(self, filename: str, dtype_schema: Optional[dict] = None, date_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Lee un archivo CSV garantizando correcta inferencia de tipos base alineados al schema relacional.
        
        Args:
            filename (str): Nombre del archivo a cargar.
            dtype_schema (Optional[dict]): Diccionario con los tipos de datos estrictos (mapping equivalente a schema_postgresql).
            date_cols (Optional[List[str]]): Lista de columnas a convertir a datetime64.
            
        Returns:
            pd.DataFrame: DataFrame crudo pero con tipos y fechas corregidas.
            
        Raises:
            FileNotFoundError: Si el archivo especificado no existe.
        """
        file_path = self.raw_dir / filename
        try:
            logger.info(f"Iniciando carga de archivo: {file_path}")
            # Se fuerza el casteo desde la ingesta para respetar el esquema DDL Original
            df = pd.read_csv(file_path, dtype=dtype_schema)
            
            if date_cols:
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"Columna '{col}' casteada a datetime64[ns].")
                    else:
                        logger.warning(f"La columna '{col}' no existe en {filename}.")
                        
            logger.info(f"Archivo {filename} cargado exitosamente. Forma: {df.shape}")
            return df
            
        except FileNotFoundError:
            logger.error(f"El archivo no existe en el path indicado: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error inesperado al leer el archivo {file_path}: {e}")
            raise

    def load_all(self) -> dict:
        """
        Carga todos los archivos CSV del proyecto en un diccionario de DataFrames.

        Returns:
            dict: {'clientes': df, 'creditos': df, 'pagos': df, 'eventos': df}
        """
        logger.info("Cargando todos los archivos del dataset...")
        return {
            'clientes': self.read_csv(
                settings.CLIENTES_FILE,
                date_cols=['fecha_registro'],
            ),
            'creditos': self.read_csv(
                settings.CREDITOS_FILE,
                date_cols=['fecha_desembolso'],
            ),
            'pagos': self.read_csv(
                settings.PAGOS_FILE,
                date_cols=['fecha_vencimiento', 'fecha_pago'],
            ),
            'eventos': self.read_csv(
                'eventos_app.csv',
                date_cols=['fecha_evento'],
            ),
        }

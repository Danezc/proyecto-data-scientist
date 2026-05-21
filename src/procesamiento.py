import logging
import pandas as pd

logger = logging.getLogger(__name__)

class DataCleaner:
    """
    Se encarga de la limpieza de datos por entidad, tratando valores nulos e imputando outliers evidentes.
    Al encapsularse en métodos independientes, aseguramos modularidad para orquestadores.
    """
    
    def __init__(self):
        pass
        
    def clean_clientes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza de la tabla 'clientes'. 
        Imputa nulos en variables numéricas demográficas, corrige texto y remueve valores extremos.
        """
        logger.info("Iniciando procesamiento y limpieza de tabla 'clientes'.")
        df_cleaned = df.copy()
        
        # Imputación de ingresos por mediana
        if 'ingreso_mensual_estimado' in df_cleaned.columns:
            median_ingresos = df_cleaned['ingreso_mensual_estimado'].median()
            df_cleaned['ingreso_mensual_estimado'] = df_cleaned['ingreso_mensual_estimado'].fillna(median_ingresos)
            logger.info("Imputados valores nulos en 'ingreso_mensual_estimado' utilizando la mediana.")
            
        # Imputación de score_externo por mediana
        if 'score_externo' in df_cleaned.columns:
            median_score = df_cleaned['score_externo'].median()
            df_cleaned['score_externo'] = df_cleaned['score_externo'].fillna(median_score)
            logger.info("Imputados valores nulos en 'score_externo' utilizando la mediana.")

        # Estandarización de ciudad (mapeo robusto para inconsistencias)
        if 'ciudad' in df_cleaned.columns:
            ciudad_map = {
                'bogot': 'Bogotá', 'bogota': 'Bogotá', 'bogota d.c.': 'Bogotá', 'bogotá d.c.': 'Bogotá', 'bogota dc': 'Bogotá', 'bogotá dc': 'Bogotá',
                'medellin ': 'Medellín', 'medellin': 'Medellín', 'medellín': 'Medellín',
                'barranquila': 'Barranquilla', 'barranquilla': 'Barranquilla',
                'cali': 'Cali', 'ibagué': 'Ibagué', 'soacha': 'Soacha', 'villavicencio': 'Villavicencio',
                'manizales': 'Manizales', 'pereira': 'Pereira', 'bucaramanga': 'Bucaramanga',
                'cúcuta': 'Cúcuta', 'cartagena': 'Cartagena'
            }
            df_cleaned['ciudad'] = df_cleaned['ciudad'].astype(str).str.strip().str.lower().map(ciudad_map).fillna(df_cleaned['ciudad'].str.title())
            logger.info("Ciudad estandarizada: variantes y errores ortográficos unificados.")
            
        # Corrección de outliers por límite lógico
        if 'edad' in df_cleaned.columns:
            # Reemplaza edades menores a los 18 o sin sentido con la mediana
            invalid_age_mask = (df_cleaned['edad'] < 18) | (df_cleaned['edad'] > 110)
            if invalid_age_mask.any():
                med_edad = df_cleaned['edad'].median()
                df_cleaned.loc[invalid_age_mask, 'edad'] = med_edad
                logger.info(f"Corregidos {invalid_age_mask.sum()} outliers en 'edad'.")
                
        return df_cleaned

    def clean_creditos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza de la tabla 'creditos'.
        """
        logger.info("Iniciando procesamiento y limpieza de tabla 'creditos'.")
        df_cleaned = df.copy()
        
        # Imputa nulos en tipo de producto
        if 'producto_credito' in df_cleaned.columns:
            df_cleaned['producto_credito'] = df_cleaned['producto_credito'].fillna('Desconocido')
            logger.info("Imputados valores nulos en 'producto_credito' como 'Desconocido'.")
            
        # Elimina registros donde la llave base o el monto esté completamente ausente
        required_cols = ['credito_id', 'cliente_id', 'monto_credito']
        cols_present = [c for c in required_cols if c in df_cleaned.columns]
        
        initial_len = len(df_cleaned)
        df_cleaned = df_cleaned.dropna(subset=cols_present)
        
        if initial_len != len(df_cleaned):
            logger.info(f"Eliminados {initial_len - len(df_cleaned)} registros en 'creditos' nulos.")
            
        return df_cleaned
        
    def clean_pagos(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpieza de la tabla 'pagos'.
        """
        logger.info("Iniciando procesamiento y limpieza de tabla 'pagos'.")
        df_cleaned = df.copy()
        
        if 'monto_pago' in df_cleaned.columns:
            df_cleaned['monto_pago'] = df_cleaned['monto_pago'].fillna(0.0)
        elif 'valor_pagado' in df_cleaned.columns:
            df_cleaned['valor_pagado'] = df_cleaned['valor_pagado'].fillna(0.0)
            
        if 'medio_pago' in df_cleaned.columns:
            df_cleaned['medio_pago'] = df_cleaned['medio_pago'].fillna('Desconocido')
            
        if 'dias_mora' in df_cleaned.columns:
            # Evita periodos de mora negativos (inconsistencias por pagos adelantados)
            df_cleaned.loc[df_cleaned['dias_mora'] < 0, 'dias_mora'] = 0
            
        return df_cleaned

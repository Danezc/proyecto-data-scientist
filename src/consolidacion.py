import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

class DataConsolidator:
    """
    Agrupa la lógica de negocio final.
    Calcula variables objetivo y características predictivas previniendo el "data leakage".
    """
    
    def __init__(self, cutoff_date: str):
        """
        Inicializa con una fecha de corte para prevenir fuga de información y
        asegurar backtesting temporal robusto.
        
        Args:
            cutoff_date (str): Fecha de corte en formato YYYY-MM-DD.
        """
        self.cutoff_date = pd.to_datetime(cutoff_date)
        logger.info(f"DataConsolidator inicializado con fecha de corte (cutoff): {self.cutoff_date.date()}")

    def calculate_target(self, df_pagos: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula la variable target 'es_moroso'.
        Definición de negocio: 1 si registró max(dias_mora) > 30 antes o igual a la fecha de corte.
        """
        logger.info("Calculando variable 'es_moroso' a partir del histórico de pagos.")
        
        df_eval = df_pagos.copy()
        # Filtro estricto para fuga de información (data leakage):
        # Solo analizar morosidad evidenciada HASTA la fecha de corte.
        if 'fecha_pago' in df_eval.columns:
            df_eval = df_eval[pd.to_datetime(df_eval['fecha_pago']) <= self.cutoff_date]
            
        if 'dias_mora' not in df_eval.columns:
            raise ValueError("El DataFrame de pagos carece de la métrica 'dias_mora'.")
            
        # Agregación para target: si alguna vez tuvo más de 30 días de mora
        target_df = df_eval.groupby('credito_id')['dias_mora'].max().reset_index()
        target_df['es_moroso'] = (target_df['dias_mora'] > 30).astype(int)
        
        return target_df[['credito_id', 'es_moroso']]

    def build_analytical_base_table(
        self, 
        df_clientes: pd.DataFrame, 
        df_creditos: pd.DataFrame, 
        df_pagos: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Construye la Tabla Analítica Final (ABT) acoplando clientes, créditos y el target.
        ESTRICTO ANTI-DATA LEAKAGE: Utilizamos solo demografía (clientes) y originación (creditos) como predictores.
        Los pagos solo se usan para definir la variable objetivo histórica, NO como predictors.
        """
        logger.info("Construyendo la Analytical Base Table (ABT) consolidada.")
        
        # 1. Definir la etiqueta (Target) a la fecha de corte
        target_df = self.calculate_target(df_pagos)
        
        # 2. Unir créditos (Originación) con el target
        # Todo cliente con crédito entra, con o sin mora
        abt = pd.merge(df_creditos, target_df, on='credito_id', how='left')
        
        # Un crédito sin info de pagos será NO moroso por defecto (0)
        abt['es_moroso'] = abt['es_moroso'].fillna(0.0).astype(int) 
        
        # 3. Unir variables demográficas de la base Cliente (Socioeconómicas)
        abt = pd.merge(abt, df_clientes, on='cliente_id', how='left')
        
        logger.info(f"ABT Final creada exitosamente con {abt.shape[0]} filas y {abt.shape[1]} columnas.")
        return abt

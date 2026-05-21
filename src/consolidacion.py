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
        Definición de negocio: 1 si registró max(dias_mora) > 30 en cuotas cuya fecha de vencimiento es previa o igual al corte.
        """
        logger.info("Calculando variable 'es_moroso' a partir del histórico de pagos (excluyendo vencimientos futuros).")
        
        df_eval = df_pagos.copy()
        
        # Filtro correcto anti-leakage: solo evaluar cuotas vencidas antes o igual a la fecha de corte
        if 'fecha_vencimiento' in df_eval.columns:
            df_eval = df_eval[pd.to_datetime(df_eval['fecha_vencimiento']) <= self.cutoff_date]
            
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
        df_pagos: pd.DataFrame,
        df_eventos: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Construye la Tabla Analítica Final (ABT) acoplando clientes, créditos, eventos digitales y target.
        ESTRICTO ANTI-DATA LEAKAGE:
        1. Los pagos futuros a la fecha de corte se excluyen del target.
        2. Los eventos digitales posteriores a la fecha de desembolso de cada crédito se excluyen de las features.
        3. Excluimos variables como estado_credito_operativo en el entrenamiento del modelo.
        """
        logger.info("Construyendo la Analytical Base Table (ABT) consolidada.")
        
        # 1. Definir la etiqueta (Target) a la fecha de corte
        target_df = self.calculate_target(df_pagos)
        
        # 2. Unir créditos (Originación) con el target
        abt = pd.merge(df_creditos, target_df, on='credito_id', how='left')
        abt['es_moroso'] = abt['es_moroso'].fillna(0).astype(int) 
        
        # 3. Unir variables demográficas de la base Cliente
        abt = pd.merge(abt, df_clientes, on='cliente_id', how='left')
        
        # 4. Ingeniería de características de originación
        # Tenure de la relación (días desde registro hasta desembolso)
        if 'fecha_desembolso' in abt.columns and 'fecha_registro' in abt.columns:
            abt['antiguedad_cliente_dias'] = (pd.to_datetime(abt['fecha_desembolso']) - pd.to_datetime(abt['fecha_registro'])).dt.days
            
            # Estacionalidad del desembolso
            abt['mes_desembolso'] = pd.to_datetime(abt['fecha_desembolso']).dt.month
            abt['dia_semana_desembolso'] = pd.to_datetime(abt['fecha_desembolso']).dt.dayofweek
            
        # 5. Agregar comportamiento digital previo al desembolso (si viene df_eventos)
        if df_eventos is not None and not df_eventos.empty:
            logger.info("Realizando ingeniería de características sobre eventos_app...")
            # Unir créditos con eventos para filtrar cronológicamente
            events_merged = pd.merge(
                df_creditos[['credito_id', 'cliente_id', 'fecha_desembolso']],
                df_eventos,
                on='cliente_id',
                how='inner'
            )
            # Solo mantener eventos ocurridos ANTES del desembolso (Anti-leakage estricto)
            events_prev = events_merged[pd.to_datetime(events_merged['fecha_evento']) < pd.to_datetime(events_merged['fecha_desembolso'])]
            
            if not events_prev.empty:
                # Contar eventos por tipo por crédito
                event_counts = events_prev.groupby(['credito_id', 'tipo_evento']).size().unstack(fill_value=0).reset_index()
                event_cols = [c for c in event_counts.columns if c != 'credito_id']
                event_counts = event_counts.rename(columns={c: f'prev_evento_{c}' for c in event_cols})
                
                # Estadísticas de duración de sesión de eventos anteriores
                session_time = events_prev.groupby('credito_id')['duracion_sesion_seg'].agg(['sum', 'mean']).reset_index()
                session_time = session_time.rename(columns={'sum': 'prev_evento_sesion_seg_tot', 'mean': 'prev_evento_sesion_seg_avg'})
                
                # Integrar en la ABT
                abt = pd.merge(abt, event_counts, on='credito_id', how='left')
                abt = pd.merge(abt, session_time, on='credito_id', how='left')
                
                # Rellenar con 0 si el cliente no tenía eventos previos
                new_event_cols = [f'prev_evento_{c}' for c in event_cols] + ['prev_evento_sesion_seg_tot', 'prev_evento_sesion_seg_avg']
                for col in new_event_cols:
                    if col in abt.columns:
                        abt[col] = abt[col].fillna(0)
                        
            logger.info("Características de comportamiento digital agregadas exitosamente.")

        logger.info(f"ABT Final creada exitosamente con {abt.shape[0]} filas y {abt.shape[1]} columnas.")
        return abt

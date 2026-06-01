import logging
import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config import settings

logger = logging.getLogger(__name__)

class MoraModelTrainer:
    """
    Pipeline de entrenamiento de modelo predictivo de riesgo de mora empleando LightGBM.
    """
    
    def __init__(self, data_path: Path, model_output_path: Path):
        self.data_path = data_path
        self.model_output_path = model_output_path
        # Crear directorios si no existen
        self.model_output_path.parent.mkdir(parents=True, exist_ok=True)
        
    def load_data(self) -> pd.DataFrame:
        """Carga la Tabla Analítica Final (ABT)."""
        logger.info(f"Cargando ABT desde {self.data_path}")
        if self.data_path.suffix == '.csv':
            return pd.read_csv(self.data_path)
        return pd.read_parquet(self.data_path)
        
    def train_test_split_custom(self, df: pd.DataFrame, target_col: str = 'es_moroso') -> Tuple:
        """
        Divide los datos en entrenamiento y prueba asegurando validación robusta y estratificada.
        Elimina columnas datetime, variables con fuga (leakage) e identificadores irrelevantes.
        Codifica strings como 'category' para LightGBM.
        """
        # Columnas de ID/target que no son predictores
        id_cols = {'cliente_id', 'credito_id', 'pago_id', 'email_hash', target_col}
        
        # Filtro estricto anti-data leakage:
        # 'estado_credito_operativo' refleja la morosidad ocurrida después de la originación.
        leakage_cols = {'estado_credito_operativo', 'probabilidad_mora', 'prediccion_mora'}

        # Excluir también todas las columnas datetime (fechas sin feature-engineering)
        date_cols = set(df.select_dtypes(include=['datetime', 'datetime64']).columns.tolist())
        # También buscar nombres de fecha si son de tipo object pero no convertidos
        for col in df.columns:
            if 'fecha' in col:
                date_cols.add(col)

        drop_cols = id_cols | leakage_cols | date_cols
        features = [c for c in df.columns if c not in drop_cols]

        X = df[features].copy()
        y = df[target_col]

        # Convertir strings a 'category' → LightGBM los maneja nativamente sin one-hot
        # include=['object', 'str'] cubre pandas 2 (object) y pandas 3 (StringDtype)
        for col in X.select_dtypes(include=['object', 'str', 'category']).columns:
            X[col] = X[col].astype('category')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        logger.info(f"Dataset particionado. Train={X_train.shape[0]}, Test={X_test.shape[0]}.")
        logger.info(f"Features utilizadas en el modelo ({len(features)}): {features}")
        return X_train, X_test, y_train, y_test

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> LGBMClassifier:
        """Entrena el clasificador LightGBM con parámetros de regularización."""
        logger.info("Entrenando modelo LightGBM...")
        model = LGBMClassifier(
            objective='binary',
            class_weight='balanced',
            n_estimators=80,
            learning_rate=0.03,
            max_depth=4,
            num_leaves=15,
            min_child_samples=30,
            random_state=42
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model: LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series):
        """Calcula métricas clave previendo escenarios de mora."""
        logger.info("Evaluando el modelo...")
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        logger.info("\nMatriz de Confusión:\n" + str(confusion_matrix(y_test, y_pred)))
        logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
        logger.info(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
        
        # Feature Importance
        feature_importances = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)
        logger.info(f"\nTop 5 Variables más importantes:\n{feature_importances.head(5).to_string(index=False)}")
        
    def export_model(self, model: LGBMClassifier, feature_names: list):
        """Exporta el modelo entrenado y los feature names necesarios para inferencia."""
        artifact = {
            'model': model,
            'features': feature_names
        }
        joblib.dump(artifact, self.model_output_path)
        logger.info(f"Modelo exportado en {self.model_output_path}")

    def run_pipeline(self):
        """Ejecuta el pipeline completo de entrenamiento."""
        df = self.load_data()
        X_train, X_test, y_train, y_test = self.train_test_split_custom(df)
        model = self.train_model(X_train, y_train)
        self.evaluate_model(model, X_test, y_test)
        self.export_model(model, list(X_train.columns))

if __name__ == "__main__":
    abt_path = settings.DATA_PROCESSED_DIR / "abt.csv"
    model_path = settings.MODELS_DIR / "modelo_mora.pkl"
    trainer = MoraModelTrainer(abt_path, model_path)
    trainer.run_pipeline()

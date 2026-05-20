# Usar versión slim para un contenedor más ligero en producción
FROM python:3.10-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema operativo (libgomp1 es fundamental para LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copiar requerimientos y cachear capa en Docker
# Se asume que requirements.txt existe, si no, se instalan directamente abajo
COPY requirements.txt .

# Instalar librerías
RUN pip install --no-cache-dir -r requirements.txt || \
    pip install --no-cache-dir fastapi uvicorn pydantic pydantic-settings pandas scikit-learn lightgbm openai joblib pyarrow

# Copiar el código del proyecto
COPY ./src ./src

# Exponer el puerto de Uvicorn
EXPOSE 8000

# Variable de entorno de Python
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Levantar el servidor FastAPI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]

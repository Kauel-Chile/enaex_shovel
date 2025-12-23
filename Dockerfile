# Etapa 1: Usamos una imagen base ligera de Python
# Asegúrate de que la versión coincida con tu .python-version (ej. 3.11 o 3.10)
FROM python:3.11-slim-bookworm

# Copiamos el binario de uv desde la imagen oficial de Astral
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Configuraciones de entorno para Python
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Definimos el directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- CAPA DE DEPENDENCIAS (Para aprovechar caché) ---

# Copiamos solo los archivos de configuración de dependencias primero
COPY pyproject.toml uv.lock ./

# Instalamos las dependencias
# --frozen: Asegura que se instalen las versiones exactas del uv.lock
# --no-install-project: No instalamos el código fuente todavía, solo dependencias
RUN uv sync --frozen --no-install-project

# --- CAPA DE CÓDIGO ---

# Copiamos el resto del código del proyecto
COPY . .

# (Opcional) Si tu proyecto necesita instalarse a sí mismo como paquete, descomenta:
# RUN uv sync --frozen

# Añadimos el entorno virtual al PATH
# uv crea el entorno en .venv por defecto dentro del contenedor
ENV PATH="/app/.venv/bin:$PATH"

# Comando de inicio
# Como ya añadimos el venv al PATH, podemos llamar a python directamente
CMD ["python", "main.py"]
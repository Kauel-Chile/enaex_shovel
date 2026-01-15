#!/bin/bash

# Configuración
ACR="acrfragnexbackend"
ENAEX_SUBSCRIPTION="Enaex TD"
IMAGE_NAME="ia_worker"
IMAGE_TAG="1.0.4"

# 1. Verificar Login de Azure
echo "Verificando sesión de Azure..."
if ! az account show &> /dev/null; then
    echo "No iniciado. Iniciando sesión..."
    az login
fi

# 2. Configurar Suscripción
echo "Configurando suscripción: $ENAEX_SUBSCRIPTION"
if ! az account set --subscription "$ENAEX_SUBSCRIPTION"; then
    echo "Error: No se pudo encontrar la suscripción $ENAEX_SUBSCRIPTION"
    exit 1
fi

# 3. Login en el ACR (Forma recomendada)
# Este comando configura automáticamente tu config.json de Docker 
# con las credenciales temporales del registro.
echo "Iniciando sesión en ACR: $ACR"
if ! az acr login --name "$ACR"; then
    echo "Error: No se pudo autenticar con el ACR"
    exit 1
fi

# 4. Construir la imagen
echo "Construyendo imagen $IMAGE_NAME:$IMAGE_TAG..."
docker build . -t "$IMAGE_NAME:$IMAGE_TAG"

# 5. Taggear para ACR
FULL_IMAGE_NAME="$ACR.azurecr.io/$IMAGE_NAME:$IMAGE_TAG"
echo "Taggeando imagen como $FULL_IMAGE_NAME..."
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$FULL_IMAGE_NAME"

# 6. Push a Azure
echo "Subiendo imagen a Azure..."
if docker push "$FULL_IMAGE_NAME"; then
    echo "✅ Éxito: Imagen subida correctamente."
else
    echo "❌ Error: Falló el push a ACR."
    exit 1
fi
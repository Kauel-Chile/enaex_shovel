"""
Nodos del pipeline para la interacción con Azure Blob Storage.

Este módulo contiene los nodos responsables de descargar los datos de entrada
(imágenes) desde un contenedor de blobs y de subir los artefactos de resultado
(visualizaciones, datos JSON, etc.) al mismo u otro contenedor.
"""
import io
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np
from azure.storage.blob import BlobServiceClient
from matplotlib.figure import Figure
from typing import Any, Dict

from src.nodes.base import PipelineNode


class BlobStorageUploaderNode(PipelineNode):
    """
    Sube artefactos generados (como figuras) a un contenedor de Azure Blob Storage.
    """
    def __init__(
        self,
        blob_service_client: BlobServiceClient,
        container_name: str,
        img_format: str = "png",
        dpi: int = 300,
        name: str = "blob_uploader_node"
    ):
        """
        Inicializa el nodo de subida.

        Args:
            blob_service_client (BlobServiceClient): Cliente de Blob Storage
                pre-inicializado.
            container_name (str): Nombre del contenedor de destino.
            img_format (str): Formato de la imagen a guardar (e.g., 'png', 'jpg').
            dpi (int): Resolución en puntos por pulgada para la imagen guardada.
            name (str): Nombre del nodo.
        """
        super().__init__(name)
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self.img_format = img_format
        self.dpi = dpi

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma una figura de Matplotlib del contexto y la sube como imagen a Blob Storage.

        Context Inputs:
            - `output_figure` (matplotlib.figure.Figure): La figura a subir.
            - `metadata.filename` (str): Nombre base para el archivo de salida.

        Context Outputs:
            - `uploaded_url` (str): La URL pública del blob subido.

        Side Effects:
            - Cierra la figura de Matplotlib para liberar memoria.
            - Elimina `output_figure` del contexto.
        """
        fig: Figure = context.get('output_figure')
        if fig is None:
            logging.warning("[%s] No se encontró 'output_figure' en el contexto. Omitiendo subida.", self.name)
            return context

        metadata = context.get('metadata', {})
        base_name = metadata.get('filename', 'unnamed_analysis')
        if '.' in base_name:
            base_name = base_name.rsplit('.', 1)[0]
        blob_name = f"{base_name}_result.{self.img_format}"

        try:
            logging.info("[%s] Preparando la subida de: %s", self.name, blob_name)
            fig_stream = io.BytesIO()
            fig.savefig(fig_stream, dpi=self.dpi, format=self.img_format, bbox_inches="tight")
            fig_stream.seek(0)

            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name,
            )
            blob_client.upload_blob(fig_stream.read(), overwrite=True)

            context['uploaded_url'] = blob_client.url
            logging.info("[%s] Subida exitosa a: %s", self.name, blob_client.url)

        except Exception as e:
            logging.error("[%s] Error al subir la figura: %s", self.name, e, exc_info=True)
            raise
        finally:
            # Es crucial cerrar la figura para liberar memoria en un servicio de larga duración.
            plt.close(fig)
            if 'output_figure' in context:
                del context['output_figure']

        return context


class BlobStorageDownloaderNode(PipelineNode):
    """
    Descarga un archivo (imagen) desde Azure Blob Storage y lo carga como un
    array de numpy.
    """
    def __init__(
        self,
        blob_service_client: BlobServiceClient,
        container_name: str,
        input_key: str = "image_url",
        output_key: str = "image",
        name: str = "blob_downloader"
    ):
        """
        Inicializa el nodo de descarga.

        Args:
            blob_service_client (BlobServiceClient): Cliente de Blob Storage
                pre-inicializado.
            container_name (str): Nombre del contenedor de donde se descargará.
            input_key (str): Clave en el contexto que contiene la URL/nombre del blob.
            output_key (str): Clave en el contexto donde se guardará la imagen cargada.
            name (str): Nombre del nodo.
        """
        super().__init__(name)
        self.blob_service_client = blob_service_client
        self.container_name = container_name
        self.input_key = input_key
        self.output_key = output_key

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Toma la URL de una imagen del contexto, la descarga y la decodifica.

        Context Inputs:
            - `context[self.input_key]` (str): La URL completa o el nombre del
              blob a descargar.

        Context Outputs:
            - `context[self.output_key]` (np.ndarray): La imagen decodificada
              en formato OpenCV (BGR).
            - `original_shape` (tuple): Tupla (ancho, alto) de la imagen.
            - `metadata.filename` (str): El nombre del archivo extraído del blob.
        """
        source_path = context.get(self.input_key)
        if not source_path:
            raise ValueError(f"[{self.name}] El contexto no contiene la clave de entrada '{self.input_key}'.")

        blob_name = source_path.split("/")[-1]
        logging.info("[%s] Descargando blob: %s", self.name, blob_name)

        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name,
                blob=blob_name
            )

            download_stream = blob_client.download_blob()
            image_bytes = download_stream.readall()
            np_arr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError(f"OpenCV no pudo decodificar los bytes de la imagen para el blob '{blob_name}'.")

            context[self.output_key] = image
            h, w = image.shape[:2]
            context['original_shape'] = (w, h)
            
            if 'metadata' not in context:
                context['metadata'] = {}
            context['metadata']['filename'] = blob_name
            logging.info("[%s] Descarga y decodificación completadas (%dx%d).", self.name, w, h)

        except Exception as e:
            logging.error("[%s] Error al descargar o decodificar '%s': %s", self.name, blob_name, e, exc_info=True)
            raise

        return context
"""
Define el pipeline principal de granulometría, que orquesta la secuencia de
pasos de procesamiento.
"""
import logging
import time
from typing import Any, Dict, List

from azure.storage.blob import BlobServiceClient

from config.settings import STORAGE_CONTAINER_NAME

from src.domain import CameraParameters
from src.domain.granulometry.models import RosinRammler
from src.nodes.base import PipelineNode
from src.nodes.blobstorage import BlobStorageDownloaderNode, BlobStorageUploaderNode
from src.nodes.granulometry import (FixedDistanceTransformer,
                                    GranulometryModelNode, GranulometryStatsNode)
from src.nodes.ia import OnnxInferenceNode, OnnxPostProcessingNode
from src.nodes.viz import ResultVisualizerNode


class GranulometryPipeline:
    """
    Orquesta la ejecución de la secuencia de análisis de granulometría.

    Esta clase encapsula la lógica de negocio completa. Se instancia una vez
    al inicio del servicio (ver `main.py`) y reutiliza recursos pesados como
    clientes de Azure y el modelo ONNX a través de inyección de dependencias.

    Su función principal es definir la cadena de nodos de procesamiento y
    ejecutarlos en orden para cada trabajo recibido.
    """
    def __init__(self, 
                 blob_service_client: BlobServiceClient, 
                 model_onnx: Any,
                 camera_params: CameraParameters = None):
        """
        Inicializa el pipeline de granulometría.

        Args:
            blob_service_client (BlobServiceClient): Cliente de Azure Blob
                Storage ya inicializado para ser compartido entre los nodos.
            model_onnx: Sesión de inferencia de ONNX ya cargada en memoria.
        """
        self.blob_service_client = blob_service_client
        self.onnx_session = model_onnx
        self.camera_params = camera_params 
        
        # Construye la secuencia de nodos que componen el pipeline.
        self.nodes: List[PipelineNode] = self._build_pipeline()
        logging.info("Pipeline de granulometría construido con %d nodos.", len(self.nodes))

    def _build_pipeline(self) -> List[PipelineNode]:
        """
        Define la secuencia de pasos de procesamiento del pipeline.

        Cada paso es un 'PipelineNode' que se instancia aquí, inyectando las
        dependencias necesarias (clientes, modelos, configuración). El orden
        en esta lista define el flujo de ejecución.

        Returns:
            List[PipelineNode]: Una lista ordenada de nodos listos para ser ejecutados.
        """
        return [
            # 1. Descargar la imagen desde Azure Blob Storage.
            BlobStorageDownloaderNode(
                blob_service_client=self.blob_service_client,
                container_name=STORAGE_CONTAINER_NAME,
                name="Downloader"
            ),
            # 2. Ejecutar la inferencia con el modelo ONNX para segmentar la imagen.
            OnnxInferenceNode(
                onnx_session=self.onnx_session,
                name="IA_Inference"
            ),
            # 3. Post-procesar la máscara de segmentación.
            OnnxPostProcessingNode(name="PostProcessing"),
            # 4. Transformar los contornos de píxeles a objetos 'Rock' con medidas reales.
            FixedDistanceTransformer(
                camera_params=self.camera_params,
                name="PolygonsToRocks"
            ),
            # 5. Calcular estadísticas de distribución de tamaños (histograma).
            GranulometryStatsNode(
                nbins=30, 
                name="Statistics_Raw"
            ),
            # 6. Ajustar un modelo matemático (Rosin-Rammler) a la distribución.
            GranulometryModelNode(
                model_strategy=RosinRammler(), 
                name="Statistics_Modeled"
            ),
            # 7. Generar visualizaciones de los resultados.
            ResultVisualizerNode(name="Visualization"),
            # 8. Subir los artefactos resultantes (imágenes, JSON) a Blob Storage.
            BlobStorageUploaderNode(
                blob_service_client=self.blob_service_client,
                container_name=STORAGE_CONTAINER_NAME,
                img_format="png",
                dpi=300,
                name="Uploader"
            )
        ]

    def run(self, initial_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta la secuencia completa de nodos del pipeline sobre un contexto dado.

        Args:
            initial_context (Dict[str, Any]): El contexto inicial del trabajo.
                Debe contener como mínimo una clave 'image_url'.

        Returns:
            Dict[str, Any]: El contexto final, enriquecido con los resultados
            de todos los nodos del pipeline, incluyendo los tiempos de ejecución.

        Raises:
            Exception: Si cualquier nodo del pipeline falla, la excepción se
                       propaga hacia arriba para ser gestionada por el Worker.
        """
        context = initial_context.copy()
        context['execution_times'] = {} 

        job_id = context.get('job_id', f"job_{int(time.time())}")
        logging.info(">>> Iniciando JOB: %s", job_id)
        
        total_start_time = time.perf_counter()

        for node in self.nodes:
            node_start_time = time.perf_counter()
            try:
                # Cada nodo recibe el contexto, lo modifica y lo retorna.
                context = node.run(context)
                
                # Registra el tiempo de ejecución para monitoreo.
                duration = time.perf_counter() - node_start_time
                context['execution_times'][node.name] = duration
                
            except Exception as e:
                logging.error(
                    "!!! Error en nodo '%s' (Job: %s): %s",
                    node.name, job_id, e, exc_info=True
                )
                raise

        total_duration = time.perf_counter() - total_start_time
        context['execution_times']['total_pipeline'] = total_duration
        
        logging.info("<<< JOB %s finalizado en %.4f segundos.", job_id, total_duration)
        
        return context
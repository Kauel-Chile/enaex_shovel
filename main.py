import logging
import sys
from azure.storage.blob import BlobServiceClient

from config.settings import STORAGE_CONN_STR, QUEUE_CONN_STR, QUEUE_INPUT_NAME, PATH_MODEL
from src.utils import load_model, load_camera_params
from src.worker import Worker

# Configuración de Logging Global
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
# Silenciamos logs ruidosos de librerías externas
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("uamqp").setLevel(logging.WARNING)

if __name__ == "__main__":
    logging.info("Iniciando servicio de Granulometría...")

    # Verificación crítica de configuración
    if not STORAGE_CONN_STR or not QUEUE_CONN_STR:
        logging.error("Faltan variables de entorno críticas (Storage o Service Bus).")
        sys.exit(1)

    worker = None
    try:
        logging.info("Cargando modelos y parámetros...")
        model = load_model(PATH_MODEL)
        camera_params = load_camera_params()
        blob_service_client = BlobServiceClient.from_connection_string(STORAGE_CONN_STR)
        
        worker = Worker(
            sb_conn_str=QUEUE_CONN_STR,
            queue_name=QUEUE_INPUT_NAME,
            blob_service_client=blob_service_client,
            model_onnx=model, 
            camera_params=camera_params
        )

        worker.start()

    except KeyboardInterrupt:
        logging.warning("\nInterrupción manual detectada.")
        if worker:
            worker.stop()
        
    except Exception as e:
        logging.critical(f"Error fatal en el ciclo principal: {e}", exc_info=True)
        if worker:
            worker.stop()
        sys.exit(1)